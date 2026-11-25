// APEX Blackwell Next-Gen — Two-Regime Zero-HBM-Traffic Collective-GEMM Fusion (4/8/16-GPU IMEX)
// Regime 1: SMEM-Coalesced `Epi-NVLS-TilePipe` (`multimem.red.v4.bf16x2` + per-strip arrival counter)
// Regime 2: Tile-Sharded `Epi-P2P-RS` + Deterministic FP32 Local Sum + Single-Writer `multimem.st` (0 XR16 diffs)
// Target: NVIDIA Blackwell GB200 NVL72 (sm_100a, CUDA 13.0+, NVSwitch Gen5 / IMEX FABRIC)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace apex_blackwell {

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)((smem_ptr & 0x3FFFFu) >> 4);
    d |= ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16);
    d |= ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32);
    d |= (1ULL << 46);
    return d;
}

__host__ __device__ __forceinline__ uint32_t make_idesc_bf16_128x128() {
    return (1u << 4) | (1u << 7) | (1u << 10) | (16u << 17) | (8u << 24);
}

__global__ void k_nvls_barrier(uint32_t* mc_f, uint32_t* uc_f, uint32_t* ep, uint32_t W) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t tgt = (*ep) + W;
        *ep = tgt;
        asm volatile("fence.release.sys;" ::: "memory");
        asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" :: "l"(mc_f), "r"(1u) : "memory");
        uint32_t v = 0;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(uc_f) : "memory");
        } while (v < tgt);
    }
}

// EPI_MODE: 0=LocalHBM, 1=Epi-NVLS (Regime 1), 2=Epi-P2P-RS (Regime 2)
template <int EPI_MODE>
__global__ __launch_bounds__(128, 1)
void k_tma_tcgen05_two_regime(
    const uint8_t* __restrict__ A_packed,
    const uint8_t* __restrict__ B_packed,
    void* __restrict__ dst_out,
    const uint64_t* __restrict__ peer_va,
    uint32_t* __restrict__ mc_row_flag,
    int M, int N, int K, int my_rank, int W
) {
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* smA[2] = { smem, smem + 32768 };
    uint8_t* smB[2] = { smem + 16384, smem + 49152 };
    uint8_t* smEpi  = smem + 65536;
    __shared__ uint32_t tmem_base;
    __shared__ uint64_t mbar_tma[2], mbar_mma;
    uint32_t tid = threadIdx.x, warp = tid >> 5;
    uint32_t mt0 = (uint32_t)__cvta_generic_to_shared(&mbar_tma[0]);
    uint32_t mt1 = (uint32_t)__cvta_generic_to_shared(&mbar_tma[1]);
    uint32_t mm  = (uint32_t)__cvta_generic_to_shared(&mbar_mma);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mt0));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mt1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mm));
    }
    if (warp == 0) asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" :: "l"(&tmem_base));
    __syncthreads();

    uint32_t idesc = make_idesc_bf16_128x128();
    int tiles_m = (M + 127) / 128, tiles_n = N / 128, total_tiles = tiles_m * tiles_n;
    int tn_per_rank = tiles_n / W, N_shard = N / W, k_steps = K / 64;
    uint32_t smA_a[2] = { (uint32_t)__cvta_generic_to_shared(smA[0]), (uint32_t)__cvta_generic_to_shared(smA[1]) };
    uint32_t smB_a[2] = { (uint32_t)__cvta_generic_to_shared(smB[0]), (uint32_t)__cvta_generic_to_shared(smB[1]) };
    uint32_t mt_a[2]  = { mt0, mt1 };
    int tma_par[2] = {0, 0}, mma_par = 0;

    auto issue_tma_k = [&](int tm, int tn, int ki, int stg) {
        if (tid == 0) {
            const void* gA = A_packed + ((size_t)tm * k_steps + ki) * 16384ULL;
            const void* gB = B_packed + ((size_t)tn * k_steps + ki) * 16384ULL;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 32768;" :: "r"(mt_a[stg]));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 16384, [%2];"
                         :: "r"(smA_a[stg]), "l"(gA), "r"(mt_a[stg]) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 16384, [%2];"
                         :: "r"(smB_a[stg]), "l"(gB), "r"(mt_a[stg]) : "memory");
        }
    };
    int shift_tn = my_rank * tn_per_rank;

    for (int t_idx = blockIdx.x; t_idx < total_tiles; t_idx += gridDim.x) {
        int tm = t_idx / tiles_n, tn = (t_idx % tiles_n + shift_tn) % tiles_n;
        int row_base = tm * 128, col_base = tn * 128;

        issue_tma_k(tm, tn, 0, 0);
        for (int ki = 0; ki < k_steps; ki++) {
            int cur = ki & 1, nxt = cur ^ 1;
            if (ki + 1 < k_steps) issue_tma_k(tm, tn, ki + 1, nxt);
            if (tid == 0) {
                uint32_t m_cur = mt_a[cur]; int pr = tma_par[cur];
                asm volatile("{\n .reg .pred P1;\n L_TK:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @!P1 bra L_TK;\n}\n"
                             :: "r"(m_cur), "r"(pr));
                tma_par[cur] ^= 1;
                #pragma unroll
                for (int ks = 0; ks < 4; ks++) {
                    uint64_t da = make_smem_desc(smA_a[cur] + ks * 4096, 128, 256);
                    uint64_t db = make_smem_desc(smB_a[cur] + ks * 4096, 128, 256);
                    uint32_t acc = (ki > 0 || ks > 0) ? 1u : 0u;
                    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
                                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                                 :: "r"(tmem_base), "l"(da), "l"(db), "r"(idesc), "r"(acc));
                }
                if (ki == k_steps - 1 || ki + 2 < k_steps) {
                    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mm));
                    asm volatile("{\n .reg .pred P1;\n L_MK:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @!P1 bra L_MK;\n}\n"
                                 :: "r"(mm), "r"(mma_par));
                    mma_par ^= 1;
                }
            }
        }
        __syncthreads();

        uint4* sm_u4 = (uint4*)smEpi;
        #pragma unroll
        for (int c_blk = 0; c_blk < 4; c_blk++) {
            uint32_t r[32];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]),
                  "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]),
                  "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]),
                  "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31])
                : "r"(tmem_base + c_blk * 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                __nv_bfloat162 b0 = __floats2bfloat162_rn(__uint_as_float(r[g*8+0]), __uint_as_float(r[g*8+1]));
                __nv_bfloat162 b1 = __floats2bfloat162_rn(__uint_as_float(r[g*8+2]), __uint_as_float(r[g*8+3]));
                __nv_bfloat162 b2 = __floats2bfloat162_rn(__uint_as_float(r[g*8+4]), __uint_as_float(r[g*8+5]));
                __nv_bfloat162 b3 = __floats2bfloat162_rn(__uint_as_float(r[g*8+6]), __uint_as_float(r[g*8+7]));
                sm_u4[tid * 16 + c_blk * 4 + g] = make_uint4(*reinterpret_cast<uint32_t*>(&b0), *reinterpret_cast<uint32_t*>(&b1),
                                                             *reinterpret_cast<uint32_t*>(&b2), *reinterpret_cast<uint32_t*>(&b3));
            }
        }
        __syncthreads();

        int rows_valid = min(128, M - row_base), col_u4 = tid & 15, r_grp = tid >> 4;
        if constexpr (EPI_MODE == 0) {
            for (int rr = r_grp; rr < rows_valid; rr += 8)
                *((uint4*)((__nv_bfloat16*)dst_out + (size_t)(row_base + rr) * N + col_base) + col_u4) = sm_u4[rr * 16 + col_u4];
        } else if constexpr (EPI_MODE == 1) {
            for (int rr = r_grp; rr < rows_valid; rr += 8) {
                uint4 v = sm_u4[rr * 16 + col_u4];
                uint4* p = (uint4*)((__nv_bfloat16*)dst_out + (size_t)(row_base + rr) * N + col_base) + col_u4;
                asm volatile("multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1,%2,%3,%4};"
                             :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
            }
        } else if constexpr (EPI_MODE == 2) {
            int owner = tn / tn_per_rank;
            int local_col = (tn - owner * tn_per_rank) * 128;
            __nv_bfloat16* owner_slot = (__nv_bfloat16*)(peer_va[owner]) + (size_t)my_rank * M * N_shard;
            for (int rr = r_grp; rr < rows_valid; rr += 8) {
                uint4 v = sm_u4[rr * 16 + col_u4];
                uint4* p = (uint4*)(owner_slot + (size_t)(row_base + rr) * N_shard + local_col) + col_u4;
                asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1,%2,%3,%4};"
                             :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
            }
        }
        __syncthreads();
        if (mc_row_flag && tid == 0) {
            asm volatile("fence.release.sys;" ::: "memory");
            asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" :: "l"(mc_row_flag + tm), "r"(1u) : "memory");
        }
    }
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(tmem_base));
}

template <int W_RANKS, int DO_MC_BCAST>
__global__ void k_owner_det_sum_and_mc(
    const uint4* __restrict__ local_inbox,
    __nv_bfloat16* __restrict__ dst_out,
    int M, int N, int N_shard, int my_rank
) {
    size_t shard_u4 = ((size_t)M * N_shard) / 8;
    int col_off = my_rank * N_shard, n_sh_u4 = N_shard / 8;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < shard_u4; i += (size_t)gridDim.x * blockDim.x) {
        float acc[8] = {0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int r = 0; r < W_RANKS; r++) {
            uint4 v = local_inbox[(size_t)r * shard_u4 + i];
            uint32_t u[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u[k]));
                acc[2*k+0] += f.x; acc[2*k+1] += f.y;
            }
        }
        __nv_bfloat162 b0 = __floats2bfloat162_rn(acc[0], acc[1]), b1 = __floats2bfloat162_rn(acc[2], acc[3]);
        __nv_bfloat162 b2 = __floats2bfloat162_rn(acc[4], acc[5]), b3 = __floats2bfloat162_rn(acc[6], acc[7]);
        uint4 pk = make_uint4(*reinterpret_cast<uint32_t*>(&b0), *reinterpret_cast<uint32_t*>(&b1),
                              *reinterpret_cast<uint32_t*>(&b2), *reinterpret_cast<uint32_t*>(&b3));
        if constexpr (!DO_MC_BCAST) {
            ((uint4*)dst_out)[i] = pk;
        } else {
            int row = (int)(i / n_sh_u4), c4 = (int)(i % n_sh_u4);
            float4* mc_p = (float4*)(dst_out + (size_t)row * N + col_off) + c4;
            float4 fv = *reinterpret_cast<float4*>(&pk);
            asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};"
                         :: "l"(mc_p), "f"(fv.x), "f"(fv.y), "f"(fv.z), "f"(fv.w) : "memory");
        }
    }
}

} // namespace apex_blackwell
