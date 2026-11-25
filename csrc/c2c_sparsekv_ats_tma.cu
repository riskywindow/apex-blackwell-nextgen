// APEX Blackwell Next-Gen — C2C-SparseKV-NVFP4: 1D TMA Over Unpinned Hardware-ATS + Native cvt.e2m1x2
// Target: NVIDIA Grace-Blackwell GB200 (sm_100a, CUDA 13.0+)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <math.h>

namespace apex_blackwell {

// Double-buffered 1D TMA-over-Unpinned-ATS NVFP4 Sparse Flash-Decode (P=16 tokens/page, D=128).
// Per-token layout per KV head: 160 bytes (64B K_fp4 + 8B K_sc + 8B pad + 64B V_fp4 + 8B V_sc + 8B pad)
// -> 2,560 B/page/head (3.20x smaller than BF16). Reads BOTH `ats_pids` and `KV_fp4_pool` directly
// from unpinned NUMA-local Grace LPDDR5X via Hardware ATS with ZERO cudaMemcpy.
__global__ __launch_bounds__(128, 2)
void k_c2c_ats_tma_nvfp4_p16(
    const __nv_bfloat16* __restrict__ Q,          // [H_q = 4*H_kv, 128]
    const uint8_t* __restrict__ KV_fp4_pool,      // [N_pages, H_kv, 16, 160B] in Grace LPDDR5X (unpinned ATS!)
    const int32_t* __restrict__ ats_pids,         // [n_pg] in Grace LPDDR5X (written by Grace sdot indexer!)
    float* __restrict__ part_m,                   // [H_q, S_splits]
    float* __restrict__ part_l,                   // [H_q, S_splits]
    float* __restrict__ part_o,                   // [H_q, S_splits, 128]
    int n_pg, int page_mask, int S_splits
) {
    extern __shared__ __align__(128) uint8_t sm_raw[];
    constexpr int P = 16, TOK_B = 160, PB = P * TOK_B; // 2,560 B/page/head
    uint8_t* sm_buf[2] = { sm_raw, sm_raw + PB };
    __shared__ uint64_t mbar[2];
    int sp = blockIdx.x, hkv = blockIdx.y;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    uint32_t mbar_a[2] = { (uint32_t)__cvta_generic_to_shared(&mbar[0]), (uint32_t)__cvta_generic_to_shared(&mbar[1]) };
    uint32_t sm_a[2]   = { (uint32_t)__cvta_generic_to_shared(sm_buf[0]), (uint32_t)__cvta_generic_to_shared(sm_buf[1]) };
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_a[0]));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_a[1]));
    }
    __syncthreads();

    float q0[4], q1[4], q2[4], q3[4];
    constexpr float SC = 0.08838834764831845f;
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const __nv_bfloat162* qp = (const __nv_bfloat162*)(Q + (size_t)(hkv * 4 + g) * 128);
        float2 a = __bfloat1622float2(qp[2 * lane]), b = __bfloat1622float2(qp[2 * lane + 1]);
        if (g == 0) { q0[0]=a.x*SC; q0[1]=a.y*SC; q0[2]=b.x*SC; q0[3]=b.y*SC; }
        if (g == 1) { q1[0]=a.x*SC; q1[1]=a.y*SC; q1[2]=b.x*SC; q1[3]=b.y*SC; }
        if (g == 2) { q2[0]=a.x*SC; q2[1]=a.y*SC; q2[2]=b.x*SC; q2[3]=b.y*SC; }
        if (g == 3) { q3[0]=a.x*SC; q3[1]=a.y*SC; q3[2]=b.x*SC; q3[3]=b.y*SC; }
    }
    float wm[4] = {-1e30f,-1e30f,-1e30f,-1e30f}, wl[4] = {0,0,0,0}, wo[4][4];
    #pragma unroll
    for (int g = 0; g < 4; g++) for (int j = 0; j < 4; j++) wo[g][j] = 0.f;

    int p_per = (n_pg + S_splits - 1) / S_splits, p_beg = sp * p_per, p_end = min(n_pg, p_beg + p_per);
    size_t p_str = (size_t)gridDim.y * PB, h_off = (size_t)hkv * PB;
    int blk16 = lane >> 2;

    auto issue = [&](int pi, int stg) {
        if (tid == 0 && pi < p_end) {
            int pg = ats_pids[pi] & page_mask;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mbar_a[stg]), "r"(PB));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(sm_a[stg]), "l"(KV_fp4_pool + (size_t)pg * p_str + h_off), "r"(PB), "r"(mbar_a[stg]) : "memory");
        }
    };
    int par[2] = {0, 0};
    if (p_beg < p_end) issue(p_beg, 0);
    for (int pi = p_beg; pi < p_end; pi++) {
        int cur = (pi - p_beg) & 1, nxt = cur ^ 1;
        if (pi + 1 < p_end) issue(pi + 1, nxt);
        if (tid == 0) {
            uint32_t ma = mbar_a[cur]; int pr = par[cur];
            asm volatile("{\n .reg .pred P1;\n L_F4:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n @!P1 bra L_F4;\n}\n"
                         :: "r"(ma), "r"(pr));
            par[cur] ^= 1;
        }
        __syncthreads();
        const uint8_t* psm = sm_buf[cur];
        for (int t = warp; t < P; t += 4) {
            const uint8_t* tok = psm + t * TOK_B;
            __nv_fp4x2_storage_t k0 = tok[2*lane], k1 = tok[2*lane+1];
            __nv_fp4x2_storage_t v0 = tok[80+2*lane], v1 = tok[80+2*lane+1];
            __nv_fp8_e4m3 sk_r, sv_r;
            sk_r.__x = tok[64 + blk16]; sv_r.__x = tok[144 + blk16];
            float sk = float(sk_r), sv = float(sv_r);
            __half2_raw hka = __nv_cvt_fp4x2_to_halfraw2(k0, __NV_E2M1), hkb = __nv_cvt_fp4x2_to_halfraw2(k1, __NV_E2M1);
            __half2_raw hva = __nv_cvt_fp4x2_to_halfraw2(v0, __NV_E2M1), hvb = __nv_cvt_fp4x2_to_halfraw2(v1, __NV_E2M1);
            float2 ka = __half22float2(*reinterpret_cast<__half2*>(&hka)), kb = __half22float2(*reinterpret_cast<__half2*>(&hkb));
            float2 va = __half22float2(*reinterpret_cast<__half2*>(&hva)), vb = __half22float2(*reinterpret_cast<__half2*>(&hvb));
            ka.x*=sk; ka.y*=sk; kb.x*=sk; kb.y*=sk;
            va.x*=sv; va.y*=sv; vb.x*=sv; vb.y*=sv;
            float dots[4] = {
                q0[0]*ka.x+q0[1]*ka.y+q0[2]*kb.x+q0[3]*kb.y, q1[0]*ka.x+q1[1]*ka.y+q1[2]*kb.x+q1[3]*kb.y,
                q2[0]*ka.x+q2[1]*ka.y+q2[2]*kb.x+q2[3]*kb.y, q3[0]*ka.x+q3[1]*ka.y+q3[2]*kb.x+q3[3]*kb.y
            };
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) dots[g] += __shfl_xor_sync(0xffffffffu, dots[g], off);
                float nm = fmaxf(wm[g], dots[g]), al = __expf(wm[g] - nm), p = __expf(dots[g] - nm);
                wl[g] = wl[g]*al + p;
                wo[g][0]=wo[g][0]*al+p*va.x; wo[g][1]=wo[g][1]*al+p*va.y;
                wo[g][2]=wo[g][2]*al+p*vb.x; wo[g][3]=wo[g][3]*al+p*vb.y;
                wm[g] = nm;
            }
        }
        __syncthreads();
    }
    __shared__ float sm_m[4][4], sm_l[4][4], sm_o[4][4][128];
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        if (lane == 0) { sm_m[g][warp] = wm[g]; sm_l[g][warp] = wl[g]; }
        sm_o[g][warp][4*lane]=wo[g][0]; sm_o[g][warp][4*lane+1]=wo[g][1];
        sm_o[g][warp][4*lane+2]=wo[g][2]; sm_o[g][warp][4*lane+3]=wo[g][3];
    }
    __syncthreads();
    if (warp < 4) {
        int g = warp;
        float gm = fmaxf(fmaxf(sm_m[g][0], sm_m[g][1]), fmaxf(sm_m[g][2], sm_m[g][3]));
        float w0=__expf(sm_m[g][0]-gm), w1=__expf(sm_m[g][1]-gm), w2=__expf(sm_m[g][2]-gm), w3=__expf(sm_m[g][3]-gm);
        int hq = hkv * 4 + g;
        if (lane == 0) { part_m[hq*S_splits+sp] = gm; part_l[hq*S_splits+sp] = sm_l[g][0]*w0+sm_l[g][1]*w1+sm_l[g][2]*w2+sm_l[g][3]*w3; }
        float* d = part_o + ((size_t)hq * S_splits + sp) * 128;
        #pragma unroll
        for (int j = 0; j < 4; j++) { int idx = 4*lane+j; d[idx] = sm_o[g][0][idx]*w0+sm_o[g][1][idx]*w1+sm_o[g][2][idx]*w2+sm_o[g][3][idx]*w3; }
    }
}

} // namespace apex_blackwell
