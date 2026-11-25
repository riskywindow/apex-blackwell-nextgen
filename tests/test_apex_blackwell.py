"""Verification suite for APEX Blackwell Next-Gen (algebraic identities, patcher, allocators & JSON consistency)."""

import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import pytest
except ImportError:
    pytest = None

from apex_blackwell import (
    allocate_tmem_dfs_slots,
    build_int8_subcentroids,
    decompose_dual_nvfp4,
    deterministic_p2p_rs_reference,
    dual_nvfp4_matmul,
    gated_deltanet_tree_reference,
    patch_cutlass_sm100_mainloop,
    quantize_kv_nvfp4_pages,
    quantize_nvfp4_1x16,
    select_collective_regime,
    select_topk_pages_int8,
    sqnr_db,
    transposed_state_chunkwise_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


def test_dual_nvfp4_outlier_immunity():
    torch.manual_seed(2026)
    M, K, N = 128, 512, 256
    X = torch.randn((M, K))
    X[:, :4] *= 200.0  # extreme 200x activation outliers
    W = torch.randn((N, K)) * 0.05
    Y_ref = X @ W.T

    Y_1t = dual_nvfp4_matmul(X, W, terms=1)
    Y_2t = dual_nvfp4_matmul(X, W, terms=2)
    Y_3t = dual_nvfp4_matmul(X, W, terms=3)
    Y_4t = dual_nvfp4_matmul(X, W, terms=4)

    s1, s2, s3, s4 = sqnr_db(Y_ref, Y_1t), sqnr_db(Y_ref, Y_2t), sqnr_db(Y_ref, Y_3t), sqnr_db(Y_ref, Y_4t)
    assert s3 > s1 + 15.0, f"Dual3 ({s3:.2f} dB) must crush 1-term NVFP4 ({s1:.2f} dB) by >15 dB"
    assert s3 > 34.0, f"Dual3 expected >34 dB under 200x outliers, got {s3:.2f} dB"
    assert s4 >= s3, f"4-term ({s4:.2f}) should bound 3-term ({s3:.2f})"
    assert s2 > s1, f"Dual2 ({s2:.2f}) should improve over 1-term ({s1:.2f})"


def test_dual_nvfp4_cutlass_patcher_roundtrip(tmp_path):
    # Create synthetic minimal CUTLASS header matching the exact anchors patched on GB200
    hdr_dir = tmp_path / "cutlass_in"
    rel = Path("cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp")
    src = hdr_dir / rel
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "// mock header\n"
        "    if constexpr (IsOverlappingAccum) {\n"
        "      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);\n"
        "    }\n"
        "    while (k_tile_count > 0) {\n"
        "      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);\n"
        "    }\n"
        "// end\n"
    )
    out = patch_cutlass_sm100_mainloop(hdr_dir, tmp_path / "shadow")
    patched = out.read_text()
    assert "#if defined(APEX_DUAL_NVFP4_TERMS)" in patched
    assert "tCrA(_,_,k_block,s1), tCrB(_,_,k_block,s0)" in patched
    assert "tCrA(_,_,k_block,s0), tCrB(_,_,k_block,s1)" in patched


def test_c2c_int8_subcentroid_needle_retrieval_and_nvfp4_kv():
    torch.manual_seed(42)
    L, D, P = 4096, 128, 16
    K = torch.randn((L, D)) * 0.2
    V = torch.randn((L, D)) * 0.2
    q = torch.randn((D,))
    q = q / q.norm()
    # Plant 3 distinct 4-token needles aligned with query
    needles = [400, 2000, 3600]
    for pos in needles:
        K[pos : pos + 4] = 2.0 * q

    c_i8, sc = build_int8_subcentroids(K, page_size=P, sub_size=4)
    assert c_i8.dtype == torch.int8
    top_pages = set(select_topk_pages_int8(q, c_i8, sc, k_pages=32, page_size=P, sub_size=4).tolist())
    for pos in needles:
        assert (pos // P) in top_pages, f"Missed needle at page {pos // P}"

    # Verify NVFP4 KV quantization fidelity
    Kq, Vq = quantize_kv_nvfp4_pages(K, V)
    assert sqnr_db(K, Kq) > 16.0
    assert sqnr_db(V, Vq) > 16.0


def test_epi_two_regime_selector_and_bit_exact_p2p_rs():
    assert select_collective_regime(16, 8192, 16, "allreduce") == "epi_nvls_tilepipe"
    assert select_collective_regime(64, 8192, 16, "allreduce") == "epi_p2p_rs_mc_ar"
    assert select_collective_regime(4096, 8192, 16, "allreduce") == "epi_p2p_rs_mc_ar"
    assert select_collective_regime(16, 8192, 16, "reducescatter") == "epi_p2p_rs_sp"
    assert select_collective_regime(4096, 8192, 16, "reducescatter") == "epi_p2p_rs_sp"

    torch.manual_seed(7)
    W, M, N = 16, 32, 256
    partials = [torch.randn((M, N), dtype=torch.bfloat16) for _ in range(W)]
    out_ranks = deterministic_p2p_rs_reference(partials, broadcast=True)
    # 0 cross-rank bit differences across all 16 ranks
    ref_bits = out_ranks[0].view(torch.int16)
    for r in range(1, W):
        assert torch.equal(out_ranks[r].view(torch.int16), ref_bits)


def test_tmem_ssm_zero_transpose_and_skewed_eagle2_dfs_allocator():
    torch.manual_seed(99)
    # 1. Verify transposed-state H = S^T chunkwise recurrence matches sequential step-by-step recurrence!
    H_dim, L, D = 2, 256, 32
    Q = torch.randn((H_dim, L, D)) * 0.1
    K = torch.randn((H_dim, L, D)) * 0.1
    U = torch.randn((H_dim, L, D)) * 0.1
    g = 0.99
    O_chunk, H_chunk = transposed_state_chunkwise_reference(Q, K, U, g_tok=g, chunk_size=128)

    # Naive sequential token-by-token ground truth
    h = torch.zeros((H_dim, D, D))
    O_seq = torch.empty_like(Q)
    for t in range(L):
        h = g * h + torch.einsum("hd,hk->hdk", U[:, t], K[:, t])
        O_seq[:, t] = torch.einsum("hdk,hk->hd", h, Q[:, t])
    assert torch.allclose(O_chunk, O_seq, atol=1e-4, rtol=1e-4)
    assert torch.allclose(H_chunk, h, atol=1e-4, rtol=1e-4)

    # 2. Skewed depth-7 EAGLE-2 tree DFS slot allocation & exact Householder tree execution
    # Spine of depth 7 (nodes 1..7) + secondary branches off nodes 1, 2, 3
    parents = [0, 1, 2, 3, 4, 5, 6, 1, 8, 2, 10, 3]
    dfs_order, s_src, s_dst = allocate_tmem_dfs_slots(parents, max_slots=8)
    assert len(dfs_order) == len(parents)
    assert max(s_dst) == 7
    try:
        allocate_tmem_dfs_slots(parents, max_slots=4)  # depth 7 cannot fit in 4 FP32 slots
        raise AssertionError("Expected ValueError for depth-7 tree on 4-slot TMEM")
    except ValueError:
        pass

    # Simulate TMEM slot execution in DFS order vs exact reference!
    W = len(parents)
    H0 = torch.randn((2, 16, 16)) * 0.05
    Qt = torch.randn((W, 2, 16)) * 0.1
    Kt = F.normalize(torch.randn((W, 2, 16)), dim=-1)
    Vt = torch.randn((W, 2, 16)) * 0.1
    O_ref, _ = gated_deltanet_tree_reference(H0, Qt, Kt, Vt, parents, g=0.99, beta=0.15)

    slots = [None] * 8
    slots[0] = H0.clone()
    O_tmem = torch.empty_like(O_ref)
    for node_idx, ss, sd in zip(dfs_order, s_src, s_dst):
        hp = slots[ss] * 0.99
        pred_v = torch.einsum("hij,hj->hi", hp, Kt[node_idx])
        delta = 0.15 * (Vt[node_idx] - pred_v)
        hn = hp + torch.einsum("hi,hj->hij", delta, Kt[node_idx])
        O_tmem[node_idx] = torch.einsum("hij,hj->hi", hn, Qt[node_idx])
        slots[sd] = hn
    assert torch.allclose(O_tmem, O_ref, atol=1e-6)


def test_archived_wave1_to_wave4_silicon_json_integrity():
    expected_files = [
        "t1_dual_nvfp4_results.json",
        "t2_c2c_sparsekv_results.json",
        "t3_v2_coalesced_results.json",
        "t4_v2_tmem_ssm_results.json",
        "w2_t1_48l_qwen3_results.json",
        "w2_t2_c2c_4gpu_48l_results.json",
        "w2_t3_8gpu_imex_results.json",
        "w2_t4_tmem_ssm_results.json",
        "w3_t1_cutlass_dual_results.json",
        "w3_t2_c2c_e2e_results.json",
        "w3_t3_two_regime_8gpu_results.json",
        "w3_t4_hybrid32l_results.json",
        "w4_t1_real_qwen_cutlass_dual_results.json",
        "w4_t2_c2c_nvfp4_results.json",
        "w4_t3_16gpu_imex_results.json",
        "w4_t4_capstone_results.json",
    ]
    for fn in expected_files:
        p = RESULTS_DIR / fn
        assert p.exists(), f"Missing archived silicon artifact: {fn}"
        data = json.loads(p.read_text())
        assert isinstance(data, dict)

    # Spot-check headline silicon claims against archived JSON ground truth
    w3_t1 = json.loads((RESULTS_DIR / "w3_t1_cutlass_dual_results.json").read_text())
    d3_ffn = next(s for s in w3_t1["runs"]["Dual3_CUTLASS_SMEM"]["shapes"] if s["tag"] == "LLM_FFN_4Kx14Kx4K")
    assert d3_ffn["gemm_tf"] > 2600.0 and d3_ffn["e2e_vs_bf16"] > 1.30 and d3_ffn["d0"] == d3_ffn["expected_d0"]

    w4_t1 = json.loads((RESULTS_DIR / "w4_t1_real_qwen_cutlass_dual_results.json").read_text())
    assert all(r["dual3_gain_over_mxfp8_db"] > 7.5 for r in w4_t1["rows"])

    w4_t2 = json.loads((RESULTS_DIR / "w4_t2_c2c_nvfp4_results.json").read_text())
    b1 = next(r for r in w4_t2["multibatch_48l_nvfp4_vs_bf16"] if r["B"] == 1)
    assert b1["nvfp4_ov_pct"] < 0.0  # faster than HBM_BF16!

    w4_t3 = json.loads((RESULTS_DIR / "w4_t3_16gpu_imex_results.json").read_text())
    assert w4_t3["verify_xr16"]["xr16_diffs"] == 0
    assert all(r["speedup_ar_vs_nccl16"] >= 1.40 for r in w4_t3["sweep"])
    assert all(r["speedup_rs_vs_nccl16"] >= 2.30 for r in w4_t3["sweep"])

    w4_t4 = json.loads((RESULTS_DIR / "w4_t4_capstone_results.json").read_text())
    assert w4_t4["eagle2_depth7_verify_cos"] > 0.99999
    assert all(r["speedup"] >= 1.09 for r in w4_t4["prefetched_c128_prefill"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as _td:
        for _name, _fn in sorted(list(globals().items())):
            if _name.startswith("test_") and callable(_fn):
                if "tmp_path" in _fn.__code__.co_varnames:
                    _fn(Path(_td))
                else:
                    _fn()
                print(f"PASS: {_name}")
    print("ALL_TESTS_PASSED")
