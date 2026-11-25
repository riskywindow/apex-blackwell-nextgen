"""Dual-NVFP4: Rotation-Free Residual-Microscaled FP4 & CUTLASS SM100 2SM Patcher."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import torch

_E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_EDGES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

CUTLASS_OLD_LOOP_SENTINEL = "if constexpr (IsOverlappingAccum) {"
CUTLASS_REL_HEADER = "cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp"


def quantize_nvfp4_1x16(x: torch.Tensor, sr: bool = False) -> torch.Tensor:
    """Exact Blackwell NVFP4 (1x16 E4M3 micro-scaled E2M1) fake-quantizer matching sm_100a hardware."""
    orig = x.shape
    if orig[-1] % 16 != 0:
        raise ValueError(f"Last dimension must be divisible by 16 for NVFP4, got {orig[-1]}")
    dev = x.device
    pos = torch.tensor(_E2M1_POS, dtype=torch.float32, device=dev)
    xf = x.float().reshape(-1, 16)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    gmax = amax.max().clamp(min=1e-12)
    g_sc = gmax / (6.0 * 256.0)
    b_sc = (amax / (6.0 * g_sc)).clamp(min=1.5625e-2, max=448.0).to(torch.float8_e4m3fn).float() * g_sc
    b_sc = b_sc.clamp(min=1e-12)
    y = xf / b_sc
    a = y.abs().clamp(max=6.0)
    if not sr:
        edges = torch.tensor(_E2M1_EDGES, dtype=torch.float32, device=dev)
        idx = torch.bucketize(a, edges)
        q = y.sign() * pos[idx]
    else:
        idx_hi = torch.bucketize(a, pos[1:]).clamp(max=6) + 1
        lo, hi = pos[idx_hi - 1], pos[idx_hi]
        p = ((a - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
        q = y.sign() * torch.where(torch.rand_like(a) < p, hi, lo)
    return (q * b_sc).reshape(orig)


def decompose_dual_nvfp4(
    x: torch.Tensor,
    sr_residual: bool = False,
    adaptive_skip_ratio: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose tensor `x` into `(A0, A1)` with independent 1x16 E4M3 micro-scales."""
    a0 = quantize_nvfp4_1x16(x, sr=False)
    r = x.float() - a0
    if adaptive_skip_ratio > 0.0:
        orig = r.shape
        rf = r.reshape(-1, 16)
        norms = rf.abs().amax(dim=-1)
        thresh = torch.quantile(norms, adaptive_skip_ratio)
        r = torch.where((norms >= thresh).unsqueeze(-1), rf, torch.zeros_like(rf)).reshape(orig)
    a1 = quantize_nvfp4_1x16(r, sr=sr_residual)
    return a0, a1


def dual_nvfp4_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    terms: int = 3,
    sr_residual: bool = False,
    adaptive_skip_ratio: float = 0.0,
) -> torch.Tensor:
    """Compute `a @ b.T` using 2-term (`A0 B0 + A1 B0`), 3-term (`+ A0 B1`), or 4-term Dual-NVFP4."""
    if terms not in (1, 2, 3, 4):
        raise ValueError(f"terms must be in (1, 2, 3, 4), got {terms}")
    a0, a1 = decompose_dual_nvfp4(a, sr_residual=sr_residual, adaptive_skip_ratio=adaptive_skip_ratio)
    if terms == 1:
        b0 = quantize_nvfp4_1x16(b, sr=False)
        return a0 @ b0.T
    if terms == 2:
        b0 = quantize_nvfp4_1x16(b, sr=False)
        return (a0 @ b0.T) + (a1 @ b0.T)
    b0, b1 = decompose_dual_nvfp4(b, sr_residual=sr_residual, adaptive_skip_ratio=adaptive_skip_ratio)
    out = (a0 @ b0.T) + (a1 @ b0.T) + (a0 @ b1.T)
    if terms == 4:
        out = out + (a1 @ b1.T)
    return out


def sqnr_db(ref: torch.Tensor, est: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB."""
    r = ref.float()
    e = r - est.float()
    return 10.0 * math.log10(max(float((r * r).mean().item()), 1e-30) / max(float((e * e).mean().item()), 1e-30))


def patch_cutlass_sm100_mainloop(cutlass_include_dir: str | Path, shadow_out_dir: str | Path) -> Path:
    """Generate a non-destructive shadow copy of CUTLASS SM100 blockscaled mainloop with Dual-NVFP4 SMEM reuse."""
    src = Path(cutlass_include_dir) / CUTLASS_REL_HEADER
    if not src.exists():
        raise FileNotFoundError(f"CUTLASS header not found: {src}")
    text = src.read_text()
    if "APEX_DUAL_NVFP4_TERMS" in text:
        out = Path(shadow_out_dir) / CUTLASS_REL_HEADER
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        return out
    anchor = "    if constexpr (IsOverlappingAccum) {"
    end_anchor = "      mainloop_pipeline.consumer_release(curr_mainloop_pipe_consumer_state);\n    }"
    i0 = text.find(anchor)
    i1 = text.find(end_anchor, i0)
    if i0 < 0 or i1 < 0:
        raise RuntimeError("Unexpected CUTLASS sm100_blockscaled_mma_warpspecialized.hpp structure")
    # Locate the second consumer_release (end of the main while loop)
    i2 = text.find(end_anchor, i1 + len(end_anchor))
    if i2 < 0:
        raise RuntimeError("Could not locate main while-loop end in CUTLASS header")
    i_end = i2 + len(end_anchor)
    orig_block = text[i0:i_end]
    dual_block = """#if defined(APEX_DUAL_NVFP4_TERMS)
    accumulator_pipeline.producer_acquire(accumulator_pipe_producer_state);
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count >= 2) {
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, barrier_token);
      int s0 = mainloop_pipe_consumer_state.index();
      auto state_s0 = mainloop_pipe_consumer_state;
      ++mainloop_pipe_consumer_state; --k_tile_count;
      auto tok_s1 = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, false);
      if (cute::elect_one_sync()) {
        copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,s0), thr_tCtSFA_s2t);
        copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_,_,_,_,s0), thr_tCtSFB_s2t);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_,_,k_block), tCtSFB_mma(_,_,k_block)),
                   tCrA(_,_,k_block,s0), tCrB(_,_,k_block,s0), accumulators);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      mainloop_pipeline.consumer_wait(mainloop_pipe_consumer_state, tok_s1);
      int s1 = mainloop_pipe_consumer_state.index();
      auto state_s1 = mainloop_pipe_consumer_state;
      ++mainloop_pipe_consumer_state; --k_tile_count;
      skip_wait = k_tile_count <= 0;
      barrier_token = mainloop_pipeline.consumer_try_wait(mainloop_pipe_consumer_state, skip_wait);
      if (cute::elect_one_sync()) {
        copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,s1), thr_tCtSFA_s2t);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_,_,k_block), tCtSFB_mma(_,_,k_block)),
                   tCrA(_,_,k_block,s1), tCrB(_,_,k_block,s0), accumulators);
      }
#if APEX_DUAL_NVFP4_TERMS >= 3
      if (cute::elect_one_sync()) {
        copy(tiled_copy_s2t_SFA, thr_tCsSFA_s2t(_,_,_,_,s0), thr_tCtSFA_s2t);
        copy(tiled_copy_s2t_SFB, thr_tCsSFB_s2t(_,_,_,_,s1), thr_tCtSFB_s2t);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        cute::gemm(tiled_mma.with(tiled_mma.accumulate_, tCtSFA(_,_,k_block), tCtSFB_mma(_,_,k_block)),
                   tCrA(_,_,k_block,s0), tCrB(_,_,k_block,s1), accumulators);
      }
#endif
      mainloop_pipeline.consumer_release(state_s0);
      mainloop_pipeline.consumer_release(state_s1);
    }
#else
""" + orig_block + "\n#endif"
    out = Path(shadow_out_dir) / CUTLASS_REL_HEADER
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text[:i0] + dual_block + text[i_end:])
    return out
