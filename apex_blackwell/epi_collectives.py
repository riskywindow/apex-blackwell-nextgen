"""Epi-NVLS & Epi-P2P-RS: Two-Regime Zero-HBM-Traffic Collective-GEMM Dispatch & Reference."""

from __future__ import annotations

from typing import List, Literal

import torch


def select_collective_regime(
    M: int,
    N: int,
    world_size: int,
    op: Literal["allreduce", "reducescatter"] = "allreduce",
    dtype_bytes: int = 2,
) -> Literal["epi_nvls_tilepipe", "epi_p2p_rs_sp", "epi_p2p_rs_mc_ar"]:
    """Hardware-calibrated Two-Regime dispatch law for NVIDIA GB200 NVL72 (4/8/16 GPUs).

    - ReduceScatter (Megatron SP): ALWAYS use `epi_p2p_rs_sp` (2.33x-6.40x vs NCCL16 RS).
    - AllReduce:
      * Latency-bound small batches (`M <= 32` or payload <= 512 KiB on W >= 8):
        `epi_nvls_tilepipe` (direct `multimem.red` + per-strip RMSNorm, up to 10.72x vs NCCL16).
      * Bandwidth-bound batches (`M >= 64`):
        `epi_p2p_rs_mc_ar` (direct IMEX NVLink-5 P2P push -> deterministic FP32 local sum ->
        single-writer `multimem.st` broadcast; avoids Wx NVSwitch write amplification, 0 XR diffs).
    """
    if op == "reducescatter":
        return "epi_p2p_rs_sp"
    payload_bytes = M * N * dtype_bytes
    if world_size <= 4:
        return "epi_nvls_tilepipe"
    if M <= 32 or payload_bytes <= 512 * 1024:
        return "epi_nvls_tilepipe"
    return "epi_p2p_rs_mc_ar"


def deterministic_p2p_rs_reference(
    rank_partials: List[torch.Tensor],
    broadcast: bool = True,
) -> List[torch.Tensor]:
    """Exact bit-identical reference for `Epi-P2P-RS` (+ optional single-writer `multimem.st` AG).

    Every owner rank `o` sums its column shard `N // W` across all `W` ranks in canonical
    rank order `r = 0..W-1` in FP32, then casts once to BF16 -> guarantees 0 cross-rank bit diffs.
    """
    W = len(rank_partials)
    M, N = rank_partials[0].shape
    if N % W != 0:
        raise ValueError(f"N={N} must be divisible by world_size={W}")
    n_sh = N // W
    shards: List[torch.Tensor] = []
    for owner in range(W):
        c0, c1 = owner * n_sh, (owner + 1) * n_sh
        acc = torch.zeros((M, n_sh), dtype=torch.float32, device=rank_partials[0].device)
        for r in range(W):
            acc = acc + rank_partials[r][:, c0:c1].to(torch.bfloat16).float()
        shards.append(acc.to(torch.bfloat16))
    if not broadcast:
        return shards
    full = torch.cat(shards, dim=1)
    return [full.clone() for _ in range(W)]
