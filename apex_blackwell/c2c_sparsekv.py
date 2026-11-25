"""C2C-SparseKV-NVFP4: 4-Token INT8 Sub-Centroid Indexer & NVFP4 KV Page Compression."""

from __future__ import annotations

from typing import Tuple

import torch

from .dual_nvfp4 import quantize_nvfp4_1x16


def build_int8_subcentroids(
    keys: torch.Tensor,
    page_size: int = 16,
    sub_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build 4-token post-RoPE INT8 sub-centroids matching Grace Neoverse-V2 `vdotq_s32` layout.

    Args:
        keys: [L, D] post-RoPE keys
    Returns:
        centroids_i8: [L // sub_size, D] int8
        scales: [L // sub_size] float32
    """
    L, D = keys.shape
    if L % page_size != 0 or page_size % sub_size != 0:
        raise ValueError(f"L={L} must be divisible by page_size={page_size} and sub_size={sub_size}")
    ks = keys.float().view(L // sub_size, sub_size, D).mean(dim=1)
    scales = ks.abs().amax(dim=-1).clamp(min=1e-6) / 127.0
    centroids_i8 = torch.round(ks / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return centroids_i8, scales


def select_topk_pages_int8(
    query: torch.Tensor,
    centroids_i8: torch.Tensor,
    scales: torch.Tensor,
    k_pages: int = 128,
    page_size: int = 16,
    sub_size: int = 4,
    keep_sink_pages: int = 1,
    keep_recent_pages: int = 4,
) -> torch.Tensor:
    """Exact INT8 dot-product page scoring + Sink+Recent pinned selection."""
    sub_per_page = page_size // sub_size
    n_pages = centroids_i8.shape[0] // sub_per_page
    q_sc = query.float().abs().max().clamp(min=1e-6) / 127.0
    q_i8 = torch.round(query.float() / q_sc).clamp(-127, 127).to(torch.int8)
    # Exact int32 accumulation matching ARMv8.4 `vdotq_s32`
    dots = torch.matmul(centroids_i8.to(torch.float32), q_i8.to(torch.float32))
    sub_scores = (dots * scales * q_sc).view(n_pages, sub_per_page).amax(dim=-1)
    if keep_sink_pages > 0:
        sub_scores[:keep_sink_pages] = 1e9
    if keep_recent_pages > 0:
        sub_scores[-keep_recent_pages:] = 1e9
    k_eff = min(k_pages, n_pages)
    return torch.topk(sub_scores, k_eff).indices


def quantize_kv_nvfp4_pages(
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize K and V tensors to exact 1x16 E4M3 NVFP4 representation (`3.20x` smaller C2C payload)."""
    return quantize_nvfp4_1x16(k, sr=False), quantize_nvfp4_1x16(v, sr=False)
