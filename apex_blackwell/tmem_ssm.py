"""TMEM-SSM: Zero-Transpose A-in-TMEM Chunkwise SSM & Zero-HBM Speculative Tree Slot Allocator."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def allocate_tmem_dfs_slots(
    parents: Sequence[int],
    max_slots: int = 8,
) -> Tuple[List[int], List[int], List[int]]:
    """Convert any arbitrary speculative draft tree (`parents[w]` in `{0..w}`, 0=root)
    into a DFS traversal order `(dfs_order, slot_src, slot_dst)` fitting inside `max_slots` TMEM slots.

    Raises `ValueError` if tree depth exceeds `max_slots - 1` (depth <= 3 for 4-slot FP32,
    depth <= 7 for 8-slot FP16-SR inside Blackwell's 256 KB TMEM).
    """
    W = len(parents)
    children: Dict[int, List[int]] = {i: [] for i in range(W + 1)}
    for idx, p in enumerate(parents, start=1):
        if not (0 <= p < idx):
            raise ValueError(f"Invalid parent {p} for node {idx}")
        children[p].append(idx)

    dfs_order: List[int] = []
    slot_src: List[int] = []
    slot_dst: List[int] = []

    def dfs(node: int, depth: int) -> None:
        for ch in children[node]:
            if depth + 1 >= max_slots:
                raise ValueError(f"Tree depth {depth + 1} exceeds TMEM capacity {max_slots - 1}")
            dfs_order.append(ch - 1)
            slot_src.append(depth)
            slot_dst.append(depth + 1)
            dfs(ch, depth + 1)

    dfs(0, 0)
    return dfs_order, slot_src, slot_dst


def gated_deltanet_tree_reference(
    H0: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    parents: Sequence[int],
    g: float = 0.99,
    beta: float = 0.15,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Exact Householder Gated-DeltaNet tree verification reference.

    Args:
        H0: [H, d_v, d_k] base transposed state `H = S^T`
        Q, K, V: [W, H, d] draft tree node vectors
        parents: length-W parent index list (0 = root H0, 1..W = prior node)
    Returns:
        O: [W, H, d_v] verified outputs in original node order
        states: list of W post-node states
    """
    W, H_dim, d = Q.shape
    states: List[torch.Tensor] = [H0.float()]
    outs = torch.empty((W, H_dim, d), dtype=torch.float32, device=Q.device)
    for w in range(W):
        hp = states[parents[w]]
        qw, kw, vw = Q[w].float(), K[w].float(), V[w].float()
        h_dec = hp * g
        pred_v = torch.einsum("hij,hj->hi", h_dec, kw)
        delta = beta * (vw - pred_v)
        hw = h_dec + torch.einsum("hi,hj->hij", delta, kw)
        outs[w] = torch.einsum("hij,hj->hi", hw, qw)
        states.append(hw)
    return outs, states[1:]


def transposed_state_chunkwise_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    U: torch.Tensor,
    g_tok: float = 0.99,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-transpose `H = S^T` chunkwise recurrence matching Blackwell A-in-TMEM `tcgen05`."""
    H_dim, L, D = Q.shape
    if L % chunk_size != 0:
        raise ValueError(f"Sequence length {L} must be divisible by chunk_size {chunk_size}")
    nc = L // chunk_size
    Qc = Q.float().view(H_dim, nc, chunk_size, D)
    Kc = K.float().view(H_dim, nc, chunk_size, D)
    Uc = U.float().view(H_dim, nc, chunk_size, D)
    dev = Q.device
    idx = torch.arange(chunk_size, device=dev, dtype=torch.float32)
    wq = (g_tok ** (idx + 1)).view(1, 1, chunk_size, 1)
    wk = (g_tok ** (chunk_size - 1 - idx)).view(1, 1, chunk_size, 1)
    diff = idx[:, None] - idx[None, :]
    M = torch.where(diff >= 0, g_tok ** torch.clamp(diff, min=0), torch.zeros_like(diff))
    gamma_c = float(g_tok ** chunk_size)

    O_chunks = []
    h = torch.zeros((H_dim, D, D), dtype=torch.float32, device=dev)  # H = S^T
    for c in range(nc):
        # Inter-chunk: O_inter^T = H @ (Q_c * wq)^T -> O_inter = (Q_c * wq) @ H^T
        o_inter = torch.bmm(Qc[:, c] * wq[0], h.transpose(1, 2))
        p = torch.bmm(Qc[:, c], Kc[:, c].transpose(1, 2)) * M
        o_intra = torch.bmm(p, Uc[:, c])
        O_chunks.append(o_inter + o_intra)
        h = h * gamma_c + torch.bmm(Uc[:, c].transpose(1, 2), Kc[:, c] * wk[0])
    return torch.cat(O_chunks, dim=1), h
