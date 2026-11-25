"""APEX Blackwell Next-Gen — Publication-Grade GB200 (sm_100a) Systems & Algorithms Suite.

Four orthogonal NVIDIA Blackwell (GB200 NVL72) primitives validated across Waves 1-4:
  1. `dual_nvfp4`     — Residual-microscaled Dual-NVFP4 + CUTLASS SM100 2SM SMEM-reused patch
  2. `c2c_sparsekv`   — Tierless Hardware-ATS 1D TMA NVFP4 KV + Grace 1.41 TB/s INT8 sdot indexer
  3. `epi_collectives`— Two-Regime Zero-HBM Collective-GEMM (`Epi-NVLS` + bit-exact `Epi-P2P-RS`)
  4. `tmem_ssm`       — Zero-transpose `A-in-TMEM` chunkwise SSM & Zero-HBM Spec-Tree verification
"""

from .dual_nvfp4 import (
    decompose_dual_nvfp4,
    dual_nvfp4_matmul,
    patch_cutlass_sm100_mainloop,
    quantize_nvfp4_1x16,
    sqnr_db,
)
from .c2c_sparsekv import (
    build_int8_subcentroids,
    select_topk_pages_int8,
    quantize_kv_nvfp4_pages,
)
from .epi_collectives import (
    select_collective_regime,
    deterministic_p2p_rs_reference,
)
from .tmem_ssm import (
    allocate_tmem_dfs_slots,
    gated_deltanet_tree_reference,
    transposed_state_chunkwise_reference,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "quantize_nvfp4_1x16",
    "decompose_dual_nvfp4",
    "dual_nvfp4_matmul",
    "patch_cutlass_sm100_mainloop",
    "sqnr_db",
    "build_int8_subcentroids",
    "select_topk_pages_int8",
    "quantize_kv_nvfp4_pages",
    "select_collective_regime",
    "deterministic_p2p_rs_reference",
    "allocate_tmem_dfs_slots",
    "gated_deltanet_tree_reference",
    "transposed_state_chunkwise_reference",
]
