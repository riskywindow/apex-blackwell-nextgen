# APEX Blackwell Next-Gen (`GB200` / `sm_100a` / `NVL72 IMEX`)

Publication-grade systems & algorithms suite exploiting four orthogonal NVIDIA Grace–Blackwell hardware primitives, validated across **Waves 1–4** on **up to 6 live GB200 NVL72 trays (24× GB200, including 4-tray 16× GB200 IMEX `ComputeDomain`)**.

---

## Four Papers & Headline Live-Silicon Results

| Paper | Target Venue | Hardware Primitive | Headline Live `GB200` Result | PDF |
|---|---|---|---|---|
| **Paper 3 — `Dual-NVFP4`** | NeurIPS / ICML / MLSys | Patched `CUTLASS SM100 2SM` (`tcgen05` `NVFP4`) + SMEM/TMEM reuse | **`2.28–2.62 PF/s`** (**`1.18×–1.54×` > `cuBLAS BF16`**, **`1.01×–1.46×` E2E incl. `5.58 TB/s` quantizer**, `+14.7%..+42.2%` vs K-stack); live hardware on real `Qwen3-30B-A3B` weights hits **`36.3–40.3 dB` (`+7.8..+11.8 dB` > live `MXFP8`)**; **`2.06×` lower 48L KL than `MXFP8`** (zero Hadamard!) | [`paper3_dual_nvfp4.pdf`](papers/paper3_dual_nvfp4.pdf) |
| **Paper 4 — `C2C-SparseKV-NVFP4`** | SOSP / OSDI / ASPLOS | Unpinned 1D `TMA`-over-Hardware-ATS + `cvt.e2m1x2` + Grace `1.41 TB/s` `INT8 sdot` | **Faster than local `HBM3e BF16`** on 48L `TP=4` at `B=1` (`6.56 vs 6.67 ms`, **`-1.6%`**) & `B=4` (`8.58 vs 8.62 ms`, **`-0.5%`**), **`+8.7%` at `B=16`** (`867 tok/s`), `3.91×` 4-GPU scaling (`321.7 GB/s`, `0.0%` SVE2 contention), **`>12M` tok/tray**, **`100%` (`64K`/`256K`) / `91.7%` (`1M`)** real `Qwen3` multi-needle recall (`+50 pp` vs Quest) | [`paper4_c2c_sparsekv.pdf`](papers/paper4_c2c_sparsekv.pdf) |
| **Paper 5 — `Epi-NVLS` + `Epi-P2P-RS`** | SOSP / OSDI / SC | `tcgen05` TMEM epilogue → `multimem.red` & `775 GB/s` IMEX NVLink-5 P2P push | Live **4-tray `16× GB200` IMEX (`TP=16`, `0 / 1,048,576` XR16 diffs):** beats `cuBLAS + NCCL16 + RMSNorm` across **all `M=16..4096` by `1.40×–10.72×`** (`24.0 vs 257.1 µs` @ `M=16`; `267.6 vs 374.7 µs` @ `M=4096` `64 MiB` AR16); beats `cuBLAS + NCCL16 RS` across **all `M` by `2.33×–6.40×`** | [`paper5_epi_nvls_p2p_rs.pdf`](papers/paper5_epi_nvls_p2p_rs.pdf) |
| **Paper 6 — `TMEM-SSM`** | MLSys / ASPLOS / NeurIPS | `256 KB/SM` TMEM (`240.6 TB/s` `.x64`) + `A-in-TMEM` `tcgen05.mma` + `cvt.rs.f16x2` | `A-in-TMEM` **`1.132×`** faster than A-in-SMEM (`cos=1.000000`); **`6.97×`** Householder Gated-DeltaNet tree verify (`W=64`, `1.265 GB`/L saved); **`2.07×` full 32L Hybrid-LLM E2E** (`+38.4%` peak tok/s shifting `W*=16→32`, `1.35×–1.91×` across `B=1..16` on skewed depth-7 EAGLE-2 trees); **`1.10×` faster** `C=128` prefill (`247.9 TF`) saving **`245 GB`/model** HBM state | [`paper6_tmem_ssm.pdf`](papers/paper6_tmem_ssm.pdf) |

---

## Repository Layout

```text
release/apex-blackwell-nextgen/
├── papers/                          # Camera-ready IEEE LaTeX sources & compiled PDFs (Papers 3–6)
├── csrc/                            # Standalone sm_100a CUDA & ARM Neoverse-V2 C++ kernels
│   ├── dual_nvfp4_quant.cuh         # 5.58 TB/s vectorized Dual-NVFP4 quantizer
│   ├── c2c_sparsekv_ats_tma.cu      # Unpinned Hardware-ATS 1D TMA NVFP4 Flash-Decode
│   ├── grace_int8_sdot_indexer.cpp  # 1.41 TB/s Grace Neoverse-V2 vdotq_s32 4-tok sub-centroid indexer
│   ├── epi_two_regime_imex.cu       # Two-Regime Epi-NVLS + bit-exact Epi-P2P-RS (4/8/16-GPU IMEX)
│   └── tmem_ssm_kernels.cu          # 4-slot FP32 & 8-slot FP16-SR Householder TMEM Spec-Tree engine
├── apex_blackwell/                  # Python reference implementations, CUTLASS patcher & allocators
├── benchmarks/results/              # All 18 raw live-GB200 silicon JSON artifacts (Waves 1–4)
└── tests/test_apex_blackwell.py     # Unit, algebraic-identity, patcher & JSON consistency tests
```

## Quick Verification

```bash
pip install -e .
pytest -v tests/test_apex_blackwell.py
```
