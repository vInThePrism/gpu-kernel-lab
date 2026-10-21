# Release validation

Validated on 2026-10-21 on macOS arm64 with Python 3.14.0. The isolated validation environment used pytest 9.1.1, NumPy 2.5.4, and PyTorch 2.14.1 where needed. These are the versions exercised for this release, not a compatibility matrix.

Original private datasets, raw cluster traces, and historical run logs are not bundled. Historical measurements in notes were not independently reproduced in this release.

## Checks completed

- `python tools/attn_check/attn_check.py`: **12/12 CPU SDPA cases passed**, including output and dQ/dK/dV checks.
- Zero-reference gradients use a maximum absolute residual threshold of **1e-5**; nonzero references use normalized L2 ≤ **1e-2**. An `a` prefix in the table marks absolute error. NaN/Inf in candidate or reference fails.
- `python demos/window_attention_varlen/window_attention_varlen.py`: CPU output comparison yielded normalized L2 **0.0**. CPU batching was slower for this run; this demo does not establish a CUDA speedup.
- `python tools/triton_smem_budget/smem_budget.py --device H800 --M 8192 --N 3584 --K 18944 --top 3` executes on CPU.

## Boundaries

No NVIDIA GPU or nvcc was available. CUDA binaries, compute-sanitizer, Triton grouped GEMM, FlashAttention, and the CUDA timing helper were not run. The grouped-GEMM example is forward-only. CPU SDPA checks do not validate a different GPU backend.
