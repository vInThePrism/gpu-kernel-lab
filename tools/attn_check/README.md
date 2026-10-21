# attn_check: is the attention kernel correct, and is the speedup real?

> **Status:** the CPU SDPA candidate passes all 12 edge cases in this release. GPU FlashAttention and CUDA timing paths remain unverified; see [release validation](../../VALIDATION.md).

## `attn_check.py` (correct)

Tests any function with the `flash_attn_varlen_func` signature against an fp32 reference, forward and
backward (out, dQ, dK, dV).

- **Edge cases sampled by bug probability:** tail blocks (lengths not a multiple of the tile), an empty
  sequence inside a varlen batch, causal with q_len < k_len (bottom-right aligned), q_len > k_len,
  GQA 8:1, single-token sequences.
- **The 48-shape production grid** (`--full`): {ViT MHA, LLM GQA} × {varlen, causal, varlen+causal} ×
  {8K, 16K, 32K, 64K} × head dim {64, 128}.
- **Three checks together:** normalized L2 ≤ 1e-2 for nonzero references, maximum absolute residual ≤ 1e-5 for zero references, and NaN/Inf = fail. The table prefixes absolute errors with `a`; max absolute error is also retained in each result.
  dQ is usually the hardest to match because it is accumulated across CTAs with atomics.

```bash
python attn_check.py                 # CPU: SDPA-per-segment candidate, edge cases
python attn_check.py --impl flash    # GPU: flash_attn_varlen_func
python attn_check.py --impl flash --full
```

Any failing configuration should fall back to a known-good backend for that architecture.

## `bench.py` (real)

CUDA-event timing with 50 warm-up / 200 timed / 3 runs, input rotation over 2× the L2 size so small
shapes cannot live in cache, median + CV, and a geometric-mean helper. Lock clocks outside Python
(`nvidia-smi -lgc`). After choosing a config, re-measure inside a fixed end-to-end training job.
