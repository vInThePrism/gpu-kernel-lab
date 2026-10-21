# Varlen-M grouped GEMM for MoE (Triton)

One launch per linear layer serves every expert, with no device → host sync.

- **Weights** have the same shape for every expert → found by **indexing** (`w + e·stride_e`).
- **Tokens per expert** differ → found by a **prefix sum** of the counts, computed on the GPU.
- All experts' M-tiles form one pool; each program finds its expert from the prefix sum of tiles per
  expert. The grid uses the static bound `ceil(rows / BM) + E`, so the host never reads the counts.

`moe_ffn_grouped` wraps it into a forward-only MoE FFN example (route → permute → gate/up GEMM → SwiGLU → down GEMM →
unpermute → weighted combine) and includes checks against `moe_ffn_loop`, the per-expert loop it replaces,
including an expert with 0 tokens and one with 1 token.

```bash
python grouped_gemm.py                      # GPU
TRITON_INTERPRET=1 python grouped_gemm.py   # CPU, through the Triton interpreter
```

> **Status:** the program → expert → rows mapping has been checked against a per-expert reference in a
> NumPy model of the kernel (empty experts, 1-row experts, multi-tile experts). The Triton file itself has
> not yet been run; do the `TRITON_INTERPRET=1` run above before relying on it.

This is a teaching kernel (simple tiles, no TMA / warp specialization / persistent scheduling). For
production use SonicMoE, Megatron-Core's grouped GEMM, or `torch._grouped_mm`. Background:
[notes/03](https://github.com/vInThePrism/ai-infra-notebook/blob/main/notes/03-moe-grouped-gemm-and-expert-parallelism.md).

No custom backward implementation is supplied. CUDA Graph compatibility of the complete routing path has not been verified; the static launch bound applies to the grouped-GEMM stage.
