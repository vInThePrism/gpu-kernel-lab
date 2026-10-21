# triton_smem_budget

Shared-memory budget for pipelined Triton GEMM tiles, so tiles tuned on one GPU don't crash (or get
silently downgraded) on another.

```
smem = buffers × (BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) × bytes  (+ extra operands)
```

`buffers` is `num_stages` or `num_stages − 1` depending on the Triton version. Calibrate it from a real
error message instead of guessing:

```bash
python smem_budget.py --calibrate 245760 --BM 64 --BN 256 --BK 128 --stages 4
#   buffers = num_stages      ->   327,680 B   diff -81,920 B
#   buffers = num_stages - 1  ->   245,760 B   MATCH
#   does not fit on H800 (232,448 B)
```

Then list, per shape, the deepest pipeline that fits for each tile (with an fp32-accumulator register
check), and paste the top ones into `@triton.autotune`:

```bash
python smem_budget.py --device H800 --M 8192 --N 3584 --K 18944
```

Why per shape: a library fallback that halves BLOCK_N for *all* shapes when one config doesn't fit
removes the crash and slows every shape. Co-searching BLOCK_N and num_stages per projection shape
recovered a 1.21× speedup on a representative fused-LoRA GEMM (M=8192, N=3584, K=18944: 2.089 → 1.721 ms).
