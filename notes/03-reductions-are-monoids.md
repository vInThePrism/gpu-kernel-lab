# Reductions are monoids (and that removes most boundary bugs)

## What a parallel reduction actually requires

A sequential loop `acc = op(acc, x[i])` works for any `op`. A parallel tree regroups the operations: `((a·b)·(c·d))` instead of `(((a·b)·c)·d)`. That regrouping is valid exactly when `op` is **associative**.

Two more properties matter in practice:

- An **identity element** `e` with `op(e, x) = x`. Threads that own no data (because `n` is not a multiple of the block size, or `N < blockDim.x`) contribute `e`. The tree never has to know which threads are "real".
- **Commutativity** is *not* required by a tree reduction that keeps element order. It is required by anything that reorders, for example atomics, where blocks finish in any order.

Associative + identity is a **monoid**. Writing every reduction as a monoid gave me one rule for all boundary handling: *pad with the identity, never skip*.

| Operation | combine | identity | where used |
|---|---|---|---|
| sum | `a + b` | `0` | reduce, scan, dot products, RMSNorm |
| max | `fmaxf(a, b)` | `-INFINITY` | reduce max, softmax, absmax for quantization |
| argmax | larger value wins, ties → smaller index | `(-INFINITY, -1)` | top-1 |
| online softmax | see below | `(m = -INFINITY, l = 0)` | softmax, FlashAttention |
| Welford (mean, M2, count) | parallel Welford merge | `(0, 0, 0)` | numerically stable variance / LayerNorm |

In code, this becomes a template parameter: `block_allreduce<SumOp>`, `block_allreduce<MaxOp>`, with `Op::identity()` used for empty lanes ([primitives.cuh](../kernels/cuda/common/primitives.cuh)).

The classic boundary bug is padding with the *wrong* identity: 0 for a max reduction (fails on all-negative rows), or the max-identity for a sum.

## Online softmax as a monoid

Softmax needs the row max `m` (for stability) and `l = Σ exp(xᵢ − m)`. The state is the pair `(m, l)`:

- a single element `x` is `(x, 1)` — because `exp(x − x) = 1`. Writing `(x, 0)` was one of my repeated bugs.
- merging two partial states:

```
m = max(m₁, m₂)
l = l₁ · exp(m₁ − m) + l₂ · exp(m₂ − m)
```

Each side's sum was computed relative to its *own* max, so each is rescaled to the new common max. This merge is associative, and `(−∞, 0)` is its identity — mathematically. Numerically there is one trap:

> If **both** sides are `(−∞, 0)`, then `m = −∞` and `exp(m₁ − m) = exp(−∞ − (−∞)) = exp(NaN) = NaN`, and `0 · NaN = NaN`.

So the merge needs exactly one guard: `if (m == -INFINITY) return identity;`. My first version guarded `if (m₁ == -INFINITY)`, which is wrong in the other direction: when only one side is empty, the formula already works (`exp(−∞) = 0` drops that side), and the guard discarded valid data. The test that exposes it is `N < blockDim.x`, e.g. `N = 100` with 256 threads, where most threads hold the identity.

A related trap when writing the output: for a **fully masked row** (all −∞, as with padding or causal masks in attention), even after the reduction `m = −∞`, so `exp(x − m)` is NaN for every element. Setting `1/l` to 0 doesn't help, because `NaN · 0 = NaN`. The masked case has to skip the exponent entirely. (I found this one while writing the clean version of the kernel for this repo.)

This same monoid is why FlashAttention works: the softmax over a row of attention scores can be accumulated tile by tile over K/V, carrying `(m, l, O)` and rescaling `O` the same way as `l`.

## Two ways to all-reduce in a warp

```cuda
for (int off = 16; off > 0; off /= 2) v = op(v, __shfl_xor_sync(FULL, v, off));   // butterfly
for (int off = 16; off > 0; off /= 2) v = op(v, __shfl_down_sync(FULL, v, off));  // tree
```

`lane ^ off` is an involution, so the XOR version pairs lanes symmetrically, and after five rounds **every** lane holds the result. With `shfl_down`, lanes near the top read from out-of-range sources (which return their own value), so only lane 0 is guaranteed correct. For softmax and RMSNorm, every thread needs the result, so the butterfly saves a broadcast through shared memory.

## Floating point is not associative

`(a + b) + c ≠ a + (b + c)` in floating point. A tree reduction is a *different* summation order from a sequential loop, so results differ in the last bits — usually more accurate, but different. Practical consequences:

- **Determinism:** a fixed grid and block size give a fixed tree, so the result is bitwise reproducible across runs. One `atomicAdd` per block is *not* reproducible, because blocks finish in arbitrary order. [02_reduce.cu](../kernels/cuda/02_reduce.cu) has both.
- **Testing:** compare against a double-precision CPU reference with a relative tolerance, or use small integers stored in floats (sums are exact), which turns rounding noise into a clear pass/fail. The scan test does the latter.

## Appendix: RMSNorm backward

Forward, for one row of length N:

```
r = (mean(x²) + ε)^(−1/2),      y_j = x_j · r · γ_j
```

Let `u_j = γ_j · dy_j`. Then `L` depends on `x` through `y_j = x_j · r · γ_j`, and `r` depends on every `x_i`:

```
∂r/∂x_i = −½ (mean(x²) + ε)^(−3/2) · (2 x_i / N) = −(r³ / N) · x_i

dx_i = Σ_j dy_j · ∂y_j/∂x_i
     = dy_i · r · γ_i  +  Σ_j dy_j · x_j · γ_j · ∂r/∂x_i
     = r · u_i  −  (r³ / N) · x_i · Σ_j u_j x_j

dγ_j = Σ_rows dy_j · x_j · r
```

The `Σ_j u_j x_j` term is one more row reduction, so the dx kernel has the same shape as the forward kernel. dγ reduces over rows instead of columns, which calls for a different mapping (thread ↔ column). Checked against finite differences to ~1e-10; implementation in [07_rmsnorm.cu](../kernels/cuda/07_rmsnorm.cu).
