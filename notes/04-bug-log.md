# Bug log: every mistake I made writing the first ten kernels

Over about four weeks (late Sep – Oct 2026) I wrote reduce, scan, transpose, GeLU, softmax, RMSNorm, GEMV, quantization and tiled GEMM from a blank file, many times each, and had every version reviewed line by line. This is the full list of bugs from those reviews, grouped by root cause. Most of them compiled cleanly; several produce correct output on "nice" sizes and only fail on edge cases.

The point of publishing this: tutorials show correct code. The failure modes of someone learning are more useful, because they are the same failure modes an interviewer probes.

## A. Barriers and collectives

| Bug | Symptom | Fix |
|---|---|---|
| `__syncthreads()` inside `if (warp_id == 0)` | hang / undefined | barriers only in uniform control flow |
| `if (i >= n) return;` at the top of a kernel that later calls `__syncthreads()` (tiled GEMM, scan) | boundary blocks hang or corrupt smem | out-of-range threads load the identity and stay |
| calling `block_scan()` inside `if (i < n)` | partial-tile blocks deadlock | collectives are called by every thread |
| `__shfl_up_sync` inside a ternary / `if (lane >= d)` | full-mask shuffle with inactive lanes = undefined | shuffle unconditionally, predicate only the add |
| block reduce: read `smem[0]` without a barrier after warp 0 writes it | some warps see stale value | RAW barrier |
| block reduce called twice (max then sum) with no barrier after the final read | second call overwrites `smem[0]` before slow warps read it | WAR barrier ([note 02](02-every-barrier-names-a-hazard.md)) |
| online softmax: missing barrier before broadcasting the reduced `(m, l)` | wrong normalization on some rows | RAW barrier |

## B. Identity elements and boundaries

| Bug | Symptom | Fix |
|---|---|---|
| second-level warp reduce reads `smem[lane]` for lanes ≥ number of warps (uninitialized) | garbage when `blockDim < 1024` | `lane < nw ? smem[lane] : identity` |
| online softmax: single element as `(x, 0)` | everything normalizes to 0/NaN | `(x, 1)` |
| online softmax merge guarded on one side being −∞ | valid partials discarded when `N < blockDim` | guard only when the merged max is −∞ ([note 03](03-reductions-are-monoids.md)) |
| fully masked row: zeroed `1/l` but still computed `exp(x − m)` | NaN, because `NaN · 0 = NaN` | skip the exponent for masked rows |
| scan: read `in[i]` before checking `i < n` | out-of-bounds read | check, then read; pad with 0 |
| Hillis-Steele condition `lane > d` / missing | low lanes double-count | `lane >= d` |
| block scan adds `warp_sums[warp - 1]` for warp 0 | out-of-bounds read | only for `warp > 0` |
| `float4` loop `i <= N/4` | writes one vector past the end | `<` |
| bounds check `>` instead of `>=` | off-by-one at the last element | `>=` |

## C. Ownership: who reads, who writes

| Bug | Symptom | Fix |
|---|---|---|
| warp total taken from lane 0 after an *inclusive* scan | lane 0 holds only its own element | lane 31 holds the warp total |
| shared slot written as `smem[warp]`, read as `smem[warp_idx]` vs `smem[lane]` inconsistently | wrong partials combined | warp writes slot `warp`; warp 0's lane `l` reads slot `l` |
| per-block partial written to `out[blockIdx.x / 32]` | blocks overwrite each other | `out[blockIdx.x]` |
| GeLU bias pointer offset by row (`bias + blockIdx.x`) | wrong bias for every row but the first | bias is shared across rows: `bias[col]` |
| GEMV: vector `x` offset per row | reads past the end of `x` | only `W` gets the row offset |
| GEMV / RMSNorm: final write without `if (lane == 0)` / `if (threadIdx.x == 0)` | benign duplicate writes at best, wrong values when only lane 0 holds the result (`shfl_down`) | exactly one writer per output |
| GEMV: global row computed without the block offset | every block computes the same rows | `blockIdx.x * warps_per_block + warp` |
| M and K swapped in loop bounds | wrong for non-square shapes | name dimensions by meaning, not letter |
| GEMM: `grid(M/T, N/T)` instead of `grid(N/T, M/T)` | correct on square matrices, wrong otherwise | x ↔ columns ↔ N; **always test M ≠ N** |
| GEMM: accumulator declared inside the K loop; write-back inside the loop | only the last K tile survives | accumulate in a register across rounds, write once |
| `blockIdx` where `threadIdx` was meant (scan) | every thread in a block uses the same index | keep hardware and data coordinates in separate lines ([note 01](01-three-coordinates.md)) |
| quantization: read `in[i]` instead of `row_in[i]` | every block quantizes row 0 | compute the row pointer once |
| `warp_idx * warp_idx` instead of `warp_idx * 32` | wrong local index | — |

## D. Math

| Bug | Fix |
|---|---|
| RMSNorm: square of the sum instead of sum of squares | `ss += x * x` |
| RMSNorm: forgot `/ N`; then wrote `1 / N` with integers (= 0) | `ss / N` in float |
| softmax rescale missing the `l *` factor | `l = l * exp(m_old − m_new) + ...` |
| softmax output missing `exp()` | `exp(x − m) / l` |
| algebra slip: `e^(a+b) = e^a + e^b`, `e^(a−b) = e^a − e^b` | `e^(a+b) = e^a · e^b`; a numeric example kills it instantly |
| absmax without `fabsf` | negative outliers ignored |
| quantization `scale` declared `int` | scale truncates to 0, divide by zero |

## E. Types, memory and races

| Bug | Fix |
|---|---|
| `blockIdx.x * N` overflows `int` for large tensors | cast to `size_t` before multiplying |
| `float4` output written as `out4[i].x = ...; out4[i].y = ...` | assemble a local `float4`, store once (one 128-bit store instead of four 32-bit) |
| quantization: thread 0 writes `scales[row]` to global, others read it back | race; use the value already in a register |
| `__shared__ float s[warpSize]` | `warpSize` is a runtime variable; use a `constexpr 32` |
| `(m, l)` passed by value to the merge function | results lost; pass by reference or return a struct |
| `if (x = 0)` | `==` (enable warnings) |
| full mask typed as `0xfffffffff` (9 f's) | `0xffffffff`, or a named constant |

## F. Tooling and process

- **clangd silently adds `#include <__clang_cuda_builtin_vars.h>`** when you autocomplete `threadIdx`. It works in the editor and can break `nvcc`. It appeared in almost every file I wrote for two weeks.
- **Empty function bodies compile.** A `gelu()` or `block_reduce_sum()` stub with no `return` is undefined behaviour, not a compile error. I repeatedly sent stubbed code for review instead of running it; running it would have caught these in seconds.
- **Nice sizes hide bugs.** Square matrices hid the grid swap; `N` a multiple of 32 or of `blockDim` hid every boundary bug.

## The test sizes that catch almost all of the above

| Kernel type | Sizes |
|---|---|
| anything with warps | `n = 1, 31, 32, 33` |
| anything with blocks | `n = B − 1, B, B + 1` |
| multi-level (scan) | `n > B²` so recursion goes two levels deep |
| row kernels | `N = 1`, `N < blockDim` (e.g. 100 with 256 threads), `N` not a multiple of 32 |
| 2D kernels | `M ≠ N`, neither a multiple of the tile (e.g. 1000 × 777) |
| softmax | large magnitudes (±1000), a fully masked row |
| reductions with max | all-negative input |

Every test in [`kernels/`](../kernels/cuda) uses these.
