# Tiled GEMM: what the K loop is actually doing

I was stuck on the shared-memory tiled GEMM for three days. Not on the idea ("load tiles into shared memory, reuse them"), but on the picture: which thread copies what, why there's a loop over K, and what each thread holds after each round. This note is the explanation that finally worked, written for the version of me that was stuck.

Setup: `C[M][N] = A[M][K] · B[K][N]`, row-major, one block per `T × T` tile of C, one thread per C element in that tile (`T = 32` in the code, `T = 4` in the examples).

## Step 0: why tile at all

Naive: thread `(row, col)` loops over all of K, reading a row of A and a column of B from global memory. Every element of A is read N times in total, every element of B M times. Threads in the same block read the *same* rows and columns over and over.

Shared memory lets a block read each needed element from global **once** and serve it to every thread that needs it. But a block's output tile needs a `T × K` band of A and a `K × T` band of B, and K can be 4096 — too large for shared memory. So we cut K into chunks of `T` and process one chunk at a time.

That's the only reason the K loop exists: **shared memory is too small for the whole band**. Without the capacity limit, one load and one sync would do.

## The picture that was missing

I thought of the K loop as "some threads compute part of C". It is not. In **every** round:

- **every** thread of the block participates,
- **every** C element of the tile receives **one more partial sum** — the contribution of that K-chunk.

```
round 0:  C_tile += A[:, 0:T]   · B[0:T, :]
round 1:  C_tile += A[:, T:2T]  · B[T:2T, :]
...
```

The parallelism is over C (all T × T elements at once). The K loop is sequential, and the running sum stays in a register in each thread across rounds. That's why the accumulator must be declared **outside** the loop and written out **after** it — two of my bugs were the opposite.

## One round, from one thread's point of view

Thread `(ty, tx)` in a 4 × 4 tile, round `k0`:

1. **Copy 2 numbers.** `As[ty][tx] = A[row][k0 + tx]` and `Bs[ty][tx] = B[k0 + ty][col]`. Only two loads per thread per round — together, the 16 threads fill both 4 × 4 tiles.
2. **Wait** (`__syncthreads`): the next step reads elements that other threads copied.
3. **Compute** with a row of `As` and a column of `Bs`: `acc += As[ty][k] * Bs[k][tx]` for `k = 0..3`. Thread `(ty, tx)` reads things it did not load — that's the reuse.
4. **Wait** again: the next round's copy overwrites tiles that slower threads may still be reading.

The coordinates:

| Quantity | Formula | Why |
|---|---|---|
| my C element | `row = by·T + ty`, `col = bx·T + tx` | block picks the tile, thread picks the element |
| my A element this round | `A[row][k0 + tx]` | same row as my C element; the column walks K |
| my B element this round | `B[k0 + ty][col]` | same column as my C element; the row walks K |

Three rules I keep re-deriving:

- `x` ↔ column, `y` ↔ row (so `threadIdx.x` walks contiguous memory; `grid.x` covers N).
- Row-major index = `row · stride + col`, where stride is the number of **columns** of that matrix (K for A, N for B and C).
- Global coordinate = tile origin + local coordinate.

## Boundaries without deadlock

When M, N or K aren't multiples of T, some threads map outside the matrix. They **must not** `return`: there are two barriers in every round, and they must reach both. Instead they load `0` into the tile — 0 is the identity of the dot product, so padded entries contribute nothing — and skip only the final write.

## Testing

Use `M ≠ N` and sizes that aren't multiples of T (e.g. 100 × 37 × 250). Swapping `grid.x` and `grid.y` gives correct results on square matrices.

## What's next on the ladder

The tiled kernel does 2 shared-memory loads per FMA in the inner loop, so it's limited by shared-memory bandwidth, not FP32 throughput. Register tiling has each thread compute a `TM × TN` block of outputs, so `TM + TN` shared loads feed `TM · TN` FMAs. Then vectorized loads, double buffering, and Tensor Cores (`mma.sync`). Code: [09_sgemm_tiled.cu](../kernels/cuda/09_sgemm_tiled.cu).
