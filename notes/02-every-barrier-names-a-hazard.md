# Every `__syncthreads()` should name a hazard

I used to place barriers by feel: "after writing shared memory, sync". That rule is incomplete in both directions. It misses barriers that are needed and adds ones that aren't. What worked was borrowing the vocabulary of CPU pipeline hazards and labelling every barrier with the hazard it closes.

For a shared-memory slot touched by different threads:

| Hazard | Pattern | What goes wrong without a barrier |
|---|---|---|
| **RAW** (read after write) | thread A writes `s[i]`, thread B then reads it | B reads stale data |
| **WAR** (write after read) | thread A reads `s[i]`, thread B then overwrites it | A reads the *new* value, or B's write lands before A's read |
| WAW | two writes to the same slot | final value depends on ordering (usually a design bug, not a barrier issue) |

**Rule:** a barrier is needed exactly when two *different* threads touch the same shared slot and at least one of the accesses is a write. If you can't name the pair of accesses, either the barrier isn't needed or a different one is missing.

## Worked example: block all-reduce

```cuda
template <class Op>
__device__ float block_allreduce(float v) {
    __shared__ float partial[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = blockDim.x / 32;

    v = warp_allreduce<Op>(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();             // (1) RAW: warp 0 reads every warp's slot

    if (warp == 0) {
        float t = (lane < nw) ? partial[lane] : Op::identity();
        t = warp_allreduce<Op>(t);
        if (lane == 0) partial[0] = t;
    }
    __syncthreads();             // (2) RAW: every thread reads partial[0]

    float r = partial[0];
    __syncthreads();             // (3) WAR: see below
    return r;
}
```

Barriers (1) and (2) are the ones everybody writes. Barrier (3) is the one I missed, and it only matters when the function is **called twice in a row**, which is exactly what softmax and RMSNorm do (max, then sum; or forward then something else). The `__shared__` array is the same storage on both calls, so without (3):

```
warp 3 (slow)                       warp 0 (fast)
-------------                       -------------
                                    call 1: writes partial[0] = result1
                                    passes barrier (2), reads partial[0]
                                    call 2: lane 0 writes partial[0] = warp-0 partial of call 2
reads partial[0]  <-- gets call 2's partial, not result1
```

Warp 0's write in call 2 overtakes warp 3's read from call 1: a WAR hazard across calls. Reading into a register and putting a barrier after the read closes it.

The same reasoning explains the two barriers in a tiled GEMM's K loop: one RAW (compute reads tile elements other threads loaded) and one WAR (the next round's loads overwrite elements still being read).

## Barriers I *don't* need

- **Within one warp exchanging through shuffles.** `__shfl_*_sync` moves values register-to-register and synchronizes the named lanes. No shared memory, no hazard.
- **When each thread reads only what it wrote itself.** Example: in a block scan, warp 0 scans `warp_sums[]` in place. Each lane reads slot `lane` and later writes slot `lane`. No other thread touches those slots during that step, so no barrier is needed between the read and the write.
- **In purely element-wise kernels.** No shared state at all. A `__syncthreads()` in a GeLU kernel is just a stall.

## The other half: who must *reach* the barrier

`__syncthreads()` is a contract that **every thread of the block** arrives. Two patterns break that contract and are undefined behaviour (in practice: hangs, or silently wrong results):

1. **A barrier inside a thread-dependent branch.**
   ```cuda
   if (warp == 0) { ...; __syncthreads(); }   // warps 1..n never arrive
   ```
2. **An early return before a barrier**, usually as a bounds check.
   ```cuda
   if (i >= n) return;      // boundary threads leave...
   ...
   __syncthreads();         // ...and the rest wait for them
   ```

The same applies to anything that *contains* a barrier or a full-mask shuffle: `block_allreduce`, `block_inclusive_scan`, CUB's `BlockReduce`. These are **collectives**: call them unconditionally from every thread.

The fix is never "skip the boundary threads"; it is "let them participate with a harmless value". That harmless value is the operation's identity element (0 for sum, −∞ for max), which is the subject of the next note: [reductions are monoids](03-reductions-are-monoids.md).

When is an early return fine? When nothing after it cooperates. `add_offset` in the scan, the dgamma kernel in RMSNorm, and the naive GEMM all return early safely. In GEMV with warp-per-row, `if (row >= M) return;` is safe for a subtler reason: `row` is the same for all 32 lanes of a warp, so whole warps leave together and the full-mask shuffles in the remaining warps are still valid.

## Checking it mechanically

`compute-sanitizer --tool racecheck` reports shared-memory RAW/WAR/WAW hazards; `--tool synccheck` reports divergent barriers. Both run on a T4, and the [Makefile](../kernels/cuda/Makefile) has a `sanitize` target.
