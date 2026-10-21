// Cooperative primitives used by every kernel in this repo.
//
// Two ideas shape this file (see notes/02 and notes/03):
//
//   1. A reduction is defined by a *monoid*: an associative combine plus an
//      identity element. The identity is what out-of-range threads contribute,
//      so padding is never a special case: sum pads with 0, max pads with -inf,
//      online softmax pads with (m = -inf, l = 0).
//
//   2. Every __syncthreads() below exists to close one specific hazard on a
//      shared-memory slot, and is labelled RAW (read-after-write) or WAR
//      (write-after-read). If you cannot name the hazard, the barrier is
//      either missing somewhere else or not needed.
//
// Contract for every block-level function in this file:
//   * blockDim.x is a multiple of 32 (so every warp is full and the full mask
//     0xffffffff is correct).
//   * EVERY thread of the block calls it, unconditionally. It contains
//     __syncthreads() and full-mask shuffles; calling it from inside a
//     thread-dependent branch (or after an early `return`) is undefined
//     behaviour and in practice a hang or garbage. Out-of-range threads must
//     participate and contribute the identity instead.
#pragma once
#include <cmath>
#include <cuda_runtime.h>

constexpr int WARP = 32;
constexpr unsigned FULL_MASK = 0xffffffffu;

// ---------------------------------------------------------------------------
// Monoids
// ---------------------------------------------------------------------------
struct SumOp {
    static __device__ __forceinline__ float identity() { return 0.0f; }
    static __device__ __forceinline__ float combine(float a, float b) { return a + b; }
};

struct MaxOp {
    static __device__ __forceinline__ float identity() { return -INFINITY; }
    static __device__ __forceinline__ float combine(float a, float b) { return fmaxf(a, b); }
};

// Online-softmax state: m = running max, l = sum of exp(x - m) over the
// elements seen so far. A single element x is (x, 1), not (x, 0).
struct MaxSum {
    float m;
    float l;
};

struct OnlineSoftmaxOp {
    static __device__ __forceinline__ MaxSum identity() { return {-INFINITY, 0.0f}; }
    static __device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
        float m = fmaxf(a.m, b.m);
        // Both sides empty: exp(-inf - (-inf)) = exp(NaN). This is the ONLY
        // case that needs a guard. If just one side is -inf, its exp() term
        // is exp(-inf) = 0 and contributes nothing, which is correct.
        if (m == -INFINITY) return {-INFINITY, 0.0f};
        return {m, a.l * __expf(a.m - m) + b.l * __expf(b.m - m)};
    }
};

// ---------------------------------------------------------------------------
// Warp level
// ---------------------------------------------------------------------------

// Butterfly all-reduce. XOR pairs lanes (lane ^ off is an involution), so after
// 5 rounds every lane holds the full result -- unlike __shfl_down_sync, where
// only lane 0 is guaranteed correct.
template <class Op>
__device__ __forceinline__ float warp_allreduce(float v) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off /= 2)
        v = Op::combine(v, __shfl_xor_sync(FULL_MASK, v, off));
    return v;
}

__device__ __forceinline__ MaxSum warp_allreduce_ms(MaxSum s) {
#pragma unroll
    for (int off = WARP / 2; off > 0; off /= 2) {
        MaxSum other{__shfl_xor_sync(FULL_MASK, s.m, off),
                     __shfl_xor_sync(FULL_MASK, s.l, off)};
        s = OnlineSoftmaxOp::combine(s, other);
    }
    return s;
}

// Hillis-Steele inclusive scan inside one warp.
// Invariant after the round with offset d: lane i holds the sum of the window
// [i - 2d + 1, i] (clipped at 0). Offsets go 1, 2, 4, 8, 16 so windows double.
// The shuffle is executed by ALL lanes; only the add is predicated. Putting the
// shuffle itself inside `if (lane >= d)` would violate the full mask.
__device__ __forceinline__ float warp_inclusive_scan(float v) {
    int lane = threadIdx.x % WARP;
#pragma unroll
    for (int d = 1; d < WARP; d *= 2) {
        float up = __shfl_up_sync(FULL_MASK, v, d);
        if (lane >= d) v += up;
    }
    return v;
}

// ---------------------------------------------------------------------------
// Block level (contract at top of file applies)
// ---------------------------------------------------------------------------

// Two-level all-reduce; the result is broadcast to every thread.
template <class Op>
__device__ float block_allreduce(float v) {
    __shared__ float partial[WARP];  // one slot per warp, at most 32 warps
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int num_warps = blockDim.x / WARP;

    v = warp_allreduce<Op>(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();  // RAW: warp 0 is about to read every warp's slot

    if (warp == 0) {
        float t = (lane < num_warps) ? partial[lane] : Op::identity();
        t = warp_allreduce<Op>(t);
        if (lane == 0) partial[0] = t;
    }
    __syncthreads();  // RAW: everyone is about to read partial[0]

    float result = partial[0];
    __syncthreads();  // WAR: a second call would overwrite partial[0] while
                      // slower warps are still reading it (softmax/RMSNorm
                      // call this twice in a row)
    return result;
}

__device__ MaxSum block_allreduce_ms(MaxSum s) {
    __shared__ float pm[WARP];
    __shared__ float pl[WARP];
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int num_warps = blockDim.x / WARP;

    s = warp_allreduce_ms(s);
    if (lane == 0) { pm[warp] = s.m; pl[warp] = s.l; }
    __syncthreads();  // RAW

    if (warp == 0) {
        MaxSum t = (lane < num_warps) ? MaxSum{pm[lane], pl[lane]} : OnlineSoftmaxOp::identity();
        t = warp_allreduce_ms(t);
        if (lane == 0) { pm[0] = t.m; pl[0] = t.l; }
    }
    __syncthreads();  // RAW

    MaxSum result{pm[0], pl[0]};
    __syncthreads();  // WAR
    return result;
}

// Block-wide inclusive scan. Also returns the block total in *block_total
// (the value held by the last thread), which is what a multi-block scan needs.
__device__ float block_inclusive_scan(float v, float* block_total) {
    __shared__ float warp_sums[WARP];
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int num_warps = blockDim.x / WARP;

    // Step 1: scan inside each warp.
    v = warp_inclusive_scan(v);

    // Step 2: the warp total lives in lane 31 after an INCLUSIVE scan
    // (lane 0 only holds its own element).
    if (lane == WARP - 1) warp_sums[warp] = v;
    __syncthreads();  // RAW: warp 0 reads all warp totals

    // Step 3: scan the warp totals in place. Safe without another barrier:
    // only warp 0 touches warp_sums here, and each lane reads and writes only
    // its own slot.
    if (warp == 0) {
        float t = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        t = warp_inclusive_scan(t);
        if (lane < num_warps) warp_sums[lane] = t;  // warp_sums[w] = total of warps 0..w
    }
    __syncthreads();  // RAW: every warp reads the scanned totals

    // Step 4: add the total of all EARLIER warps (w - 1, not w: the warp's own
    // total is already inside v).
    if (warp > 0) v += warp_sums[warp - 1];
    if (block_total) *block_total = warp_sums[num_warps - 1];
    __syncthreads();  // WAR: the next call rewrites warp_sums
    return v;
}
