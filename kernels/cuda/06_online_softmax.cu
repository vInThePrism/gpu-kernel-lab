// Row-wise softmax, x: [M, N] -> y: [M, N].
//
// Class (notes/01): (3) row reduction, one block per row.
//
// Safe softmax needs max and sum-of-exp: naively 3 passes over the row
// (max, sum, write). Online softmax carries the pair (m, l) and rescales l
// whenever m grows, so max and sum come out of ONE pass:
//
//   single element x      -> (x, 1)            (not (x, 0): exp(x - x) = 1)
//   merge (m1,l1),(m2,l2) -> m = max(m1,m2),  l = l1*e^(m1-m) + l2*e^(m2-m)
//   identity              -> (-inf, 0)
//
// The merge is associative with an identity, i.e. a monoid, so it plugs into
// the same warp/block tree as a plain sum (primitives.cuh). The numerators
// exp(x - m) are never stored; pass 2 recomputes them.
//
// HBM traffic: 3-pass = 3 reads + 1 write of the row; online = 2 reads + 1 write.
// (Caching the row in registers/smem when N is small gets it to 1 + 1.)
#include "common/check.cuh"
#include "common/primitives.cuh"

__global__ void online_softmax(const float* __restrict__ x, float* __restrict__ y, int M, int N) {
    // 1. hardware coordinates
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    // 2. f: block <-> row; inside the row, thread <-> columns t, t + blockDim.x, ...
    const float* xr = x + (size_t)row * N;
    float* yr = y + (size_t)row * N;

    // Pass 1: per-thread online accumulation. Threads with no columns (N < blockDim)
    // keep the identity (-inf, 0) -- the merge guard handles it.
    MaxSum s = OnlineSoftmaxOp::identity();
    for (int c = t; c < N; c += blockDim.x)
        s = OnlineSoftmaxOp::combine(s, MaxSum{xr[c], 1.0f});

    // Cooperate: everyone receives the row's (m, l).
    s = block_allreduce_ms(s);

    // Fully masked row (all -inf): define the output as 0 rather than NaN.
    // This is the convention attention kernels use for padded queries.
    // Careful: exp(x - m) with x = m = -inf is exp(NaN), and NaN * 0 is still
    // NaN, so the masked case must skip the exp entirely, not just zero inv_l.
    // The condition is block-uniform, so this branch causes no divergence.
    if (s.m == -INFINITY) {
        for (int c = t; c < N; c += blockDim.x) yr[c] = 0.0f;
        return;  // safe: no barriers after this point
    }
    const float inv_l = 1.0f / s.l;

    // Pass 2: recompute numerators and write.
    for (int c = t; c < N; c += blockDim.x)
        yr[c] = __expf(xr[c] - s.m) * inv_l;
}

static void run_case(int M, int N, float scale, bool masked_row = false) {
    auto h = random_vector((size_t)M * N, -scale, scale, 9);
    if (masked_row)
        for (int c = 0; c < N; ++c) h[c] = -INFINITY;  // row 0 fully masked
    std::vector<float> ref((size_t)M * N);
    for (int r = 0; r < M; ++r) {
        const float* xr = &h[(size_t)r * N];
        double m = -INFINITY, l = 0;
        for (int c = 0; c < N; ++c) m = std::fmax(m, (double)xr[c]);
        for (int c = 0; c < N; ++c) l += std::exp((double)xr[c] - m);
        for (int c = 0; c < N; ++c)
            ref[(size_t)r * N + c] = (m == -INFINITY) ? 0.0f : (float)(std::exp((double)xr[c] - m) / l);
    }
    DeviceBuffer<float> dx(h), dy((size_t)M * N);
    online_softmax<<<M, 256>>>(dx, dy, M, N);
    CUDA_CHECK_LAUNCH();
    char name[96];
    std::snprintf(name, sizeof name, "online softmax M=%d N=%d scale=%g%s", M, N, scale,
                  masked_row ? " masked" : "");
    report(all_close(dy.download(), ref, 1e-4f, 1e-7f, name), name);
}

int main() {
    run_case(4, 1, 1);         // single element per row -> exactly 1.0
    run_case(4, 100, 1);       // N < blockDim: most threads hold the identity
    run_case(64, 1000, 1);
    run_case(64, 32000, 10);   // vocab-sized rows
    run_case(8, 4096, 1000);   // naive exp() would overflow; online/safe must not
    run_case(8, 300, 1, true); // fully masked row -> 0, not NaN
    return 0;
}
