// Global sum and max of a float array.
//
// Class (notes/01): (2) global reduction. Inside a block, threads cooperate
// via shuffles + shared memory; ACROSS blocks there is no barrier inside a
// kernel, so the last step needs one of:
//   (a) a second kernel launch over the per-block partials  -> deterministic
//   (b) one atomic per block                                 -> nondeterministic for floats
// Both are shown. Every thread first accumulates many elements serially
// (grid-stride, modifier a), which amortizes the fixed cost of the tree.
#include "common/check.cuh"
#include "common/primitives.cuh"

// Pass 1: each block writes one partial result.
template <class Op>
__global__ void reduce_partial(const float* __restrict__ in, float* __restrict__ partial, size_t n) {
    // 1. hardware coordinates
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    // 2. f: block b owns output partial[b]; inside it, this thread owns the
    //    strided slice tid, tid + stride, ... of the input.
    // 3. g: partial[b] = Op over that block's slices.
    float acc = Op::identity();                // threads with no elements contribute the identity
    for (size_t i = tid; i < n; i += stride) acc = Op::combine(acc, in[i]);
    // 4. cooperate. Called by ALL threads, even those whose loop never ran.
    acc = block_allreduce<Op>(acc);
    if (threadIdx.x == 0) partial[blockIdx.x] = acc;  // one writer per output
}

// atomicMax has no float overload. For IEEE floats, the integer order of the
// bit patterns matches the float order for non-negatives and is reversed for
// negatives, so: positives -> signed int max, negatives -> unsigned int min.
__device__ void atomic_max_float(float* addr, float v) {
    if (v >= 0.0f)
        atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
    else
        atomicMin(reinterpret_cast<unsigned int*>(addr), __float_as_uint(v));
}

// Single launch: one atomic per BLOCK (not per element). Contention is
// grid-size atomics on one address instead of n.
__global__ void reduce_sum_atomic(const float* __restrict__ in, float* out, size_t n) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = tid; i < n; i += stride) acc += in[i];
    acc = block_allreduce<SumOp>(acc);
    if (threadIdx.x == 0) atomicAdd(out, acc);
}

__global__ void reduce_max_atomic(const float* __restrict__ in, float* out, size_t n) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = -INFINITY;
    for (size_t i = tid; i < n; i += stride) acc = fmaxf(acc, in[i]);
    acc = block_allreduce<MaxOp>(acc);
    if (threadIdx.x == 0) atomic_max_float(out, acc);
}

// Deterministic two-pass driver: pass 1 with many blocks, pass 2 with one.
template <class Op>
float reduce_two_pass(const float* d_in, size_t n) {
    const int block = 256;
    const int grid = 512;  // fixed grid => fixed summation order => bitwise reproducible
    DeviceBuffer<float> partial(grid), result(1);
    reduce_partial<Op><<<grid, block>>>(d_in, partial, n);
    CUDA_CHECK_LAUNCH();
    reduce_partial<Op><<<1, block>>>(partial, result, grid);
    CUDA_CHECK_LAUNCH();
    return result.download()[0];
}

static void run_case(size_t n) {
    auto h = random_vector(n, -1.f, 1.f, 7);
    if (n > 3) h[n / 3] = 5.0f;  // known max somewhere in the middle
    double ref_sum = 0;
    float ref_max = -INFINITY;
    for (float v : h) { ref_sum += v; ref_max = std::fmax(ref_max, v); }

    DeviceBuffer<float> d(h);
    char name[96];

    float s = reduce_two_pass<SumOp>(d, n);
    std::snprintf(name, sizeof name, "reduce_sum two-pass n=%zu", n);
    report(std::fabs(s - ref_sum) <= 1e-3 + 1e-5 * std::fabs(ref_sum), name);

    float m = reduce_two_pass<MaxOp>(d, n);
    std::snprintf(name, sizeof name, "reduce_max two-pass n=%zu", n);
    report(m == ref_max, name);

    DeviceBuffer<float> out(1);
    std::vector<float> zero{0.0f}, ninf{-INFINITY};
    out.upload(zero);
    reduce_sum_atomic<<<512, 256>>>(d, out, n);
    CUDA_CHECK_LAUNCH();
    std::snprintf(name, sizeof name, "reduce_sum atomic n=%zu", n);
    report(std::fabs(out.download()[0] - ref_sum) <= 1e-3 + 1e-5 * std::fabs(ref_sum), name);

    out.upload(ninf);
    reduce_max_atomic<<<512, 256>>>(d, out, n);
    CUDA_CHECK_LAUNCH();
    std::snprintf(name, sizeof name, "reduce_max atomic n=%zu", n);
    report(out.download()[0] == ref_max, name);
}

int main() {
    for (size_t n : {size_t(1), size_t(31), size_t(33), size_t(1000), size_t(1) << 22, size_t(10000019)})
        run_case(n);
    // All-negative input exercises the negative branch of atomic_max_float.
    std::vector<float> neg = {-3.f, -1.5f, -7.f, -2.f};
    DeviceBuffer<float> d(neg), out(1);
    std::vector<float> ninf{-INFINITY};
    out.upload(ninf);
    reduce_max_atomic<<<2, 64>>>(d, out, neg.size());
    CUDA_CHECK_LAUNCH();
    report(out.download()[0] == -1.5f, "reduce_max atomic all-negative");
    return 0;
}
