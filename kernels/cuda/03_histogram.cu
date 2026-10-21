// Histogram of byte values: hist[b] = #{ i : in[i] == b }, 256 bins.
//
// Class (notes/01): (6) data-dependent placement -- WHICH output an input
// touches depends on its value, so writes collide. The ladder:
//   v0  every element does atomicAdd on global hist          n global atomics
//   v1  privatize: per-block histogram in shared memory,     n shared atomics +
//       then merge it into global with one atomic per bin    grid*256 global atomics
//
// Why v1 wins: global atomics execute at L2. Many threads hitting the same
// address serialize at one L2 slice -- the atomic itself is not slow, the
// CONTENTION is. Shared-memory atomics run inside the SM and only contend
// within one block. This is the same shape as DDP: accumulate gradients
// locally over micro-batches, communicate once (notes/06).
#include "common/check.cuh"

constexpr int BINS = 256;

__global__ void histogram_global_atomic(const unsigned char* __restrict__ in,
                                        unsigned int* __restrict__ hist, size_t n) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) atomicAdd(&hist[in[i]], 1u);
}

__global__ void histogram_privatized(const unsigned char* __restrict__ in,
                                     unsigned int* __restrict__ hist, size_t n) {
    __shared__ unsigned int local[BINS];

    // Phase 0: clear the private copy (block-stride: works for any blockDim).
    for (int b = threadIdx.x; b < BINS; b += blockDim.x) local[b] = 0;
    __syncthreads();  // RAW: no thread may increment a bin before it is zeroed

    // Phase 1: accumulate privately.
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n; i += stride) atomicAdd(&local[in[i]], 1u);
    __syncthreads();  // RAW: the merge reads bins other threads incremented

    // Phase 2: commit. One global atomic per non-empty bin per block.
    for (int b = threadIdx.x; b < BINS; b += blockDim.x)
        if (local[b]) atomicAdd(&hist[b], local[b]);
}

static void run_case(size_t n, bool skewed) {
    std::vector<unsigned char> h(n);
    std::mt19937 gen(3);
    for (auto& v : h) v = skewed ? (gen() % 8 == 0 ? (unsigned char)(gen() % BINS) : 7)  // heavy contention
                                 : (unsigned char)(gen() % BINS);
    std::vector<unsigned int> ref(BINS, 0);
    for (auto v : h) ref[v]++;

    DeviceBuffer<unsigned char> d(h);
    DeviceBuffer<unsigned int> hist(BINS);
    std::vector<unsigned int> zeros(BINS, 0);
    char name[96];

    hist.upload(zeros);
    histogram_global_atomic<<<256, 256>>>(d, hist, n);
    CUDA_CHECK_LAUNCH();
    std::snprintf(name, sizeof name, "histogram global n=%zu%s", n, skewed ? " skewed" : "");
    report(hist.download() == ref, name);

    hist.upload(zeros);
    histogram_privatized<<<256, 192>>>(d, hist, n);  // blockDim < BINS on purpose
    CUDA_CHECK_LAUNCH();
    std::snprintf(name, sizeof name, "histogram privatized n=%zu%s", n, skewed ? " skewed" : "");
    report(hist.download() == ref, name);
}

int main() {
    run_case(1, false);
    run_case(1000, false);
    run_case(size_t(1) << 24, false);
    run_case(size_t(1) << 24, true);
    return 0;
}
