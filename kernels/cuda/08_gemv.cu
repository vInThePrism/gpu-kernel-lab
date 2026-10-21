// y = W x,  W: [M, K] row-major, x: [K], y: [M]
//
// Class (notes/01): (3) row reduction -- GEMM with N = 1, so there is no reuse
// of W at all: every element of W is read exactly once. Arithmetic intensity
// is ~2 FLOP per 4 bytes => firmly memory-bound; the goal is HBM bandwidth.
// x is read by every row, but it is small and stays in L2/L1 after first touch,
// so it does NOT need a per-row offset and need not be staged in smem.
//
// Which unit owns a row depends on K:
//   warp-per-row  : K small/medium. No smem, no __syncthreads, one block = several rows.
//   block-per-row : K large. More threads in flight per row; two-level reduce.
// (Decode-time GEMV in LLM inference adds split-K and fused dequant on top.)
#include "common/check.cuh"
#include "common/primitives.cuh"

__global__ void gemv_warp_per_row(const float* __restrict__ W, const float* __restrict__ x,
                                  float* __restrict__ y, int M, int K) {
    // 1. hardware coordinates
    const int lane = threadIdx.x % WARP;
    const int warp = threadIdx.x / WARP;
    const int warps_per_block = blockDim.x / WARP;
    // 2. f: warp <-> row; lane <-> columns lane, lane+32, ...
    const int row = blockIdx.x * warps_per_block + warp;
    // Early exit is safe only because `row` is WARP-uniform: either the whole
    // warp leaves or none of it does, so the full-mask shuffle stays valid.
    if (row >= M) return;
    const float* wr = W + (size_t)row * K;  // size_t: row*K overflows int at 2^31

    // 3. g: y[row] = sum_k W[row][k] * x[k]   4. work
    float acc = 0.0f;
    for (int k = lane; k < K; k += WARP) acc += wr[k] * x[k];  // consecutive lanes, consecutive k
    acc = warp_allreduce<SumOp>(acc);
    if (lane == 0) y[row] = acc;            // exactly one writer per output
}

__global__ void gemv_block_per_row(const float* __restrict__ W, const float* __restrict__ x,
                                   float* __restrict__ y, int M, int K) {
    const int row = blockIdx.x;             // f: block <-> row (grid = M, so no bound check)
    const float* wr = W + (size_t)row * K;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) acc += wr[k] * x[k];
    acc = block_allreduce<SumOp>(acc);
    if (threadIdx.x == 0) y[row] = acc;
}

void gemv(const float* W, const float* x, float* y, int M, int K) {
    if (K <= 1024) {
        const int block = 256;  // 8 rows per block
        gemv_warp_per_row<<<ceil_div(M, block / WARP), block>>>(W, x, y, M, K);
    } else {
        gemv_block_per_row<<<M, 256>>>(W, x, y, M, K);
    }
    CUDA_CHECK_LAUNCH();
}

static void run_case(int M, int K) {
    auto hW = random_vector((size_t)M * K, -1, 1, 31);
    auto hx = random_vector(K, -1, 1, 32);
    std::vector<float> ref(M);
    for (int r = 0; r < M; ++r) {
        double acc = 0;
        for (int k = 0; k < K; ++k) acc += (double)hW[(size_t)r * K + k] * hx[k];
        ref[r] = (float)acc;
    }
    DeviceBuffer<float> dW(hW), dx(hx), dy(M);
    gemv(dW, dx, dy, M, K);
    char name[64];
    std::snprintf(name, sizeof name, "gemv M=%d K=%d (%s)", M, K, K <= 1024 ? "warp/row" : "block/row");
    report(all_close(dy.download(), ref, 1e-4f, 1e-4f, name), name);
}

int main() {
    run_case(1, 1);
    run_case(1000, 100);    // M not a multiple of rows-per-block
    run_case(4096, 1024);
    run_case(1000, 4097);
    run_case(64, 32000);
    return 0;
}
