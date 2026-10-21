// C = A B,  A: [M, K], B: [K, N], C: [M, N], all row-major FP32.
//
// Class (notes/01): (5) blocked accumulation.
//
// naive : thread <-> one C element, loops over all of K reading A and B from
//         global. Each A element is read N times, each B element M times.
// tiled : block <-> a TILE x TILE output tile. Shared memory cannot hold the
//         full TILE x K row band of A and K x TILE column band of B, so we walk K
//         in chunks of TILE. Per round:
//             load A-tile, B-tile (each thread copies ONE element of each)
//             __syncthreads()        RAW: compute reads elements others loaded
//             acc += row(A-tile) . col(B-tile)
//             __syncthreads()        WAR: next round's load overwrites tiles still being read
//
// The insight that made this click (notes/08): in each round EVERY C element
// of the tile gets one more partial sum. The K loop is not "some threads do
// part of C"; it is "all of C, one slice of K at a time". The accumulator lives
// in a register across rounds. Global reads drop by a factor of TILE.
//
// Next step (not here): register tiling -- each thread computes TM x TN
// outputs so the inner loop does TM*TN FMAs per TM+TN smem loads.
#include "common/check.cuh"

constexpr int TILE = 32;

__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
    // x <-> column (contiguous in C and B) so a warp's B reads / C writes coalesce
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;  // no barriers in this kernel -> OK
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = acc;
}

__global__ void sgemm_smem_tiled(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // 1. hardware coordinates
    const int tx = threadIdx.x, ty = threadIdx.y;
    // 2. f: this thread owns C[row][col]
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;  // declared OUTSIDE the K loop: it accumulates across rounds
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 3. g, one round: needs A[row][k0..k0+TILE) and B[k0..k0+TILE)[col].
        //    Each thread copies one element of each tile. Out-of-range threads
        //    load 0 instead of returning: they must still reach both barriers,
        //    and 0 is the identity of the dot product.
        const int a_col = k0 + tx;
        const int b_row = k0 + ty;
        As[ty][tx] = (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.0f;
        __syncthreads();  // RAW

        // As[ty][k]: same address across the warp (ty fixed) -> broadcast, no conflict
        // Bs[k][tx]: consecutive tx -> consecutive banks, no conflict
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();  // WAR
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

static void run_case(int M, int N, int K) {
    auto hA = random_vector((size_t)M * K, -1, 1, 41);
    auto hB = random_vector((size_t)K * N, -1, 1, 42);
    std::vector<float> ref((size_t)M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double acc = 0;
            for (int k = 0; k < K; ++k) acc += (double)hA[(size_t)i * K + k] * hB[(size_t)k * N + j];
            ref[(size_t)i * N + j] = (float)acc;
        }
    DeviceBuffer<float> dA(hA), dB(hB), dC((size_t)M * N);
    // grid.x walks N (columns), grid.y walks M (rows). Swapping these is the
    // classic bug that square test matrices never catch -- hence M != N below.
    dim3 block(TILE, TILE), grid(ceil_div(N, TILE), ceil_div(M, TILE));
    char name[96];

    sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_LAUNCH();
    std::snprintf(name, sizeof name, "sgemm naive M=%d N=%d K=%d", M, N, K);
    report(all_close(dC.download(), ref, 1e-3f, 1e-3f, name), name);

    sgemm_smem_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK_LAUNCH();
    std::snprintf(name, sizeof name, "sgemm smem-tiled M=%d N=%d K=%d", M, N, K);
    report(all_close(dC.download(), ref, 1e-3f, 1e-3f, name), name);
}

int main() {
    run_case(1, 1, 1);
    run_case(64, 64, 64);
    run_case(100, 37, 250);   // nothing divisible by 32, M != N
    run_case(512, 256, 1024);
    return 0;
}
