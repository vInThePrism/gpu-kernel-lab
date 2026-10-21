// out = in^T,  in: [M, N] row-major, out: [N, M] row-major.
//
// Class (notes/01): (4) fixed permutation. g is known at compile time:
// out[r][c] = in[c][r]. The problem is that coalescing wants threadIdx.x on the
// contiguous (column) dimension on BOTH sides, but the read side's columns are
// the write side's rows. Shared memory decouples the two access patterns:
//
//   read  in  : tx walks a row of the input tile  -> coalesced
//   write smem: row-wise
//   read  smem: column-wise  (this is where bank conflicts appear)
//   write out : tx walks a row of the output tile -> coalesced
//
// The transpose happens in two independent places:
//   tile level   : swap blockIdx.x / blockIdx.y when computing output coordinates
//   element level: swap the smem indices on the read side
// threadIdx is NEVER swapped in global addresses -- that would break coalescing.
//
// Bank conflicts: tile[32][32] floats, reading a column means 32 lanes hit
// addresses 32 floats apart -> all in one bank -> 32-way conflict.
//   padding: tile[32][33], stride 33 is odd, so lanes map to 32 distinct banks.
//   XOR swizzle: store logical (r, c) at physical column c ^ r. For a fixed
//     row, c -> c ^ r is a bijection (rows stay complete); for a fixed column,
//     r -> c ^ r is also a bijection (columns spread over all banks). It is a
//     Latin square. No wasted smem, and alignment of power-of-two chunks is
//     preserved (matters later for float4 / ldmatrix / TMA layouts). See notes/09.
#include "common/check.cuh"

constexpr int TILE = 32;     // = number of banks, power of two (required by XOR)
constexpr int ROWS = 8;      // block is 32 x 8; each thread moves TILE/ROWS = 4 elements

template <bool kSwizzle>
__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out,
                                int M, int N) {
    __shared__ float tile[TILE][kSwizzle ? TILE : TILE + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;

    // ---- read phase: block (bx, by) owns input tile rows by*32.., cols bx*32..
    const int in_c = blockIdx.x * TILE + tx;
    for (int j = 0; j < TILE; j += ROWS) {
        const int r = ty + j;                     // local row in the tile
        const int in_r = blockIdx.y * TILE + r;
        if (in_r < M && in_c < N) {
            const int pc = kSwizzle ? (tx ^ r) : tx;  // physical column
            tile[r][pc] = in[(size_t)in_r * N + in_c];
        }
    }
    // Outside any `if`: every thread must reach the barrier.
    __syncthreads();  // RAW: each thread reads elements written by OTHER threads

    // ---- write phase: same block owns output tile rows bx*32.., cols by*32..
    const int out_c = blockIdx.y * TILE + tx;     // tile-level swap of blockIdx
    for (int j = 0; j < TILE; j += ROWS) {
        const int r = ty + j;
        const int out_r = blockIdx.x * TILE + r;
        if (out_r < N && out_c < M) {
            // out[out_r][out_c] = in[out_c][out_r] = logical tile[tx][r]
            const int pc = kSwizzle ? (r ^ tx) : r;
            out[(size_t)out_r * M + out_c] = tile[tx][pc];  // element-level swap
        }
    }
}

template <bool kSwizzle>
void transpose(const float* in, float* out, int M, int N) {
    dim3 block(TILE, ROWS);
    dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));  // x walks columns of the input
    transpose_tiled<kSwizzle><<<grid, block>>>(in, out, M, N);
    CUDA_CHECK_LAUNCH();
}

static void run_case(int M, int N) {
    auto h = random_vector((size_t)M * N, -1, 1, 11);
    std::vector<float> ref((size_t)M * N);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) ref[(size_t)c * M + r] = h[(size_t)r * N + c];
    DeviceBuffer<float> din(h), dout((size_t)M * N);
    char name[96];

    transpose<false>(din, dout, M, N);
    std::snprintf(name, sizeof name, "transpose padding %dx%d", M, N);
    report(all_close(dout.download(), ref, 0, 0, name), name);

    transpose<true>(din, dout, M, N);
    std::snprintf(name, sizeof name, "transpose xor-swizzle %dx%d", M, N);
    report(all_close(dout.download(), ref, 0, 0, name), name);
}

int main() {
    run_case(1, 1);
    run_case(32, 32);
    run_case(1000, 777);   // neither dimension a multiple of 32
    run_case(33, 4097);
    run_case(4096, 4096);
    return 0;
}
