// Inclusive prefix sum over n floats, any n.
//
// Class (notes/01): (2) global scan. Same two-level shape as the block scan in
// primitives.cuh, lifted one level up, with kernel boundaries playing the role
// that __syncthreads() plays inside a block:
//
//   A  scan_tiles   : each block scans its tile, writes the tile total
//   B  scan(tile_totals)   -- recursive: there may be more tiles than one block
//   C  add_offset   : tile t adds the scanned total of tiles 0..t-1
//
// (Single-pass "decoupled look-back", as in CUB, removes the extra passes; not
// implemented here.)
#include "common/check.cuh"
#include "common/primitives.cuh"

constexpr int BLOCK = 512;  // multiple of 32 (primitives contract)

__global__ void scan_tiles(const float* __restrict__ in, float* __restrict__ out,
                           float* __restrict__ tile_totals, int n) {
    // 1. hardware coordinates
    const int tile = blockIdx.x;
    const int t = threadIdx.x;
    // 2. f: block <-> tile, thread <-> one element of the tile
    const int i = tile * BLOCK + t;
    // 3. g (this stage): local[t] = in[tile_base] + ... + in[tile_base + t]
    // 4. work. Out-of-range threads do NOT return: block_inclusive_scan has
    //    barriers. They read the identity 0 instead, which leaves every sum intact.
    float v = (i < n) ? in[i] : 0.0f;
    float total;
    v = block_inclusive_scan(v, &total);
    if (i < n) out[i] = v;
    // Because padding is 0, the last tile's total is still correct even when it
    // is partially full -- no need to find the "last valid thread".
    if (t == 0 && tile_totals) tile_totals[tile] = total;
}

// Class (1) element-wise: no cooperation, so an early return is fine here.
__global__ void add_offset(float* __restrict__ out, const float* __restrict__ scanned_totals, int n) {
    const int tile = blockIdx.x;
    const int i = tile * BLOCK + threadIdx.x;
    if (tile == 0 || i >= n) return;
    out[i] += scanned_totals[tile - 1];  // total of all EARLIER tiles
}

// Host driver. Recursion handles n up to BLOCK^k for any k.
void scan(const float* d_in, float* d_out, int n) {
    const int tiles = ceil_div(n, BLOCK);
    if (tiles == 1) {
        scan_tiles<<<1, BLOCK>>>(d_in, d_out, nullptr, n);
        CUDA_CHECK_LAUNCH();
        return;
    }
    DeviceBuffer<float> totals(tiles), scanned(tiles);
    scan_tiles<<<tiles, BLOCK>>>(d_in, d_out, totals, n);
    CUDA_CHECK_LAUNCH();
    scan(totals, scanned, tiles);  // stage B
    add_offset<<<tiles, BLOCK>>>(d_out, scanned, n);
    CUDA_CHECK_LAUNCH();
}

static void run_case(int n) {
    // Small integers keep float sums exact, so the comparison can be strict and
    // any indexing bug shows up as a wrong integer rather than rounding noise.
    std::vector<float> h(n);
    std::mt19937 gen(5);
    for (auto& v : h) v = (float)(gen() % 7) - 3.0f;
    std::vector<float> ref(n);
    double acc = 0;
    for (int i = 0; i < n; ++i) { acc += h[i]; ref[i] = (float)acc; }
    DeviceBuffer<float> din(h), dout(n);
    scan(din, dout, n);
    char name[64];
    std::snprintf(name, sizeof name, "inclusive scan n=%d", n);
    report(all_close(dout.download(), ref, 0, 0, name), name);
}

int main() {
    // 1, 31/32/33: warp-offset edges; 511/512/513: tile edges;
    // 262145 = 512*512 + 1 forces two levels of recursion.
    for (int n : {1, 31, 32, 33, 511, 512, 513, 100000, 262145, 1 << 22}) run_case(n);
    return 0;
}
