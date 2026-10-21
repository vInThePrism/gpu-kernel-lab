// RMSNorm forward + backward.  x, y, dy, dx: [M, N];  gamma, dgamma: [N]
//
//   r_i   = 1 / sqrt(mean_j(x_ij^2) + eps)          (per row; saved as rstd)
//   y_ij  = x_ij * r_i * gamma_j
//
// Backward (derivation in notes/03-reductions-are-monoids.md, appendix):
//   let  u_ij = gamma_j * dy_ij
//   dx_ij     = r_i * u_ij  -  (r_i^3 / N) * x_ij * sum_k(u_ik * x_ik)
//   dgamma_j  = sum_i dy_ij * x_ij * r_i
//
// Classes (notes/01): forward and dx are (3) row reductions, one block per row.
// dgamma reduces over ROWS instead -- a column reduction -- so it gets its own
// mapping: thread <-> column, loop over rows (consecutive threads read
// consecutive columns, so every row read is coalesced).
//
// Memory-bound: forward reads x twice + gamma, writes y. ~0.5 FLOP/byte.
#include "common/check.cuh"
#include "common/primitives.cuh"

__global__ void rmsnorm_fwd(const float* __restrict__ x, const float* __restrict__ gamma,
                            float* __restrict__ y, float* __restrict__ rstd, int M, int N, float eps) {
    const int row = blockIdx.x;                       // f: block <-> row
    const float* xr = x + (size_t)row * N;
    float* yr = y + (size_t)row * N;

    float ss = 0.0f;                                  // sum of squares, not square of sum
    for (int c = threadIdx.x; c < N; c += blockDim.x) ss += xr[c] * xr[c];
    ss = block_allreduce<SumOp>(ss);                  // broadcast to all threads

    const float r = rsqrtf(ss / N + eps);             // float division: never 1/N on ints
    if (threadIdx.x == 0) rstd[row] = r;              // saved for backward
    for (int c = threadIdx.x; c < N; c += blockDim.x) yr[c] = xr[c] * r * gamma[c];
}

__global__ void rmsnorm_bwd_dx(const float* __restrict__ x, const float* __restrict__ gamma,
                               const float* __restrict__ dy, const float* __restrict__ rstd,
                               float* __restrict__ dx, int M, int N) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * N;
    const float* dyr = dy + (size_t)row * N;
    float* dxr = dx + (size_t)row * N;
    const float r = rstd[row];

    float dot = 0.0f;                                 // sum_k gamma_k dy_k x_k
    for (int c = threadIdx.x; c < N; c += blockDim.x) dot += gamma[c] * dyr[c] * xr[c];
    dot = block_allreduce<SumOp>(dot);

    const float coef = r * r * r * dot / N;
    for (int c = threadIdx.x; c < N; c += blockDim.x)
        dxr[c] = r * gamma[c] * dyr[c] - coef * xr[c];
}

__global__ void rmsnorm_bwd_dgamma(const float* __restrict__ x, const float* __restrict__ dy,
                                   const float* __restrict__ rstd, float* __restrict__ dgamma,
                                   int M, int N) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;  // f: thread <-> column
    if (c >= N) return;                                   // no cooperation -> early exit OK
    float acc = 0.0f;
    for (int r = 0; r < M; ++r) acc += dy[(size_t)r * N + c] * x[(size_t)r * N + c] * rstd[r];
    dgamma[c] = acc;
    // For large M this serial loop is the bottleneck; the production fix is
    // split-M (modifier b): partial dgamma per row-chunk, then a second reduce.
}

static void run_case(int M, int N) {
    const float eps = 1e-6f;
    auto hx = random_vector((size_t)M * N, -2, 2, 21);
    auto hg = random_vector(N, 0.5f, 1.5f, 22);
    auto hdy = random_vector((size_t)M * N, -1, 1, 23);

    std::vector<float> ry((size_t)M * N), rdx((size_t)M * N), rdg(N, 0.f);
    std::vector<double> dg(N, 0.0);
    for (int i = 0; i < M; ++i) {
        const float* xr = &hx[(size_t)i * N];
        const float* dyr = &hdy[(size_t)i * N];
        double ss = 0, dot = 0;
        for (int j = 0; j < N; ++j) ss += (double)xr[j] * xr[j];
        double r = 1.0 / std::sqrt(ss / N + eps);
        for (int j = 0; j < N; ++j) {
            ry[(size_t)i * N + j] = (float)(xr[j] * r * hg[j]);
            dot += (double)hg[j] * dyr[j] * xr[j];
            dg[j] += (double)dyr[j] * xr[j] * r;
        }
        for (int j = 0; j < N; ++j)
            rdx[(size_t)i * N + j] = (float)(r * hg[j] * dyr[j] - r * r * r / N * xr[j] * dot);
    }
    for (int j = 0; j < N; ++j) rdg[j] = (float)dg[j];

    DeviceBuffer<float> dx_in(hx), dgam(hg), ddy(hdy), dy_out((size_t)M * N), drstd(M),
        ddx((size_t)M * N), ddg(N);
    rmsnorm_fwd<<<M, 256>>>(dx_in, dgam, dy_out, drstd, M, N, eps);
    CUDA_CHECK_LAUNCH();
    rmsnorm_bwd_dx<<<M, 256>>>(dx_in, dgam, ddy, drstd, ddx, M, N);
    CUDA_CHECK_LAUNCH();
    rmsnorm_bwd_dgamma<<<ceil_div(N, 256), 256>>>(dx_in, ddy, drstd, ddg, M, N);
    CUDA_CHECK_LAUNCH();

    char name[96];
    std::snprintf(name, sizeof name, "rmsnorm fwd M=%d N=%d", M, N);
    report(all_close(dy_out.download(), ry, 1e-4f, 1e-5f, name), name);
    std::snprintf(name, sizeof name, "rmsnorm bwd dx M=%d N=%d", M, N);
    report(all_close(ddx.download(), rdx, 1e-3f, 1e-4f, name), name);
    std::snprintf(name, sizeof name, "rmsnorm bwd dgamma M=%d N=%d", M, N);
    report(all_close(ddg.download(), rdg, 1e-3f, 1e-3f, name), name);
}

int main() {
    run_case(1, 1);
    run_case(3, 100);      // N < blockDim
    run_case(64, 4096);    // LLaMA-ish hidden size
    run_case(128, 5000);   // not a multiple of 32 or of blockDim
    return 0;
}
