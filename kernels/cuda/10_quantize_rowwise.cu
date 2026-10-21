// Per-row (per-channel) symmetric quantization of x: [M, N] to INT8 and FP8-E4M3.
//
//   scale_r = absmax_r / QMAX           (QMAX = 127 for INT8, 448 for E4M3)
//   q_ij    = quantize(x_ij / scale_r)
//   x_ij   ~= q_ij * scale_r            (dequantize)
//
// Class (notes/01): (3) row reduction (absmax) followed by (1) element-wise,
// with modifier d (low precision). Two passes over the row; memory-bound.
//
// INT8 vs E4M3: INT8 levels are evenly spaced, so a single outlier stretches
// the scale and crushes small values. E4M3 levels are log-spaced, so relative
// error is roughly constant (~2^-4 with 3 mantissa bits) across magnitudes.
//
// __NV_SATFINITE: E4M3 has no infinity. Without saturation, a value that
// rounds past 448 becomes NaN and poisons everything downstream. With exact
// absmax scaling nothing should exceed 448, but rounding of x/scale can land
// just above it, so saturate anyway.
#include "common/check.cuh"
#include "common/primitives.cuh"
#include <cuda_fp8.h>
#include <cstdint>

constexpr float INT8_MAX_F = 127.0f;
constexpr float E4M3_MAX_F = 448.0f;
constexpr float SCALE_EPS = 1e-12f;  // all-zero row -> scale 0 -> 0/0; clamp it

template <bool kFp8>
__global__ void quantize_rowwise(const float* __restrict__ x, uint8_t* __restrict__ q,
                                 float* __restrict__ scales, int M, int N) {
    const int row = blockIdx.x;                         // f: block <-> row
    const float* xr = x + (size_t)row * N;
    uint8_t* qr = q + (size_t)row * N;

    float amax = 0.0f;                                  // |x| >= 0, so 0 is a valid identity
    for (int c = threadIdx.x; c < N; c += blockDim.x) amax = fmaxf(amax, fabsf(xr[c]));
    amax = block_allreduce<MaxOp>(amax);                // every thread gets it

    const float scale = fmaxf(amax / (kFp8 ? E4M3_MAX_F : INT8_MAX_F), SCALE_EPS);
    const float inv_scale = 1.0f / scale;
    // Use the register copy of `scale` below. Re-reading scales[row] from
    // global after thread 0 writes it would be a cross-thread race.
    if (threadIdx.x == 0) scales[row] = scale;

    for (int c = threadIdx.x; c < N; c += blockDim.x) {
        float v = xr[c] * inv_scale;
        if (kFp8) {
            qr[c] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        } else {
            int iv = __float2int_rn(v);                 // round to nearest even
            iv = max(-127, min(127, iv));               // symmetric: never use -128
            qr[c] = (uint8_t)(int8_t)iv;
        }
    }
}

template <bool kFp8>
__global__ void dequantize_rowwise(const uint8_t* __restrict__ q, const float* __restrict__ scales,
                                   float* __restrict__ x, int M, int N) {
    const size_t total = (size_t)M * N;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const float s = scales[i / N];
        float v;
        if (kFp8) v = __half2float(__half(__nv_cvt_fp8_to_halfraw(q[i], __NV_E4M3)));
        else      v = (float)(int8_t)q[i];
        x[i] = v * s;
    }
}

template <bool kFp8>
static void run_case(int M, int N, bool with_outlier, bool zero_row) {
    auto h = random_vector((size_t)M * N, -1, 1, 51);
    if (with_outlier) h[3] = 80.0f;                     // stretches row 0's scale
    if (zero_row) for (int c = 0; c < N; ++c) h[(size_t)(M - 1) * N + c] = 0.0f;

    DeviceBuffer<float> dx(h), ds(M), dback((size_t)M * N);
    DeviceBuffer<uint8_t> dq((size_t)M * N);
    quantize_rowwise<kFp8><<<M, 256>>>(dx, dq, ds, M, N);
    CUDA_CHECK_LAUNCH();
    dequantize_rowwise<kFp8><<<256, 256>>>(dq, ds, dback, M, N);
    CUDA_CHECK_LAUNCH();
    auto back = dback.download();
    auto sc = ds.download();

    // Error bounds. INT8: rounding error <= scale/2. E4M3: relative error
    // <= 2^-4 for normals, plus an absolute floor for the subnormal range
    // (smallest subnormal is 2^-9, so absolute error <= scale * 2^-10).
    bool ok = true;
    for (int r = 0; r < M && ok; ++r)
        for (int c = 0; c < N; ++c) {
            size_t i = (size_t)r * N + c;
            float err = std::fabs(back[i] - h[i]);
            float tol = kFp8 ? 0.0625f * std::fabs(h[i]) + sc[r] * 0.0009765625f + 1e-30f
                             : 0.5f * sc[r] * 1.0001f + 1e-30f;
            if (!(err <= tol)) {
                std::printf("  row %d col %d: x=%g back=%g err=%g tol=%g\n", r, c, h[i], back[i], err, tol);
                ok = false;
                break;
            }
        }
    char name[96];
    std::snprintf(name, sizeof name, "quantize %s M=%d N=%d%s%s", kFp8 ? "fp8-e4m3" : "int8", M, N,
                  with_outlier ? " outlier" : "", zero_row ? " zero-row" : "");
    report(ok, name);
}

int main() {
    run_case<false>(4, 100, false, false);
    run_case<false>(64, 4096, true, true);
    run_case<true>(4, 100, false, false);
    run_case<true>(64, 4096, true, true);
    return 0;
}
