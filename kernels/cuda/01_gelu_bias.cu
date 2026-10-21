// y[r][c] = GeLU(x[r][c] + bias[c]),  x, y: [M, N] row-major, bias: [N]
//
// Class (notes/01): (1) element-wise  + modifier a (vectorized)  + c (fused).
// Why fuse: unfused bias-add then GeLU moves 4 * M*N floats through HBM
// (read x, write tmp, read tmp, write y); fused moves 2 * M*N. The kernel is
// memory-bound (~10 FLOP per 8 bytes), so traffic is the runtime.
#include "common/check.cuh"

// GeLU(x) = x * Phi(x) = 0.5 x (1 + erf(x / sqrt(2))). Multiply by the constant
// 1/sqrt(2) instead of dividing by sqrtf(2): division is a multi-instruction
// sequence on the GPU and the compiler may not fold it under IEEE rules.
__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

// v1: scalar, grid-stride. Works for any M, N.
__global__ void gelu_bias_scalar(const float* __restrict__ x, const float* __restrict__ bias,
                                 float* __restrict__ y, int M, int N) {
    const size_t total = (size_t)M * N;  // M*N can exceed 2^31 for real activations
    // 1. hardware coordinates
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    // 2. f: this thread owns outputs tid, tid + stride, ...
    for (size_t i = tid; i < total; i += stride) {
        // 3. g: y[i] depends on x[i] and bias[col]
        int col = (int)(i % N);
        // 4. work, data coordinates only
        y[i] = gelu(x[i] + bias[col]);
    }
}

// v2: float4. Preconditions: N % 4 == 0 (so a float4 never straddles two rows,
// and bias4 is aligned) and 16-byte aligned pointers (cudaMalloc guarantees
// 256 B). One 128-bit load replaces four 32-bit loads: same bytes, 4x fewer
// memory instructions, more bytes in flight per thread.
__global__ void gelu_bias_vec4(const float4* __restrict__ x4, const float4* __restrict__ bias4,
                               float4* __restrict__ y4, int M, int N) {
    const size_t total4 = (size_t)M * N / 4;
    const int N4 = N / 4;
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < total4; i += stride) {
        float4 v = x4[i];                 // 1 x LDG.128 into registers
        float4 b = bias4[i % N4];
        float4 o;                         // assemble in registers ...
        o.x = gelu(v.x + b.x);
        o.y = gelu(v.y + b.y);
        o.z = gelu(v.z + b.z);
        o.w = gelu(v.w + b.w);
        y4[i] = o;                        // ... then 1 x STG.128. Writing y4[i].x
                                          // etc. directly would emit 4 scalar stores.
    }
}

void gelu_bias(const float* x, const float* bias, float* y, int M, int N) {
    const int block = 256;
    // Enough blocks to fill the GPU several times over; grid-stride covers the rest.
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int grid = sms * 8;
    if (N % 4 == 0)
        gelu_bias_vec4<<<grid, block>>>(reinterpret_cast<const float4*>(x),
                                        reinterpret_cast<const float4*>(bias),
                                        reinterpret_cast<float4*>(y), M, N);
    else
        gelu_bias_scalar<<<grid, block>>>(x, bias, y, M, N);
    CUDA_CHECK_LAUNCH();
}

static bool run_case(int M, int N) {
    auto hx = random_vector((size_t)M * N, -4.f, 4.f, 1);
    auto hb = random_vector(N, -1.f, 1.f, 2);
    std::vector<float> ref((size_t)M * N);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c) {
            double v = (double)hx[(size_t)r * N + c] + hb[c];
            ref[(size_t)r * N + c] = (float)(0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))));
        }
    DeviceBuffer<float> dx(hx), db(hb), dy((size_t)M * N);
    gelu_bias(dx, db, dy, M, N);
    return all_close(dy.download(), ref, 1e-5f, 1e-6f, "gelu_bias");
}

int main() {
    report(run_case(1, 4), "gelu_bias M=1 N=4 (vec4)");
    report(run_case(1000, 1024), "gelu_bias M=1000 N=1024 (vec4)");
    report(run_case(1000, 1023), "gelu_bias M=1000 N=1023 (scalar fallback)");
    report(run_case(3, 1), "gelu_bias M=3 N=1 (scalar fallback)");
    return 0;
}
