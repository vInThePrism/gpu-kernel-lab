// End-to-end H2D -> kernel -> D2H pipeline, three ways. Read with notes/05.
//
//   serial          : one stream, whole array: copy all in, compute, copy all out
//   streams+pinned  : C chunks round-robin over S streams; chunk i's compute
//                     overlaps chunk i+1's H2D and chunk i-1's D2H
//   streams+pageable: identical code, but host buffers from plain new[].
//                     The async copies silently stop being async (staging through
//                     a driver-owned pinned buffer, CPU in the loop) and the
//                     overlap collapses. Same code, different memory = different timeline.
//
// Upper bound on speedup: with H2D, compute, D2H times h, c, d, the pipelined
// steady state costs ~max(h, c, d) per chunk instead of h + c + d -- if the
// device has separate copy engines for each direction (check asyncEngineCount;
// T4 reports 3, but H2D and D2H still share the PCIe link budget).
//
// NVTX ranges label each phase so the Nsight Systems timeline reads like the code.
// Run:  nsys profile -t cuda,nvtx ./11_async_pipeline
#include "common/check.cuh"
#include <nvtx3/nvToolsExt.h>
#include <algorithm>

__device__ __forceinline__ float gelu(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

// Deliberately a bit heavier than one GeLU so compute is comparable to PCIe time.
__global__ void work(const float* __restrict__ in, float* __restrict__ out, size_t n, int iters) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float v = in[i];
        for (int k = 0; k < iters; ++k) v = gelu(v) + 0.001f;
        out[i] = v;
    }
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { cudaEventRecord(a); }
    // Events on the legacy default stream wait for all blocking streams, so
    // this measures the whole pipeline. Host-side std::chrono around async
    // calls would only measure launch overhead.
    float stop_ms() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

static float run_serial(const float* h_in, float* h_out, float* d_in, float* d_out, size_t n, int iters) {
    Timer t;
    t.start();
    nvtxRangePushA("serial");
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));
    work<<<1024, 256>>>(d_in, d_out, n, iters);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();
    return t.stop_ms();
}

static float run_streams(const float* h_in, float* h_out, float* d_in, float* d_out, size_t n,
                         int iters, int num_chunks, cudaStream_t* streams, int num_streams) {
    Timer t;
    t.start();
    nvtxRangePushA("pipelined");
    const size_t chunk = (n + num_chunks - 1) / num_chunks;
    for (int c = 0; c < num_chunks; ++c) {
        const size_t off = c * chunk;
        if (off >= n) break;
        const size_t len = std::min(chunk, n - off);
        cudaStream_t s = streams[c % num_streams];
        // Within one stream the three steps are ordered (a TRUE dependency).
        // Across streams nothing is ordered, so different chunks overlap.
        CUDA_CHECK(cudaMemcpyAsync(d_in + off, h_in + off, len * sizeof(float), cudaMemcpyHostToDevice, s));
        work<<<256, 256, 0, s>>>(d_in + off, d_out + off, len, iters);
        CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, len * sizeof(float), cudaMemcpyDeviceToHost, s));
    }
    nvtxRangePop();
    return t.stop_ms();  // see Timer: waits for every stream
}

int main() {
    const size_t n = size_t(1) << 26;  // 256 MB of floats each way
    const int iters = 8, chunks = 16, nstreams = 4;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("%s  asyncEngineCount=%d\n", prop.name, prop.asyncEngineCount);

    float *p_in, *p_out;  // pinned
    CUDA_CHECK(cudaMallocHost(&p_in, n * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&p_out, n * sizeof(float)));
    std::vector<float> g_in(n), g_out(n);  // pageable
    for (size_t i = 0; i < n; ++i) p_in[i] = g_in[i] = (float)((i % 1000) / 500.0 - 1.0);

    DeviceBuffer<float> d_in(n), d_out(n);
    cudaStream_t streams[nstreams];
    for (auto& s : streams) CUDA_CHECK(cudaStreamCreate(&s));

    run_serial(p_in, p_out, d_in, d_out, n, iters);  // warm-up (context, clocks)

    float t_serial = run_serial(p_in, p_out, d_in, d_out, n, iters);
    std::vector<float> ref(p_out, p_out + n);
    float t_pinned = run_streams(p_in, p_out, d_in, d_out, n, iters, chunks, streams, nstreams);
    bool ok_pinned = std::equal(ref.begin(), ref.end(), p_out);
    float t_pageable = run_streams(g_in.data(), g_out.data(), d_in, d_out, n, iters, chunks, streams, nstreams);
    bool ok_pageable = (g_out == ref);

    std::printf("serial (1 stream)         : %8.2f ms\n", t_serial);
    std::printf("%d streams, pinned host    : %8.2f ms  (%.2fx)\n", nstreams, t_pinned, t_serial / t_pinned);
    std::printf("%d streams, pageable host  : %8.2f ms  (%.2fx)\n", nstreams, t_pageable, t_serial / t_pageable);
    report(ok_pinned, "pipeline pinned matches serial");
    report(ok_pageable, "pipeline pageable matches serial");

    for (auto& s : streams) cudaStreamDestroy(s);
    cudaFreeHost(p_in);
    cudaFreeHost(p_out);
    return 0;
}
