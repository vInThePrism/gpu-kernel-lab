// Small host-side helpers shared by every kernel file.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(err_), __FILE__, __LINE__);        \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Launch errors are reported lazily (launches are asynchronous), so check both
// the launch itself and the first synchronization after it.
#define CUDA_CHECK_LAUNCH()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

inline std::vector<float> random_vector(size_t n, float lo = -1.f, float hi = 1.f,
                                        unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(gen);
    return v;
}

// Relative + absolute tolerance comparison. Prints the first mismatch.
inline bool all_close(const std::vector<float>& got, const std::vector<float>& ref,
                      float rtol = 1e-4f, float atol = 1e-5f, const char* name = "") {
    if (got.size() != ref.size()) {
        std::printf("[%s] size mismatch %zu vs %zu\n", name, got.size(), ref.size());
        return false;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        float tol = atol + rtol * std::fabs(ref[i]);
        if (!(std::fabs(got[i] - ref[i]) <= tol)) {  // also catches NaN
            std::printf("[%s] mismatch at %zu: got %.7g, ref %.7g\n", name, i, got[i], ref[i]);
            return false;
        }
    }
    return true;
}

inline void report(bool ok, const char* name) {
    std::printf("%-48s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) std::exit(1);
}

// Device buffer RAII wrapper: one less place to leak on an early exit.
template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t n = 0;
    explicit DeviceBuffer(size_t count) : n(count) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
    DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) { upload(host); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const std::vector<T>& host) {
        CUDA_CHECK(cudaMemcpy(ptr, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> host(n);
        CUDA_CHECK(cudaMemcpy(host.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }
    operator T*() const { return ptr; }
};
