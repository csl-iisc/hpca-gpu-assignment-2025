#pragma once
#include <cuda_runtime.h>
#include <cassert>

struct CudaEventTimer {
    cudaEvent_t start{}, stop{};
    CudaEventTimer() {
        cudaEventCreateWithFlags(&start, cudaEventDefault);
        cudaEventCreateWithFlags(&stop,  cudaEventDefault);
    }
    ~CudaEventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void record_start(cudaStream_t s = 0) { cudaEventRecord(start, s); }
    float record_stop_and_elapsed_ms(cudaStream_t s = 0) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

inline void ck(cudaError_t st, const char* where) {
    if (st != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(st));
        std::abort();
    }
}
#define CK(x) ck((x), #x)