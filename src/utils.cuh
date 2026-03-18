// utils.cuh
// Error checking, verification, and benchmark utilities for HGEMM (FP16)

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CURAND(call) \
    do { \
        curandStatus_t status = call; \
        if (status != CURAND_STATUS_SUCCESS) { \
            std::cerr << "cuRAND error at " << __FILE__ << ":" << __LINE__ \
                      << ": " << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void float_to_half_kernel(const float* src, __half* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

inline void FillRandomDevice(__half* d_ptr, size_t n, unsigned long long seed = 42)
{
    float* d_tmp;
    CHECK_CUDA(cudaMalloc(&d_tmp, n * sizeof(float)));

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CHECK_CURAND(curandGenerateUniform(gen, d_tmp, n));
    CHECK_CURAND(curandDestroyGenerator(gen));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads>>>(d_tmp, d_ptr, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_tmp));
}

inline bool VerifyGEMM(const __half* d_C, const __half* d_C_ref, int size, float threshold = 5.0f)
{
    std::vector<__half> h_C(size);
    std::vector<__half> h_C_ref(size);

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, size * sizeof(__half), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C_ref, size * sizeof(__half), cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    double avg_diff = 0.0;
    for (int i = 0; i < size; ++i) {
        double val = (double)__half2float(h_C[i]);
        double ref = (double)__half2float(h_C_ref[i]);
        double diff = std::fabs(val - ref);
        max_diff = std::fmax(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= size;

    printf("  [verify] max_diff: %.6f, avg_diff: %.6f", max_diff, avg_diff);

    if (avg_diff > threshold) {
        printf(" FAIL\n");
        return false;
    }
    printf(" OK\n");
    return true;
}
