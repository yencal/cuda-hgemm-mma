// autotune.cuh
// Autotuning framework for PTX MMA HGEMM kernels (FP16)

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <functional>
#include <cfloat>
#include <cstdio>
#include <string>

#include "utils.cuh"
#include "01a_mma_direct.cuh"
#include "01b_mma_ldmatrix.cuh"
#include "02_mma_swizzle.cuh"
#include "03_mma_multistage.cuh"
#include "04_mma_pipelining.cuh"

struct TuneConfig {
    const char* name;
    std::function<void(int, int, int, __half, const __half*, const __half*, __half, __half*)> run;
};

struct MMADirectTag {};
struct MMALdmatrixTag {};
struct MMASwizzleTag {};
struct MMAMultistageTag {};
struct MMAPipeliningTag {};

template<typename Tag>
struct Autotuned {
    static inline TuneConfig config;

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        config.run(M, N, K, alpha, A, B, beta, C);
    }
};

#define TUNE_CONFIG(Kernel, BM, BN, BK, WM, WN) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #WM "x" #WN, Kernel<BM, BN, BK, WM, WN>::Run}

#define TUNE_CONFIG_MULTISTAGE(Kernel, BM, BN, BK, WM, WN, STAGES) \
    TuneConfig{#BM "x" #BN "x" #BK "_" #WM "x" #WN "_S" #STAGES, \
               Kernel<BM, BN, BK, WM, WN, STAGES>::Run}

// =========================================================================
// Kernels 1a, 1b: Direct / Ldmatrix (no swizzle, BK can be 32 or 64)
// =========================================================================
template<template<int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetMMABasicVariants() {
    return {
        // 128 threads
        TUNE_CONFIG(Kernel, 128, 128, 64, 64, 64),
        // 256 threads
        TUNE_CONFIG(Kernel, 256, 128, 64, 64, 64),
        TUNE_CONFIG(Kernel, 128, 256, 64, 64, 64),
        TUNE_CONFIG(Kernel, 128, 128, 64, 64, 32),
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 64),
        // 512 threads
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 32),
        TUNE_CONFIG(Kernel, 256, 128, 64, 64, 32),
    };
}

// =========================================================================
// Kernel 02: Swizzle (BK >= 64 enforced by static_assert)
// Same configs as basic since all use BK=64
// =========================================================================
template<template<int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetMMASwizzleVariants() {
    return {
        // 128 threads
        TUNE_CONFIG(Kernel, 128, 128, 64, 64, 64),
        // 256 threads
        TUNE_CONFIG(Kernel, 256, 128, 64, 64, 64),
        TUNE_CONFIG(Kernel, 128, 256, 64, 64, 64),
        TUNE_CONFIG(Kernel, 128, 128, 64, 64, 32),
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 64),
        // 512 threads
        TUNE_CONFIG(Kernel, 128, 128, 64, 32, 32),
        TUNE_CONFIG(Kernel, 256, 128, 64, 64, 32),
    };
}

// =========================================================================
// Kernels 03-04: Multistage / Pipelining
// BK >= 64, STAGES parameter controls pipeline depth vs smem usage
// =========================================================================
template<template<int, int, int, int, int, int> class Kernel>
inline std::vector<TuneConfig> GetMMAMultistageVariants() {
    return {
        // 128 threads, STAGES sweep
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 64, 64, 64, 2),
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 64, 64, 64, 3),
        // 256 threads
        TUNE_CONFIG_MULTISTAGE(Kernel, 256, 128, 64, 64, 64, 2),
        TUNE_CONFIG_MULTISTAGE(Kernel, 256, 128, 64, 64, 64, 3),
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 256, 64, 64, 64, 2),
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 64, 64, 32, 2),
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 64, 32, 64, 2),
        // 512 threads
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 64, 32, 32, 2),
        TUNE_CONFIG_MULTISTAGE(Kernel, 128, 128, 64, 32, 32, 3),
        TUNE_CONFIG_MULTISTAGE(Kernel, 256, 128, 64, 64, 32, 2),
    };
}

// =========================================================================
// Autotune engine
// =========================================================================

inline TuneConfig Autotune(
    const std::vector<TuneConfig>& variants,
    int M, int N, int K, __half alpha,
    const __half* A, const __half* B,
    __half beta, __half* C,
    int warmup = 5, int iters = 10)
{
    float best_time = FLT_MAX;
    TuneConfig best = variants[0];

    printf("\n[Autotune] Testing %zu configurations on %dx%dx%d...\n",
           variants.size(), M, N, K);

    for (const auto& config : variants) {
        for (int i = 0; i < warmup; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  %-40s SKIP (%s)\n", config.name, cudaGetErrorString(err));
            continue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) {
            config.run(M, N, K, alpha, A, B, beta, C);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= iters;

        double tflops = (2.0 * M * N * K) / (ms * 1e9);
        printf("  %-40s %7.3f ms  %6.2f TFLOPS\n", config.name, ms, tflops);

        if (ms < best_time) {
            best_time = ms;
            best = config;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    double best_tflops = (2.0 * M * N * K) / (best_time * 1e9);
    printf("[Autotune] Best: %s (%.3f ms, %.2f TFLOPS)\n\n",
           best.name, best_time, best_tflops);

    return best;
}

template<typename Tag>
inline void RunAutotune(
    const std::vector<TuneConfig>& variants,
    int tuneN = 8192,
    __half alpha = __float2half(1.0f),
    __half beta = __float2half(0.0f))
{
    __half *tune_A, *tune_B, *tune_C;
    CHECK_CUDA(cudaMalloc(&tune_A, (size_t)tuneN * tuneN * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&tune_B, (size_t)tuneN * tuneN * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&tune_C, (size_t)tuneN * tuneN * sizeof(__half)));

    FillRandomDevice(tune_A, (size_t)tuneN * tuneN);
    FillRandomDevice(tune_B, (size_t)tuneN * tuneN);

    Autotuned<Tag>::config = Autotune(
        variants, tuneN, tuneN, tuneN, alpha, tune_A, tune_B, beta, tune_C);

    CHECK_CUDA(cudaFree(tune_A));
    CHECK_CUDA(cudaFree(tune_B));
    CHECK_CUDA(cudaFree(tune_C));
}
