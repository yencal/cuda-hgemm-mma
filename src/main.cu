// main.cu
// Benchmark runner for PTX MMA HGEMM implementations (FP16)
//
// NOTE: All kernels expect B in standard layout B[K,N] row-major.

#include <iostream>
#include <vector>

#include "utils.cuh"
#include "00_cublas.cuh"
#include "01a_mma_direct.cuh"
#include "01b_mma_ldmatrix.cuh"
#include "02_mma_swizzle.cuh"
#include "03_mma_multistage.cuh"
#include "04_mma_pipelining.cuh"
#include "autotune.cuh"

int main(int argc, char** argv)
{
    std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
    std::vector<BenchmarkResult> results;

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // ========================================
    // Benchmark loop - autotune all kernels per size
    // ========================================
    for (int N : sizes) {
        int M = N, K = N;

        std::cout << "\n========================================" << std::endl;
        std::cout << "N = " << N << " (" << (2.0 * M * N * K / 1e9) << " GFLOPs)" << std::endl;
        std::cout << "========================================" << std::endl;

        __half *d_A, *d_B, *d_C, *d_C_ref;
        CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&d_C_ref, (size_t)M * N * sizeof(__half)));

        FillRandomDevice(d_A, (size_t)M * K);
        FillRandomDevice(d_B, (size_t)K * N);

        // Generate reference
        HGEMMCuBLAS::Run(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);
        CHECK_CUDA(cudaDeviceSynchronize());

        // 00: cuBLAS reference
        results.push_back(RunCuBLASBenchmark<HGEMMCuBLAS>(
            "00_cuBLAS", handle, M, N, K, alpha, d_A, d_B, beta, d_C));

        // 1a: MMADirect
        printf("\nAutotuning 1a_MMADirect for N=%d\n", N);
        RunAutotune<MMADirectTag>(GetMMABasicVariants<MMADirect>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<MMADirectTag>>(
            "1a_MMADirect", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 1b: MMALdmatrix
        printf("\nAutotuning 1b_MMALdmatrix for N=%d\n", N);
        RunAutotune<MMALdmatrixTag>(GetMMABasicVariants<MMALdmatrix>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<MMALdmatrixTag>>(
            "1b_MMALdmatrix", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 02: MMASwizzle
        printf("\nAutotuning 02_MMASwizzle for N=%d\n", N);
        RunAutotune<MMASwizzleTag>(GetMMASwizzleVariants<MMASwizzle>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<MMASwizzleTag>>(
            "02_MMASwizzle", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 03: MMAMultistage
        printf("\nAutotuning 03_MMAMultistage for N=%d\n", N);
        RunAutotune<MMAMultistageTag>(GetMMAMultistageVariants<MMAMultistage>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<MMAMultistageTag>>(
            "03_MMAMultistage", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        // 04: MMAPipelining
        printf("\nAutotuning 04_MMAPipelining for N=%d\n", N);
        RunAutotune<MMAPipeliningTag>(GetMMAMultistageVariants<MMAPipelining>(), N);
        CHECK_CUDA(cudaMemset(d_C, 0, (size_t)M * N * sizeof(__half)));
        results.push_back(RunBenchmark<Autotuned<MMAPipeliningTag>>(
            "04_MMAPipelining", M, N, K, alpha, d_A, d_B, beta, d_C, d_C_ref));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaFree(d_C_ref));
    }
    CHECK_CUBLAS(cublasDestroy(handle));

    WriteCSV(results, "hgemm_results.csv");
    std::cout << "\nResults saved to hgemm_results.csv" << std::endl;

    return 0;
}
