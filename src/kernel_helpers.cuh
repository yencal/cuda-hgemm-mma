// kernel_helpers.cuh
// Shared memory tile loaders using cp.async (global -> shared)
//
// Non-swizzled: loadTileA_async, loadTileB_async (used by 01a, 01b)
// Swizzled:     loadTileA_async_swizzled, loadTileB_async_swizzled (used by 02+)
//
// NOTE: B is in standard layout B[K,N] row-major.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include "smem_swizzle.cuh"

// =========================================================================
// Tile Loading A: cp.async, float4 = 16 bytes = 8 halves per copy
// A is [M,K] row-major, As is [BM][BK] in shared memory
// =========================================================================

template <int BM, int BK, int NUM_THREADS>
__device__ void loadTileA_async(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((BM * BK) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BK % 8 == 0, "BK must be divisible by 8 for vectorized loads");

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);
        __pipeline_memcpy_async(
            &As[row * BK + col8 * 8],
            &A[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Tile Loading B: cp.async, float4 = 16 bytes = 8 halves per copy
// B is [K,N] row-major, Bs is [BK][BN] in shared memory
// =========================================================================

template <int BK, int BN, int NUM_THREADS>
__device__ void loadTileB_async(
    const __half *B,
    __half *Bs,
    int N,
    uint tid)
{
    constexpr int TOTAL_VEC = (BK * BN) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((BK * BN) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized loads");

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BN / 8);
        uint col8 = idx % (BN / 8);
        __pipeline_memcpy_async(
            &Bs[row * BN + col8 * 8],
            &B[row * N + col8 * 8],
            sizeof(float4)
        );
    }
}

// =========================================================================
// Swizzled Tile Loaders (02_mma_swizzle and beyond)
// Same cp.async pattern but destination address is swizzled.
// SwizzleT must provide: static int apply(int byte_offset)
// =========================================================================

template <int BM, int BK, int NUM_THREADS, typename SwizzleT>
__device__ void loadTileA_async_swizzled(
    const __half *A,
    __half *As,
    int K,
    uint tid)
{
    constexpr int TOTAL_VEC = (BM * BK) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((BM * BK) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BK % 8 == 0, "BK must be divisible by 8 for vectorized loads");

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BK / 8);
        uint col8 = idx % (BK / 8);
        int byte_offset = (row * BK + col8 * 8) * sizeof(__half);
        int swizzled = SwizzleT::apply(byte_offset);
        __pipeline_memcpy_async(
            reinterpret_cast<__half*>(reinterpret_cast<char*>(As) + swizzled),
            &A[row * K + col8 * 8],
            sizeof(float4)
        );
    }
}

template <int BK, int BN, int NUM_THREADS, typename SwizzleT>
__device__ void loadTileB_async_swizzled(
    const __half *B,
    __half *Bs,
    int N,
    uint tid)
{
    constexpr int TOTAL_VEC = (BK * BN) / 8;
    constexpr int VEC_PER_THREAD = TOTAL_VEC / NUM_THREADS;

    static_assert((BK * BN) % 8 == 0, "Tile size must be divisible by 8");
    static_assert(TOTAL_VEC % NUM_THREADS == 0, "vec count must be divisible by NUM_THREADS");
    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized loads");

    #pragma unroll
    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        uint idx = tid + i * NUM_THREADS;
        uint row = idx / (BN / 8);
        uint col8 = idx % (BN / 8);
        int byte_offset = (row * BN + col8 * 8) * sizeof(__half);
        int swizzled = SwizzleT::apply(byte_offset);
        __pipeline_memcpy_async(
            reinterpret_cast<__half*>(reinterpret_cast<char*>(Bs) + swizzled),
            &B[row * N + col8 * 8],
            sizeof(float4)
        );
    }
}
