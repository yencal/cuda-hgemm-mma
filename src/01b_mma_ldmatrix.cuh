// 01b_mma_ldmatrix.cuh
// PTX MMA HGEMM with ldmatrix warp-collective loads from shared memory.
// Same structure as 01a_mma_direct, only the smem->register load changes:
//   load_direct.cuh  -> load_ldmatrix.cuh
//
// - cp.async for global -> shared (vectorized float4)
// - ldmatrix.x4 for A, ldmatrix.x2.trans for B (the improvement over 01a)
// - PTX mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// - Vectorized epilogue (fragments -> smem -> coalesced float4 global stores)
//
// NOTE: B is in standard layout B[K,N] row-major.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fragment.cuh"
#include "mma_ops.cuh"
#include "load_ldmatrix.cuh"
#include "epilogue.cuh"
#include "kernel_helpers.cuh"

template <int BM, int BN, int BK, int WM, int WN>
__global__ void mma_ldmatrix_kernel(
    int M, int N, int K, __half alpha,
    const __half *__restrict__ A,
    const __half *__restrict__ B,
    __half beta,
    __half *__restrict__ C)
{
    constexpr int MMA_M_TILES = WM / MMA_M;
    constexpr int MMA_N_TILES = WN / MMA_N;
    constexpr int WARPS_M = BM / WM;
    constexpr int WARPS_N = BN / WN;
    constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;

    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(BK % MMA_K == 0, "BK must be divisible by MMA_K (16)");
    static_assert(WM % MMA_M == 0, "WM must be divisible by MMA_M (16)");
    static_assert(WN % MMA_N == 0, "WN must be divisible by MMA_N (8)");
    static_assert((BM * BK) % (NUM_THREADS * 8) == 0, "A tile must divide evenly for vec4 loads");
    static_assert((BK * BN) % (NUM_THREADS * 8) == 0, "B tile must divide evenly for vec4 loads");

    extern __shared__ __half smem[];
    __half* As = smem;
    __half* Bs = smem + BM * BK;

    const uint tid = threadIdx.x;
    const uint warpId = tid / 32;
    const uint warpM = warpId / WARPS_N;
    const uint warpN = warpId % WARPS_N;

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    FragmentC acc[MMA_M_TILES][MMA_N_TILES];
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m)
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n)
            acc[m][n].fill_zero();

    for (int tileK = 0; tileK < K; tileK += BK) {
        loadTileA_async<BM, BK, NUM_THREADS>(A, As, K, tid);
        loadTileB_async<BK, BN, NUM_THREADS>(B, Bs, N, tid);
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();

        #pragma unroll
        for (int innerK = 0; innerK < BK; innerK += MMA_K) {
            FragmentA a_frag[MMA_M_TILES];
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m) {
                const __half* As_ptr = &As[(warpM * WM + m * MMA_M) * BK + innerK];
                load_fragment_ldmatrix<BK>(a_frag[m], As_ptr);
            }

            FragmentB b_frag[MMA_N_TILES];
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n) {
                const __half* Bs_ptr = &Bs[innerK * BN + warpN * WN + n * MMA_N];
                load_fragment_ldmatrix<BN>(b_frag[n], Bs_ptr);
            }

            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m)
                #pragma unroll
                for (int n = 0; n < MMA_N_TILES; ++n)
                    mma_sync(acc[m][n], a_frag[m], b_frag[n]);
        }

        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // Epilogue: fragments -> smem -> coalesced global stores
    epilogue_vec4<BM, BN, WM, WN, MMA_M_TILES, MMA_N_TILES, NUM_THREADS>(
        acc, smem, C, N, alpha, tid, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN>
struct MMALdmatrix {
    static constexpr int WARPS_M = BM / WM;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
    static constexpr size_t SMEM_AB = (BM * BK + BK * BN) * sizeof(__half);
    static constexpr size_t SMEM_C  = BM * BN * sizeof(__half);
    static constexpr size_t SMEM_SIZE = SMEM_AB > SMEM_C ? SMEM_AB : SMEM_C;

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(
                mma_ldmatrix_kernel<BM, BN, BK, WM, WN>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_SIZE
            );
            configured = true;
        }
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        mma_ldmatrix_kernel<BM, BN, BK, WM, WN><<<grid, block, SMEM_SIZE>>>(
            M, N, K, alpha, A, B, beta, C);
    }
};
