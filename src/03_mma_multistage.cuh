// 03_mma_multistage.cuh
// PTX MMA HGEMM with multi-stage async pipeline (GMEM->SMEM overlap).
// Builds on 02_mma_swizzle: adds multi-stage buffering for compute/load overlap.
//
// - Multi-stage smem buffers (default STAGES=2)
// - Prologue fills STAGES-1 buffers
// - Main loop overlaps cp.async of next tile with mma.sync of current tile
// - XOR swizzle on all smem accesses (inherited from 02)
//
// NOTE: B is in standard layout B[K,N] row-major.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "fragment.cuh"
#include "mma_ops.cuh"
#include "smem_swizzle.cuh"
#include "load_ldmatrix.cuh"
#include "epilogue.cuh"
#include "kernel_helpers.cuh"

template <int BM, int BN, int BK, int WM, int WN, int STAGES>
__global__ void mma_multistage_kernel(
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

    constexpr int M_PARAM = 4;
    constexpr int S_A = clog2(BK * 2) - M_PARAM;
    constexpr int B_A = S_A < 3 ? S_A : 3;
    constexpr int S_B = clog2(BN * 2) - M_PARAM;
    constexpr int B_B = S_B < 3 ? S_B : 3;

    using SwzA = Swizzle<B_A, M_PARAM, S_A>;
    using SwzB = Swizzle<B_B, M_PARAM, S_B>;

    static_assert(BM % WM == 0);
    static_assert(BN % WN == 0);
    static_assert(BK % MMA_K == 0);
    static_assert(WM % MMA_M == 0);
    static_assert(WN % MMA_N == 0);
    static_assert(BK >= 64, "BK must be >= 64 for full swizzle (B=3)");
    static_assert((BM * BK) % (NUM_THREADS * 8) == 0);
    static_assert((BK * BN) % (NUM_THREADS * 8) == 0);

    constexpr int A_STAGE_SIZE = BM * BK;
    constexpr int B_STAGE_SIZE = BK * BN;

    extern __shared__ __half smem[];
    __half* As = smem;
    __half* Bs = smem + STAGES * A_STAGE_SIZE;

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

    const int numTiles = K / BK;

    // ====== PROLOGUE: fill the pipeline ======
    const int prologueStages = min(STAGES - 1, numTiles);
    #pragma unroll
    for (int s = 0; s < prologueStages; ++s) {
        __half* As_stage = As + s * A_STAGE_SIZE;
        __half* Bs_stage = Bs + s * B_STAGE_SIZE;
        loadTileA_async_swizzled<BM, BK, NUM_THREADS, SwzA>(A + s * BK, As_stage, K, tid);
        loadTileB_async_swizzled<BK, BN, NUM_THREADS, SwzB>(B + s * BK * N, Bs_stage, N, tid);
        cp_async_commit();
    }

    // ====== MAIN LOOP ======
    int loadTile = STAGES - 1;

    for (int tile = 0; tile < numTiles; ++tile) {
        int computeStage = tile % STAGES;

        // --- Async load: prefetch future tile ---
        if (loadTile < numTiles) {
            int loadStage = loadTile % STAGES;
            __half* As_stage = As + loadStage * A_STAGE_SIZE;
            __half* Bs_stage = Bs + loadStage * B_STAGE_SIZE;
            loadTileA_async_swizzled<BM, BK, NUM_THREADS, SwzA>(A + loadTile * BK, As_stage, K, tid);
            loadTileB_async_swizzled<BK, BN, NUM_THREADS, SwzB>(B + loadTile * BK * N, Bs_stage, N, tid);
            cp_async_commit();
            ++loadTile;
        }

        // Wait for compute stage data
        if (loadTile < numTiles) {
            cp_async_wait<STAGES - 1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // Compute pointers for this stage
        const __half* As_tile = As + computeStage * A_STAGE_SIZE;
        const __half* Bs_tile = Bs + computeStage * B_STAGE_SIZE;

        // --- Compute: MMA on current tile ---
        #pragma unroll
        for (int innerK = 0; innerK < BK; innerK += MMA_K) {
            FragmentA a_frag[MMA_M_TILES];
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m) {
                int row_base = warpM * WM + m * MMA_M;
                int col_base = innerK;
                load_fragment_ldmatrix_swizzled<BK, SwzA>(a_frag[m], As_tile, row_base, col_base);
            }

            FragmentB b_frag[MMA_N_TILES];
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n) {
                int row_base = innerK;
                int col_base = warpN * WN + n * MMA_N;
                load_fragment_ldmatrix_swizzled<BN, SwzB>(b_frag[n], Bs_tile, row_base, col_base);
            }

            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m)
                #pragma unroll
                for (int n = 0; n < MMA_N_TILES; ++n)
                    mma_sync(acc[m][n], a_frag[m], b_frag[n]);
        }

        __syncthreads();
    }

    // Epilogue: reuse smem for C staging
    epilogue_vec4_swizzled<BM, BN, WM, WN, MMA_M_TILES, MMA_N_TILES, NUM_THREADS, SwzB>(
        acc, smem, C, N, alpha, tid, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN, int STAGES = 2>
struct MMAMultistage {
    static constexpr int WARPS_M = BM / WM;
    static constexpr int WARPS_N = BN / WN;
    static constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
    static constexpr size_t SMEM_STAGES = STAGES * (BM * BK + BK * BN) * sizeof(__half);
    static constexpr size_t SMEM_C  = BM * BN * sizeof(__half);
    static constexpr size_t SMEM_SIZE = SMEM_STAGES > SMEM_C ? SMEM_STAGES : SMEM_C;

    static void Run(int M, int N, int K, __half alpha,
                    const __half* A, const __half* B,
                    __half beta, __half* C) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(
                mma_multistage_kernel<BM, BN, BK, WM, WN, STAGES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_SIZE
            );
            configured = true;
        }
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        mma_multistage_kernel<BM, BN, BK, WM, WN, STAGES><<<grid, block, SMEM_SIZE>>>(
            M, N, K, alpha, A, B, beta, C);
    }
};
