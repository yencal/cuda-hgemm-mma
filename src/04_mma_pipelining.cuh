// 04_mma_pipelining.cuh
// PTX MMA HGEMM with interleaved load/compute software pipeline.
// Builds on 03_mma_multistage: adds fragment double-buffering and
// interleaved async copy for smem->register and gmem->smem overlap.
//
// - Fragment double buffering (smem->register overlap within k-loop)
// - Async copy issued at k-loop midpoint (overlaps DMA with MMA)
// - Cross-tile fragment prefetch (next tile k=0 overlaps with current tile last MMA)
// - XOR swizzle on all smem accesses (inherited from 02)
//
// NOTE: B is in standard layout B[K,N] row-major.
//       BK must be a multiple of 32 (K_STEPS must be even for double-buffering).

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
__global__ void mma_pipelining_kernel(
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
    constexpr int K_STEPS = BK / MMA_K;

    static_assert(K_STEPS >= 1 && K_STEPS <= 8);
    static_assert(K_STEPS % 2 == 0, "K_STEPS must be even for double-buffering (BK must be multiple of 32)");

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

    // Double-buffered fragments
    FragmentA a_frag[2][MMA_M_TILES];
    FragmentB b_frag[2][MMA_N_TILES];

    const int numTiles = K / BK;

    // ====== PROLOGUE: fill pipeline ======
    #pragma unroll
    for (int s = 0; s < STAGES - 1 && s < numTiles; ++s) {
        loadTileA_async_swizzled<BM, BK, NUM_THREADS, SwzA>(
            A + s * BK, As + s * A_STAGE_SIZE, K, tid);
        loadTileB_async_swizzled<BK, BN, NUM_THREADS, SwzB>(
            B + s * BK * N, Bs + s * B_STAGE_SIZE, N, tid);
        cp_async_commit();
    }

    cp_async_wait<0>();
    __syncthreads();

    // Load initial k=0 fragments from stage 0
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m)
        load_fragment_ldmatrix_swizzled<BK, SwzA>(
            a_frag[0][m], As, warpM * WM + m * MMA_M, 0);
    #pragma unroll
    for (int n = 0; n < MMA_N_TILES; ++n)
        load_fragment_ldmatrix_swizzled<BN, SwzB>(
            b_frag[0][n], Bs, 0, warpN * WN + n * MMA_N);

    int loadTile = STAGES - 1;

    // ====== MAIN LOOP ======
    for (int tile = 0; tile < numTiles; ++tile) {
        const int stage = tile % STAGES;
        const __half* As_tile = As + stage * A_STAGE_SIZE;
        const __half* Bs_tile = Bs + stage * B_STAGE_SIZE;

        // Phase 1: load k+1 fragments, MMA k (k=0..K_STEPS-2)
        #pragma unroll
        for (int k = 1; k < K_STEPS; ++k) {
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m)
                load_fragment_ldmatrix_swizzled<BK, SwzA>(
                    a_frag[k % 2][m], As_tile, warpM * WM + m * MMA_M, k * MMA_K);
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n)
                load_fragment_ldmatrix_swizzled<BN, SwzB>(
                    b_frag[k % 2][n], Bs_tile, k * MMA_K, warpN * WN + n * MMA_N);

            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m)
                #pragma unroll
                for (int n = 0; n < MMA_N_TILES; ++n)
                    mma_sync(acc[m][n], a_frag[(k - 1) % 2][m], b_frag[(k - 1) % 2][n]);

            // Issue async copy at midpoint
            constexpr int ASYNC_ISSUE_K = K_STEPS / 2;
            if (k == ASYNC_ISSUE_K && loadTile < numTiles) {
                int loadStage = loadTile % STAGES;
                loadTileA_async_swizzled<BM, BK, NUM_THREADS, SwzA>(
                    A + loadTile * BK, As + loadStage * A_STAGE_SIZE, K, tid);
                loadTileB_async_swizzled<BK, BN, NUM_THREADS, SwzB>(
                    B + loadTile * BK * N, Bs + loadStage * B_STAGE_SIZE, N, tid);
                cp_async_commit();
                ++loadTile;
            }
        }

        // Barrier: ensure next tile's data is ready
        if (loadTile < numTiles) {
            cp_async_wait<STAGES - 2>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // Phase 2: load k=0 of next tile, MMA last k-step of current tile
        if (tile + 1 < numTiles) {
            int nextStage = (tile + 1) % STAGES;
            const __half* As_next = As + nextStage * A_STAGE_SIZE;
            const __half* Bs_next = Bs + nextStage * B_STAGE_SIZE;
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m)
                load_fragment_ldmatrix_swizzled<BK, SwzA>(
                    a_frag[0][m], As_next, warpM * WM + m * MMA_M, 0);
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n)
                load_fragment_ldmatrix_swizzled<BN, SwzB>(
                    b_frag[0][n], Bs_next, 0, warpN * WN + n * MMA_N);
        }

        #pragma unroll
        for (int m = 0; m < MMA_M_TILES; ++m)
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n)
                mma_sync(acc[m][n], a_frag[(K_STEPS - 1) % 2][m],
                         b_frag[(K_STEPS - 1) % 2][n]);
    }

    // Epilogue: reuse smem for C staging
    epilogue_vec4_swizzled<BM, BN, WM, WN, MMA_M_TILES, MMA_N_TILES, NUM_THREADS, SwzB>(
        acc, smem, C, N, alpha, tid, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN, int STAGES = 3>
struct MMAPipelining {
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
                mma_pipelining_kernel<BM, BN, BK, WM, WN, STAGES>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_SIZE
            );
            configured = true;
        }
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        mma_pipelining_kernel<BM, BN, BK, WM, WN, STAGES><<<grid, block, SMEM_SIZE>>>(
            M, N, K, alpha, A, B, beta, C);
    }
};
