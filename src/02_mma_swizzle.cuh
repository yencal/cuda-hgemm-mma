// 02_mma_swizzle.cuh
// PTX MMA HGEMM with XOR swizzle for bank conflict elimination.
// Builds on 01b_mma_ldmatrix: adds swizzle to both write (cp.async) and
// read (ldmatrix) paths. Same kernel structure otherwise.
//
// - cp.async global -> shared with XOR swizzle on destination address
// - ldmatrix.x4 / .x2.trans with swizzled source address
// - PTX mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// - Vectorized epilogue (fragments -> smem -> coalesced float4 global stores)
//
// Swizzle parameters derived from tile dimensions:
//   M = 4 (ldmatrix 16-byte alignment)
//   S = log2(stride_bytes) - M
//   B = min(3, S)
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

template <int BM, int BN, int BK, int WM, int WN>
__global__ void mma_swizzle_kernel(
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

    // Derive swizzle parameters from tile dimensions
    // B = S = log2(stride_bytes) - M: use all available column bits
    constexpr int M_PARAM = 4;
    constexpr int S_A = clog2(BK * 2) - M_PARAM;
    constexpr int B_A = S_A < 3 ? S_A : 3;
    constexpr int S_B = clog2(BN * 2) - M_PARAM;
    constexpr int B_B = S_B < 3 ? S_B : 3;

    using SwzA = Swizzle<B_A, M_PARAM, S_A>;
    using SwzB = Swizzle<B_B, M_PARAM, S_B>;

    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(BK % MMA_K == 0, "BK must be divisible by MMA_K (16)");
    static_assert(WM % MMA_M == 0, "WM must be divisible by MMA_M (16)");
    static_assert(WN % MMA_N == 0, "WN must be divisible by MMA_N (8)");
    static_assert(BK >= 64, "BK must be >= 64 for full swizzle (B=3)");
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
        // Global -> Shared with swizzle
        loadTileA_async_swizzled<BM, BK, NUM_THREADS, SwzA>(A, As, K, tid);
        loadTileB_async_swizzled<BK, BN, NUM_THREADS, SwzB>(B, Bs, N, tid);
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();

        #pragma unroll
        for (int innerK = 0; innerK < BK; innerK += MMA_K) {
            // Load A fragments with swizzled ldmatrix
            FragmentA a_frag[MMA_M_TILES];
            #pragma unroll
            for (int m = 0; m < MMA_M_TILES; ++m) {
                int row_base = warpM * WM + m * MMA_M;
                int col_base = innerK;
                load_fragment_ldmatrix_swizzled<BK, SwzA>(a_frag[m], As, row_base, col_base);
            }

            // Load B fragments with swizzled ldmatrix
            FragmentB b_frag[MMA_N_TILES];
            #pragma unroll
            for (int n = 0; n < MMA_N_TILES; ++n) {
                int row_base = innerK;
                int col_base = warpN * WN + n * MMA_N;
                load_fragment_ldmatrix_swizzled<BN, SwzB>(b_frag[n], Bs, row_base, col_base);
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

    // Epilogue: fragments -> swizzled smem -> coalesced global stores
    // C staging stride = BN, same swizzle as B tile
    epilogue_vec4_swizzled<BM, BN, WM, WN, MMA_M_TILES, MMA_N_TILES, NUM_THREADS, SwzB>(
        acc, smem, C, N, alpha, tid, warpM, warpN);
}

template<int BM, int BN, int BK, int WM, int WN>
struct MMASwizzle {
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
                mma_swizzle_kernel<BM, BN, BK, WM, WN>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                SMEM_SIZE
            );
            configured = true;
        }
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 block(NUM_THREADS);
        mma_swizzle_kernel<BM, BN, BK, WM, WN><<<grid, block, SMEM_SIZE>>>(
            M, N, K, alpha, A, B, beta, C);
    }
};
