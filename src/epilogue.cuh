// epilogue.cuh
// Vectorized epilogue: fragments -> shared memory -> coalesced global stores.
//
// 1. Each warp scatters its C fragments into shared memory (no coalescing needed)
// 2. __syncthreads()
// 3. All threads cooperatively copy smem -> global using float4 (fully coalesced)

#pragma once

#include <cuda_fp16.h>
#include "fragment.cuh"
#include "smem_swizzle.cuh"

// Store a single C fragment to shared memory (scatter, no coalescing concern)
// smem_ptr points to the top-left of this fragment's tile in shared memory.
// SMEM_STRIDE is the leading dimension of the C staging buffer in smem (= BN).
template<int SMEM_STRIDE>
__device__ __forceinline__
void store_fragment_to_smem(__half* smem_ptr, const FragmentC& frag, __half alpha) {
    int lane = threadIdx.x & 31;
    const __half* elements = reinterpret_cast<const __half*>(frag.reg);

    #pragma unroll
    for (int i = 0; i < FragmentC::num_elements; ++i) {
        int row = FragmentC::get_row(lane, i);
        int col = FragmentC::get_col(lane, i);
        smem_ptr[row * SMEM_STRIDE + col] = __hmul(alpha, elements[i]);
    }
}

// Vectorized epilogue: store all accumulators through smem with coalesced global writes.
// Reuses the existing smem buffer (As/Bs are dead after the K-loop).
// C_smem must have at least BM * BN halves.
template<int BM, int BN, int WM, int WN, int MMA_M_TILES, int MMA_N_TILES, int NUM_THREADS>
__device__ __forceinline__
void epilogue_vec4(
    FragmentC acc[MMA_M_TILES][MMA_N_TILES],
    __half* C_smem,
    __half* C,
    int N,
    __half alpha,
    uint tid,
    uint warpM,
    uint warpN)
{
    // Step 1: scatter fragments into shared memory
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            __half* smem_ptr = &C_smem[(warpM * WM + m * MMA_M) * BN
                                       + (warpN * WN + n * MMA_N)];
            store_fragment_to_smem<BN>(smem_ptr, acc[m][n], alpha);
        }
    }

    __syncthreads();

    // Step 2: coalesced float4 copy from smem to global
    constexpr int TOTAL_ELEMENTS = BM * BN;
    constexpr int ELEMENTS_PER_VEC = 8;  // float4 = 16 bytes = 8 halves
    constexpr int TOTAL_VECS = TOTAL_ELEMENTS / ELEMENTS_PER_VEC;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;

    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized stores");
    static_assert(TOTAL_VECS % NUM_THREADS == 0, "Vectors must divide evenly among threads");

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        int vec_idx = tid + i * NUM_THREADS;
        int vecs_per_row = BN / ELEMENTS_PER_VEC;
        int row = vec_idx / vecs_per_row;
        int col8 = vec_idx % vecs_per_row;

        float4 val = *reinterpret_cast<float4*>(&C_smem[row * BN + col8 * 8]);
        *reinterpret_cast<float4*>(&C[row * N + col8 * 8]) = val;
    }
}

// =========================================================================
// Swizzled epilogue (02_mma_swizzle and beyond)
// Same two-phase approach but C staging buffer uses XOR swizzle to
// eliminate bank conflicts during the fragment scatter writes.
// SwizzleT must provide: static int apply(int byte_offset)
// =========================================================================

template<int SMEM_STRIDE, typename SwizzleT>
__device__ __forceinline__
void store_fragment_to_smem_swizzled(__half* tile_base, const FragmentC& frag,
                                     int row_base, int col_base, __half alpha) {
    int lane = threadIdx.x & 31;
    const __half* elements = reinterpret_cast<const __half*>(frag.reg);

    #pragma unroll
    for (int i = 0; i < FragmentC::num_elements; ++i) {
        int row = row_base + FragmentC::get_row(lane, i);
        int col = col_base + FragmentC::get_col(lane, i);
        int byte_offset = (row * SMEM_STRIDE + col) * sizeof(__half);
        int swizzled = SwizzleT::apply(byte_offset);
        *reinterpret_cast<__half*>(reinterpret_cast<char*>(tile_base) + swizzled) =
            __hmul(alpha, elements[i]);
    }
}

template<int BM, int BN, int WM, int WN, int MMA_M_TILES, int MMA_N_TILES,
         int NUM_THREADS, typename SwizzleT>
__device__ __forceinline__
void epilogue_vec4_swizzled(
    FragmentC acc[MMA_M_TILES][MMA_N_TILES],
    __half* C_smem,
    __half* C,
    int N,
    __half alpha,
    uint tid,
    uint warpM,
    uint warpN)
{
    // Step 1: scatter fragments into swizzled shared memory
    #pragma unroll
    for (int m = 0; m < MMA_M_TILES; ++m) {
        #pragma unroll
        for (int n = 0; n < MMA_N_TILES; ++n) {
            int row_base = warpM * WM + m * MMA_M;
            int col_base = warpN * WN + n * MMA_N;
            store_fragment_to_smem_swizzled<BN, SwizzleT>(
                C_smem, acc[m][n], row_base, col_base, alpha);
        }
    }

    __syncthreads();

    // Step 2: coalesced float4 copy from swizzled smem to global
    constexpr int TOTAL_ELEMENTS = BM * BN;
    constexpr int ELEMENTS_PER_VEC = 8;
    constexpr int TOTAL_VECS = TOTAL_ELEMENTS / ELEMENTS_PER_VEC;
    constexpr int VECS_PER_THREAD = TOTAL_VECS / NUM_THREADS;

    static_assert(BN % 8 == 0, "BN must be divisible by 8 for vectorized stores");
    static_assert(TOTAL_VECS % NUM_THREADS == 0, "Vectors must divide evenly among threads");

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        int vec_idx = tid + i * NUM_THREADS;
        int vecs_per_row = BN / ELEMENTS_PER_VEC;
        int row = vec_idx / vecs_per_row;
        int col8 = vec_idx % vecs_per_row;

        int byte_offset = (row * BN + col8 * 8) * sizeof(__half);
        int swizzled = SwizzleT::apply(byte_offset);
        float4 val = *reinterpret_cast<float4*>(
            reinterpret_cast<char*>(C_smem) + swizzled);
        *reinterpret_cast<float4*>(&C[row * N + col8 * 8]) = val;
    }
}
