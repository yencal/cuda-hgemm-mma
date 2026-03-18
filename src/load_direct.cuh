// load_direct.cuh
// Element-by-element fragment loads from shared memory.
// Educational: shows exactly which element each thread reads.
// Used in 01a_mma_direct only; replaced by ldmatrix in 01b.

#pragma once

#include "fragment.cuh"

// Load A fragment from shared memory (direct element-by-element)
// smem_ptr points to the top-left of the 16x16 sub-tile in shared memory.
// STRIDE is the leading dimension of the shared memory tile (= BK).
template<int STRIDE>
__device__ __forceinline__
void load_fragment_direct(FragmentA& frag, const __half* smem_ptr) {
    int lane = threadIdx.x & 31;
    __half* elements = reinterpret_cast<__half*>(frag.reg);

    #pragma unroll
    for (int i = 0; i < FragmentA::num_elements; ++i) {
        int row = FragmentA::get_row(lane, i);
        int col = FragmentA::get_col(lane, i);
        elements[i] = smem_ptr[row * STRIDE + col];
    }
}

// Load B fragment from shared memory (direct element-by-element)
// smem_ptr points to the top-left of the 16x8 sub-tile in shared memory.
// STRIDE is the leading dimension of the shared memory tile (= BN).
template<int STRIDE>
__device__ __forceinline__
void load_fragment_direct(FragmentB& frag, const __half* smem_ptr) {
    int lane = threadIdx.x & 31;
    __half* elements = reinterpret_cast<__half*>(frag.reg);

    #pragma unroll
    for (int i = 0; i < FragmentB::num_elements; ++i) {
        int row = FragmentB::get_row(lane, i);
        int col = FragmentB::get_col(lane, i);
        elements[i] = smem_ptr[row * STRIDE + col];
    }
}
