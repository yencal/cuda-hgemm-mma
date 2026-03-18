// load_ldmatrix.cuh
// Warp-collective fragment loads using PTX ldmatrix instructions.
// Replaces element-by-element loads (load_direct.cuh) with hardware-accelerated
// shared memory -> register transfers.
//
// ldmatrix.x4:       loads 4 x 8x8 tiles (= 16x16 A fragment), no transpose
// ldmatrix.x2.trans: loads 2 x 8x8 tiles (= 16x8 B fragment), with transpose
//
// Each lane provides an address to one row of an 8x8 sub-tile.
// The hardware shuffles data across lanes so registers match the MMA layout.

#pragma once

#include <cuda_fp16.h>
#include "fragment.cuh"
#include "smem_swizzle.cuh"

// Load A fragment (16x16) using ldmatrix.x4
// smem_ptr points to the top-left of the 16x16 sub-tile in shared memory.
// STRIDE is the leading dimension of the shared memory tile (= BK).
//
// Tile layout in the 16x16 matrix (tile n -> reg[n]):
//   tile 0 (lanes 0-7):   rows 0-7,  cols 0-7    (top-left)
//   tile 1 (lanes 8-15):  rows 8-15, cols 0-7    (bottom-left)
//   tile 2 (lanes 16-23): rows 0-7,  cols 8-15   (top-right)
//   tile 3 (lanes 24-31): rows 8-15, cols 8-15   (bottom-right)
template<int STRIDE>
__device__ __forceinline__
void load_fragment_ldmatrix(FragmentA& frag, const __half* smem_ptr) {
    int lane = threadIdx.x & 31;

    int tile_id = lane >> 3;
    int row_in_tile = lane & 7;
    int row = (tile_id & 1) * 8 + row_in_tile;
    int col = (tile_id >> 1) * 8;

    uint32_t addr = __cvta_generic_to_shared(smem_ptr + row * STRIDE + col);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]),
          "=r"(frag.reg[2]), "=r"(frag.reg[3])
        : "r"(addr)
    );
}

// Load B fragment (16x8) using ldmatrix.x2.trans
// smem_ptr points to the top-left of the 16x8 sub-tile in shared memory.
// STRIDE is the leading dimension of the shared memory tile (= BN).
//
// B is stored row-major in smem (K contiguous along rows, N along columns).
// .trans transposes the 8x8 tiles during load so registers end up in the
// column-major layout that mma.sync(.row.col) expects for operand B.
//
// Tile layout:
//   tile 0 (lanes 0-7):  rows 0-7
//   tile 1 (lanes 8-15): rows 8-15
//   Lanes 16-31 wrap (lane & 15), addresses ignored by hardware but must be valid.
template<int STRIDE>
__device__ __forceinline__
void load_fragment_ldmatrix(FragmentB& frag, const __half* smem_ptr) {
    int lane = threadIdx.x & 31;

    int row = lane & 15;

    uint32_t addr = __cvta_generic_to_shared(smem_ptr + row * STRIDE);

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(addr)
    );
}

// =========================================================================
// Swizzled ldmatrix loaders (02_mma_swizzle and beyond)
// tile_base: base pointer of the full smem tile (As or Bs)
// row_base, col_base: offset to the sub-tile within the full tile
// STRIDE: leading dimension of the full tile (BK for A, BN for B)
// SwizzleT: must provide static int apply(int byte_offset)
// =========================================================================

template<int STRIDE, typename SwizzleT>
__device__ __forceinline__
void load_fragment_ldmatrix_swizzled(FragmentA& frag, const __half* tile_base,
                                     int row_base, int col_base) {
    int lane = threadIdx.x & 31;

    int tile_id = lane >> 3;
    int row_in_tile = lane & 7;
    int row = row_base + (tile_id & 1) * 8 + row_in_tile;
    int col = col_base + (tile_id >> 1) * 8;

    int byte_offset = (row * STRIDE + col) * sizeof(__half);
    int swizzled = SwizzleT::apply(byte_offset);
    uint32_t addr = __cvta_generic_to_shared(tile_base) + swizzled;

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1]),
          "=r"(frag.reg[2]), "=r"(frag.reg[3])
        : "r"(addr)
    );
}

template<int STRIDE, typename SwizzleT>
__device__ __forceinline__
void load_fragment_ldmatrix_swizzled(FragmentB& frag, const __half* tile_base,
                                     int row_base, int col_base) {
    int lane = threadIdx.x & 31;

    int row = row_base + (lane & 15);
    int col = col_base;

    int byte_offset = (row * STRIDE + col) * sizeof(__half);
    int swizzled = SwizzleT::apply(byte_offset);
    uint32_t addr = __cvta_generic_to_shared(tile_base) + swizzled;

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(frag.reg[0]), "=r"(frag.reg[1])
        : "r"(addr)
    );
}
