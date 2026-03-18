// fragment.cuh
// Fragment structs for mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
//
// Each fragment maps thread lanes to matrix elements via get_row()/get_col().
// Registers hold packed FP16 pairs (two half values per uint32_t).

#pragma once

#include <cuda_fp16.h>

// MMA tile dimensions (hardware fixed)
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Fragment A: 16x16 (M x K), 8 elements per thread, 4 registers
struct FragmentA {
    static constexpr int num_elements = 8;
    static constexpr int num_regs = 4;

    uint32_t reg[num_regs];

    // Element index i (0-7) -> matrix row
    //   i=0,1: row = group_id       i=2,3: row = group_id + 8
    //   i=4,5: row = group_id       i=6,7: row = group_id + 8
    static __device__ int get_row(int lane, int i) {
        int group_id = lane >> 2;
        return group_id + 8 * ((i >> 1) & 1);
    }

    // Element index i (0-7) -> matrix column
    //   i=0-3: cols 0-7             i=4-7: cols 8-15
    static __device__ int get_col(int lane, int i) {
        int tid_in_group = lane & 3;
        return tid_in_group * 2 + (i & 1) + 8 * (i >> 2);
    }
};

// Fragment B: 16x8 (K x N), 4 elements per thread, 2 registers
struct FragmentB {
    static constexpr int num_elements = 4;
    static constexpr int num_regs = 2;

    uint32_t reg[num_regs];

    // Element index i (0-3) -> matrix row (K dimension)
    //   i=0,1: rows 0-7             i=2,3: rows 8-15
    static __device__ int get_row(int lane, int i) {
        return (lane & 3) * 2 + (i & 1) + 8 * (i >> 1);
    }

    // Element index i (0-3) -> matrix column (N dimension)
    static __device__ int get_col(int lane, int i) {
        return lane >> 2;
    }
};

// Fragment C/D: 16x8 (M x N), 4 elements per thread, 2 registers
struct FragmentC {
    static constexpr int num_elements = 4;
    static constexpr int num_regs = 2;

    uint32_t reg[num_regs];

    // Element index i (0-3) -> matrix row
    //   i=0,1: rows 0-7             i=2,3: rows 8-15
    static __device__ int get_row(int lane, int i) {
        return (lane >> 2) + 8 * (i >> 1);
    }

    // Element index i (0-3) -> matrix column
    static __device__ int get_col(int lane, int i) {
        return (lane & 3) * 2 + (i & 1);
    }

    __device__ void fill_zero() {
        reg[0] = 0;
        reg[1] = 0;
    }
};
