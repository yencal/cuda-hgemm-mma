// smem_swizzle.cuh
// XOR-based shared memory swizzle for bank conflict elimination.
// Follows CUTLASS Swizzle<B, M, S> convention.
//
// Applied to byte offsets within a shared memory tile:
//   swizzled = offset ^ (((offset >> (M+S)) & ((1<<B)-1)) << M)
//
// Parameters:
//   B: number of bits to XOR (2^B rows differentiated)
//   M: destination bit position (must be >= 4 for ldmatrix = 16-byte aligned)
//   S: distance from M to source bits (source at M+S)

#pragma once

template<int B, int M, int S>
struct Swizzle {
    static_assert(M >= 4, "M must be >= 4 for ldmatrix compatibility (16-byte alignment)");
    static_assert(S >= B, "S must be >= B to avoid source/destination bit overlap");

    static constexpr int MASK = (1 << B) - 1;

    __device__ __forceinline__
    static int apply(int byte_offset) {
        return byte_offset ^ (((byte_offset >> (M + S)) & MASK) << M);
    }
};

// constexpr log2 for power-of-2 values
constexpr __host__ __device__ int clog2(int n) {
    int r = 0;
    while (n > 1) { n >>= 1; r++; }
    return r;
}
