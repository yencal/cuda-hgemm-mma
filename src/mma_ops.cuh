// mma_ops.cuh
// PTX mma.sync wrapper for m16n8k16.row.col.f16.f16.f16.f16

#pragma once

#include "fragment.cuh"

// D = A * B + C
__device__ __forceinline__
void mma_sync(FragmentC& D,
              const FragmentA& A,
              const FragmentB& B,
              const FragmentC& C) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
        : "=r"(D.reg[0]), "=r"(D.reg[1])
        : "r"(A.reg[0]), "r"(A.reg[1]), "r"(A.reg[2]), "r"(A.reg[3]),
          "r"(B.reg[0]), "r"(B.reg[1]),
          "r"(C.reg[0]), "r"(C.reg[1])
    );
}

// In-place accumulation: acc = A * B + acc
__device__ __forceinline__
void mma_sync(FragmentC& acc,
              const FragmentA& A,
              const FragmentB& B) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(acc.reg[0]), "+r"(acc.reg[1])
        : "r"(A.reg[0]), "r"(A.reg[1]), "r"(A.reg[2]), "r"(A.reg[3]),
          "r"(B.reg[0]), "r"(B.reg[1])
    );
}
