// Int8 prefill GEMM on the CPU: Y[t][r] = sum_k W[r][k] * X[t][k] for the
// weight formats a prefill meets in practice, unpacked to exact integers
// (see matmul.cpp, i8_gemm).
//
// Layout: the kernels work on groups of 8 weight rows interleaved 4 bytes at
// a time, so that one 256-bit register holds 4 consecutive weights of 8
// different rows. A 4-byte slice of one token's activation, broadcast to
// every lane, meets all 8 rows in a single int8 dot instruction: 8 outputs
// per register, no horizontal sums, and the float work (scales, offsets)
// done once per 32 weights for 8 rows at a time instead of once per output.
//
// The same implementation (i8gemm_impl.h) is compiled once per instruction
// set, each copy in its own namespace, and picked at run time:
//   i8_base    AVX2 (maddubs + madd), or NEON sdot on ARM with dotprod
//   i8_vnni    AVX-VNNI (vpdpbusd, VEX)      - Intel 12th gen+, Zen 5
//   i8_vnni512 AVX512-VNNI at 256 bits (EVEX) - Zen 4, Xeon
#pragma once

#include <cstddef>
#include <cstdint>

namespace desireeia {

struct I8GemmJob {
    const uint8_t* data = nullptr;   // weights, `rows` rows of `row_bytes`
    size_t row_bytes = 0;
    size_t rows = 0, cols = 0;
    // Decodes one row into q[cols] (int8), a[2*nb] (scale of each 16),
    // m[2*nb] (offset of each 16; only when has_m).
    void (*unpack)(int src, const uint8_t* row, size_t cols, int8_t* q, float* a, float* m) = nullptr;
    int src = 0;
    bool has_m = false;      // w = a*q - m (Q4_K/Q5_K); halves share a and m
    bool split = false;      // the two 16-wide halves of a block have different scales (Q6_K)
    bool wide = false;       // values reach +-127 (Q8_0); otherwise |q| <= 63
    const float* x = nullptr;
    size_t n_tok = 0;
    float* y = nullptr;
};

namespace i8_base    { bool available(); void run(const I8GemmJob& job); }
namespace i8_vnni    { bool available(); void run(const I8GemmJob& job); }
namespace i8_vnni512 { bool available(); void run(const I8GemmJob& job); }

// Best implementation for this CPU, or false when none applies (the caller
// then uses the portable kernels).
bool i8_gemm_interleaved(const I8GemmJob& job);

// Which one i8_gemm_interleaved uses ("avx-vnni", "avx512-vnni", "avx2",
// "neon-dotprod" or "none"), for logs and benchmarks.
const char* i8_gemm_isa();

} // namespace desireeia
