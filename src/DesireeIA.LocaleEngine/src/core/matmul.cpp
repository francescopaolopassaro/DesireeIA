// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "engine.h"
#include "profile.h"
#include "../quant/quant.h"
#include <algorithm>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace desireeia {

#if defined(__AVX2__)
// Faster signed int8 dot-product pairing after profiling showed our
// earlier prefill throughput had real headroom left on the table. The
// previous version converted both int8 operands to int16 (cvtepi8_epi16)
// before multiplying — correct, but nearly 2x more instructions than
// needed. _mm256_maddubs_epi16 multiplies unsigned x signed bytes and
// sums adjacent pairs in a single instruction; since both of our operands
// are actually signed, we use the standard sign trick: ax=|x| (unsigned),
// sy=sign(y,x) (negate y wherever x was negative), so that
// unsigned(ax)*signed(sy) == x*y exactly.
static inline __m256i mul_add_i8_pairs_avx2(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy); // 16x int16, summed in adjacent pairs
}
static inline __m256i sum_i16_pairs_i32_avx2(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, x); // 8x int32
}

// Horizontal reduction of an 8-lane float accumulator. Done with shuffles
// instead of storing to an array and doing 8 scalar sums: the latter form
// forces a round-trip through memory (store-to-load forwarding) on the
// value that was just computed.
static inline float hsmax_ps_avx2(__m256 v) {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

static inline int32_t hsum_epi32_avx2(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

static inline float hsum_ps_avx2(__m256 v) {
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

static float dot8_avx2(const int8_t* a, const int8_t* b, size_t n) {
    __m256i sum = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        sum = _mm256_add_epi32(sum, sum_i16_pairs_i32_avx2(mul_add_i8_pairs_avx2(va, vb)));
    }
    int32_t r[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(r), sum);
    int32_t acc = 0;
    for (int k = 0; k < 8; ++k) acc += r[k];
    for (; i < n; ++i) acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    return static_cast<float>(acc);
}

// Like dot8_avx2 but for a fixed n=32 and WITHOUT the final horizontal
// reduction (returns the vector of 8 partial int32 sums, not yet summed
// into a scalar). Used when the caller needs to accumulate many dots of
// the same order of magnitude in a row (e.g. the eight 32-wide sub-blocks
// of a Q4_K super-block): doing the horizontal reduction (store to memory
// + 8 scalar sums) once per row instead of once per sub-block avoids
// ~64 redundant reductions per row on a typical model (n_super=8, 8
// sub-dots per super-block). The combination with the per-block scale
// still happens in floating point after the conversion (cvtepi32_ps), so
// the result remains numerically equivalent (up to floating-point
// reordering, same as for the 2-row tiling).
static __m256i dot8_avx2_i32(const int8_t* a, const int8_t* b) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    return sum_i16_pairs_i32_avx2(mul_add_i8_pairs_avx2(va, vb));
}

// Dedicated int8x16 dot (SSE4.1/SSSE3, always available when __AVX2__ is
// defined). dot8_avx2 requires n>=32 to enter its AVX2 loop (below that
// threshold everything falls back to the scalar tail loop): matmul_q6_k,
// however, works on sub-blocks of exactly 16 elements (per-sub-block scale
// for Q6_K), so with dot8_avx2 those dots were NEVER vectorized. Discovered
// by measuring with the profiler (core/profile.h), not guessed. Same
// maddubs technique as dot8_avx2 above, at 128 bits.
static inline __m128i mul_add_i8_pairs_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy); // 8x int16
}
static inline __m128i sum_i16_pairs_i32_sse(const __m128i x) {
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, x); // 4x int32
}

static float dot8_16(const int8_t* a, const int8_t* b) {
    const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    const __m128i sum = sum_i16_pairs_i32_sse(mul_add_i8_pairs_sse(va, vb));
    int32_t r[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(r), sum);
    return static_cast<float>(r[0] + r[1] + r[2] + r[3]);
}

// Like dot8_16 but without the final horizontal reduction (see
// dot8_avx2_i32: same principle, applied to the 16-wide dimension used by
// Q6_K). matmul_q6_k does 16 of these dots per super-block (4 quadrants x
// 2 halves): deferring the reduction to once per row instead of once per
// dot avoids that many redundant reductions.
static __m128i dot8_16_i32(const int8_t* a, const int8_t* b) {
    const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    return sum_i16_pairs_i32_sse(mul_add_i8_pairs_sse(va, vb));
}

// Note for future work: an explicit AVX2 version of the nibble extraction
// above (16-bit shift + AND 0x0F, same trick as dot8_avx2) was tried for
// both Q4_0 and Q4_K. Measured: no real gain (~6% slower over 5 repeated
// runs), most likely because the scalar loop is already well
// auto-vectorized by -O3 and the manual intrinsics only add store/reload
// latency. Removed, not kept: see docs/engine_gap_analysis.md for the
// measurement details.
#endif

#if defined(__ARM_NEON)
// Dot product of two int8x16 vectors -> int32x4, accumulated into acc.
//
// With __ARM_FEATURE_DOTPROD (ARMv8.2+: all Apple Silicon chips, and most
// modern ARM server/desktop parts) this is a single hardware vdotq_s32
// instruction. Without it (e.g. Raspberry Pi 4 / Cortex-A72, ARMv8.0), we
// fall back to widening multiply-add: widen each 8-lane half to int16 with
// vmull_s8, then pairwise-widen-accumulate into int32 with vpadalq_s16.
// Both paths were checked to produce identical results, not just by
// construction: the selftest (dequantize+float vs fused kernel) was run
// under QEMU aarch64 for both variants (see docs/engine_gap_analysis.md).
static inline int32x4_t dot_i8x16_neon(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpadalq_s16(vpadalq_s16(acc, lo), hi);
#endif
}

// NEON equivalents of dot8_avx2/dot8_16/dot8_16_i32 (defined above for
// AVX2): same signature, same semantics, so the ~15 call sites scattered
// through the file can add an `#elif defined(__ARM_NEON)` branch that
// calls these instead of duplicating the nibble-extraction logic every
// time.
//
// dot8_neon works in steps of 16 (one NEON register), not 32 like
// dot8_avx2 (which processes 2 at once, since AVX2 is 256 bits wide): the
// final scalar tail covers both the case where n is not a multiple of 16
// and, when needed, the last 16 elements of a 32-wide block.
static float dot8_neon(const int8_t* a, const int8_t* b, size_t n) {
    int32x4_t sum = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        sum = dot_i8x16_neon(sum, vld1q_s8(a + i), vld1q_s8(b + i));
    }
    int32_t acc = vaddvq_s32(sum);
    for (; i < n; ++i) acc += (int32_t) a[i] * (int32_t) b[i];
    return (float) acc;
}

// NEON equivalent of dot8_16: fixed n=16, WITH reduction to scalar.
static inline float dot8_16_neon(const int8_t* a, const int8_t* b) {
    const int32x4_t sum = dot_i8x16_neon(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b));
    return (float) vaddvq_s32(sum);
}

// Fixed n=16, without horizontal reduction (equivalent of dot8_16_i32):
// used where the caller accumulates many 16-element dots in a row before
// converting to float just once (same idea as dot8_avx2_i32/dot8_16_i32,
// here at 128 bits — which is already NEON's native width, so no separate
// "wide" variant is needed like on AVX2).
static inline int32x4_t dot16_neon_i32(const int8_t* a, const int8_t* b) {
    return dot_i8x16_neon(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b));
}
#endif

struct QRow {
    std::vector<int8_t> q;
    std::vector<float> scales;
};

static void quantize_row(const float* src, size_t n, QRow& out) {
    std::vector<uint8_t> sign;
    quantize_q8_0(src, n, out.q, out.scales, sign);
}

int quantized_matmul(const std::vector<float>& a, const std::vector<float>& b,
                     size_t M, size_t K, size_t N, std::vector<float>& out) {
    if (a.size() < M * K || b.size() < K * N) {
        return DESIREEIA_ERR_INVALID_ARG;
    }

    std::vector<QRow> qa(M);
    for (size_t m = 0; m < M; ++m) {
        quantize_row(a.data() + m * K, K, qa[m]);
    }

    std::vector<QRow> qbt(N);
    std::vector<float> bt(N * K);
    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            bt[n * K + k] = b[k * N + n];
        }
        quantize_row(bt.data() + n * K, K, qbt[n]);
    }

    out.assign(M * N, 0.0f);

    constexpr size_t BLOCK = 32;

    for (size_t m = 0; m < M; ++m) {
        const int8_t* pa = qa[m].q.data();
        const float* sa = qa[m].scales.data();
        for (size_t n = 0; n < N; ++n) {
            const int8_t* pb = qbt[n].q.data();
            const float* sb = qbt[n].scales.data();
            float acc = 0.0f;
#if defined(__AVX2__)
            for (size_t k = 0; k + BLOCK <= K; k += BLOCK) {
                acc += sa[k / BLOCK] * sb[k / BLOCK] * dot8_avx2(pa + k, pb + k, BLOCK);
            }
#elif defined(__ARM_NEON)
            for (size_t k = 0; k + BLOCK <= K; k += BLOCK) {
                acc += sa[k / BLOCK] * sb[k / BLOCK] * dot8_neon(pa + k, pb + k, BLOCK);
            }
#else
            for (size_t k = 0; k < K; ++k) {
                float da = sa[k / BLOCK];
                float db = sb[k / BLOCK];
                acc += da * db * static_cast<float>(
                    static_cast<int32_t>(pa[k]) * static_cast<int32_t>(pb[k]));
            }
#endif
            size_t tail = K % BLOCK;
            if (tail > 0) {
                size_t base = K - tail;
                float da = sa[base / BLOCK];
                float db = sb[base / BLOCK];
                for (size_t k = 0; k < tail; ++k) {
                    acc += da * db * static_cast<float>(
                        static_cast<int32_t>(pa[base + k]) * static_cast<int32_t>(pb[base + k]));
                }
            }
            out[m * N + n] = acc;
        }
    }
    return DESIREEIA_OK;
}

namespace {
// Same function as in quant.cpp (not exposed in quant.h): extracts the
// 6-bit scale and min from the scales[12] block of a Q4_K/Q5_K super-block.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}
}

int matmul_q4_0(const uint8_t* q4_data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q4_0);

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
    }

    {
    ScopedTimer t(profile_counters().ns_q40_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = q4_data + r * row_bytes;
            float acc = 0.0f;
            for (size_t b = 0; b < nb; ++b) {
                const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(row_ptr + b * sizeof(block_q4_0));
                const float wd = desireeia_fp16_to_fp32(blk->d);

                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    wblk[j]      = (int8_t) ((blk->qs[j] & 0x0F) - 8);
                    wblk[j + 16] = (int8_t) ((blk->qs[j] >> 4)   - 8);
                }

                const int8_t* xblk = xq.data() + b * 32;
#if defined(__AVX2__)
                float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                float dot = dot8_neon(wblk, xblk, 32);
#else
                int32_t idot = 0;
                for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                float dot = (float) idot;
#endif
                acc += wd * xscale[b] * dot;
            }
            y[r] = acc;
        }
    });
    }
    profile_counters().calls_q40.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q8_0(const uint8_t* q8_data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q8_0);

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
    }

    {
    ScopedTimer t(profile_counters().ns_q80_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        // 2-row tiling (same technique as matmul_q4_k/matmul_q6_k): no
        // nibble to unpack here (Q8_0 is already int8), so the gain is
        // just from giving 2 independent dot chains per iteration.
        size_t r = r0;
        for (; r + 1 < r1; r += 2) {
            const uint8_t* row0 = q8_data + (r + 0) * row_bytes;
            const uint8_t* row1 = q8_data + (r + 1) * row_bytes;
            float acc0 = 0.0f, acc1 = 0.0f;
            for (size_t b = 0; b < nb; ++b) {
                const block_q8_0* blk0 = reinterpret_cast<const block_q8_0*>(row0 + b * sizeof(block_q8_0));
                const block_q8_0* blk1 = reinterpret_cast<const block_q8_0*>(row1 + b * sizeof(block_q8_0));
                const float wd0 = desireeia_fp16_to_fp32(blk0->d);
                const float wd1 = desireeia_fp16_to_fp32(blk1->d);
                const int8_t* xblk = xq.data() + b * 32;
#if defined(__AVX2__)
                const float dot0 = dot8_avx2(blk0->qs, xblk, 32);
                const float dot1 = dot8_avx2(blk1->qs, xblk, 32);
#elif defined(__ARM_NEON)
                const float dot0 = dot8_neon(blk0->qs, xblk, 32);
                const float dot1 = dot8_neon(blk1->qs, xblk, 32);
#else
                int32_t idot0 = 0, idot1 = 0;
                for (int j = 0; j < 32; ++j) {
                    idot0 += (int32_t) blk0->qs[j] * (int32_t) xblk[j];
                    idot1 += (int32_t) blk1->qs[j] * (int32_t) xblk[j];
                }
                const float dot0 = (float) idot0, dot1 = (float) idot1;
#endif
                acc0 += wd0 * xscale[b] * dot0;
                acc1 += wd1 * xscale[b] * dot1;
            }
            y[r] = acc0;
            y[r + 1] = acc1;
        }
        for (; r < r1; ++r) {
            const uint8_t* row_ptr = q8_data + r * row_bytes;
            float acc = 0.0f;
            for (size_t b = 0; b < nb; ++b) {
                const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(row_ptr + b * sizeof(block_q8_0));
                const float wd = desireeia_fp16_to_fp32(blk->d);
                const int8_t* xblk = xq.data() + b * 32;
#if defined(__AVX2__)
                float dot = dot8_avx2(blk->qs, xblk, 32);
#elif defined(__ARM_NEON)
                float dot = dot8_neon(blk->qs, xblk, 32);
#else
                int32_t idot = 0;
                for (int j = 0; j < 32; ++j) idot += (int32_t) blk->qs[j] * (int32_t) xblk[j];
                float dot = (float) idot;
#endif
                acc += wd * xscale[b] * dot;
            }
            y[r] = acc;
        }
    });
    }
    profile_counters().calls_q80.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q8_0_batch(const uint8_t* q8_data, size_t rows, size_t cols,
                       const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q8_0(q8_data, rows, cols, x, y);
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q8_0);

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
        }
    }

    {
    ScopedTimer t(profile_counters().ns_q80_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        // 2x4-tile GEMM (RM=2 rows, RN=4 tokens). Unlike the earlier
        // version (outer loop over b, inner loop over tokens, with
        // std::vector accumulators that had to survive across b
        // iterations and so lived in memory/cache, never in registers),
        // here the loop over the TOKEN BLOCK is outer and the loop over
        // b (super-blocks) is inner: the 8 tile accumulators are named
        // local variables (not an indexed array), so the compiler can
        // keep them in registers for the whole reduction over b, the way
        // a real GEMM microkernel does.
        // Cost: the weight bytes get re-read once per block of 4 tokens
        // instead of just once for the whole row (weight-read traffic
        // divided by RN=4 instead of by n_tok) — but they almost certainly
        // stay in L1/L2 between one token block and the next (small rows,
        // a few dozen bytes), so the real cost is low compared to the gain
        // from having accumulators that truly live in registers. Measured
        // before keeping it (development rule): see
        // docs/engine_gap_analysis.md.
        size_t r = r0;
        for (; r + 1 < r1; r += 2) {
            const uint8_t* row_ptr0 = q8_data + (r + 0) * row_bytes;
            const uint8_t* row_ptr1 = q8_data + (r + 1) * row_bytes;
            size_t tk = 0;
            for (; tk + 4 <= n_tok; tk += 4) {
                const int8_t* xq0 = xq[tk + 0].data();
                const int8_t* xq1 = xq[tk + 1].data();
                const int8_t* xq2 = xq[tk + 2].data();
                const int8_t* xq3 = xq[tk + 3].data();
                const float* xs0 = xscale[tk + 0].data();
                const float* xs1 = xscale[tk + 1].data();
                const float* xs2 = xscale[tk + 2].data();
                const float* xs3 = xscale[tk + 3].data();
                float a00 = 0, a01 = 0, a02 = 0, a03 = 0;
                float a10 = 0, a11 = 0, a12 = 0, a13 = 0;
                for (size_t b = 0; b < nb; ++b) {
                    const block_q8_0* blk0 = reinterpret_cast<const block_q8_0*>(row_ptr0 + b * sizeof(block_q8_0));
                    const block_q8_0* blk1 = reinterpret_cast<const block_q8_0*>(row_ptr1 + b * sizeof(block_q8_0));
                    const float wd0 = desireeia_fp16_to_fp32(blk0->d);
                    const float wd1 = desireeia_fp16_to_fp32(blk1->d);
                    const int8_t* x0 = xq0 + b * 32;
                    const int8_t* x1 = xq1 + b * 32;
                    const int8_t* x2 = xq2 + b * 32;
                    const int8_t* x3 = xq3 + b * 32;
#if defined(__AVX2__)
                    a00 += wd0 * xs0[b] * dot8_avx2(blk0->qs, x0, 32);
                    a01 += wd0 * xs1[b] * dot8_avx2(blk0->qs, x1, 32);
                    a02 += wd0 * xs2[b] * dot8_avx2(blk0->qs, x2, 32);
                    a03 += wd0 * xs3[b] * dot8_avx2(blk0->qs, x3, 32);
                    a10 += wd1 * xs0[b] * dot8_avx2(blk1->qs, x0, 32);
                    a11 += wd1 * xs1[b] * dot8_avx2(blk1->qs, x1, 32);
                    a12 += wd1 * xs2[b] * dot8_avx2(blk1->qs, x2, 32);
                    a13 += wd1 * xs3[b] * dot8_avx2(blk1->qs, x3, 32);
#elif defined(__ARM_NEON)
                    a00 += wd0 * xs0[b] * dot8_neon(blk0->qs, x0, 32);
                    a01 += wd0 * xs1[b] * dot8_neon(blk0->qs, x1, 32);
                    a02 += wd0 * xs2[b] * dot8_neon(blk0->qs, x2, 32);
                    a03 += wd0 * xs3[b] * dot8_neon(blk0->qs, x3, 32);
                    a10 += wd1 * xs0[b] * dot8_neon(blk1->qs, x0, 32);
                    a11 += wd1 * xs1[b] * dot8_neon(blk1->qs, x1, 32);
                    a12 += wd1 * xs2[b] * dot8_neon(blk1->qs, x2, 32);
                    a13 += wd1 * xs3[b] * dot8_neon(blk1->qs, x3, 32);
#else
                    auto sdot = [](const int8_t* a, const int8_t* c) {
                        int32_t s = 0;
                        for (int j = 0; j < 32; ++j) s += (int32_t) a[j] * (int32_t) c[j];
                        return (float) s;
                    };
                    a00 += wd0 * xs0[b] * sdot(blk0->qs, x0);
                    a01 += wd0 * xs1[b] * sdot(blk0->qs, x1);
                    a02 += wd0 * xs2[b] * sdot(blk0->qs, x2);
                    a03 += wd0 * xs3[b] * sdot(blk0->qs, x3);
                    a10 += wd1 * xs0[b] * sdot(blk1->qs, x0);
                    a11 += wd1 * xs1[b] * sdot(blk1->qs, x1);
                    a12 += wd1 * xs2[b] * sdot(blk1->qs, x2);
                    a13 += wd1 * xs3[b] * sdot(blk1->qs, x3);
#endif
                }
                y[(tk + 0) * rows + r] = a00; y[(tk + 1) * rows + r] = a01;
                y[(tk + 2) * rows + r] = a02; y[(tk + 3) * rows + r] = a03;
                y[(tk + 0) * rows + r + 1] = a10; y[(tk + 1) * rows + r + 1] = a11;
                y[(tk + 2) * rows + r + 1] = a12; y[(tk + 3) * rows + r + 1] = a13;
            }
            for (; tk < n_tok; ++tk) {
                float acc0 = 0.0f, acc1 = 0.0f;
                const int8_t* xqt = xq[tk].data();
                const float* xst = xscale[tk].data();
                for (size_t b = 0; b < nb; ++b) {
                    const block_q8_0* blk0 = reinterpret_cast<const block_q8_0*>(row_ptr0 + b * sizeof(block_q8_0));
                    const block_q8_0* blk1 = reinterpret_cast<const block_q8_0*>(row_ptr1 + b * sizeof(block_q8_0));
                    const int8_t* xt = xqt + b * 32;
#if defined(__AVX2__)
                    acc0 += desireeia_fp16_to_fp32(blk0->d) * xst[b] * dot8_avx2(blk0->qs, xt, 32);
                    acc1 += desireeia_fp16_to_fp32(blk1->d) * xst[b] * dot8_avx2(blk1->qs, xt, 32);
#elif defined(__ARM_NEON)
                    acc0 += desireeia_fp16_to_fp32(blk0->d) * xst[b] * dot8_neon(blk0->qs, xt, 32);
                    acc1 += desireeia_fp16_to_fp32(blk1->d) * xst[b] * dot8_neon(blk1->qs, xt, 32);
#else
                    int32_t i0 = 0, i1 = 0;
                    for (int j = 0; j < 32; ++j) { i0 += (int32_t) blk0->qs[j] * (int32_t) xt[j]; i1 += (int32_t) blk1->qs[j] * (int32_t) xt[j]; }
                    acc0 += desireeia_fp16_to_fp32(blk0->d) * xst[b] * (float) i0;
                    acc1 += desireeia_fp16_to_fp32(blk1->d) * xst[b] * (float) i1;
#endif
                }
                y[tk * rows + r] = acc0;
                y[tk * rows + r + 1] = acc1;
            }
        }
        for (; r < r1; ++r) {
            const uint8_t* row_ptr = q8_data + r * row_bytes;
            for (size_t tk = 0; tk < n_tok; ++tk) {
                float acc = 0.0f;
                const int8_t* xqt = xq[tk].data();
                const float* xst = xscale[tk].data();
                for (size_t b = 0; b < nb; ++b) {
                    const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(row_ptr + b * sizeof(block_q8_0));
                    const int8_t* xt = xqt + b * 32;
#if defined(__AVX2__)
                    acc += desireeia_fp16_to_fp32(blk->d) * xst[b] * dot8_avx2(blk->qs, xt, 32);
#elif defined(__ARM_NEON)
                    acc += desireeia_fp16_to_fp32(blk->d) * xst[b] * dot8_neon(blk->qs, xt, 32);
#else
                    int32_t idot = 0;
                    for (int j = 0; j < 32; ++j) idot += (int32_t) blk->qs[j] * (int32_t) xt[j];
                    acc += desireeia_fp16_to_fp32(blk->d) * xst[b] * (float) idot;
#endif
                }
                y[tk * rows + r] = acc;
            }
        }
    });
    }
    profile_counters().calls_q80.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q4_1(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q4_1);

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    std::vector<int32_t> xsum(nb);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
        for (size_t b = 0; b < nb; ++b) {
            int32_t s = 0;
            for (int j = 0; j < 32; ++j) s += (int32_t) xq[b * 32 + j];
            xsum[b] = s;
        }
    }

    {
    ScopedTimer t(profile_counters().ns_legacy_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            float acc = 0.0f;
            for (size_t b = 0; b < nb; ++b) {
                const block_q4_1* blk = reinterpret_cast<const block_q4_1*>(row_ptr + b * sizeof(block_q4_1));
                const float wd = desireeia_fp16_to_fp32(blk->d);
                const float wm = desireeia_fp16_to_fp32(blk->m);
                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    wblk[j]      = (int8_t) (blk->qs[j] & 0x0F);
                    wblk[j + 16] = (int8_t) (blk->qs[j] >> 4);
                }
                const int8_t* xblk = xq.data() + b * 32;
#if defined(__AVX2__)
                float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                float dot = dot8_neon(wblk, xblk, 32);
#else
                int32_t idot = 0;
                for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                float dot = (float) idot;
#endif
                acc += xscale[b] * (wd * dot + wm * (float) xsum[b]);
            }
            y[r] = acc;
        }
    });
    }
    profile_counters().calls_legacy.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q4_1_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q4_1(data, rows, cols, x, y);
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q4_1);

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    std::vector<std::vector<int32_t>> xsum(n_tok, std::vector<int32_t>(nb));
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
            for (size_t b = 0; b < nb; ++b) {
                int32_t s = 0;
                for (int j = 0; j < 32; ++j) s += (int32_t) xq[tk][b * 32 + j];
                xsum[tk][b] = s;
            }
        }
    }

    {
    ScopedTimer t(profile_counters().ns_legacy_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> acc(n_tok);
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t b = 0; b < nb; ++b) {
                const block_q4_1* blk = reinterpret_cast<const block_q4_1*>(row_ptr + b * sizeof(block_q4_1));
                const float wd = desireeia_fp16_to_fp32(blk->d);
                const float wm = desireeia_fp16_to_fp32(blk->m);
                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    wblk[j]      = (int8_t) (blk->qs[j] & 0x0F);
                    wblk[j + 16] = (int8_t) (blk->qs[j] >> 4);
                }
                for (size_t tk = 0; tk < n_tok; ++tk) {
                    const int8_t* xblk = xq[tk].data() + b * 32;
#if defined(__AVX2__)
                    float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                    float dot = dot8_neon(wblk, xblk, 32);
#else
                    int32_t idot = 0;
                    for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                    float dot = (float) idot;
#endif
                    acc[tk] += xscale[tk][b] * (wd * dot + wm * (float) xsum[tk][b]);
                }
            }
            for (size_t tk = 0; tk < n_tok; ++tk) y[tk * rows + r] = acc[tk];
        }
    });
    }
    profile_counters().calls_legacy.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q5_0(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q5_0);

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
    }

    {
    ScopedTimer t(profile_counters().ns_legacy_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            float acc = 0.0f;
            for (size_t b = 0; b < nb; ++b) {
                const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(row_ptr + b * sizeof(block_q5_0));
                const float wd = desireeia_fp16_to_fp32(blk->d);
                uint32_t qh; std::memcpy(&qh, blk->qh, sizeof(qh));
                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    const uint8_t xh0 = (uint8_t) (((qh >> (j + 0))  << 4) & 0x10);
                    const uint8_t xh1 = (uint8_t) (((qh >> (j + 12))     ) & 0x10);
                    wblk[j]      = (int8_t) (((blk->qs[j] & 0x0F) | xh0) - 16);
                    wblk[j + 16] = (int8_t) (((blk->qs[j] >> 4)   | xh1) - 16);
                }
                const int8_t* xblk = xq.data() + b * 32;
#if defined(__AVX2__)
                float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                float dot = dot8_neon(wblk, xblk, 32);
#else
                int32_t idot = 0;
                for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                float dot = (float) idot;
#endif
                acc += wd * xscale[b] * dot;
            }
            y[r] = acc;
        }
    });
    }
    profile_counters().calls_legacy.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q5_0_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q5_0(data, rows, cols, x, y);
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q5_0);

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
        }
    }

    {
    ScopedTimer t(profile_counters().ns_legacy_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> acc(n_tok);
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t b = 0; b < nb; ++b) {
                const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(row_ptr + b * sizeof(block_q5_0));
                const float wd = desireeia_fp16_to_fp32(blk->d);
                uint32_t qh; std::memcpy(&qh, blk->qh, sizeof(qh));
                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    const uint8_t xh0 = (uint8_t) (((qh >> (j + 0))  << 4) & 0x10);
                    const uint8_t xh1 = (uint8_t) (((qh >> (j + 12))     ) & 0x10);
                    wblk[j]      = (int8_t) (((blk->qs[j] & 0x0F) | xh0) - 16);
                    wblk[j + 16] = (int8_t) (((blk->qs[j] >> 4)   | xh1) - 16);
                }
                for (size_t tk = 0; tk < n_tok; ++tk) {
                    const int8_t* xblk = xq[tk].data() + b * 32;
#if defined(__AVX2__)
                    float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                    float dot = dot8_neon(wblk, xblk, 32);
#else
                    int32_t idot = 0;
                    for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                    float dot = (float) idot;
#endif
                    acc[tk] += wd * xscale[tk][b] * dot;
                }
            }
            for (size_t tk = 0; tk < n_tok; ++tk) y[tk * rows + r] = acc[tk];
        }
    });
    }
    profile_counters().calls_legacy.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q5_1(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q5_1);

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    std::vector<int32_t> xsum(nb);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
        for (size_t b = 0; b < nb; ++b) {
            int32_t s = 0;
            for (int j = 0; j < 32; ++j) s += (int32_t) xq[b * 32 + j];
            xsum[b] = s;
        }
    }

    {
    ScopedTimer t(profile_counters().ns_legacy_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            float acc = 0.0f;
            for (size_t b = 0; b < nb; ++b) {
                const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(row_ptr + b * sizeof(block_q5_1));
                const float wd = desireeia_fp16_to_fp32(blk->d);
                const float wm = desireeia_fp16_to_fp32(blk->m);
                uint32_t qh; std::memcpy(&qh, blk->qh, sizeof(qh));
                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    const uint8_t xh0 = (uint8_t) (((qh >> (j + 0))  << 4) & 0x10);
                    const uint8_t xh1 = (uint8_t) (((qh >> (j + 12))     ) & 0x10);
                    wblk[j]      = (int8_t) ((blk->qs[j] & 0x0F) | xh0);
                    wblk[j + 16] = (int8_t) ((blk->qs[j] >> 4)   | xh1);
                }
                const int8_t* xblk = xq.data() + b * 32;
#if defined(__AVX2__)
                float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                float dot = dot8_neon(wblk, xblk, 32);
#else
                int32_t idot = 0;
                for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                float dot = (float) idot;
#endif
                acc += xscale[b] * (wd * dot + wm * (float) xsum[b]);
            }
            y[r] = acc;
        }
    });
    }
    profile_counters().calls_legacy.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q5_1_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q5_1(data, rows, cols, x, y);
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q5_1);

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    std::vector<std::vector<int32_t>> xsum(n_tok, std::vector<int32_t>(nb));
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
            for (size_t b = 0; b < nb; ++b) {
                int32_t s = 0;
                for (int j = 0; j < 32; ++j) s += (int32_t) xq[tk][b * 32 + j];
                xsum[tk][b] = s;
            }
        }
    }

    {
    ScopedTimer t(profile_counters().ns_legacy_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> acc(n_tok);
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t b = 0; b < nb; ++b) {
                const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(row_ptr + b * sizeof(block_q5_1));
                const float wd = desireeia_fp16_to_fp32(blk->d);
                const float wm = desireeia_fp16_to_fp32(blk->m);
                uint32_t qh; std::memcpy(&qh, blk->qh, sizeof(qh));
                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    const uint8_t xh0 = (uint8_t) (((qh >> (j + 0))  << 4) & 0x10);
                    const uint8_t xh1 = (uint8_t) (((qh >> (j + 12))     ) & 0x10);
                    wblk[j]      = (int8_t) ((blk->qs[j] & 0x0F) | xh0);
                    wblk[j + 16] = (int8_t) ((blk->qs[j] >> 4)   | xh1);
                }
                for (size_t tk = 0; tk < n_tok; ++tk) {
                    const int8_t* xblk = xq[tk].data() + b * 32;
#if defined(__AVX2__)
                    float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                    float dot = dot8_neon(wblk, xblk, 32);
#else
                    int32_t idot = 0;
                    for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                    float dot = (float) idot;
#endif
                    acc[tk] += xscale[tk][b] * (wd * dot + wm * (float) xsum[tk][b]);
                }
            }
            for (size_t tk = 0; tk < n_tok; ++tk) y[tk * rows + r] = acc[tk];
        }
    });
    }
    profile_counters().calls_legacy.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q8_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q8_K);
    const size_t n_sub = cols / 32;

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
    }

    ScopedTimer t(profile_counters().ns_kquant2_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            float acc = 0.0f;
            for (size_t si = 0; si < n_super; ++si) {
                const block_q8_K& blk = *reinterpret_cast<const block_q8_K*>(row_ptr + si * sizeof(block_q8_K));
                const float wd = blk.d;
                const size_t sub0 = si * (QK_K / 32);
                for (size_t sb = 0; sb < QK_K / 32; ++sb) {
                    const int8_t* wblk = blk.qs + sb * 32;
                    const int8_t* xblk = xq.data() + (sub0 + sb) * 32;
#if defined(__AVX2__)
                    float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                    float dot = dot8_neon(wblk, xblk, 32);
#else
                    int32_t idot = 0;
                    for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                    float dot = (float) idot;
#endif
                    acc += wd * xscale[sub0 + sb] * dot;
                }
            }
            y[r] = acc;
        }
    });
    profile_counters().calls_kquant2.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q8_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q8_k(data, rows, cols, x, y);
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q8_K);

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
        }
    }

    ScopedTimer t(profile_counters().ns_kquant2_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> acc(n_tok);
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t si = 0; si < n_super; ++si) {
                const block_q8_K& blk = *reinterpret_cast<const block_q8_K*>(row_ptr + si * sizeof(block_q8_K));
                const float wd = blk.d;
                const size_t sub0 = si * (QK_K / 32);
                for (size_t sb = 0; sb < QK_K / 32; ++sb) {
                    const int8_t* wblk = blk.qs + sb * 32;
                    for (size_t tk = 0; tk < n_tok; ++tk) {
                        const int8_t* xblk = xq[tk].data() + (sub0 + sb) * 32;
#if defined(__AVX2__)
                        float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                        float dot = dot8_neon(wblk, xblk, 32);
#else
                        int32_t idot = 0;
                        for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                        float dot = (float) idot;
#endif
                        acc[tk] += wd * xscale[tk][sub0 + sb] * dot;
                    }
                }
            }
            for (size_t tk = 0; tk < n_tok; ++tk) y[tk * rows + r] = acc[tk];
        }
    });
    profile_counters().calls_kquant2.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q2_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q2_K);
    const size_t n_sub16 = cols / 16;

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
    }
    // Q2_K sub-blocks are 16 wide (not 32 like the activation): xscale/xq
    // stay block-32, but the "min" term has to be summed over ranges of 16,
    // which requires the sum of the quantized activations over 16 elements
    // (not the 32 used by Q4_K/Q5_K). Note: since this is a sub-range of a
    // 32-wide block with the SAME xscale (the activation's Q8_0 block is
    // wider than the Q2_K sub-block), no separate scale is needed, just a
    // finer-grained sum.
    std::vector<int32_t> xsum16(n_sub16);
    for (size_t sb = 0; sb < n_sub16; ++sb) {
        int32_t s = 0;
        for (int j = 0; j < 16; ++j) s += (int32_t) xq[sb * 16 + j];
        xsum16[sb] = s;
    }

    ScopedTimer t(profile_counters().ns_kquant2_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            float acc = 0.0f;
            for (size_t si = 0; si < n_super; ++si) {
                const block_q2_K& blk = *reinterpret_cast<const block_q2_K*>(row_ptr + si * sizeof(block_q2_K));
                const float d = desireeia_fp16_to_fp32(blk.d);
                const float dmin = desireeia_fp16_to_fp32(blk.dmin);
                const uint8_t* q = blk.qs;
                const size_t sub0_32 = si * (QK_K / 32);
                int is = 0;
                for (int n = 0; n < QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        for (int half = 0; half < 2; ++half) {
                            const uint8_t sc = blk.scales[is++];
                            const float dl = d * (sc & 0xF);
                            const float ml = dmin * (sc >> 4);
                            const uint8_t* qh = q + half * 16;
                            int8_t wblk[16];
                            for (int l = 0; l < 16; ++l) wblk[l] = (int8_t) ((qh[l] >> shift) & 3);
                            const size_t out_pos = (size_t) sub0_32 * 32 + (size_t) n + (size_t) j * 32 + (size_t) half * 16;
                            const size_t sub16 = out_pos / 16;
                            const size_t sub32 = out_pos / 32;
                            const int8_t* xblk = xq.data() + out_pos;
#if defined(__AVX2__)
                            const float dot = dot8_16(wblk, xblk);
#elif defined(__ARM_NEON)
                            const float dot = dot8_16_neon(wblk, xblk);
#else
                            int32_t idot = 0;
                            for (int l = 0; l < 16; ++l) idot += (int32_t) wblk[l] * (int32_t) xblk[l];
                            float dot = (float) idot;
#endif
                            acc += xscale[sub32] * (dl * dot - ml * (float) xsum16[sub16]);
                        }
                        shift += 2;
                    }
                    q += 32;
                }
            }
            y[r] = acc;
        }
    });
    profile_counters().calls_kquant2.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q2_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q2_k(data, rows, cols, x, y);
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q2_K);
    const size_t n_sub16 = cols / 16;

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    std::vector<std::vector<int32_t>> xsum16(n_tok, std::vector<int32_t>(n_sub16));
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
            for (size_t sb = 0; sb < n_sub16; ++sb) {
                int32_t s = 0;
                for (int j = 0; j < 16; ++j) s += (int32_t) xq[tk][sb * 16 + j];
                xsum16[tk][sb] = s;
            }
        }
    }

    ScopedTimer t(profile_counters().ns_kquant2_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> acc(n_tok);
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t si = 0; si < n_super; ++si) {
                const block_q2_K& blk = *reinterpret_cast<const block_q2_K*>(row_ptr + si * sizeof(block_q2_K));
                const float d = desireeia_fp16_to_fp32(blk.d);
                const float dmin = desireeia_fp16_to_fp32(blk.dmin);
                const uint8_t* q = blk.qs;
                const size_t sub0_32 = si * (QK_K / 32);
                int is = 0;
                for (int n = 0; n < QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        for (int half = 0; half < 2; ++half) {
                            const uint8_t sc = blk.scales[is++];
                            const float dl = d * (sc & 0xF);
                            const float ml = dmin * (sc >> 4);
                            const uint8_t* qh = q + half * 16;
                            int8_t wblk[16];
                            for (int l = 0; l < 16; ++l) wblk[l] = (int8_t) ((qh[l] >> shift) & 3);
                            const size_t out_pos = (size_t) sub0_32 * 32 + (size_t) n + (size_t) j * 32 + (size_t) half * 16;
                            const size_t sub16 = out_pos / 16;
                            const size_t sub32 = out_pos / 32;
                            for (size_t tk = 0; tk < n_tok; ++tk) {
                                const int8_t* xblk = xq[tk].data() + out_pos;
#if defined(__AVX2__)
                                const float dot = dot8_16(wblk, xblk);
#elif defined(__ARM_NEON)
                                const float dot = dot8_16_neon(wblk, xblk);
#else
                                int32_t idot = 0;
                                for (int l = 0; l < 16; ++l) idot += (int32_t) wblk[l] * (int32_t) xblk[l];
                                float dot = (float) idot;
#endif
                                acc[tk] += xscale[tk][sub32] * (dl * dot - ml * (float) xsum16[tk][sub16]);
                            }
                        }
                        shift += 2;
                    }
                    q += 32;
                }
            }
            for (size_t tk = 0; tk < n_tok; ++tk) y[tk * rows + r] = acc[tk];
        }
    });
    profile_counters().calls_kquant2.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

namespace {
// Unpacks the 12 bytes of block_q3_K::scales into 16 signed 6-bit values
// (the -32 offset is applied by the caller). Same bit-shuffling as
// dequantize_row_q3_K in quant.cpp, kept in sync rather than re-derived:
// 4 groups of 4 low scale bits (kmask2, 4 bits) plus 4 groups of 4 high
// bits unpacked from tmp=aux[2] (kmask1, 2 bits each).
inline void unpack_q3_k_scales(const uint8_t* raw12, int8_t out16[16]) {
    uint32_t aux[4];
    std::memcpy(aux, raw12, 12);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    std::memcpy(out16, aux, 16);
}
}

int matmul_q3_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q3_K);

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
    }

    ScopedTimer t(profile_counters().ns_kquant2_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            float acc = 0.0f;
            for (size_t si = 0; si < n_super; ++si) {
                const block_q3_K& blk = *reinterpret_cast<const block_q3_K*>(row_ptr + si * sizeof(block_q3_K));
                const float d_all = desireeia_fp16_to_fp32(blk.d);
                const uint8_t* q = blk.qs;
                const uint8_t* hm = blk.hmask;
                int8_t scales[16];
                unpack_q3_k_scales(blk.scales, scales);
                uint8_t m = 1;
                const size_t sub0_32 = si * (QK_K / 32);
                int is = 0;
                for (int n = 0; n < QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        for (int half = 0; half < 2; ++half) {
                            const float dl = d_all * (float) (scales[is++] - 32);
                            const uint8_t* qh = q + half * 16;
                            const uint8_t* hh = hm + half * 16;
                            int8_t wblk[16];
                            for (int l = 0; l < 16; ++l) {
                                wblk[l] = (int8_t) (((qh[l] >> shift) & 3) - ((hh[l] & m) ? 0 : 4));
                            }
                            const size_t out_pos = (size_t) sub0_32 * 32 + (size_t) n + (size_t) j * 32 + (size_t) half * 16;
                            const size_t sub32 = out_pos / 32;
                            const int8_t* xblk = xq.data() + out_pos;
#if defined(__AVX2__)
                            const float dot = dot8_16(wblk, xblk);
#elif defined(__ARM_NEON)
                            const float dot = dot8_16_neon(wblk, xblk);
#else
                            int32_t idot = 0;
                            for (int l = 0; l < 16; ++l) idot += (int32_t) wblk[l] * (int32_t) xblk[l];
                            float dot = (float) idot;
#endif
                            acc += xscale[sub32] * dl * dot;
                        }
                        shift += 2;
                        m = (uint8_t) (m << 1);
                    }
                    q += 32;
                }
            }
            y[r] = acc;
        }
    });
    profile_counters().calls_kquant2.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q3_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q3_k(data, rows, cols, x, y);
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q3_K);

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
        }
    }

    ScopedTimer t(profile_counters().ns_kquant2_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> acc(n_tok);
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = data + r * row_bytes;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t si = 0; si < n_super; ++si) {
                const block_q3_K& blk = *reinterpret_cast<const block_q3_K*>(row_ptr + si * sizeof(block_q3_K));
                const float d_all = desireeia_fp16_to_fp32(blk.d);
                const uint8_t* q = blk.qs;
                const uint8_t* hm = blk.hmask;
                int8_t scales[16];
                unpack_q3_k_scales(blk.scales, scales);
                uint8_t m = 1;
                const size_t sub0_32 = si * (QK_K / 32);
                int is = 0;
                for (int n = 0; n < QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        for (int half = 0; half < 2; ++half) {
                            const float dl = d_all * (float) (scales[is++] - 32);
                            const uint8_t* qh = q + half * 16;
                            const uint8_t* hh = hm + half * 16;
                            int8_t wblk[16];
                            for (int l = 0; l < 16; ++l) {
                                wblk[l] = (int8_t) (((qh[l] >> shift) & 3) - ((hh[l] & m) ? 0 : 4));
                            }
                            const size_t out_pos = (size_t) sub0_32 * 32 + (size_t) n + (size_t) j * 32 + (size_t) half * 16;
                            const size_t sub32 = out_pos / 32;
                            for (size_t tk = 0; tk < n_tok; ++tk) {
                                const int8_t* xblk = xq[tk].data() + out_pos;
#if defined(__AVX2__)
                                const float dot = dot8_16(wblk, xblk);
#elif defined(__ARM_NEON)
                                const float dot = dot8_16_neon(wblk, xblk);
#else
                                int32_t idot = 0;
                                for (int l = 0; l < 16; ++l) idot += (int32_t) wblk[l] * (int32_t) xblk[l];
                                float dot = (float) idot;
#endif
                                acc[tk] += xscale[tk][sub32] * dl * dot;
                            }
                        }
                        shift += 2;
                        m = (uint8_t) (m << 1);
                    }
                    q += 32;
                }
            }
            for (size_t tk = 0; tk < n_tok; ++tk) y[tk * rows + r] = acc[tk];
        }
    });
    profile_counters().calls_kquant2.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

// Q8_K-style activation quantization: ONE float scale per 256-value
// super-block, instead of one scale per 32-value sub-block like Q8_0.
// The speed win isn't a bigger GEMM tile — it's that all 8 six-bit scales
// of the 32-value sub-blocks can be applied in INTEGER arithmetic
// (accumulated as sc_i*dot_i, m_i*xsum_i on integers), so the float
// conversion/scale multiply happens ONCE per super-block instead of once
// per sub-block (8x fewer int->float conversions and float multiplies per
// super-block row). This gets the same result while reusing the existing
// dot8_avx2_i32/dot8_16_i32 kernels as-is — only the ORDER of operations
// changes, no new SIMD primitives, same formula. Declared in engine.h
// (not anonymous) so dense_forward.cpp can quantize the activation shared
// by wq/wk/wv/wo/ffn_gate/ffn_up once per layer instead of per matmul.
void quantize_q8_k_super(const float* x, size_t cols, std::vector<int8_t>& q, std::vector<float>& dscale) {
    const size_t n_super = cols / QK_K;
    q.resize(cols);
    dscale.resize(n_super);
    for (size_t s = 0; s < n_super; ++s) {
        const float* xs = x + s * QK_K;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) amax = std::max(amax, std::fabs(xs[j]));
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        dscale[s] = d;
        int8_t* qs = q.data() + s * QK_K;
        for (int j = 0; j < QK_K; ++j) {
            int v = (int) std::lrintf(xs[j] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qs[j] = (int8_t) v;
        }
    }
}

// Quantizes the activation Q8_K-style (ONE scale every 256 values) but
// exposes the same per-32 arrays the kernels used with Q8_0: the scale is
// REPLICATED across the 8 slots of the super-block.
//
// The point isn't saving on the quantization itself — it's that, with a
// constant scale inside the super-block, the kernel can accumulate ALL 8
// sub-blocks in the integer domain and do a single float conversion/
// multiply every 256 weights instead of eight. With the per-32 scale this
// is impossible, because every sub-block has to be brought back to float
// before it can be summed with the others.
//
// The replication keeps the kernels that haven't been converted yet
// (Q5_K, Q6_K) compatible, since they keep reading xscale[sub] without
// knowing it is now piecewise-constant: no double quantization, no
// signature change.
void quantize_act_q8k_rep(const float* x, size_t cols, std::vector<int8_t>& q,
                          std::vector<float>& xscale32, std::vector<int32_t>& xsum32) {
    const size_t n_super = cols / QK_K;
    const size_t n_sub = cols / 32;
    q.resize(cols);
    xscale32.resize(n_sub);
    xsum32.resize(n_sub);
    for (size_t s = 0; s < n_super; ++s) {
        const float* xs = x + s * QK_K;
        int8_t* qs = q.data() + s * QK_K;
#if defined(__AVX2__)
        // Vectorized version. The scalar version cost ~19 cycles per value
        // (measured at 3.5 ms per token over ~550000 values, the heaviest
        // serial item after the activation itself): scalar lrintf is slow
        // and the compiler can't auto-vectorize it because of the clamps.
        const __m256i absmask = _mm256_set1_epi32(0x7FFFFFFF);
        __m256 vmax = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 8) {
            const __m256 v = _mm256_loadu_ps(xs + j);
            vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, _mm256_castsi256_ps(absmask)));
        }
        const float amax = hsmax_ps_avx2(vmax);
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 127.0f / amax : 0.0f;

        const __m256 vid = _mm256_set1_ps(id);
        for (int sb = 0; sb < QK_K / 32; ++sb) {
            // The 32 sub-block values in four vectors, converted to int32
            // with round-to-even (like lrintf in its default mode), then
            // packed into int8 with saturation.
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 +  0), vid));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 +  8), vid));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 + 16), vid));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 + 24), vid));

            // The sub-block sum is derived here in int32, without a second
            // read pass over the already-written bytes.
            const __m256i vsum = _mm256_add_epi32(_mm256_add_epi32(i0, i1),
                                                  _mm256_add_epi32(i2, i3));

            // packs saturates to [-128,127]; -128 never occurs because
            // |x*id| <= 127 by construction of id.
            __m256i p01 = _mm256_packs_epi32(i0, i1);   // lanes: [i0.lo i1.lo | i0.hi i1.hi]
            __m256i p23 = _mm256_packs_epi32(i2, i3);
            __m256i p   = _mm256_packs_epi16(p01, p23);
            // packs operates per 128-bit lane: this puts the groups back in order.
            p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(qs + sb * 32), p);

            const size_t idx = s * (QK_K / 32) + (size_t) sb;
            xscale32[idx] = d;
            xsum32[idx] = hsum_epi32_avx2(vsum);
        }
#elif defined(__ARM_NEON)
        // Same logic as the AVX2 version above, with the same guarantees:
        // `id` is computed so that |x*id| <= 127 by construction, so the
        // sum of the values BEFORE saturation to int8 coincides exactly
        // with the sum of the FINAL values — so the pre-narrow int32
        // vector is summed, no need for a second pass over the written
        // bytes.
        //
        // vcvtnq_s32_f32/vmaxvq_f32 are AArch64 intrinsics (not ARMv7):
        // consistent with the rest of the file, which already assumes
        // aarch64 (vaddvq_s32 in dot_i8x16_neon) — the project's target is
        // macOS Apple Silicon and Linux ARM, both aarch64.
        float32x4_t vmax4 = vdupq_n_f32(0.0f);
        for (int j = 0; j < QK_K; j += 4) {
            vmax4 = vmaxq_f32(vmax4, vabsq_f32(vld1q_f32(xs + j)));
        }
        const float amax = vmaxvq_f32(vmax4);
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 127.0f / amax : 0.0f;

        for (int sb = 0; sb < QK_K / 32; ++sb) {
            const float* xb = xs + sb * 32;
            int32x4_t sum32 = vdupq_n_s32(0);
            // The 32 sub-block activations are converted in two halves of
            // 16 (4 float32x4 vectors each), then narrowed in cascade:
            // int32 -> int16 (vqmovn, saturates) -> int8 (vqmovn,
            // saturates again — the double saturation is harmless because
            // by construction the values never leave [-127,127]).
            int8x16_t out[2];
            for (int half = 0; half < 2; ++half) {
                const int32x4_t i0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(xb + half * 16 +  0), id));
                const int32x4_t i1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(xb + half * 16 +  4), id));
                const int32x4_t i2 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(xb + half * 16 +  8), id));
                const int32x4_t i3 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(xb + half * 16 + 12), id));
                sum32 = vaddq_s32(sum32, vaddq_s32(vaddq_s32(i0, i1), vaddq_s32(i2, i3)));
                const int16x8_t nab = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                const int16x8_t ncd = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                out[half] = vcombine_s8(vqmovn_s16(nab), vqmovn_s16(ncd));
            }
            vst1q_s8(qs + sb * 32, out[0]);
            vst1q_s8(qs + sb * 32 + 16, out[1]);

            const size_t idx = s * (QK_K / 32) + (size_t) sb;
            xscale32[idx] = d;
            xsum32[idx] = vaddvq_s32(sum32);
        }
#else
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) amax = std::max(amax, std::fabs(xs[j]));
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 127.0f / amax : 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            int v = (int) std::lrintf(xs[j] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            qs[j] = (int8_t) v;
        }
        for (int sb = 0; sb < QK_K / 32; ++sb) {
            int32_t acc = 0;
            for (int j = 0; j < 32; ++j) acc += (int32_t) qs[sb * 32 + j];
            xscale32[s * (QK_K / 32) + (size_t) sb] = d;
            xsum32[s * (QK_K / 32) + (size_t) sb] = acc;
        }
#endif
    }
}

namespace {
// Core of matmul_q4_k, extracted so it can be called both from the normal
// path (quantizes the activation internally) and from the
// "pre-quantized" path matmul_q4_k_pq ("eliminate redundant
// re-quantization" phase, 2026-09-07): wq/wk/wo/ffn_gate/ffn_up of a
// Q4_K_M model ALL read from the same activation (the output of
// attn_norm/ffn_norm respectively), but being different formats
// (Q4_K/Q6_K) each one used to call its own matmul_qX_k, which
// requantized the activation from scratch — up to 5 times the same
// quantization per layer. Given that
// Q4_K/Q5_K/Q6_K now share the same activation quantization scheme
// (quantize_q8_k_super), so it can be computed once in dense_forward.cpp
// and reused across every matmul_qX_k_pq call in the layer.
//
// CORRECTNESS REVERT (2026-09-07): fusing the scales into integer math
// with Q8_K-style activation quantization (one float scale per 256
// elements) measured FAST but was BROKEN — tested against a real prompt,
// not just a benchmark string, the output collapsed into garbage from the
// very first generated token, not after many steps. Likely cause:
// activation outliers (a known LLM phenomenon — a few dimensions end up
// much larger than the rest after RMSNorm). A 256-wide block with a
// single scale is far more vulnerable to that than a 32-wide one, because
// the outlier crushes 250 other values toward zero during quantization
// instead of 30. This isn't a fundamentally broken idea — the tradeoff
// just isn't acceptable here: the explicit priority is an engine that
// produces sensible output, not maximum tok/s. Reverted to per-32
// sub-block quantization (quantize_q8_0, the same scheme already used by
// matmul_q4_0/Q8_0 and proven consistent earlier in this project), scale
// applied in float per sub-block as before the fusion attempt.
//
// ATTEMPT REMOVED (2026-09-07): four rows in flight at once (same
// activations loaded once, reused across four rows; four independent DRAM
// read streams; plus software prefetch of the next super-block). The idea
// was to raise memory-level parallelism, not compute parallelism.
// Measured: +67% on the microbenchmark but -2% on the actual ENGINE (9.92
// vs 10.12 tok/s at 16 threads). Removed, not kept.
// The lesson matters more than the code: the microbenchmark reused a
// 15 MB matrix on every iteration, which fits in this machine's 24 MB L3
// cache — measuring a "hot weights in cache" regime that decode never
// actually sees. That's what motivated the "DRAM" regime added to
// bench/kernel_bench.cpp. When a microbenchmark and the real engine
// disagree, trust the engine.

// row_lo/row_hi select a slice of rows to compute on the CALLING thread,
// without dispatching to the pool. row_hi == 0 keeps the original behaviour
// (compute every row, in parallel). The slice mode is what lets several
// independent matrices share a single dispatch — see matmul_fused_pq.
int matmul_q4_k_core(const uint8_t* q4k_data, size_t rows, size_t cols,
                      const int8_t* xq, const float* xscale, const int32_t* xsum, float* y,
                      size_t row_lo = 0, size_t row_hi = 0) {
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q4_K);

    auto row_super_dot = [&](const block_q4_K& blk, size_t sub0) -> float {
        const float d = desireeia_fp16_to_fp32(blk.d);
        const float dmin = desireeia_fp16_to_fp32(blk.dmin);
        const uint8_t* q = blk.qs;

        // SCALAR PHASE, all done up front and outside the vector loop.
        // Previously the 8 scales/mins were unpacked INSIDE the loop, two
        // per group: get_scale_min_k4 is full of branches and bit
        // manipulation, and interleaving it with the SIMD instructions
        // kept the vector units idle waiting. Counted ~20 scalar operations
        // per 32 bytes of weights, against ~16 vector ones: the loop was
        // actually limited by the scalar part, not by the useful compute.
        // By separating them, the out-of-order engine can overlap the two
        // phases. Here the final product d*scale*xscale is precomputed
        // directly, so inside the loop only a broadcast from memory
        // remains. The activation is Q8_K-quantized: a single scale per
        // super-block, replicated across the 8 per-32 slots (see
        // quantize_act_q8k_rep). This is what enables the integer
        // accumulation further below.
        const float dx = xscale[sub0];
        float min_acc = 0.0f;
        uint8_t sc8[QK_K / 32];
        for (int i = 0; i < QK_K / 32; ++i) {
            uint8_t sc, m;
            get_scale_min_k4(i, blk.scales, &sc, &m);
            sc8[i] = sc;
            min_acc -= dmin * (float) m * dx * (float) xsum[sub0 + (size_t) i];
        }

        int is = 0;
#if defined(__AVX2__)
        // TWO independent accumulators, not one. With a single accumulator
        // the two add_ps per group form a serial dependency chain: each
        // add has to wait for the previous one (~4 cycles of latency), and
        // the loop becomes latency-bound instead of throughput-bound.
        // Measured: we were running at 24% of the available memory
        // bandwidth at ANY thread count — a symptom of a dependency stall,
        // not of memory saturation. The two sub-blocks A and B are
        // independent, so they can accumulate in parallel and only be
        // summed at the end.
        //
        // The accumulators are INTEGER, not float: the weight's 6-bit
        // scale is applied in the integer domain with madd_epi16 (same
        // technique already used in q6k_group128_avx2), so all 8
        // sub-blocks get summed together without ever passing through
        // floating point. Only a single cvtepi32_ps + multiply remains
        // every 256 weights instead of eight. No overflow: maddubs stays
        // within +-3810, times the scale (<=63) gives 240030, summed over
        // 8 sub-blocks stays well inside int32.
        __m256i sumiA = _mm256_setzero_si256();
        __m256i sumiB = _mm256_setzero_si256();
#elif defined(__ARM_NEON)
        // Different scheme than the AVX2 path above: here the scale is
        // multiplied in RIGHT AFTER reducing the dot-product to a scalar
        // (vaddvq_s32), accumulating into one scalar int32_t for A and one
        // for B, instead of carrying the scale through the SIMD domain
        // like the AVX2 path does. This is the simpler scheme, and it's
        // viable specifically because vaddvq_s32 (4-lane horizontal
        // reduction) is a single instruction on NEON, whereas on AVX2
        // (8 lanes) the reduction is more expensive — which is exactly
        // what the SIMD accumulation was avoiding repeating there.
        // No overflow risk: max |idot| per sub-block is ~15*127*32=60960,
        // times scale (<=63) is ~3.84M, times 8 sub-blocks is ~30.7M —
        // comfortably inside int32 (the sum here stays within a plain
        // int32 without needing the "wide" integer accumulation the AVX2
        // path uses).
        int32_t sumiA = 0;
        int32_t sumiB = 0;
#else
        float acc = 0.0f;
#endif
        for (int grp = 0; grp < QK_K; grp += 64) {
            const size_t subA = sub0 + (size_t) grp / 32;
            const size_t subB = subA + 1;
            const int8_t* xA = xq + subA * 32;
            const int8_t* xB = xq + subB * 32;
#if defined(__AVX2__)
            // The 32 packed bytes contain 64 weights: the low nibbles are
            // sub-block A, the high ones sub-block B. They are unpacked
            // IN-REGISTER and fed straight into maddubs, without ever
            // passing through a stack array: this is the difference from
            // the attempt described at the top of the file (which unpacked
            // in-register but then re-stored into wA[]/wB[] and reloaded,
            // cancelling the gain with a store-to-load).
            //
            // On top of that, this exploits a fact specific to Q4_K that
            // the generic dot8_avx2_i32 path can't: the weights are
            // UNSIGNED 0..15 (the min is handled separately via dmin), and
            // maddubs wants exactly one unsigned and one signed operand.
            // So they go in directly, without the sign_epi8 pair needed
            // for the both-signed trick. No overflow: the max per pair is
            // 2*15*127 = 3810, well within int16.
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i wA = _mm256_and_si256(packed, _mm256_set1_epi8(0x0F));
            const __m256i wB = _mm256_and_si256(_mm256_srli_epi16(packed, 4), _mm256_set1_epi8(0x0F));
            const __m256i vxA = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xA));
            const __m256i vxB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xB));
            // The weight's scale enters here, in int16, instead of later in float.
            sumiA = _mm256_add_epi32(sumiA, _mm256_madd_epi16(
                _mm256_set1_epi16((short) sc8[is + 0]), _mm256_maddubs_epi16(wA, vxA)));
            sumiB = _mm256_add_epi32(sumiB, _mm256_madd_epi16(
                _mm256_set1_epi16((short) sc8[is + 1]), _mm256_maddubs_epi16(wB, vxB)));
#elif defined(__ARM_NEON)
            // 32 packed bytes = 64 weights (32 for A in the low nibbles, 32
            // for B in the high ones), read in two 16-byte halves because
            // NEON registers are 128 bits wide against AVX2's 256. The
            // 0..15 weights remain valid as signed int8 without needing
            // maddubs's unsigned/signed trick (the sign bit is never set
            // for a nibble).
            const uint8x16_t p0 = vld1q_u8(q);
            const uint8x16_t p1 = vld1q_u8(q + 16);
            const uint8x16_t m4 = vdupq_n_u8(0x0F);
            const int8x16_t wA0 = vreinterpretq_s8_u8(vandq_u8(p0, m4));
            const int8x16_t wA1 = vreinterpretq_s8_u8(vandq_u8(p1, m4));
            const int8x16_t wB0 = vreinterpretq_s8_u8(vshrq_n_u8(p0, 4));
            const int8x16_t wB1 = vreinterpretq_s8_u8(vshrq_n_u8(p1, 4));

            const int8x16_t xA0 = vld1q_s8(xA);
            const int8x16_t xA1 = vld1q_s8(xA + 16);
            const int8x16_t xB0 = vld1q_s8(xB);
            const int8x16_t xB1 = vld1q_s8(xB + 16);

            int32x4_t dA = dot_i8x16_neon(vdupq_n_s32(0), wA0, xA0);
            dA = dot_i8x16_neon(dA, wA1, xA1);
            int32x4_t dB = dot_i8x16_neon(vdupq_n_s32(0), wB0, xB0);
            dB = dot_i8x16_neon(dB, wB1, xB1);

            sumiA += vaddvq_s32(dA) * (int32_t) sc8[is + 0];
            sumiB += vaddvq_s32(dB) * (int32_t) sc8[is + 1];
#else
            int32_t idotA = 0, idotB = 0;
            for (int l = 0; l < 32; ++l) {
                idotA += (int32_t) (q[l] & 0x0F) * (int32_t) xA[l];
                idotB += (int32_t) (q[l] >> 4)   * (int32_t) xB[l];
            }
            acc += d * dx * ((float) sc8[is + 0] * (float) idotA
                           + (float) sc8[is + 1] * (float) idotB);
#endif
            q += 32;
            is += 2;
        }
#if defined(__AVX2__)
        // A single conversion + multiplication per super-block.
        const float acc = d * dx * hsum_ps_avx2(
            _mm256_cvtepi32_ps(_mm256_add_epi32(sumiA, sumiB)));
#elif defined(__ARM_NEON)
        const float acc = d * dx * (float) (sumiA + sumiB);
#endif
        return acc + min_acc;
    };

    auto compute_fn = [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(q4k_data + r * row_bytes);
            float acc = 0.0f;
            for (size_t si = 0; si < n_super; ++si) {
                acc += row_super_dot(blocks[si], si * (QK_K / 32));
            }
            y[r] = acc;
        }
    };
    if (row_hi > row_lo) {
        // Slice mode: the caller already owns a parallel region.
        compute_fn(row_lo, row_hi);
        return DESIREEIA_OK;
    }
    {
        ScopedTimer t(profile_counters().ns_q4k_compute);
        parallel_rows(rows, compute_fn);
    }
    profile_counters().calls_q4k.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}
}

int matmul_q4_k(const uint8_t* q4k_data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_sub = cols / 32;

    (void) n_sub;
    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<int32_t> xsum;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_act_q8k_rep(x, cols, xq, xscale, xsum);
    }
    return matmul_q4_k_core(q4k_data, rows, cols, xq.data(), xscale.data(), xsum.data(), y);
}

int matmul_q4_k_pq(const uint8_t* q4k_data, size_t rows, size_t cols,
                    const int8_t* xq, const float* xscale, const int32_t* xsum, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    return matmul_q4_k_core(q4k_data, rows, cols, xq, xscale, xsum, y);
}

int matmul_q4_0_batch(const uint8_t* q4_data, size_t rows, size_t cols,
                       const float* x, size_t n_tok, float* y) {
    if (n_tok == 0) return DESIREEIA_OK;
    if (n_tok == 1) return matmul_q4_0(q4_data, rows, cols, x, y);
    if (cols == 0 || cols % 32 != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t nb = cols / 32;
    const size_t row_bytes = nb * sizeof(block_q4_0);

    std::vector<std::vector<int8_t>> xq(n_tok);
    std::vector<std::vector<float>> xscale(n_tok);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        std::vector<uint8_t> unused_signs;
        for (size_t tk = 0; tk < n_tok; ++tk) {
            quantize_q8_0(x + tk * cols, cols, xq[tk], xscale[tk], unused_signs);
        }
    }

    {
    ScopedTimer t(profile_counters().ns_q40_compute);
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> acc(n_tok);
        for (size_t r = r0; r < r1; ++r) {
            const uint8_t* row_ptr = q4_data + r * row_bytes;
            std::fill(acc.begin(), acc.end(), 0.0f);
            for (size_t b = 0; b < nb; ++b) {
                const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(row_ptr + b * sizeof(block_q4_0));
                const float wd = desireeia_fp16_to_fp32(blk->d);

                int8_t wblk[32];
                for (int j = 0; j < 16; ++j) {
                    wblk[j]      = (int8_t) ((blk->qs[j] & 0x0F) - 8);
                    wblk[j + 16] = (int8_t) ((blk->qs[j] >> 4)   - 8);
                }

                for (size_t tk = 0; tk < n_tok; ++tk) {
                    const int8_t* xblk = xq[tk].data() + b * 32;
#if defined(__AVX2__)
                    float dot = dot8_avx2(wblk, xblk, 32);
#elif defined(__ARM_NEON)
                    float dot = dot8_neon(wblk, xblk, 32);
#else
                    int32_t idot = 0;
                    for (int j = 0; j < 32; ++j) idot += (int32_t) wblk[j] * (int32_t) xblk[j];
                    float dot = (float) idot;
#endif
                    acc[tk] += wd * xscale[tk][b] * dot;
                }
            }
            for (size_t tk = 0; tk < n_tok; ++tk) y[tk * rows + r] = acc[tk];
        }
    });
    }
    profile_counters().calls_q40.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q4_k_batch(const uint8_t* q4k_data, size_t rows, size_t cols,
                       const float* x, size_t n_tok, float* y) {
    // CORRECTNESS REVERT (2026-09-07): the 2x4 tile with Q8_K quantization
    // (see the note in matmul_q4_k_core) produced corrupted output on a
    // real model. Until the tile is rewritten with fine-grained
    // quantization (Q8_0 per 32-wide sub-block), this falls back to the
    // already-correct single-column version, row by row. Slower (no reuse
    // of the decoded weight across columns), but correct.
    if (n_tok == 0) return DESIREEIA_OK;
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    for (size_t tk = 0; tk < n_tok; ++tk) {
        int rc = matmul_q4_k(q4k_data, rows, cols, x + tk * cols, y + tk * rows);
        if (rc != DESIREEIA_OK) return rc;
    }
    return DESIREEIA_OK;
}

int matmul_q6_k_batch(const uint8_t* q6k_data, size_t rows, size_t cols,
                       const float* x, size_t n_tok, float* y) {
    // Same revert as matmul_q4_k_batch above, same reasoning.
    if (n_tok == 0) return DESIREEIA_OK;
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    for (size_t tk = 0; tk < n_tok; ++tk) {
        int rc = matmul_q6_k(q6k_data, rows, cols, x + tk * cols, y + tk * rows);
        if (rc != DESIREEIA_OK) return rc;
    }
    return DESIREEIA_OK;
}

// CORRECTNESS REVERT (2026-09-07): same reasoning as matmul_q4_k above,
// quantization reverted to Q8_0 per 32-wide sub-block.
int matmul_q5_k(const uint8_t* q5k_data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q5_K);
    const size_t n_sub = cols / 32;

    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<uint8_t> unused_signs;
    std::vector<int32_t> xsum(n_sub);
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        quantize_q8_0(x, cols, xq, xscale, unused_signs);
        for (size_t sb = 0; sb < n_sub; ++sb) {
            int32_t s = 0;
            for (int j = 0; j < 32; ++j) s += (int32_t) xq[sb * 32 + j];
            xsum[sb] = s;
        }
    }

    auto row_super_dot = [&](const block_q5_K& blk, size_t sub0) -> float {
        const float d = desireeia_fp16_to_fp32(blk.d);
        const float dmin = desireeia_fp16_to_fp32(blk.dmin);
        const uint8_t* ql = blk.qs;
        const uint8_t* qh = blk.qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        float min_acc = 0.0f;
#if defined(__AVX2__)
        __m256 acc_vec = _mm256_setzero_ps();
#elif defined(__ARM_NEON)
        float acc = 0.0f;
#else
        float acc = 0.0f;
#endif
        for (int grp = 0; grp < QK_K; grp += 64) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is + 0, blk.scales, &sc1, &m1);
            get_scale_min_k4(is + 1, blk.scales, &sc2, &m2);
            const float d1 = d * sc1, mm1 = dmin * m1;
            const float d2 = d * sc2, mm2 = dmin * m2;
            int8_t wA[32], wB[32];
            for (int l = 0; l < 32; ++l) {
                wA[l] = (int8_t) ((ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0));
                wB[l] = (int8_t) ((ql[l] >> 4)   + (qh[l] & u2 ? 16 : 0));
            }
            const size_t subA = sub0 + (size_t) grp / 32;
            const size_t subB = subA + 1;
            const int8_t* xA = xq.data() + subA * 32;
            const int8_t* xB = xq.data() + subB * 32;
            min_acc += xscale[subA] * (-mm1 * (float) xsum[subA]);
            min_acc += xscale[subB] * (-mm2 * (float) xsum[subB]);
#if defined(__AVX2__)
            const __m256 dotA_f = _mm256_cvtepi32_ps(dot8_avx2_i32(wA, xA));
            const __m256 dotB_f = _mm256_cvtepi32_ps(dot8_avx2_i32(wB, xB));
            acc_vec = _mm256_add_ps(acc_vec, _mm256_mul_ps(dotA_f, _mm256_set1_ps(d1 * xscale[subA])));
            acc_vec = _mm256_add_ps(acc_vec, _mm256_mul_ps(dotB_f, _mm256_set1_ps(d2 * xscale[subB])));
#elif defined(__ARM_NEON)
            // Q5_K weights: 5 bits (0..31), safe as signed int8 (bit 7 is
            // never set). Same scalar-accumulate scheme as the Q4_K/Q6_K
            // NEON paths: reduce with vaddvq_s32, multiply by the scale,
            // accumulate in float (not in the integer domain here, because
            // the Q5_K activation stays at per-32 scale, not per-256 Q8_K
            // like Q4_K/Q6_K — see the CORRECTNESS REVERT comment at the
            // top of this function).
            const int32x4_t dA = dot_i8x16_neon(
                dot_i8x16_neon(vdupq_n_s32(0), vld1q_s8(wA), vld1q_s8(xA)),
                vld1q_s8(wA + 16), vld1q_s8(xA + 16));
            const int32x4_t dB = dot_i8x16_neon(
                dot_i8x16_neon(vdupq_n_s32(0), vld1q_s8(wB), vld1q_s8(xB)),
                vld1q_s8(wB + 16), vld1q_s8(xB + 16));
            acc += (float) vaddvq_s32(dA) * (d1 * xscale[subA]);
            acc += (float) vaddvq_s32(dB) * (d2 * xscale[subB]);
#else
            int32_t idotA = 0, idotB = 0;
            for (int l = 0; l < 32; ++l) {
                idotA += (int32_t) wA[l] * (int32_t) xA[l];
                idotB += (int32_t) wB[l] * (int32_t) xB[l];
            }
            acc += xscale[subA] * d1 * (float) idotA;
            acc += xscale[subB] * d2 * (float) idotB;
#endif
            ql += 32; is += 2;
            u1 = (uint8_t) (u1 << 2); u2 = (uint8_t) (u2 << 2);
        }
#if defined(__AVX2__)
        float tmp[8];
        _mm256_storeu_ps(tmp, acc_vec);
        float acc = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif
        return acc + min_acc;
    };

    auto compute_fn = [&](size_t r0, size_t r1) {
        for (size_t r = r0; r < r1; ++r) {
            const block_q5_K* blocks = reinterpret_cast<const block_q5_K*>(q5k_data + r * row_bytes);
            float acc = 0.0f;
            for (size_t si = 0; si < n_super; ++si) {
                acc += row_super_dot(blocks[si], si * (QK_K / 32));
            }
            y[r] = acc;
        }
    };
    {
        ScopedTimer t(profile_counters().ns_q5k_compute);
        parallel_rows(rows, compute_fn);
    }
    profile_counters().calls_q5k.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

int matmul_q5_k_batch(const uint8_t* q5k_data, size_t rows, size_t cols,
                       const float* x, size_t n_tok, float* y) {
    // Same revert as matmul_q4_k_batch, same reasoning: falls back to the
    // already-correct single-column version.
    if (n_tok == 0) return DESIREEIA_OK;
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    for (size_t tk = 0; tk < n_tok; ++tk) {
        int rc = matmul_q5_k(q5k_data, rows, cols, x + tk * cols, y + tk * rows);
        if (rc != DESIREEIA_OK) return rc;
    }
    return DESIREEIA_OK;
}

namespace {
// CORRECTNESS REVERT (2026-09-07): see the full note in matmul_q4_k_core
// above. Reverted to per-16 sub-block quantization (quantize_q8_0 of the
// activation, then used at granularity 32 with /16 indexing for Q6_K)
// instead of the single per-256 Q8_K scale.
#if defined(__AVX2__)
// Accumulates the contribution of one group of 128 Q6_K weights (the
// format's natural group: 64 bytes of low nibbles + 32 bytes of high bits
// + 8 scales).
//
// Three differences from the previous version, all measured:
//  1. Full width. Previously this worked on 16 elements (SSE, 128 bit)
//     because the Q6_K scale is per-16: half the available vector width
//     was wasted on every single product.
//  2. The per-16 scales are applied IN THE INTEGER DOMAIN with
//     madd_epi16. The 32 bytes of an AVX2 vector split into two 128-bit
//     lanes, and the first 16 weights land exactly in the first 8 int16
//     lanes: so a single scale vector with sc[2q] in the low lane and
//     sc[2q+1] in the high one suffices. This leaves a SINGLE float
//     multiply every 32 weights instead of two every 16.
//  3. The weights' -32 offset is not applied byte by byte (which would
//     make the weights signed and force the sign_epi8 trick): w is kept
//     unsigned 0..63 and 32*x is subtracted in the int16 domain. No
//     saturation: maddubs(w,x) stays within [-16128, 16002],
//     maddubs(32,x) within [-8128, 8128], the difference within
//     [-24256, 24130].
// The accumulator is INTEGER: the activation is Q8_K (one scale per
// super-block, see quantize_act_q8k_rep), so all the groups get summed
// together in the integer domain and the conversion to float happens only
// once per super-block instead of once every 32 weights.
static inline __m256i q6k_group128_avx2(const uint8_t* ql, const uint8_t* qh,
                                        const int8_t* sc,
                                        const int8_t* x, __m256i acc) {
    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i m3  = _mm256_set1_epi8(0x03);
    const __m256i c32 = _mm256_set1_epi8(32);
    const __m256i qlL = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
    const __m256i qlH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
    const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

    // The shifts are 16-bit (there's no per-byte shift in AVX2): bits can
    // migrate between the word's two bytes, but the following AND with
    // m3/m4 cleans it up, so the per-byte result stays exact.
    __m256i w[4];
    w[0] = _mm256_or_si256(_mm256_and_si256(qlL, m4),
                           _mm256_slli_epi16(_mm256_and_si256(qhv, m3), 4));
    w[1] = _mm256_or_si256(_mm256_and_si256(qlH, m4),
                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m3), 4));
    w[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlL, 4), m4),
                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m3), 4));
    w[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(qlH, 4), m4),
                           _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m3), 4));

    for (int q = 0; q < 4; ++q) {
        const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + q * 32));
        const __m256i p  = _mm256_sub_epi16(_mm256_maddubs_epi16(w[q], vx),
                                            _mm256_maddubs_epi16(c32, vx));
        const __m256i sv = _mm256_set_m128i(_mm_set1_epi16(sc[q * 2 + 1]),
                                            _mm_set1_epi16(sc[q * 2 + 0]));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, sv));
    }
    return acc;
}

#elif defined(__ARM_NEON)
// NEON equivalent of q6k_group128_avx2 above, but structured like the
// Q4_K NEON kernel: scalar int32 accumulation (scale multiplied right
// after the horizontal reduction), not vector accumulation with the scale
// in the SIMD domain. Same reasoning: on NEON (128 bit, 4 int32 lanes)
// vaddvq_s32 is a single instruction, so there's nothing to gain by
// deferring the reduction.
//
// Difference from the AVX2 branch: there the -32 offset was subtracted
// with a second maddubs (because maddubs wants one unsigned operand, and
// the weights would become signed after -32). NEON has no such
// constraint — vmull_s8 accepts two signed operands directly — so here 32
// is subtracted ONCE with vsubq_s8, before the dot-product, not with two
// maddubs per group.
//
// The 8 values returned by qh/ql cover 128 weights in four groups of 32
// (w[0..3]), each split into two 16-wide halves with its own scale
// (sc[2q+0] for elements 0-15, sc[2q+1] for 16-31) — same indexing as the
// scalar fallback above, checked line by line against it.
static inline int32_t q6k_group128_neon(const uint8_t* ql, const uint8_t* qh,
                                        const int8_t* sc, const int8_t* x) {
    const uint8x16_t m4  = vdupq_n_u8(0x0F);
    const uint8x16_t m3  = vdupq_n_u8(0x03);
    const int8x16_t  c32 = vdupq_n_s8(32);

    const uint8x16_t qlL_a = vld1q_u8(ql);
    const uint8x16_t qlL_b = vld1q_u8(ql + 16);
    const uint8x16_t qlH_a = vld1q_u8(ql + 32);
    const uint8x16_t qlH_b = vld1q_u8(ql + 48);
    const uint8x16_t qh_a  = vld1q_u8(qh);
    const uint8x16_t qh_b  = vld1q_u8(qh + 16);

    // High bits of the four groups, already positioned in the high nibble
    // (<<4) so the OR with ql's low nibble reassembles the 6-bit value.
    const uint8x16_t h0_a = vshlq_n_u8(vandq_u8(qh_a, m3), 4);
    const uint8x16_t h0_b = vshlq_n_u8(vandq_u8(qh_b, m3), 4);
    const uint8x16_t h1_a = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 2), m3), 4);
    const uint8x16_t h1_b = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 2), m3), 4);
    const uint8x16_t h2_a = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 4), m3), 4);
    const uint8x16_t h2_b = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 4), m3), 4);
    const uint8x16_t h3_a = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_a, 6), m3), 4);
    const uint8x16_t h3_b = vshlq_n_u8(vandq_u8(vshrq_n_u8(qh_b, 6), m3), 4);

    const int8x16_t w0_a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlL_a, m4), h0_a)), c32);
    const int8x16_t w0_b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlL_b, m4), h0_b)), c32);
    const int8x16_t w1_a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlH_a, m4), h1_a)), c32);
    const int8x16_t w1_b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qlH_b, m4), h1_b)), c32);
    const int8x16_t w2_a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlL_a, 4), h2_a)), c32);
    const int8x16_t w2_b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlL_b, 4), h2_b)), c32);
    const int8x16_t w3_a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlH_a, 4), h3_a)), c32);
    const int8x16_t w3_b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qlH_b, 4), h3_b)), c32);

    const int32x4_t z = vdupq_n_s32(0);
    int32_t sum = 0;
    sum += vaddvq_s32(dot_i8x16_neon(z, w0_a, vld1q_s8(x +   0))) * (int32_t) sc[0];
    sum += vaddvq_s32(dot_i8x16_neon(z, w0_b, vld1q_s8(x +  16))) * (int32_t) sc[1];
    sum += vaddvq_s32(dot_i8x16_neon(z, w1_a, vld1q_s8(x +  32))) * (int32_t) sc[2];
    sum += vaddvq_s32(dot_i8x16_neon(z, w1_b, vld1q_s8(x +  48))) * (int32_t) sc[3];
    sum += vaddvq_s32(dot_i8x16_neon(z, w2_a, vld1q_s8(x +  64))) * (int32_t) sc[4];
    sum += vaddvq_s32(dot_i8x16_neon(z, w2_b, vld1q_s8(x +  80))) * (int32_t) sc[5];
    sum += vaddvq_s32(dot_i8x16_neon(z, w3_a, vld1q_s8(x +  96))) * (int32_t) sc[6];
    sum += vaddvq_s32(dot_i8x16_neon(z, w3_b, vld1q_s8(x + 112))) * (int32_t) sc[7];
    return sum;
}
#endif

// row_lo/row_hi as in matmul_q4_k_core: a non-empty range computes that
// slice inline instead of dispatching.
int matmul_q6_k_core(const uint8_t* q6k_data, size_t rows, size_t cols,
                      const int8_t* xq, const float* xscale, float* y,
                      size_t row_lo = 0, size_t row_hi = 0) {
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q6_K);

    auto row_super_dot = [&](const block_q6_K& blk, size_t out_base) -> float {
        const float d = desireeia_fp16_to_fp32(blk.d);
        const uint8_t* ql0 = blk.ql;
        const uint8_t* qh0 = blk.qh;
        const int8_t* sc0 = blk.scales;
#if defined(__AVX2__)
        __m256i acc_vec = _mm256_setzero_si256();
#elif defined(__ARM_NEON)
        int32_t acc_i = 0;
#else
        float acc = 0.0f;
#endif
        for (int n = 0; n < QK_K; n += 128) {
            const uint8_t* ql = ql0 + (n / 128) * 64;
            const uint8_t* qh = qh0 + (n / 128) * 32;
            const int8_t* sc = sc0 + (n / 128) * 8;
#if defined(__AVX2__)
            acc_vec = q6k_group128_avx2(ql, qh, sc, xq + out_base + n, acc_vec);
#elif defined(__ARM_NEON)
            acc_i += q6k_group128_neon(ql, qh, sc, xq + out_base + n);
#else
            int8_t w[4][32];
            for (int l = 0; l < 32; ++l) {
                w[0][l] = (int8_t) ((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                w[1][l] = (int8_t) ((ql[l + 32]  & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                w[2][l] = (int8_t) ((ql[l]       >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                w[3][l] = (int8_t) ((ql[l + 32]  >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            for (int q = 0; q < 4; ++q) {
                for (int half = 0; half < 2; ++half) {
                    const size_t out_pos = out_base + (size_t) n + (size_t) q * 32 + (size_t) half * 16;
                    const size_t sub = out_pos / 32;
                    const int8_t* wsub = w[q] + half * 16;
                    const int8_t* xsub = xq + out_pos;
                    const float scale = d * (float) sc[q * 2 + half] * xscale[sub];
                    int32_t idot = 0;
                    for (int t = 0; t < 16; ++t) idot += (int32_t) wsub[t] * (int32_t) xsub[t];
                    acc += scale * (float) idot;
                }
            }
#endif
        }
#if defined(__AVX2__)
        // A single conversion to float per super-block.
        const float acc = d * xscale[out_base / 32]
                        * hsum_ps_avx2(_mm256_cvtepi32_ps(acc_vec));
#elif defined(__ARM_NEON)
        const float acc = d * xscale[out_base / 32] * (float) acc_i;
#endif
        return acc;
    };

    auto compute_fn = [&](size_t r0, size_t r1) {
        size_t r = r0;
        for (; r + 1 < r1; r += 2) {
            const uint8_t* row_base0 = q6k_data + r * row_bytes;
            const uint8_t* row_base1 = q6k_data + (r + 1) * row_bytes;
#if defined(__AVX2__)
            __m256i acc0_vec = _mm256_setzero_si256();
            __m256i acc1_vec = _mm256_setzero_si256();
            float acc0f = 0.0f, acc1f = 0.0f;
#elif defined(__ARM_NEON)
            int32_t acc0_i = 0, acc1_i = 0;
            float acc0f = 0.0f, acc1f = 0.0f;
#else
            float acc0 = 0.0f, acc1 = 0.0f;
#endif
            for (size_t si = 0; si < n_super; ++si) {
                const block_q6_K& blk0 = *reinterpret_cast<const block_q6_K*>(row_base0 + si * sizeof(block_q6_K));
                const block_q6_K& blk1 = *reinterpret_cast<const block_q6_K*>(row_base1 + si * sizeof(block_q6_K));
                const float d0 = desireeia_fp16_to_fp32(blk0.d);
                const float d1v = desireeia_fp16_to_fp32(blk1.d);
                const uint8_t* ql0_0 = blk0.ql; const uint8_t* qh0_0 = blk0.qh; const int8_t* sc0_0 = blk0.scales;
                const uint8_t* ql0_1 = blk1.ql; const uint8_t* qh0_1 = blk1.qh; const int8_t* sc0_1 = blk1.scales;
                const size_t out_base = si * QK_K;
                for (int n = 0; n < QK_K; n += 128) {
                    const uint8_t* ql_0 = ql0_0 + (n / 128) * 64; const uint8_t* qh_0 = qh0_0 + (n / 128) * 32; const int8_t* sc_0 = sc0_0 + (n / 128) * 8;
                    const uint8_t* ql_1 = ql0_1 + (n / 128) * 64; const uint8_t* qh_1 = qh0_1 + (n / 128) * 32; const int8_t* sc_1 = sc0_1 + (n / 128) * 8;
#if defined(__AVX2__)
                    // The two rows share the same activations: the
                    // quantization cost and the x loads are amortized
                    // across both (this is why the 2-row tiling stays,
                    // while 4-row made things worse — see the tiling notes
                    // in docs/engine_gap_analysis.md).
                    const int8_t* x_grp = xq + out_base + n;
                    acc0_vec = q6k_group128_avx2(ql_0, qh_0, sc_0, x_grp, acc0_vec);
                    acc1_vec = q6k_group128_avx2(ql_1, qh_1, sc_1, x_grp, acc1_vec);
#elif defined(__ARM_NEON)
                    const int8_t* x_grp = xq + out_base + n;
                    acc0_i += q6k_group128_neon(ql_0, qh_0, sc_0, x_grp);
                    acc1_i += q6k_group128_neon(ql_1, qh_1, sc_1, x_grp);
#else
                    int8_t w0[4][32], w1[4][32];
                    for (int l = 0; l < 32; ++l) {
                        w0[0][l] = (int8_t) ((ql_0[l]      & 0x0F) | (((qh_0[l] >> 0) & 3) << 4)) - 32;
                        w0[1][l] = (int8_t) ((ql_0[l + 32]  & 0x0F) | (((qh_0[l] >> 2) & 3) << 4)) - 32;
                        w0[2][l] = (int8_t) ((ql_0[l]       >> 4)   | (((qh_0[l] >> 4) & 3) << 4)) - 32;
                        w0[3][l] = (int8_t) ((ql_0[l + 32]  >> 4)   | (((qh_0[l] >> 6) & 3) << 4)) - 32;
                        w1[0][l] = (int8_t) ((ql_1[l]      & 0x0F) | (((qh_1[l] >> 0) & 3) << 4)) - 32;
                        w1[1][l] = (int8_t) ((ql_1[l + 32]  & 0x0F) | (((qh_1[l] >> 2) & 3) << 4)) - 32;
                        w1[2][l] = (int8_t) ((ql_1[l]       >> 4)   | (((qh_1[l] >> 4) & 3) << 4)) - 32;
                        w1[3][l] = (int8_t) ((ql_1[l + 32]  >> 4)   | (((qh_1[l] >> 6) & 3) << 4)) - 32;
                    }
                    for (int q = 0; q < 4; ++q) {
                        for (int half = 0; half < 2; ++half) {
                            const size_t out_pos = out_base + (size_t) n + (size_t) q * 32 + (size_t) half * 16;
                            const size_t sub = out_pos / 32;
                            const int8_t* wsub0 = w0[q] + half * 16;
                            const int8_t* wsub1 = w1[q] + half * 16;
                            const int8_t* xsub = xq + out_pos;
                            const float xs = xscale[sub];
                            int32_t idot0 = 0, idot1 = 0;
                            for (int t = 0; t < 16; ++t) {
                                idot0 += (int32_t) wsub0[t] * (int32_t) xsub[t];
                                idot1 += (int32_t) wsub1[t] * (int32_t) xsub[t];
                            }
                            acc0 += d0  * (float) sc_0[q * 2 + half] * xs * (float) idot0;
                            acc1 += d1v * (float) sc_1[q * 2 + half] * xs * (float) idot1;
                        }
                    }
#endif
                }
#if defined(__AVX2__)
                // d changes at every super-block, so the integer
                // accumulator has to be flushed to float here and reset:
                // it's still ONE conversion every 256 weights instead of
                // one every 32.
                const float dx = xscale[out_base / 32];
                acc0f += d0  * dx * hsum_ps_avx2(_mm256_cvtepi32_ps(acc0_vec));
                acc1f += d1v * dx * hsum_ps_avx2(_mm256_cvtepi32_ps(acc1_vec));
                acc0_vec = _mm256_setzero_si256();
                acc1_vec = _mm256_setzero_si256();
#elif defined(__ARM_NEON)
                const float dx = xscale[out_base / 32];
                acc0f += d0  * dx * (float) acc0_i;
                acc1f += d1v * dx * (float) acc1_i;
                acc0_i = 0;
                acc1_i = 0;
#endif
            }
#if defined(__AVX2__) || defined(__ARM_NEON)
            const float acc0 = acc0f;
            const float acc1 = acc1f;
#endif
            y[r] = acc0;
            y[r + 1] = acc1;
        }
        for (; r < r1; ++r) {
            const uint8_t* row_base = q6k_data + r * row_bytes;
            float acc = 0.0f;
            for (size_t si = 0; si < n_super; ++si) {
                const block_q6_K& blk = *reinterpret_cast<const block_q6_K*>(row_base + si * sizeof(block_q6_K));
                acc += row_super_dot(blk, si * QK_K);
            }
            y[r] = acc;
        }
    };
    if (row_hi > row_lo) {
        // Slice mode: the caller already owns a parallel region.
        compute_fn(row_lo, row_hi);
        return DESIREEIA_OK;
    }
    {
        ScopedTimer t(profile_counters().ns_q6k_compute);
        parallel_rows(rows, compute_fn);
    }
    profile_counters().calls_q6k.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}
}

int matmul_q6_k(const uint8_t* q6k_data, size_t rows, size_t cols, const float* x, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    std::vector<int8_t> xq;
    std::vector<float> xscale;
    std::vector<int32_t> unused_xsum;
    {
        ScopedTimer t(profile_counters().ns_quantize_act);
        // Q8_K: the Q6_K kernel now accumulates in integers and assumes a
        // constant scale inside the super-block (see q6k_group128_avx2).
        quantize_act_q8k_rep(x, cols, xq, xscale, unused_xsum);
    }
    return matmul_q6_k_core(q6k_data, rows, cols, xq.data(), xscale.data(), y);
}

int matmul_q6_k_pq(const uint8_t* q6k_data, size_t rows, size_t cols,
                    const int8_t* xq, const float* xscale, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    return matmul_q6_k_core(q6k_data, rows, cols, xq, xscale, y);
}

bool fused_pq_supported(const FusedPqJob* jobs, size_t n_jobs) {
    if (jobs == nullptr || n_jobs < 2) return false;
    for (size_t i = 0; i < n_jobs; ++i) {
        const FusedPqJob& j = jobs[i];
        if (j.cols == 0 || j.cols % QK_K != 0 || j.rows == 0) return false;
        if (j.format != FusedPqFormat::Q4_K && j.format != FusedPqFormat::Q6_K) return false;
    }
    return true;
}

// Computes several independent matrices in ONE dispatch.
//
// Q, K and V all read the same activation and write to separate outputs, so
// nothing forces them to be three separate parallel regions — but that is what
// they were, and each region ends with every thread waiting for the slowest.
// Measured on this machine, that wait ranged from 244 ms to 3075 ms over the
// same 64-token run depending only on how the OS happened to schedule: with 20
// threads and 273 dispatches per token, one descheduled worker stalls the other
// nineteen, and the exposure is per dispatch. Merging the three into a single
// row space cuts the number of times we take that risk, and gives the dynamic
// chunker a bigger pool of rows to balance across.
//
// The rows of all jobs form one index space: global row g belongs to the job
// whose cumulative row count contains it.
int matmul_fused_pq(const FusedPqJob* jobs, size_t n_jobs) {
    if (!fused_pq_supported(jobs, n_jobs)) return DESIREEIA_ERR_NOT_SUPPORTED;

    size_t total_rows = 0;
    for (size_t i = 0; i < n_jobs; ++i) total_rows += jobs[i].rows;

    ScopedTimer t(profile_counters().ns_fused_compute);
    parallel_rows(total_rows, [&](size_t g0, size_t g1) {
        // Walk the jobs, computing the part of [g0,g1) that falls in each.
        size_t base = 0;
        for (size_t i = 0; i < n_jobs; ++i) {
            const FusedPqJob& j = jobs[i];
            const size_t lo = base;
            const size_t hi = base + j.rows;
            base = hi;
            if (g1 <= lo || g0 >= hi) continue;

            const size_t r0 = (g0 > lo ? g0 : lo) - lo;
            const size_t r1 = (g1 < hi ? g1 : hi) - lo;
            if (j.format == FusedPqFormat::Q4_K) {
                matmul_q4_k_core(j.data, j.rows, j.cols, j.xq, j.xscale, j.xsum, j.y, r0, r1);
            } else {
                matmul_q6_k_core(j.data, j.rows, j.cols, j.xq, j.xscale, j.y, r0, r1);
            }
        }
    });

    profile_counters().calls_fused.fetch_add(1, std::memory_order_relaxed);
    return DESIREEIA_OK;
}

}
