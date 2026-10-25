// Implementation of the interleaved int8 GEMM (see i8gemm.h), included by
// one translation unit per instruction set. Before including, define:
//   I8_NS     namespace of this copy (i8_base, i8_vnni, i8_vnni512)
//   I8_DOT    0 = AVX2 maddubs+madd, 1 = AVX-VNNI, 2 = AVX512-VNNI (256-bit),
//             3 = NEON sdot
//   I8_TARGET function attribute enabling the instruction set on GCC/Clang
//             (per function, never per file: a whole file built for
//             AVX-512 would leak AVX-512 copies of shared inline code)
//
// Products per 32-weight block, for one token and 8 rows:
//   x86: the activation is stored unsigned (q_x + 128) as the unsigned
//        operand of the dot, the weight as the signed one; the extra
//        128 * sum(q_w) is an exact integer subtracted once per block.
//        AVX2's maddubs saturates int16 pairs, which that shift would
//        overflow for Q8_0 weights (+-127): there, and only on AVX2, the
//        activation stays signed and the sign trick is used instead.
//   NEON: sdot is signed x signed, no shift needed.
// Then, per block and for the 8 rows at once: convert, multiply by
// scale_w(row) * scale_x, add; for Q4_K/Q5_K subtract m(row) * sum(x).

#include "i8gemm.h"
#include "engine.h"
#include "profile.h"
#include "../quant/quant.h"
#include <algorithm>
#include <cstring>
#include <vector>

#if I8_DOT == 3
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

namespace desireeia {
namespace I8_NS {
namespace {

constexpr size_t kRowsPerGroup = 8;
constexpr size_t kTileBytes = 256 * 1024;   // activations of one token tile (see run)


using Acts = I8QuantActs;

#if I8_DOT != 3
// One token's activations to int8 (one scale per 32), written the way the kernels read them. The same arithmetic as
// quantize_q8_0 step for step (largest magnitude, scale = max / 127, multiply by 1 / scale, round to nearest even,
// clamp to +-127), 32 values at a time, so the integers are identical. `shifted` stores q + 128 (the unsigned operand
// of the dot instruction). The exact float sums keep their sequential order: they are not vectorized on purpose.
I8_TARGET void quantize_row_vec(const float* xt, size_t cols, bool shifted, bool want_sums, uint8_t* dst, float* d, float* sums) {
    const size_t nb = cols / 32;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i hi = _mm256_set1_epi32(127), lo = _mm256_set1_epi32(-127);
    const __m256i fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i flip = _mm256_set1_epi8((char) 0x80);
    for (size_t b = 0; b < nb; ++b) {
        const float* p = xt + b * 32;
        const __m256 v0 = _mm256_loadu_ps(p), v1 = _mm256_loadu_ps(p + 8), v2 = _mm256_loadu_ps(p + 16), v3 = _mm256_loadu_ps(p + 24);
        __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(v0, absmask), _mm256_and_ps(v1, absmask)),
                                 _mm256_max_ps(_mm256_and_ps(v2, absmask), _mm256_and_ps(v3, absmask)));
        __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
        const float vmax = _mm_cvtss_f32(m4);
        const float dd = vmax > 0.0f ? vmax / 127.0f : 0.0f;
        const float id = dd > 0.0f ? 1.0f / dd : 0.0f;
        d[b] = dd;
        const __m256 idv = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, idv)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, idv));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, idv)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, idv));
        i0 = _mm256_max_epi32(_mm256_min_epi32(i0, hi), lo); i1 = _mm256_max_epi32(_mm256_min_epi32(i1, hi), lo);
        i2 = _mm256_max_epi32(_mm256_min_epi32(i2, hi), lo); i3 = _mm256_max_epi32(_mm256_min_epi32(i3, hi), lo);
        __m256i q = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        q = _mm256_permutevar8x32_epi32(q, fix);
        if (shifted) q = _mm256_xor_si256(q, flip);                 // q + 128 as an unsigned byte
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + b * 32), q);
        if (want_sums) {
            float sum = 0.0f;
            for (int i = 0; i < 32; ++i) sum += p[i];
            sums[b] = sum;
        }
    }
}
#endif

void quantize_acts(const I8GemmJob& j, bool shifted, Acts& a, bool want_sums) {
    const size_t cols = j.cols, nb = cols / 32;
    a.shifted = shifted;
    a.has_sums = want_sums;
    a.q.resize(j.n_tok * cols);
    a.d.resize(j.n_tok * nb);
    if (want_sums) a.s.resize(j.n_tok * nb);
    parallel_units(j.n_tok, [&](size_t t0, size_t t1) {
#if I8_DOT == 3
        std::vector<int8_t> q;
        std::vector<float> d;
        std::vector<uint8_t> unused;
#endif
        for (size_t t = t0; t < t1; ++t) {
            const float* xt = j.x + t * cols;
#if I8_DOT != 3
            quantize_row_vec(xt, cols, shifted, want_sums, a.q.data() + t * cols, a.d.data() + t * nb,
                             want_sums ? a.s.data() + t * nb : nullptr);
#else
            quantize_q8_0(xt, cols, q, d, unused);
            uint8_t* dst = a.q.data() + t * cols;
            if (shifted) {
                for (size_t i = 0; i < cols; ++i) dst[i] = (uint8_t) (q[i] + 128);
            } else {
                std::memcpy(dst, q.data(), cols);
            }
            std::memcpy(a.d.data() + t * nb, d.data(), nb * sizeof(float));
            if (want_sums) {
                for (size_t b = 0; b < nb; ++b) {
                    float s = 0.0f;
                    for (int i = 0; i < 32; ++i) s += xt[b * 32 + i];
                    a.s[t * nb + b] = s;
                }
            }
#endif
        }
    });
}

// One slice of rows, decoded and interleaved by the thread that computes it.
struct Slice {
    std::vector<int8_t> rq;     // [n][cols]           row-major decode
    std::vector<float> ra, rm;  // [n][2*nb]
    std::vector<int8_t> gq;     // [G][nb][8 chunks][8 rows][4]
    std::vector<float> ga;      // [G][nb][2][8]       scale per half, 8 rows
    std::vector<float> gm;      // [G][nb][8]          offset (has_m)
    std::vector<int32_t> gc;    // [G][nb][2][8]       MINUS 128 * sum(q) per half (shifted mode): the accumulators start from it
};

#if I8_DOT != 3
// 8 rows x 32 weights -> [chunk][row][4 weights]: an 8x8 transpose of 32-bit words (the scalar version did 64
// four-byte copies per block). The sums of the two 16-weight halves, for the shift correction, come out of the
// same registers: a multiply-add against ones adds four signed bytes per lane.
I8_TARGET inline void interleave_block(const int8_t* const* src, int8_t* dq, bool want_sums, int32_t* lo, int32_t* hi) {
    __m256i v[8];
    for (int r = 0; r < 8; ++r) v[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src[r]));
    const __m256i t0 = _mm256_unpacklo_epi32(v[0], v[1]), t1 = _mm256_unpackhi_epi32(v[0], v[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(v[2], v[3]), t3 = _mm256_unpackhi_epi32(v[2], v[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(v[4], v[5]), t5 = _mm256_unpackhi_epi32(v[4], v[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(v[6], v[7]), t7 = _mm256_unpackhi_epi32(v[6], v[7]);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    __m256i* out = reinterpret_cast<__m256i*>(dq);
    _mm256_storeu_si256(out + 0, _mm256_permute2x128_si256(u0, u4, 0x20));
    _mm256_storeu_si256(out + 1, _mm256_permute2x128_si256(u1, u5, 0x20));
    _mm256_storeu_si256(out + 2, _mm256_permute2x128_si256(u2, u6, 0x20));
    _mm256_storeu_si256(out + 3, _mm256_permute2x128_si256(u3, u7, 0x20));
    _mm256_storeu_si256(out + 4, _mm256_permute2x128_si256(u0, u4, 0x31));
    _mm256_storeu_si256(out + 5, _mm256_permute2x128_si256(u1, u5, 0x31));
    _mm256_storeu_si256(out + 6, _mm256_permute2x128_si256(u2, u6, 0x31));
    _mm256_storeu_si256(out + 7, _mm256_permute2x128_si256(u3, u7, 0x31));
    if (want_sums) {
        const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
        for (int r = 0; r < 8; ++r) {
            const __m256i four = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, v[r]), ones16);   // 8 sums of 4 bytes
            const __m256i two = _mm256_hadd_epi32(four, four);                                   // lanes 0,1 | 4,5 hold pairs
            const __m256i one = _mm256_hadd_epi32(two, two);                                     // lane 0 = bytes 0-15, lane 4 = 16-31
            lo[r] = _mm_cvtsi128_si32(_mm256_castsi256_si128(one));
            hi[r] = _mm_cvtsi128_si32(_mm256_extracti128_si256(one, 1));
        }
    }
}
#endif

I8_TARGET void build_slice(const I8GemmJob& j, size_t r0, size_t n, bool shifted, Slice& s) {
    const size_t cols = j.cols, nb = cols / 32;
    const size_t G = (n + kRowsPerGroup - 1) / kRowsPerGroup;
    // Rows [0, n) are fully overwritten by the decode below; only the padding rows of the last group must be zero.
    s.rq.resize(G * kRowsPerGroup * cols);
    s.ra.resize(G * kRowsPerGroup * nb * 2);
    if (j.has_m) s.rm.resize(G * kRowsPerGroup * nb * 2);
    if (n < G * kRowsPerGroup) {
        std::memset(s.rq.data() + n * cols, 0, (G * kRowsPerGroup - n) * cols);
        std::fill(s.ra.begin() + (std::ptrdiff_t) (n * nb * 2), s.ra.end(), 0.0f);
        if (j.has_m) std::fill(s.rm.begin() + (std::ptrdiff_t) (n * nb * 2), s.rm.end(), 0.0f);
    }
    for (size_t i = 0; i < n; ++i) {
        j.unpack(j.src, j.data + (r0 + i) * j.row_bytes, cols, s.rq.data() + i * cols,
                 s.ra.data() + i * nb * 2, j.has_m ? s.rm.data() + i * nb * 2 : nullptr);
    }
    s.gq.resize(G * nb * 256);
    s.ga.resize(G * nb * 16);
    if (j.has_m) s.gm.resize(G * nb * 8);
    if (shifted) s.gc.resize(G * nb * 16);
    for (size_t g = 0; g < G; ++g) {
        for (size_t b = 0; b < nb; ++b) {
            int8_t* dq = s.gq.data() + (g * nb + b) * 256;
            float* da = s.ga.data() + (g * nb + b) * 16;
#if I8_DOT != 3
            const int8_t* rows[8];
            for (size_t rr = 0; rr < kRowsPerGroup; ++rr) rows[rr] = s.rq.data() + (g * kRowsPerGroup + rr) * cols + b * 32;
            int32_t lo[8], hi[8];
            interleave_block(rows, dq, shifted, lo, hi);
            for (size_t rr = 0; rr < kRowsPerGroup; ++rr) {
                const size_t row = g * kRowsPerGroup + rr;
                da[rr] = s.ra[row * nb * 2 + 2 * b];
                da[8 + rr] = s.ra[row * nb * 2 + 2 * b + 1];
                if (j.has_m) s.gm[(g * nb + b) * 8 + rr] = s.rm[row * nb * 2 + 2 * b];
                if (shifted) {
                    int32_t* dc = s.gc.data() + (g * nb + b) * 16;
                    // Halves apart only where their scales differ; otherwise the
                    // whole block's correction sits in the first slot.
                    dc[rr] = -128 * (j.split ? lo[rr] : lo[rr] + hi[rr]);
                    dc[8 + rr] = -128 * hi[rr];
                }
            }
#else
            for (size_t rr = 0; rr < kRowsPerGroup; ++rr) {
                const size_t row = g * kRowsPerGroup + rr;
                const int8_t* src = s.rq.data() + row * cols + b * 32;
                for (size_t c = 0; c < 8; ++c) std::memcpy(dq + c * 32 + rr * 4, src + c * 4, 4);
                da[rr] = s.ra[row * nb * 2 + 2 * b];
                da[8 + rr] = s.ra[row * nb * 2 + 2 * b + 1];
                if (j.has_m) s.gm[(g * nb + b) * 8 + rr] = s.rm[row * nb * 2 + 2 * b];
                if (shifted) {
                    int32_t lo = 0, hi = 0;
                    for (int k = 0; k < 16; ++k) { lo += src[k]; hi += src[16 + k]; }
                    int32_t* dc = s.gc.data() + (g * nb + b) * 16;
                    dc[rr] = -128 * (j.split ? lo : lo + hi);
                    dc[8 + rr] = -128 * hi;
                }
            }
#endif
        }
    }
}

inline void store_rows(float* y, size_t rows, size_t t, size_t r, size_t valid, const float* v) {
    float* dst = y + t * rows + r;
    for (size_t i = 0; i < valid; ++i) dst[i] = v[i];
}

#if I8_DOT == 3
// ── NEON (sdot): 8 rows = two int32x4 lanes-of-rows ──────────────────────
template <int NT>
I8_TARGET void kernel_group(const I8GemmJob& j, const Acts& a, const Slice& s, size_t g, size_t t, size_t r, size_t valid) {
    const size_t cols = j.cols, nb = cols / 32;
    float32x4_t fl[NT], fh[NT];
    for (int k = 0; k < NT; ++k) { fl[k] = vdupq_n_f32(0.0f); fh[k] = vdupq_n_f32(0.0f); }
    for (size_t b = 0; b < nb; ++b) {
        const int8_t* wq = s.gq.data() + (g * nb + b) * 256;
        const float* ga = s.ga.data() + (g * nb + b) * 16;
        int8x16_t x0[NT], x1[NT];
        for (int k = 0; k < NT; ++k) {
            const int8_t* xb = reinterpret_cast<const int8_t*>(a.q.data()) + (t + k) * cols + b * 32;
            x0[k] = vld1q_s8(xb);
            x1[k] = vld1q_s8(xb + 16);
        }
        for (int h = 0; h < (j.split ? 2 : 1); ++h) {
            int32x4_t il[NT], ih[NT];
            for (int k = 0; k < NT; ++k) { il[k] = vdupq_n_s32(0); ih[k] = vdupq_n_s32(0); }
            const int c0 = j.split ? 4 * h : 0, c1 = j.split ? c0 + 4 : 8;
            for (int c = c0; c < c1; ++c) {
                const int8x16_t wl = vld1q_s8(wq + c * 32);
                const int8x16_t wh = vld1q_s8(wq + c * 32 + 16);
                for (int k = 0; k < NT; ++k) {
                    const int8x16_t xv = c < 4 ? x0[k] : x1[k];
                    // Broadcast lane c of the activation (4 bytes) and one plain
                    // sdot per 4 rows: vdotq_laneq_s32 would need armv8.2 in GCC's
                    // headers even where +dotprod is enabled.
                    int32x4_t lane;
                    switch (c & 3) {
                        case 0: lane = vdupq_laneq_s32(vreinterpretq_s32_s8(xv), 0); break;
                        case 1: lane = vdupq_laneq_s32(vreinterpretq_s32_s8(xv), 1); break;
                        case 2: lane = vdupq_laneq_s32(vreinterpretq_s32_s8(xv), 2); break;
                        default: lane = vdupq_laneq_s32(vreinterpretq_s32_s8(xv), 3); break;
                    }
                    const int8x16_t xb4 = vreinterpretq_s8_s32(lane);
                    il[k] = vdotq_s32(il[k], wl, xb4);
                    ih[k] = vdotq_s32(ih[k], wh, xb4);
                }
            }
            const float32x4_t al = vld1q_f32(ga + 8 * h), ah = vld1q_f32(ga + 8 * h + 4);
            for (int k = 0; k < NT; ++k) {
                const float dx = a.d[(t + k) * nb + b];
                fl[k] = vfmaq_f32(fl[k], vcvtq_f32_s32(il[k]), vmulq_n_f32(al, dx));
                fh[k] = vfmaq_f32(fh[k], vcvtq_f32_s32(ih[k]), vmulq_n_f32(ah, dx));
            }
        }
        if (j.has_m) {
            const float32x4_t ml = vld1q_f32(s.gm.data() + (g * nb + b) * 8);
            const float32x4_t mh = vld1q_f32(s.gm.data() + (g * nb + b) * 8 + 4);
            for (int k = 0; k < NT; ++k) {
                const float sx = a.s[(t + k) * nb + b];
                fl[k] = vmlsq_n_f32(fl[k], ml, sx);
                fh[k] = vmlsq_n_f32(fh[k], mh, sx);
            }
        }
    }
    for (int k = 0; k < NT; ++k) {
        float out[8];
        vst1q_f32(out, fl[k]);
        vst1q_f32(out + 4, fh[k]);
        store_rows(j.y, j.rows, t + k, r, valid, out);
    }
}
#else
// ── x86: one __m256 = the same output for 8 rows ─────────────────────────
I8_TARGET inline __m256i dot_acc(__m256i acc, __m256i u, __m256i s) {
#if I8_DOT == 1
    return _mm256_dpbusd_avx_epi32(acc, u, s);
#elif I8_DOT == 2
    return _mm256_dpbusd_epi32(acc, u, s);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
#endif
}

I8_TARGET inline __m256i bcast4(const uint8_t* p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return _mm256_set1_epi32(v);
}

// The 4-token case written out with named registers and a fully unrolled
// chunk loop: arrays of vector accumulators indexed in loops were kept in
// memory by the compiler (a load and a store around every instruction),
// and a run-time half count stopped the unrolling.
template <bool Shifted, bool Split, bool HasM>
I8_TARGET void kernel4(const I8GemmJob& j, const Acts& a, const Slice& s, size_t g, size_t t, size_t r, size_t valid) {
    const size_t cols = j.cols, nb = cols / 32;
    __m256 f0 = _mm256_setzero_ps(), f1 = f0, f2 = f0, f3 = f0;
    const uint8_t* x0 = a.q.data() + t * cols;
    const uint8_t* x1 = x0 + cols;
    const uint8_t* x2 = x1 + cols;
    const uint8_t* x3 = x2 + cols;
    const float* d0 = a.d.data() + t * nb;
    const float* sx0 = HasM ? a.s.data() + t * nb : nullptr;
    const uint8_t* wq = reinterpret_cast<const uint8_t*>(s.gq.data()) + g * nb * 256;
    const float* ga = s.ga.data() + g * nb * 16;
    const int32_t* gc = Shifted ? s.gc.data() + g * nb * 16 : nullptr;
    const float* gm = HasM ? s.gm.data() + g * nb * 8 : nullptr;
    for (size_t b = 0; b < nb; ++b, wq += 256, ga += 16, x0 += 32, x1 += 32, x2 += 32, x3 += 32) {
        // The shift correction is the starting value of the integer sums (it was a subtraction per
        // output after them).
        __m256i i0 = Shifted ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gc + b * 16)) : _mm256_setzero_si256();
        __m256i i1 = i0, i2 = i0, i3 = i0;
        // A dot instruction hands its result on after ~5 cycles and two can start per
        // cycle, so four accumulators updated chunk after chunk left the ports idle
        // (the chain of 8 dependent dots was the limit, not the port count). Odd
        // chunks go to a second set, added back before the float step. Same integers.
        __m256i k0 = _mm256_setzero_si256(), k1 = k0, k2 = k0, k3 = k0;
#define I8_CHUNK(c)                                                                                  \
        {                                                                                            \
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wq + (c) * 32));  \
            if (Shifted) {                                                                           \
                if (!Split && ((c) & 1)) {                                                                       \
                    k0 = dot_acc(k0, bcast4(x0 + (c) * 4), w); k1 = dot_acc(k1, bcast4(x1 + (c) * 4), w); \
                    k2 = dot_acc(k2, bcast4(x2 + (c) * 4), w); k3 = dot_acc(k3, bcast4(x3 + (c) * 4), w); \
                } else {                                                                             \
                    i0 = dot_acc(i0, bcast4(x0 + (c) * 4), w); i1 = dot_acc(i1, bcast4(x1 + (c) * 4), w); \
                    i2 = dot_acc(i2, bcast4(x2 + (c) * 4), w); i3 = dot_acc(i3, bcast4(x3 + (c) * 4), w); \
                }                                                                                    \
            } else {                                                                                 \
                const __m256i aw = _mm256_sign_epi8(w, w);                                           \
                if (!Split && ((c) & 1)) {                                                                       \
                    k0 = dot_acc(k0, aw, _mm256_sign_epi8(bcast4(x0 + (c) * 4), w));                 \
                    k1 = dot_acc(k1, aw, _mm256_sign_epi8(bcast4(x1 + (c) * 4), w));                 \
                    k2 = dot_acc(k2, aw, _mm256_sign_epi8(bcast4(x2 + (c) * 4), w));                 \
                    k3 = dot_acc(k3, aw, _mm256_sign_epi8(bcast4(x3 + (c) * 4), w));                 \
                } else {                                                                             \
                    i0 = dot_acc(i0, aw, _mm256_sign_epi8(bcast4(x0 + (c) * 4), w));                 \
                    i1 = dot_acc(i1, aw, _mm256_sign_epi8(bcast4(x1 + (c) * 4), w));                 \
                    i2 = dot_acc(i2, aw, _mm256_sign_epi8(bcast4(x2 + (c) * 4), w));                 \
                    i3 = dot_acc(i3, aw, _mm256_sign_epi8(bcast4(x3 + (c) * 4), w));                 \
                }                                                                                    \
            }                                                                                        \
        }
#define I8_FLOAT(h)                                                                                  \
        {                                                                                            \
            const __m256 sa = _mm256_loadu_ps(ga + 8 * (h));                                         \
            if (!Split) { i0 = _mm256_add_epi32(i0, k0); i1 = _mm256_add_epi32(i1, k1);                            \
            i2 = _mm256_add_epi32(i2, k2); i3 = _mm256_add_epi32(i3, k3); }                          \
            f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + b)), f0);          \
            f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + nb + b)), f1);     \
            f2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i2), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + 2 * nb + b)), f2); \
            f3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i3), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + 3 * nb + b)), f3); \
        }
        if (Split) {
            I8_CHUNK(0) I8_CHUNK(1) I8_CHUNK(2) I8_CHUNK(3)
            I8_FLOAT(0)
            i0 = Shifted ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gc + b * 16 + 8)) : _mm256_setzero_si256();
            i1 = i0; i2 = i0; i3 = i0;
            k0 = _mm256_setzero_si256(); k1 = k0; k2 = k0; k3 = k0;
            I8_CHUNK(4) I8_CHUNK(5) I8_CHUNK(6) I8_CHUNK(7)
            I8_FLOAT(1)
        } else {
            I8_CHUNK(0) I8_CHUNK(1) I8_CHUNK(2) I8_CHUNK(3)
            I8_CHUNK(4) I8_CHUNK(5) I8_CHUNK(6) I8_CHUNK(7)
            I8_FLOAT(0)
        }
#undef I8_CHUNK
#undef I8_FLOAT
        if (HasM) {
            const __m256 m = _mm256_loadu_ps(gm + b * 8);
            f0 = _mm256_fnmadd_ps(m, _mm256_broadcast_ss(sx0 + b), f0);
            f1 = _mm256_fnmadd_ps(m, _mm256_broadcast_ss(sx0 + nb + b), f1);
            f2 = _mm256_fnmadd_ps(m, _mm256_broadcast_ss(sx0 + 2 * nb + b), f2);
            f3 = _mm256_fnmadd_ps(m, _mm256_broadcast_ss(sx0 + 3 * nb + b), f3);
        }
    }
    const __m256 fs[4] = { f0, f1, f2, f3 };
    for (int k = 0; k < 4; ++k) {
        if (valid == kRowsPerGroup) {
            _mm256_storeu_ps(j.y + (t + k) * j.rows + r, fs[k]);
        } else {
            float out[8];
            _mm256_storeu_ps(out, fs[k]);
            store_rows(j.y, j.rows, t + k, r, valid, out);
        }
    }
}

template <int NT, bool Shifted>
I8_TARGET void kernel_group(const I8GemmJob& j, const Acts& a, const Slice& s, size_t g, size_t t, size_t r, size_t valid) {
    const size_t cols = j.cols, nb = cols / 32;
    __m256 f[NT];
    for (int k = 0; k < NT; ++k) f[k] = _mm256_setzero_ps();
    const int halves = j.split ? 2 : 1;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* wq = reinterpret_cast<const uint8_t*>(s.gq.data()) + (g * nb + b) * 256;
        const float* ga = s.ga.data() + (g * nb + b) * 16;
        const uint8_t* xb[NT];
        for (int k = 0; k < NT; ++k) xb[k] = a.q.data() + (t + k) * cols + b * 32;
        for (int h = 0; h < halves; ++h) {
            __m256i acc[NT];
            const __m256i start = Shifted
                ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s.gc.data() + (g * nb + b) * 16 + 8 * h))
                : _mm256_setzero_si256();
            for (int k = 0; k < NT; ++k) acc[k] = start;
            const int c0 = halves == 2 ? 4 * h : 0, c1 = halves == 2 ? c0 + 4 : 8;
            for (int c = c0; c < c1; ++c) {
                const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wq + c * 32));
                if (Shifted) {
                    for (int k = 0; k < NT; ++k) acc[k] = dot_acc(acc[k], bcast4(xb[k] + c * 4), w);
                } else {
                    const __m256i aw = _mm256_sign_epi8(w, w);
                    for (int k = 0; k < NT; ++k)
                        acc[k] = dot_acc(acc[k], aw, _mm256_sign_epi8(bcast4(xb[k] + c * 4), w));
                }
            }
            const __m256 sa = _mm256_loadu_ps(ga + 8 * h);
            for (int k = 0; k < NT; ++k) {
                const __m256 fi = _mm256_cvtepi32_ps(acc[k]);
                f[k] = _mm256_fmadd_ps(fi, _mm256_mul_ps(sa, _mm256_set1_ps(a.d[(t + k) * nb + b])), f[k]);
            }
        }
        if (j.has_m) {
            const __m256 m = _mm256_loadu_ps(s.gm.data() + (g * nb + b) * 8);
            for (int k = 0; k < NT; ++k) f[k] = _mm256_fnmadd_ps(m, _mm256_set1_ps(a.s[(t + k) * nb + b]), f[k]);
        }
    }
    for (int k = 0; k < NT; ++k) {
        if (valid == kRowsPerGroup) {
            _mm256_storeu_ps(j.y + (t + k) * j.rows + r, f[k]);
        } else {
            float out[8];
            _mm256_storeu_ps(out, f[k]);
            store_rows(j.y, j.rows, t + k, r, valid, out);
        }
    }
}
#endif

} // namespace

void run(const I8GemmJob& j) {
#if I8_DOT == 3
    const bool shifted = false;
#elif I8_DOT == 0
    const bool shifted = !j.wide;                // AVX2: Q8_0 would overflow maddubs
#else
    const bool shifted = true;
#endif
    Acts local_acts;
    const Acts* acts_ptr = &local_acts;
    {
        ScopedTimer tq(profile_counters().ns_quantize_act);
        I8ActShare* sh = i8_share_acts_current();
        if (sh && sh->x == j.x && sh->cols == j.cols && sh->n_tok == j.n_tok) {
            // Shared with the other products of this layer that read the same input:
            // the sums are always kept (one of them may need them).
            if (!sh->filled || sh->acts.shifted != shifted) {
                quantize_acts(j, shifted, sh->acts, true);
                sh->filled = true;
            }
            acts_ptr = &sh->acts;
        } else {
            quantize_acts(j, shifted, local_acts, j.has_m);
        }
    }
    const Acts& acts = *acts_ptr;
    // Large slices: every slice streams ALL tokens' activations once, so with
    // the thread pool's small default slices (16 rows) the activations were
    // re-read from L3 ~160 times per product and the extra cores added
    // little. About four slices per worker, whole groups, still dynamic.
    const size_t workers = std::max<size_t>(1, ThreadPool::global().worker_count());
    constexpr size_t per_worker = 4;   // measured: 4 beats 2 by 3-6% on Q4_K (hybrid cores balance better), the same on Q8_0
    size_t chunk = (j.rows + per_worker * workers - 1) / (per_worker * workers);
    chunk = std::max<size_t>(16, (chunk + 15) / 16 * 16);
    const size_t n_chunks = (j.rows + chunk - 1) / chunk;
    parallel_units(n_chunks, [&](size_t c0, size_t c1) {
      for (size_t ci = c0; ci < c1; ++ci) {
        const size_t r0 = ci * chunk, r1 = std::min(j.rows, r0 + chunk);
        thread_local Slice s;
        const size_t n = r1 - r0;
        build_slice(j, r0, n, shifted, s);
        const size_t G = (n + kRowsPerGroup - 1) / kRowsPerGroup;
        // Tokens outer, groups inner: 4 tokens' activations stay in L1 while
        // the slice's interleaved weights stream from L1/L2.
        // Tokens are taken in tiles whose activations fit the L2 (about 256 KB):
        // inside a tile the groups are the OUTER loop, so one group's weights
        // (8 rows x cols bytes) come into L1 once and serve every token of the
        // tile, instead of the whole slice streaming from L2 at every 4 tokens.
        const size_t tile = std::max<size_t>(4, (kTileBytes / std::max<size_t>(1, j.cols)) / 4 * 4);
        size_t t = 0;
        for (size_t t0 = 0; t0 + 4 <= j.n_tok; t0 += tile) {
          const size_t t1 = std::min(j.n_tok / 4 * 4, t0 + tile);
          for (size_t g = 0; g < G; ++g) {
            const size_t valid = std::min(kRowsPerGroup, n - g * kRowsPerGroup);
            for (t = t0; t < t1; t += 4) {
#if I8_DOT == 3
                kernel_group<4>(j, acts, s, g, t, r0 + g * kRowsPerGroup, valid);
#else
                const size_t rg = r0 + g * kRowsPerGroup;
                if (shifted) {
                    if (j.split)      kernel4<true, true, false>(j, acts, s, g, t, rg, valid);
                    else if (j.has_m) kernel4<true, false, true>(j, acts, s, g, t, rg, valid);
                    else              kernel4<true, false, false>(j, acts, s, g, t, rg, valid);
                } else {
                    if (j.split)      kernel4<false, true, false>(j, acts, s, g, t, rg, valid);
                    else if (j.has_m) kernel4<false, false, true>(j, acts, s, g, t, rg, valid);
                    else              kernel4<false, false, false>(j, acts, s, g, t, rg, valid);
                }
#endif
            }
          }
        }
        t = j.n_tok / 4 * 4;
        for (; t < j.n_tok; ++t) {
            for (size_t g = 0; g < G; ++g) {
                const size_t valid = std::min(kRowsPerGroup, n - g * kRowsPerGroup);
#if I8_DOT == 3
                kernel_group<1>(j, acts, s, g, t, r0 + g * kRowsPerGroup, valid);
#else
                if (shifted) kernel_group<1, true>(j, acts, s, g, t, r0 + g * kRowsPerGroup, valid);
                else         kernel_group<1, false>(j, acts, s, g, t, r0 + g * kRowsPerGroup, valid);
#endif
            }
        }
      }
    });
}

} // namespace I8_NS
} // namespace desireeia
