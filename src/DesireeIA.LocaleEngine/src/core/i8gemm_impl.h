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

struct Acts {
    std::vector<uint8_t> q;    // [n_tok][cols]   int8, or int8 + 128 when `shifted`
    std::vector<float> d;      // [n_tok][nb]     scale of each 32
    std::vector<float> s;      // [n_tok][nb]     exact float sum of each 32 (has_m only)
    bool shifted = false;
};

void quantize_acts(const I8GemmJob& j, bool shifted, Acts& a) {
    const size_t cols = j.cols, nb = cols / 32;
    a.shifted = shifted;
    a.q.resize(j.n_tok * cols);
    a.d.resize(j.n_tok * nb);
    if (j.has_m) a.s.resize(j.n_tok * nb);
    parallel_units(j.n_tok, [&](size_t t0, size_t t1) {
        std::vector<int8_t> q;
        std::vector<float> d;
        std::vector<uint8_t> unused;
        for (size_t t = t0; t < t1; ++t) {
            const float* xt = j.x + t * cols;
            quantize_q8_0(xt, cols, q, d, unused);
            uint8_t* dst = a.q.data() + t * cols;
            if (shifted) {
                for (size_t i = 0; i < cols; ++i) dst[i] = (uint8_t) (q[i] + 128);
            } else {
                std::memcpy(dst, q.data(), cols);
            }
            std::memcpy(a.d.data() + t * nb, d.data(), nb * sizeof(float));
            if (j.has_m) {
                for (size_t b = 0; b < nb; ++b) {
                    float s = 0.0f;
                    for (int i = 0; i < 32; ++i) s += xt[b * 32 + i];
                    a.s[t * nb + b] = s;
                }
            }
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
    std::vector<int32_t> gc;    // [G][nb][2][8]       128 * sum(q) per half (shifted mode)
};

void build_slice(const I8GemmJob& j, size_t r0, size_t n, bool shifted, Slice& s) {
    const size_t cols = j.cols, nb = cols / 32;
    const size_t G = (n + kRowsPerGroup - 1) / kRowsPerGroup;
    s.rq.assign(G * kRowsPerGroup * cols, 0);
    s.ra.assign(G * kRowsPerGroup * nb * 2, 0.0f);
    if (j.has_m) s.rm.assign(G * kRowsPerGroup * nb * 2, 0.0f);
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
                    // Halves apart only where their scales differ; otherwise the
                    // whole block's correction sits in the first slot.
                    dc[rr] = 128 * (j.split ? lo : lo + hi);
                    dc[8 + rr] = 128 * hi;
                }
            }
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
        __m256i i0 = _mm256_setzero_si256(), i1 = i0, i2 = i0, i3 = i0;
#define I8_CHUNK(c)                                                                                  \
        {                                                                                            \
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wq + (c) * 32));  \
            if (Shifted) {                                                                           \
                i0 = dot_acc(i0, bcast4(x0 + (c) * 4), w); i1 = dot_acc(i1, bcast4(x1 + (c) * 4), w);\
                i2 = dot_acc(i2, bcast4(x2 + (c) * 4), w); i3 = dot_acc(i3, bcast4(x3 + (c) * 4), w);\
            } else {                                                                                 \
                const __m256i aw = _mm256_sign_epi8(w, w);                                           \
                i0 = dot_acc(i0, aw, _mm256_sign_epi8(bcast4(x0 + (c) * 4), w));                     \
                i1 = dot_acc(i1, aw, _mm256_sign_epi8(bcast4(x1 + (c) * 4), w));                     \
                i2 = dot_acc(i2, aw, _mm256_sign_epi8(bcast4(x2 + (c) * 4), w));                     \
                i3 = dot_acc(i3, aw, _mm256_sign_epi8(bcast4(x3 + (c) * 4), w));                     \
            }                                                                                        \
        }
#define I8_FLOAT(h)                                                                                  \
        {                                                                                            \
            const __m256 sa = _mm256_loadu_ps(ga + 8 * (h));                                         \
            if (Shifted) {                                                                           \
                const __m256i corr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(gc + b * 16 + 8 * (h))); \
                i0 = _mm256_sub_epi32(i0, corr); i1 = _mm256_sub_epi32(i1, corr);                    \
                i2 = _mm256_sub_epi32(i2, corr); i3 = _mm256_sub_epi32(i3, corr);                    \
            }                                                                                        \
            f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + b)), f0);          \
            f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + nb + b)), f1);     \
            f2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i2), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + 2 * nb + b)), f2); \
            f3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i3), _mm256_mul_ps(sa, _mm256_broadcast_ss(d0 + 3 * nb + b)), f3); \
        }
        if (Split) {
            I8_CHUNK(0) I8_CHUNK(1) I8_CHUNK(2) I8_CHUNK(3)
            I8_FLOAT(0)
            i0 = _mm256_setzero_si256(); i1 = i0; i2 = i0; i3 = i0;
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
            for (int k = 0; k < NT; ++k) acc[k] = _mm256_setzero_si256();
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
            __m256i corr = _mm256_setzero_si256();
            if (Shifted) corr = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s.gc.data() + (g * nb + b) * 16 + 8 * h));
            for (int k = 0; k < NT; ++k) {
                const __m256 fi = _mm256_cvtepi32_ps(Shifted ? _mm256_sub_epi32(acc[k], corr) : acc[k]);
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
    Acts acts;
    {
        ScopedTimer tq(profile_counters().ns_quantize_act);
        quantize_acts(j, shifted, acts);
    }
    // Large slices: every slice streams ALL tokens' activations once, so with
    // the thread pool's small default slices (16 rows) the activations were
    // re-read from L3 ~160 times per product and the extra cores added
    // little. About two slices per worker, whole groups, still dynamic.
    const size_t workers = std::max<size_t>(1, ThreadPool::global().worker_count());
    size_t chunk = (j.rows + 2 * workers - 1) / (2 * workers);
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
        size_t t = 0;
        for (; t + 4 <= j.n_tok; t += 4) {
            for (size_t g = 0; g < G; ++g) {
                const size_t valid = std::min(kRowsPerGroup, n - g * kRowsPerGroup);
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
