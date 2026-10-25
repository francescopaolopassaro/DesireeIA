// Prefill attention on the CPU, for a Q8_0 KV cache.
//
// The loop it replaces worked per (token, query head): for every key it
// dequantized the K row and the V row again - 8 times over for a model whose
// 8 query heads share one KV head - with a scalar exp per score. On a 2k
// prompt that made attention the slow half of the CPU prefill, and it grows
// with the square of the prompt.
//
// Here one unit of work is (KV head, block of query tokens): its rows are
// every (token, head) pair of that KV head, so each K/V tile is dequantized
// once for all of them.
//   * keys in tiles of kKeyTile: K dequantized TRANSPOSED ([dim][key]) so the
//     scores of one row against 8 keys come out of one vector FMA per head
//     dimension, with no horizontal sums; V dequantized as [key][dim];
//   * online softmax per row (running max and sum; the output is rescaled
//     only when the max grows);
//   * exp vectorized (range reduction to 2^n plus a polynomial);
//   * causal mask and sliding window per row.
// Float throughout (the cache's Q8_0 values are exact in float): same result
// as the per-token loop up to summation order.
//
// Two rows at a time (the common case: a key tile entirely before both rows'
// positions and inside their window). The scores of two rows share every load
// of the transposed keys, and the weighted sum of two rows shares every load
// of the values while the output stays in registers across the 32 keys of the
// tile, where the one-row loop loaded and stored it for every key. The order of
// the additions is the one-row loop's, so the results are identical to it.
#include "engine.h"
#include "../quant/quant.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace desireeia {
namespace {

constexpr uint32_t kKeyTile = 32;

// e^x for x <= 0 (softmax arguments), relative error ~2e-7.
inline float exp_neg(float x) {
    if (x < -87.0f) return 0.0f;
    const float n = std::floor(x * 1.44269504f + 0.5f);
    const float r = x - n * 0.693145751953125f - n * 1.428606765330187e-06f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * r * r + r + 1.0f;
    int32_t bits;
    std::memcpy(&bits, &p, 4);
    bits += (int32_t) n << 23;
    std::memcpy(&p, &bits, 4);
    return p;
}

#if defined(__AVX2__)
inline __m256 exp_neg8(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145751953125f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.428606765330187e-06f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(_mm256_mul_ps(p, r), r, _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_cvtps_epi32(n), 23);
    return _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(p), e));
}
#endif

struct Scratch {
    std::vector<float> q, o, m, l;     // [R][hd], [R][hd], [R], [R]
    std::vector<float> kt, v, s, row;  // [hd][tile], [tile][hd], [2][tile], [8][hd]
};

#if defined(__AVX2__)
// One Q8_0 row (hd values) to float: the same product the scalar loop made, 8 at a time.
inline void dequant_row_vec(const uint8_t* row, uint32_t hd, float* out) {
    const block_q8_0* b = reinterpret_cast<const block_q8_0*>(row);
    for (uint32_t i = 0; i < hd / 32; ++i) {
        const __m256 d = _mm256_set1_ps(desireeia_fp16_to_fp32(b[i].d));
        for (int k = 0; k < 32; k += 8) {
            const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b[i].qs + k));
            _mm256_storeu_ps(out + i * 32 + k, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw)), d));
        }
    }
}

inline void transpose8(__m256& r0, __m256& r1, __m256& r2, __m256& r3, __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    const __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    r0 = _mm256_permute2f128_ps(u0, u4, 0x20); r1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    r2 = _mm256_permute2f128_ps(u2, u6, 0x20); r3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    r4 = _mm256_permute2f128_ps(u0, u4, 0x31); r5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    r6 = _mm256_permute2f128_ps(u2, u6, 0x31); r7 = _mm256_permute2f128_ps(u3, u7, 0x31);
}
#endif

#if !defined(__AVX2__)
void dequant_row(const uint8_t* row, uint32_t hd, float* out) {
    const block_q8_0* b = reinterpret_cast<const block_q8_0*>(row);
    for (uint32_t i = 0; i < hd / 32; ++i) {
        const float d = desireeia_fp16_to_fp32(b[i].d);
        for (int k = 0; k < 32; ++k) out[i * 32 + k] = d * (float) b[i].qs[k];
    }
}
#endif

} // namespace

int attention_prefill_cpu_q8(const float* q_all, float* out, uint32_t n_tok, uint32_t n_head, uint32_t hpk,
                             uint32_t hd, uint32_t q_dim, const uint8_t* kbase, const uint8_t* vbase,
                             size_t row_bytes, size_t pos_bytes, uint32_t pos0, uint32_t n_swa, float inv_d) {
    if (hd == 0 || hd % 32 != 0 || hpk == 0 || n_head % hpk != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    const uint32_t n_kv = n_head / hpk;
    // Tokens per unit: enough rows to amortize each tile's dequantization,
    // but enough units to keep every worker busy on a short prompt.
    const size_t workers = std::max<size_t>(1, ThreadPool::global().worker_count());
    uint32_t tq = 16;
    while (tq > 2 && (size_t) n_kv * ((n_tok + tq - 1) / tq) < 2 * workers) tq /= 2;
    const uint32_t n_blocks = (n_tok + tq - 1) / tq;

    parallel_units((size_t) n_kv * n_blocks, [&](size_t u0, size_t u1) {
        thread_local Scratch sc;
        for (size_t u = u0; u < u1; ++u) {
            const uint32_t hkv = (uint32_t) (u % n_kv);
            const uint32_t t0 = (uint32_t) (u / n_kv) * tq;
            const uint32_t nt = std::min(tq, n_tok - t0);
            const uint32_t R = nt * hpk;                 // row = token * hpk + head-in-group
            sc.q.resize((size_t) R * hd);
            sc.o.assign((size_t) R * hd, 0.0f);
            sc.m.assign(R, -INFINITY);
            sc.l.assign(R, 0.0f);
            sc.kt.resize((size_t) hd * kKeyTile);
            sc.v.resize((size_t) kKeyTile * hd);
            sc.s.resize(2 * kKeyTile);
            sc.row.resize((size_t) 8 * hd);
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t t = r / hpk, h = hkv * hpk + r % hpk;
                const float* src = q_all + (size_t) (t0 + t) * q_dim + (size_t) h * hd;
                for (uint32_t d = 0; d < hd; ++d) sc.q[(size_t) r * hd + d] = src[d] * inv_d;
            }
            const uint32_t pos_first = pos0 + t0, pos_last = pos0 + t0 + nt - 1;
            const uint32_t k_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? pos_first + 1 - n_swa : 0;

            // Running max / sum update for one row whose scores for this tile are in s[0..32): masks them, rescales the
            // row's output when the max grows, turns the scores into weights, adds their sum. false = nothing valid.
            auto softmax_row = [&](uint32_t r, float* s, uint32_t k0, uint32_t kn, uint32_t pos, uint32_t lo) -> bool {
                float mt = -INFINITY;
                for (uint32_t k = 0; k < kKeyTile; ++k) {
                    const uint32_t kp = k0 + k;
                    if (k >= kn || kp > pos || kp < lo) s[k] = -INFINITY;
                    else mt = std::max(mt, s[k]);
                }
                if (mt == -INFINITY) return false;
                float& m = sc.m[r];
                float& l = sc.l[r];
                float* o = sc.o.data() + (size_t) r * hd;
                if (mt > m) {
                    const float corr = m == -INFINITY ? 0.0f : exp_neg(m - mt);
                    l *= corr;
                    for (uint32_t d = 0; d < hd; ++d) o[d] *= corr;
                    m = mt;
                }
#if defined(__AVX2__)
                const __m256 mv = _mm256_set1_ps(m);
                __m256 lsum = _mm256_setzero_ps();
                for (uint32_t k = 0; k < kKeyTile; k += 8) {
                    const __m256 sv = _mm256_loadu_ps(s + k);
                    const __m256 valid = _mm256_cmp_ps(sv, _mm256_set1_ps(-INFINITY), _CMP_NEQ_OQ);
                    const __m256 p = _mm256_and_ps(exp_neg8(_mm256_sub_ps(sv, mv)), valid);
                    _mm256_storeu_ps(s + k, p);
                    lsum = _mm256_add_ps(lsum, p);
                }
                __m128 hs = _mm_add_ps(_mm256_castps256_ps128(lsum), _mm256_extractf128_ps(lsum, 1));
                hs = _mm_add_ps(hs, _mm_movehl_ps(hs, hs));
                hs = _mm_add_ss(hs, _mm_shuffle_ps(hs, hs, 0x55));
                l += _mm_cvtss_f32(hs);
#else
                for (uint32_t k = 0; k < kKeyTile; ++k) {
                    const float p = s[k] == -INFINITY ? 0.0f : exp_neg(s[k] - m);
                    s[k] = p;
                    l += p;
                }
#endif
                return true;
            };

            for (uint32_t k0 = k_begin; k0 <= pos_last; k0 += kKeyTile) {
                const uint32_t kn = std::min(kKeyTile, pos_last + 1 - k0);
                // Keys transposed ([dim][key]), values as they are; keys past the end of the cache are zero.
#if defined(__AVX2__)
                for (uint32_t kg = 0; kg < kKeyTile; kg += 8) {
                    for (uint32_t i = 0; i < 8; ++i) {
                        const uint32_t k = kg + i;
                        float* rowbuf = sc.row.data() + (size_t) i * hd;
                        if (k < kn) {
                            const size_t off = (size_t) (k0 + k) * pos_bytes + (size_t) hkv * row_bytes;
                            dequant_row_vec(kbase + off, hd, rowbuf);
                            dequant_row_vec(vbase + off, hd, sc.v.data() + (size_t) k * hd);
                        } else {
                            std::fill_n(rowbuf, hd, 0.0f);
                            std::fill_n(sc.v.data() + (size_t) k * hd, hd, 0.0f);
                        }
                    }
                    for (uint32_t d = 0; d < hd; d += 8) {
                        const float* base = sc.row.data() + d;
                        __m256 x0 = _mm256_loadu_ps(base), x1 = _mm256_loadu_ps(base + hd);
                        __m256 x2 = _mm256_loadu_ps(base + 2 * hd), x3 = _mm256_loadu_ps(base + 3 * hd);
                        __m256 x4 = _mm256_loadu_ps(base + 4 * hd), x5 = _mm256_loadu_ps(base + 5 * hd);
                        __m256 x6 = _mm256_loadu_ps(base + 6 * hd), x7 = _mm256_loadu_ps(base + 7 * hd);
                        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
                        float* dst = sc.kt.data() + (size_t) d * kKeyTile + kg;
                        _mm256_storeu_ps(dst, x0);
                        _mm256_storeu_ps(dst + kKeyTile, x1);
                        _mm256_storeu_ps(dst + 2 * kKeyTile, x2);
                        _mm256_storeu_ps(dst + 3 * kKeyTile, x3);
                        _mm256_storeu_ps(dst + 4 * kKeyTile, x4);
                        _mm256_storeu_ps(dst + 5 * kKeyTile, x5);
                        _mm256_storeu_ps(dst + 6 * kKeyTile, x6);
                        _mm256_storeu_ps(dst + 7 * kKeyTile, x7);
                    }
                }
#else
                for (uint32_t k = 0; k < kKeyTile; ++k) {
                    if (k < kn) {
                        const size_t off = (size_t) (k0 + k) * pos_bytes + (size_t) hkv * row_bytes;
                        dequant_row(kbase + off, hd, sc.row.data());
                        for (uint32_t d = 0; d < hd; ++d) sc.kt[(size_t) d * kKeyTile + k] = sc.row[d];
                        dequant_row(vbase + off, hd, sc.v.data() + (size_t) k * hd);
                    } else {
                        for (uint32_t d = 0; d < hd; ++d) sc.kt[(size_t) d * kKeyTile + k] = 0.0f;
                        std::fill_n(sc.v.data() + (size_t) k * hd, hd, 0.0f);
                    }
                }
#endif
                uint32_t r = 0;
                while (r < R) {
                    const uint32_t pos = pos0 + t0 + r / hpk;
                    const uint32_t lo = (n_swa > 0 && pos + 1 > n_swa) ? pos + 1 - n_swa : 0;
#if defined(__AVX2__)
                    // Two rows at once when the whole tile is visible to both.
                    if (r + 1 < R && kn == kKeyTile) {
                        const uint32_t pos1 = pos0 + t0 + (r + 1) / hpk;
                        const uint32_t lo1 = (n_swa > 0 && pos1 + 1 > n_swa) ? pos1 + 1 - n_swa : 0;
                        if (pos >= k0 + kn - 1 && pos1 >= k0 + kn - 1 && lo <= k0 && lo1 <= k0) {
                            const float* q0 = sc.q.data() + (size_t) r * hd;
                            const float* q1 = q0 + hd;
                            float* s0 = sc.s.data();
                            float* s1 = s0 + kKeyTile;
                            __m256 a00 = _mm256_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
                            __m256 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
                            for (uint32_t d = 0; d < hd; ++d) {
                                const __m256 b0 = _mm256_set1_ps(q0[d]);
                                const __m256 b1 = _mm256_set1_ps(q1[d]);
                                const float* kr = sc.kt.data() + (size_t) d * kKeyTile;
                                const __m256 k0v = _mm256_loadu_ps(kr), k1v = _mm256_loadu_ps(kr + 8);
                                const __m256 k2v = _mm256_loadu_ps(kr + 16), k3v = _mm256_loadu_ps(kr + 24);
                                a00 = _mm256_fmadd_ps(b0, k0v, a00); a10 = _mm256_fmadd_ps(b1, k0v, a10);
                                a01 = _mm256_fmadd_ps(b0, k1v, a01); a11 = _mm256_fmadd_ps(b1, k1v, a11);
                                a02 = _mm256_fmadd_ps(b0, k2v, a02); a12 = _mm256_fmadd_ps(b1, k2v, a12);
                                a03 = _mm256_fmadd_ps(b0, k3v, a03); a13 = _mm256_fmadd_ps(b1, k3v, a13);
                            }
                            _mm256_storeu_ps(s0, a00); _mm256_storeu_ps(s0 + 8, a01);
                            _mm256_storeu_ps(s0 + 16, a02); _mm256_storeu_ps(s0 + 24, a03);
                            _mm256_storeu_ps(s1, a10); _mm256_storeu_ps(s1 + 8, a11);
                            _mm256_storeu_ps(s1 + 16, a12); _mm256_storeu_ps(s1 + 24, a13);
                            const bool ok0 = softmax_row(r, s0, k0, kn, pos, lo);
                            const bool ok1 = softmax_row(r + 1, s1, k0, kn, pos1, lo1);
                            if (ok0 && ok1) {
                                float* o0 = sc.o.data() + (size_t) r * hd;
                                float* o1 = o0 + hd;
                                for (uint32_t dc = 0; dc < hd; dc += 32) {
                                    __m256 c00 = _mm256_loadu_ps(o0 + dc), c01 = _mm256_loadu_ps(o0 + dc + 8);
                                    __m256 c02 = _mm256_loadu_ps(o0 + dc + 16), c03 = _mm256_loadu_ps(o0 + dc + 24);
                                    __m256 c10 = _mm256_loadu_ps(o1 + dc), c11 = _mm256_loadu_ps(o1 + dc + 8);
                                    __m256 c12 = _mm256_loadu_ps(o1 + dc + 16), c13 = _mm256_loadu_ps(o1 + dc + 24);
                                    for (uint32_t k = 0; k < kKeyTile; ++k) {
                                        const float* vr = sc.v.data() + (size_t) k * hd + dc;
                                        const __m256 v0 = _mm256_loadu_ps(vr), v1 = _mm256_loadu_ps(vr + 8);
                                        const __m256 v2 = _mm256_loadu_ps(vr + 16), v3 = _mm256_loadu_ps(vr + 24);
                                        const __m256 p0 = _mm256_set1_ps(s0[k]), p1 = _mm256_set1_ps(s1[k]);
                                        c00 = _mm256_fmadd_ps(p0, v0, c00); c10 = _mm256_fmadd_ps(p1, v0, c10);
                                        c01 = _mm256_fmadd_ps(p0, v1, c01); c11 = _mm256_fmadd_ps(p1, v1, c11);
                                        c02 = _mm256_fmadd_ps(p0, v2, c02); c12 = _mm256_fmadd_ps(p1, v2, c12);
                                        c03 = _mm256_fmadd_ps(p0, v3, c03); c13 = _mm256_fmadd_ps(p1, v3, c13);
                                    }
                                    _mm256_storeu_ps(o0 + dc, c00); _mm256_storeu_ps(o0 + dc + 8, c01);
                                    _mm256_storeu_ps(o0 + dc + 16, c02); _mm256_storeu_ps(o0 + dc + 24, c03);
                                    _mm256_storeu_ps(o1 + dc, c10); _mm256_storeu_ps(o1 + dc + 8, c11);
                                    _mm256_storeu_ps(o1 + dc + 16, c12); _mm256_storeu_ps(o1 + dc + 24, c13);
                                }
                            } else {
                                // Cannot happen for a fully visible tile; kept so a surprise still gives the one-row result.
                                for (int w = 0; w < 2; ++w) {
                                    const float* sw = w == 0 ? s0 : s1;
                                    float* ow = sc.o.data() + (size_t) (r + w) * hd;
                                    if (!(w == 0 ? ok0 : ok1)) continue;
                                    for (uint32_t k = 0; k < kn; ++k) {
                                        const float p = sw[k];
                                        if (p == 0.0f) continue;
                                        const __m256 pv = _mm256_set1_ps(p);
                                        const float* vr = sc.v.data() + (size_t) k * hd;
                                        for (uint32_t d = 0; d < hd; d += 8)
                                            _mm256_storeu_ps(ow + d, _mm256_fmadd_ps(pv, _mm256_loadu_ps(vr + d), _mm256_loadu_ps(ow + d)));
                                    }
                                }
                            }
                            r += 2;
                            continue;
                        }
                    }
#endif
                    // One row.
                    if (k0 > pos) { ++r; continue; }                         // tile entirely in this row's future
                    if (k0 + kn <= lo) { ++r; continue; }                    // tile entirely outside the window
                    const float* qr = sc.q.data() + (size_t) r * hd;
                    float* s = sc.s.data();
#if defined(__AVX2__)
                    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                    for (uint32_t d = 0; d < hd; ++d) {
                        const __m256 qd = _mm256_set1_ps(qr[d]);
                        const float* kr = sc.kt.data() + (size_t) d * kKeyTile;
                        a0 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr + 0), a0);
                        a1 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr + 8), a1);
                        a2 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr + 16), a2);
                        a3 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(kr + 24), a3);
                    }
                    _mm256_storeu_ps(s + 0, a0); _mm256_storeu_ps(s + 8, a1);
                    _mm256_storeu_ps(s + 16, a2); _mm256_storeu_ps(s + 24, a3);
#else
                    for (uint32_t k = 0; k < kKeyTile; ++k) s[k] = 0.0f;
                    for (uint32_t d = 0; d < hd; ++d) {
                        const float qd = qr[d];
                        const float* kr = sc.kt.data() + (size_t) d * kKeyTile;
                        for (uint32_t k = 0; k < kKeyTile; ++k) s[k] += qd * kr[k];
                    }
#endif
                    if (!softmax_row(r, s, k0, kn, pos, lo)) { ++r; continue; }
                    float* o = sc.o.data() + (size_t) r * hd;
#if defined(__AVX2__)
                    for (uint32_t k = 0; k < kn; ++k) {
                        const float p = s[k];
                        if (p == 0.0f) continue;
                        const __m256 pv = _mm256_set1_ps(p);
                        const float* vr = sc.v.data() + (size_t) k * hd;
                        for (uint32_t d = 0; d < hd; d += 8)
                            _mm256_storeu_ps(o + d, _mm256_fmadd_ps(pv, _mm256_loadu_ps(vr + d), _mm256_loadu_ps(o + d)));
                    }
#else
                    for (uint32_t k = 0; k < kn; ++k) {
                        const float p = s[k];
                        if (p == 0.0f) continue;
                        const float* vr = sc.v.data() + (size_t) k * hd;
                        for (uint32_t d = 0; d < hd; ++d) o[d] += p * vr[d];
                    }
#endif
                    ++r;
                }
            }
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t t = r / hpk, h = hkv * hpk + r % hpk;
                float* dst = out + (size_t) (t0 + t) * q_dim + (size_t) h * hd;
                const float inv = sc.l[r] > 0.0f ? 1.0f / sc.l[r] : 0.0f;
                const float* o = sc.o.data() + (size_t) r * hd;
                for (uint32_t d = 0; d < hd; ++d) dst[d] = o[d] * inv;
            }
        }
    });
    return DESIREEIA_OK;
}

} // namespace desireeia
