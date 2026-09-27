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
    std::vector<float> kt, v, s, row;  // [hd][tile], [tile][hd], [tile], [hd]
};

void dequant_row(const uint8_t* row, uint32_t hd, float* out) {
    const block_q8_0* b = reinterpret_cast<const block_q8_0*>(row);
    for (uint32_t i = 0; i < hd / 32; ++i) {
        const float d = desireeia_fp16_to_fp32(b[i].d);
        for (int k = 0; k < 32; ++k) out[i * 32 + k] = d * (float) b[i].qs[k];
    }
}

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
            sc.s.resize(kKeyTile);
            sc.row.resize(hd);
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t t = r / hpk, h = hkv * hpk + r % hpk;
                const float* src = q_all + (size_t) (t0 + t) * q_dim + (size_t) h * hd;
                for (uint32_t d = 0; d < hd; ++d) sc.q[(size_t) r * hd + d] = src[d] * inv_d;
            }
            const uint32_t pos_first = pos0 + t0, pos_last = pos0 + t0 + nt - 1;
            const uint32_t k_begin = (n_swa > 0 && pos_first + 1 > n_swa) ? pos_first + 1 - n_swa : 0;

            for (uint32_t k0 = k_begin; k0 <= pos_last; k0 += kKeyTile) {
                const uint32_t kn = std::min(kKeyTile, pos_last + 1 - k0);
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
                for (uint32_t r = 0; r < R; ++r) {
                    const uint32_t pos = pos0 + t0 + r / hpk;
                    if (k0 > pos) continue;                              // tile entirely in this row's future
                    const uint32_t lo = (n_swa > 0 && pos + 1 > n_swa) ? pos + 1 - n_swa : 0;
                    if (k0 + kn <= lo) continue;                         // tile entirely outside the window
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
                    // Mask: future keys, keys before the window, padding.
                    float mt = -INFINITY;
                    for (uint32_t k = 0; k < kKeyTile; ++k) {
                        const uint32_t kp = k0 + k;
                        if (k >= kn || kp > pos || kp < lo) s[k] = -INFINITY;
                        else mt = std::max(mt, s[k]);
                    }
                    if (mt == -INFINITY) continue;
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
                    {
                        __m128 hs = _mm_add_ps(_mm256_castps256_ps128(lsum), _mm256_extractf128_ps(lsum, 1));
                        hs = _mm_add_ps(hs, _mm_movehl_ps(hs, hs));
                        hs = _mm_add_ss(hs, _mm_shuffle_ps(hs, hs, 0x55));
                        l += _mm_cvtss_f32(hs);
                    }
                    for (uint32_t k = 0; k < kn; ++k) {
                        const float p = s[k];
                        if (p == 0.0f) continue;
                        const __m256 pv = _mm256_set1_ps(p);
                        const float* vr = sc.v.data() + (size_t) k * hd;
                        for (uint32_t d = 0; d < hd; d += 8)
                            _mm256_storeu_ps(o + d, _mm256_fmadd_ps(pv, _mm256_loadu_ps(vr + d), _mm256_loadu_ps(o + d)));
                    }
#else
                    for (uint32_t k = 0; k < kKeyTile; ++k) {
                        const float p = s[k] == -INFINITY ? 0.0f : exp_neg(s[k] - m);
                        s[k] = p;
                        l += p;
                    }
                    for (uint32_t k = 0; k < kn; ++k) {
                        const float p = s[k];
                        if (p == 0.0f) continue;
                        const float* vr = sc.v.data() + (size_t) k * hd;
                        for (uint32_t d = 0; d < hd; ++d) o[d] += p * vr[d];
                    }
#endif
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
