#include "dense_forward.h"
#include "moe_route.h"
#include "../core/profile.h"
#include "../quant/quant.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace desireeia {

namespace {

static float gelu_tanh(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * v * (1.0f + 0.044715f * v * v)));
}

static float silu(float v) {
    return v / (1.0f + expf(-v));
}

// relu(x)^2, used by Arcee/Nemotron: relu followed by squaring, exact,
// not an approximation.
static inline float relu_sqr(float v) {
    const float r = v > 0.0f ? v : 0.0f;
    return r * r;
}

#if defined(__AVX2__)
// exp(x) vettoriale. Schema classico: si porta l'esponenziale in base 2,
// si separa la parte intera k (che diventa direttamente il campo esponente
// del float) dal resto r in [-0.5, 0.5], e si valuta 2^r con un polinomio.
// Errore relativo ~1e-7 nell'intervallo utile, ben oltre quanto serve qui.
static inline __m256 exp_ps_avx2(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    // Saturazione: oltre questi limiti il risultato e' comunque 0 o infinito
    // e serve solo a evitare che k esca dal campo esponente.
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(87.0f));

    const __m256 t = _mm256_mul_ps(x, log2e);
    const __m256 k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_sub_ps(t, k);

    // 2^r su [-0.5, 0.5], polinomio di grado 5 (coefficienti della serie di
    // ln(2)^n/n!).
    __m256 p = _mm256_set1_ps(1.3327544e-3f);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(9.6181292e-3f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.0f));

    // 2^k costruito scrivendo k+127 nel campo esponente.
    const __m256i ki = _mm256_cvtps_epi32(k);
    const __m256i pow2k = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pow2k));
}
#endif

// y[i] = gelu(g[i]) * y[i] (gated GELU, fused into one pass).
//
// Uses the identity that eliminates the hyperbolic tangent entirely:
//     gelu(x) = 0.5x(1 + tanh(u)) = x / (1 + exp(-2u))
// with u = 0.7978845608 x (1 + 0.044715 x^2). That leaves ONE exponential
// per value instead of a tanhf, and it vectorizes cleanly.
//
// An earlier version used an f16-bit-indexed lookup table instead. It
// worked, but the table is 256 KB and the accesses are scattered, so
// every evaluation paid for a cache miss outside L1. The direct
// vectorized computation touches no memory at all.
static void geglu_inplace(float* y, const float* g, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256 c_sqrt2pi = _mm256_set1_ps(0.7978845608028654f);
    const __m256 c_a       = _mm256_set1_ps(0.044715f);
    const __m256 one       = _mm256_set1_ps(1.0f);
    const __m256 minus2    = _mm256_set1_ps(-2.0f);
    for (; i + 8 <= n; i += 8) {
        const __m256 x  = _mm256_loadu_ps(g + i);
        const __m256 xx = _mm256_mul_ps(x, x);
        const __m256 u  = _mm256_mul_ps(_mm256_mul_ps(c_sqrt2pi, x),
                                        _mm256_add_ps(one, _mm256_mul_ps(c_a, xx)));
        const __m256 e  = exp_ps_avx2(_mm256_mul_ps(minus2, u));
        const __m256 gelu = _mm256_div_ps(x, _mm256_add_ps(one, e));
        _mm256_storeu_ps(y + i, _mm256_mul_ps(gelu, _mm256_loadu_ps(y + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = gelu_tanh(g[i]) * y[i];
    }
}

static void rms_norm_vec(const float* x, const float* w, float* y, size_t n, float eps) {
    // La somma dei quadrati resta in double (e' una riduzione su n_embd
    // valori e la stabilita' numerica qui conta), ma con quattro
    // accumulatori vettoriali invece di una catena scalare seriale.
    double sum = 0.0;
    size_t i = 0;
#if defined(__AVX2__)
    __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 vv = _mm256_mul_ps(v, v);
        s0 = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm256_castps256_ps128(vv)));
        s1 = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm256_extractf128_ps(vv, 1)));
    }
    double tmp[4];
    _mm256_storeu_pd(tmp, _mm256_add_pd(s0, s1));
    sum = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
#endif
    for (; i < n; ++i) sum += (double) x[i] * x[i];

    const float scale = 1.0f / sqrtf((float) (sum / (double) n) + eps);
    size_t j = 0;
#if defined(__AVX2__)
    const __m256 vscale = _mm256_set1_ps(scale);
    for (; j + 8 <= n; j += 8) {
        _mm256_storeu_ps(y + j, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + j), vscale),
                                              _mm256_loadu_ps(w + j)));
    }
#endif
    for (; j < n; ++j) y[j] = x[j] * scale * w[j];
}

// Classic LayerNorm (DenseQuirks::layer_norm: stablelm, orion, etc.):
// POPULATION variance (divided by n, not n-1), scale = 1/sqrt(var+eps),
// then (x-mean)*scale*w [+ b if present — several of these architectures
// have a bias on the norm, which RMSNorm elsewhere in the engine never
// does]. Not vectorized (unlike rms_norm_vec): the architectures that use
// it are a small subset for now, correctness first.
static void layer_norm_vec(const float* x, const float* w, const float* b,
                           float* y, size_t n, float eps) {
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += (double) x[i];
    const double mean = sum / (double) n;

    double var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) x[i] - mean;
        var += d * d;
    }
    var /= (double) n;

    const float scale = 1.0f / sqrtf((float) var + eps);
    for (size_t i = 0; i < n; ++i) {
        const float v = ((float) ((double) x[i] - mean)) * scale * w[i];
        y[i] = b ? v + b[i] : v;
    }
}

// Sceglie fra rms_norm_vec e layer_norm_vec in base al quirk
// dell'architettura corrente: unico punto di dispatch, cosi' i ~15 call
// site sparsi nel forward path non devono sapere quale normalizzazione usa
// il modello caricato. `b` e' ignorato quando si usa RMSNorm (che in questo
// motore non ha mai bias).
static void norm_vec(bool use_layer_norm, const float* x, const float* w, const float* b,
                     float* y, size_t n, float eps) {
    if (use_layer_norm) layer_norm_vec(x, w, b, y, n, eps);
    else rms_norm_vec(x, w, y, n, eps);
}

static void matmul_f32(const float* w, size_t r, size_t c, const float* x, float* y) {
    ScopedTimer t(profile_counters().ns_f32_compute);
    parallel_rows(r, [&](size_t j0, size_t j1) {
        for (size_t j = j0; j < j1; ++j) {
            const float* wrow = w + j * c;
            double acc = 0.0;
            for (size_t i = 0; i < c; ++i) {
                acc += (double) wrow[i] * x[i];
            }
            y[j] = (float) acc;
        }
    });
    profile_counters().calls_f32.fetch_add(1, std::memory_order_relaxed);
}

// Dispatcher: usa il kernel quantizzato diretto se il tensore e' Q4_0 su
// disco, altrimenti ricade sul matmul float classico su dati dequantizzati.
static void matvec(const MatVec& m, size_t r, size_t c, const float* x, float* y) {
    switch (m.format) {
        case MatVecFormat::Q4_0: matmul_q4_0(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q4_1: matmul_q4_1(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q5_0: matmul_q5_0(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q5_1: matmul_q5_1(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q2_K: matmul_q2_k(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q3_K: matmul_q3_k(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q4_K: matmul_q4_k(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q5_K: matmul_q5_k(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q6_K: matmul_q6_k(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q8_0: matmul_q8_0(m.raw.data(), r, c, x, y); break;
        case MatVecFormat::Q8_K: matmul_q8_k(m.raw.data(), r, c, x, y); break;
        default:                 matmul_f32(m.f.data(), r, c, x, y);   break;
    }
}

// Come matvec ma per n_tok colonne di attivazione in una sola chiamata
// (Fase 8): x e y sono n_tok blocchi contigui da c/r elementi. Per il
// fallback Float (raro: solo tensori non quantizzati) non esiste ancora
// un matmul_f32 batched dedicato, quindi si richiama semplicemente
// matmul_f32 per colonna: nessuna regressione (stesso costo di prima),
// il guadagno del batching riguarda solo i kernel quantizzati.
static void matvec_batch(const MatVec& m, size_t r, size_t c, const float* x, size_t n_tok, float* y) {
    switch (m.format) {
        case MatVecFormat::Q4_0: matmul_q4_0_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q4_1: matmul_q4_1_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q5_0: matmul_q5_0_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q5_1: matmul_q5_1_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q2_K: matmul_q2_k_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q3_K: matmul_q3_k_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q4_K: matmul_q4_k_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q5_K: matmul_q5_k_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q6_K: matmul_q6_k_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q8_0: matmul_q8_0_batch(m.raw.data(), r, c, x, n_tok, y); break;
        case MatVecFormat::Q8_K: matmul_q8_k_batch(m.raw.data(), r, c, x, n_tok, y); break;
        default:
            for (size_t tk = 0; tk < n_tok; ++tk) {
                matmul_f32(m.f.data(), r, c, x + tk * c, y + tk * r);
            }
            break;
    }
}

// Fase "elimina ri-quantizzazione ridondante" (2026-09-07): wq/wk/wv (e
// separatamente wff_gate/wff_up) leggono tutti dalla STESSA attivazione
// (rispettivamente l'uscita di attn_norm e di ffn_norm), ma essendo
// spesso formati diversi (es. gemma3 Q4_K_M: wq/wk Q4_K, wv Q6_K)
// ciascuna matmul_qX_k la riquantizzava da capo al suo interno — fino a
// 3-5 volte la stessa quantizzazione per layer. Quantizzando una volta
// sola qui (stile Q8_0, per sotto-blocco da 32 — vedi REVERT DI
// CORRETTEZZA in matmul.cpp: lo schema Q8_K a scala singola per 256
// degradava troppo la precisione su un modello reale) e passando il
// risultato alle varianti "_pq" si elimina la ridondanza per Q4_K/Q6_K
// senza perdere precisione (gli altri formati, piu' rari in un modello
// K-quant, ricadono sul matvec normale, nessuna regressione).
static void quantize_shared_q8k(const float* x, size_t cols, std::vector<int8_t>& xq,
                                 std::vector<float>& xscale, std::vector<int32_t>& xsum) {
    ScopedTimer t(profile_counters().ns_quantize_act);
    // Q8_K (una scala ogni 256) invece di Q8_0 (una ogni 32): e' cio' che
    // abilita l'accumulazione intera dentro matmul_q4_k_core. La scala resta
    // replicata nelle caselle per-32, quindi i kernel Q5_K/Q6_K, che leggono
    // xscale[sub], continuano a funzionare senza modifiche.
    quantize_act_q8k_rep(x, cols, xq, xscale, xsum);
}

static void matvec_shared(const MatVec& m, size_t r, size_t c, const float* x,
                           const std::vector<int8_t>& xq, const std::vector<float>& xscale,
                           const std::vector<int32_t>& xsum, float* y) {
    switch (m.format) {
        case MatVecFormat::Q4_K: matmul_q4_k_pq(m.raw.data(), r, c, xq.data(), xscale.data(), xsum.data(), y); break;
        case MatVecFormat::Q6_K: matmul_q6_k_pq(m.raw.data(), r, c, xq.data(), xscale.data(), y); break;
        default: matvec(m, r, c, x, y); break;
    }
}

// One entry of a group of matvecs that share the same input activation.
struct SharedMatvec {
    const MatVec* m;
    size_t r;
    size_t c;
    float* y;
};

static bool fused_job_for(const SharedMatvec& it,
                          const std::vector<int8_t>& xq, const std::vector<float>& xscale,
                          const std::vector<int32_t>& xsum, FusedPqJob& job) {
    if (it.m->format == MatVecFormat::Q4_K)      job.format = FusedPqFormat::Q4_K;
    else if (it.m->format == MatVecFormat::Q6_K) job.format = FusedPqFormat::Q6_K;
    else return false;
    job.data   = it.m->raw.data();
    job.rows   = it.r;
    job.cols   = it.c;
    job.xq     = xq.data();
    job.xscale = xscale.data();
    job.xsum   = xsum.data();
    job.y      = it.y;
    return true;
}

// Runs a group of shared-activation matvecs in a single parallel dispatch.
//
// Q/K/V read the output of attn_norm and write to three separate buffers;
// gate/up do the same with ffn_norm. Nothing about them needs three (or two)
// separate parallel regions, and each region costs a barrier where every
// thread waits for the slowest one. On a busy machine that wait is the single
// most variable part of decode.
//
// Falls back to running them one at a time whenever any matrix is in a format
// the fused kernel doesn't cover — which is exactly the previous behaviour, so
// an unusual quantization mix loses nothing.
static void matvec_shared_group(const SharedMatvec* items, size_t n, const float* x,
                                const std::vector<int8_t>& xq, const std::vector<float>& xscale,
                                const std::vector<int32_t>& xsum) {
    FusedPqJob jobs[4];
    bool fusable = (n >= 2 && n <= 4);
    for (size_t i = 0; fusable && i < n; ++i) {
        fusable = fused_job_for(items[i], xq, xscale, xsum, jobs[i]);
    }
    if (fusable && fused_pq_supported(jobs, n) &&
        matmul_fused_pq(jobs, n) == DESIREEIA_OK) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        matvec_shared(*items[i].m, items[i].r, items[i].c, x, xq, xscale, xsum, items[i].y);
    }
}

static void add_bias(float* y, const float* b, size_t n) {
    if (b == nullptr) return;
    for (size_t i = 0; i < n; ++i) y[i] += b[i];
}

// freq_scale: scaling lineare delle posizioni (rope.scaling.factor
// invertito). 1.0 = nessuno scaling. Vedi la nota in DenseConfig.
// Cache di cos/sin per una posizione e una coppia (theta_base, freq_scale).
// Interleavata: cache[2j] = cos, cache[2j+1] = sin.
//
// Prima ogni chiamata a rope_neox calcolava, per OGNI coppia di dimensioni,
// una powf piu' una sinf e una cosf — e veniva invocata una volta per testa:
// con 8 teste Q + 4 teste K su 34 layer sono 408 invocazioni per token, cioe'
// ~52000 powf e altrettante sincos ripetute identiche. Ma il risultato non
// dipende dalla testa: dipende solo da posizione e parametri RoPE del layer.
//
// The fix does two things:
//   1. no powf per dimension — theta_scale = powf(base, -2/n_dims) gets
//      computed ONCE and then advanced by repeated multiplication;
//   2. the cache is built once and reused across every head.
// gemma3 alternates local and global layers with a different theta, so
// each token needs two caches instead of 408 recomputations.
// ext_factor/attn_factor/corr_lo/corr_hi default to (0, 1, 0, 0), which
// collapse exactly onto the classic formula already in use (see the note
// on DenseConfig::rope_ext_factor): no existing architecture is affected.
// With ext_factor != 0 this implements YaRN in full (DeepSeek2 and
// siblings need it for extended context).
static void rope_cache_init(std::vector<float>& cache, size_t n_rot, uint32_t pos,
                            float theta_base, float freq_scale,
                            float ext_factor = 0.0f, float attn_factor = 1.0f,
                            float corr_lo = 0.0f, float corr_hi = 0.0f) {
    cache.resize(n_rot);
    const float theta_scale = powf(theta_base, -2.0f / (float) n_rot);
    float theta_extrap = (float) pos;
    for (size_t j = 0; j < n_rot / 2; ++j) {
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float y = ((float) j - corr_lo) / std::max(0.001f, corr_hi - corr_lo);
            const float ramp = 1.0f - std::min(1.0f, std::max(0.0f, y));
            const float ramp_mix = ramp * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        cache[2 * j + 0] = cosf(theta) * mscale;
        cache[2 * j + 1] = sinf(theta) * mscale;
        theta_extrap *= theta_scale;
    }
}

static void rope_neox_cached(float* v, size_t n_rot, const float* cache) {
    const size_t half = n_rot / 2;
    for (size_t j = 0; j < half; ++j) {
        const float cos_t = cache[2 * j + 0];
        const float sin_t = cache[2 * j + 1];
        const float x0 = v[j];
        const float x1 = v[j + half];
        v[j]        = x0 * cos_t - x1 * sin_t;
        v[j + half] = x0 * sin_t + x1 * cos_t;
    }
}

// Prodotto scalare su vettori contigui (usato per i punteggi QK
// dell'attenzione, dove head_dim e' un multiplo di 8).
static inline float dot_f32(const float* a, const float* b, size_t n) {
    size_t i = 0;
    float acc = 0.0f;
#if defined(__AVX2__)
    // Quattro accumulatori: spezzano la catena di dipendenze fra addizioni
    // successive, che altrimenti impone ~4 cicli di latenza per passo.
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
    }
    for (; i + 8 <= n; i += 8) {
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
    acc = _mm_cvtss_f32(h);
#endif
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

// y[i] += s * x[i]  (accumulo pesato di V nell'attenzione).
static inline void axpy_f32(float* y, const float* x, float s, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256 vs = _mm256_set1_ps(s);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i),
                                              _mm256_mul_ps(vs, _mm256_loadu_ps(x + i))));
    }
#endif
    for (; i < n; ++i) y[i] += s * x[i];
}

// ALiBi (bloom, mpt): per-head slope — see the extended comment on
// DenseConfig::max_alibi_bias for the full formula.
static float alibi_slope(uint32_t h, uint32_t n_head, float max_bias) {
    if (max_bias <= 0.0f) return 0.0f;
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));
    const float m0 = powf(2.0f, -(max_bias) / (float) n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / (float) n_head_log2);
    return h < n_head_log2 ? powf(m0, (float) (h + 1)) : powf(m1, (float) (2 * (h - n_head_log2) + 1));
}

static void softmax_inplace(float* s, size_t n) {
    float m = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, s[i]);
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s[i] = expf(s[i] - m);
        sum += (double) s[i];
    }
    for (size_t i = 0; i < n; ++i) {
        s[i] /= (float) sum;
    }
}

}

bool DenseForward::open(ModelReader& rd, const ModelMeta& meta, ArchKind arch, uint64_t ram_budget_mb,
                         ExpertStore* experts) {
    const std::string arch_tag = meta.arch.empty() ? "gemma" : meta.arch;
    const std::string kp = arch_tag + ".";
    quirks_ = quirks_for(arch);
    experts_ = experts;

    uint32_t v32 = 0;
    float  f32 = 0.0f;

    cfg_.n_vocab  = meta.n_vocab;
    cfg_.n_layers = meta.n_layers;
    if (rd.meta_u32(kp + "embedding_length", v32)) cfg_.n_embd = v32;
    if (rd.meta_u32(kp + "attention.head_count", v32)) cfg_.n_head = v32;
    if (rd.meta_u32(kp + "attention.head_count_kv", v32)) {
        cfg_.n_head_kv = v32;
    } else {
        cfg_.n_head_kv = cfg_.n_head;
    }
    if (rd.meta_u32(kp + "attention.key_length", v32)) {
        cfg_.head_dim = v32;
    } else if (cfg_.n_embd > 0 && cfg_.n_head > 0) {
        cfg_.head_dim = cfg_.n_embd / cfg_.n_head;
    }
    if (rd.meta_u32(kp + "rope.dimension_count", v32)) {
        cfg_.n_rot = v32;
    } else {
        cfg_.n_rot = cfg_.head_dim;
    }
    if (rd.meta_u32(kp + "feed_forward_length", v32)) cfg_.n_ff = v32;
    if (rd.meta_f32(kp + "attention.layer_norm_rms_epsilon", f32)) cfg_.rms_eps = f32;
    // A DIFFERENT key from rms_epsilon above: used only by architectures
    // with classic LayerNorm (DenseQuirks::layer_norm).
    if (rd.meta_f32(kp + "attention.layer_norm_epsilon", f32)) cfg_.norm_eps = f32;
    if (rd.meta_f32(kp + "rope.freq_base", f32)) cfg_.rope_theta = f32;

    // Per-layer RoPE with sliding window (see the extended note in
    // DenseConfig): the presence of `attention.sliding_window` activates
    // the alternating local/global pattern. Default period 6 (5 local
    // layers + 1 global), overridable by `attention.sliding_window_pattern`.
    // The local layers' RoPE base is in `rope.freq_base_swa` if present,
    // otherwise 10000. Scaling (`rope.scaling.factor`) applies ONLY to
    // global layers: local layers use scale 1.0.
    if (rd.meta_u32(kp + "attention.sliding_window", v32) && v32 > 0) {
        cfg_.n_swa = v32;
        cfg_.swa_pattern = 6;
        if (rd.meta_u32(kp + "attention.sliding_window_pattern", v32) && v32 > 0) {
            cfg_.swa_pattern = v32;
        }
        cfg_.rope_theta_swa = 10000.0f;
        if (rd.meta_f32(kp + "rope.freq_base_swa", f32)) cfg_.rope_theta_swa = f32;
        cfg_.rope_freq_scale_swa = 1.0f;
    }
    // `rope.scaling.factor` is the context-extension factor: the scale
    // applied to theta is its inverse.
    if (rd.meta_f32(kp + "rope.scaling.factor", f32) && f32 > 0.0f) {
        cfg_.rope_freq_scale = 1.0f / f32;
    }
    rd.meta_f32(kp + "logit_scale", cfg_.logit_scale);
    if (rd.meta_u32(kp + "context_length", v32)) n_ctx_train_ = v32;
    rd.meta_f32(kp + "attention.max_alibi_bias", cfg_.max_alibi_bias);
    if (quirks_.alibi && cfg_.max_alibi_bias == 0.0f) cfg_.max_alibi_bias = 8.0f; // bloom's fixed default
    rd.meta_f32(kp + "attention.clamp_kqv", cfg_.clamp_kqv);

    // MoE: n_expert>0 moves the layer from the dense FFN (wff_*) to the
    // router + ExpertStore (see load_layer_data/step). expert_feed_forward_length
    // can differ from feed_forward_length (per-expert FFN width vs. the
    // model's "dense" width, often absent/0 in pure MoE models).
    cfg_.n_expert = meta.n_experts;
    if (cfg_.n_expert > 0) {
        if (rd.meta_u32(kp + "expert_used_count", v32)) cfg_.n_expert_used = v32;
        cfg_.n_ff_expert = cfg_.n_ff;
        if (rd.meta_u32(kp + "expert_feed_forward_length", v32)) cfg_.n_ff_expert = v32;
        cfg_.moe_norm_w = (arch_tag != "qwen2moe"); // this family skips renormalization
        cfg_.moe_w_scale = 1.0f;
        if (rd.meta_f32(kp + "expert_weights_scale", f32)) cfg_.moe_w_scale = f32;
    }

    // MLA (deepseek2/3): sostituisce head_dim/n_rot/q_dim con le dimensioni
    // dello spazio compresso. Vedi la nota estesa in arch_tags.h su
    // DenseQuirks::mla e mla_attn_layer() piu' sotto per la derivazione.
    if (quirks_.mla) {
        if (rd.meta_u32(kp + "attention.q_lora_rank", v32)) cfg_.q_lora_rank = v32;
        if (rd.meta_u32(kp + "attention.kv_lora_rank", v32)) cfg_.kv_lora_rank = v32;
        uint32_t k_mla = cfg_.head_dim; // fallback: attention.key_length gia' letta sopra
        rd.meta_u32(kp + "attention.key_length_mla", k_mla);
        uint32_t v_mla = k_mla;
        rd.meta_u32(kp + "attention.value_length_mla", v_mla);
        cfg_.n_embd_head_v_mla = v_mla;
        cfg_.n_embd_head_qk_rope = cfg_.n_rot; // rope.dimension_count, gia' letta sopra
        cfg_.n_embd_head_qk_nope = k_mla > cfg_.n_rot ? k_mla - cfg_.n_rot : 0;
        // head_dim/n_rot/n_head_kv vengono riusati come "larghezza
        // compressa" per la cache K/V generica (vedi grow_cache): dopo
        // l'assorbimento Q e K vivono entrambi in uno spazio di dimensione
        // kv_lora_rank+rope, condiviso da TUTTE le teste (MQA) — n_head_kv=1.
        cfg_.head_dim = cfg_.kv_lora_rank + cfg_.n_embd_head_qk_rope;
        cfg_.n_head_kv = 1;

        if (rd.meta_u32(kp + "leading_dense_block_count", v32)) cfg_.n_layer_dense_lead = v32;
        if (rd.meta_u32(kp + "expert_shared_count", v32)) cfg_.n_expert_shared = v32;
        if (rd.meta_u32(kp + "expert_gating_func", v32)) cfg_.moe_sigmoid_gate = (v32 == 2);

        // YaRN. rope_ext_factor==0 (default) disattiva completamente il
        // ramo YaRN in rope_cache_init: se il modello non dichiara
        // rope.scaling.type=="yarn" il RoPE resta quello classico.
        std::string rope_scaling_type;
        if (rd.meta_str(kp + "rope.scaling.type", rope_scaling_type) && rope_scaling_type == "yarn") {
            cfg_.rope_ext_factor = 1.0f;
        }
        cfg_.n_ctx_orig_yarn = 0;
        if (rd.meta_u32(kp + "context_length", v32)) cfg_.n_ctx_orig_yarn = v32;
        rd.meta_u32(kp + "rope.scaling.original_context_length", cfg_.n_ctx_orig_yarn);
        cfg_.rope_beta_fast = 32.0f;
        cfg_.rope_beta_slow = 1.0f;
        rd.meta_f32(kp + "rope.scaling.yarn_beta_fast", cfg_.rope_beta_fast);
        rd.meta_f32(kp + "rope.scaling.yarn_beta_slow", cfg_.rope_beta_slow);
        float rope_attn_factor_meta = 1.0f;
        rd.meta_f32(kp + "rope.scaling.attn_factor", rope_attn_factor_meta);
        // The conversion tooling for this format writes this value
        // already multiplied by 0.1, so it needs to be cancelled back out
        // here.
        cfg_.rope_yarn_log_mul = 0.0f;
        if (rd.meta_f32(kp + "rope.scaling.yarn_log_multiplier", f32)) cfg_.rope_yarn_log_mul = f32 / 0.1f;

        // Final YaRN attn_factor: with ext_factor!=0, apply get_mscale
        // twice (once with mscale_all_dims for the "cancel" correction,
        // once with mscale=1 or =rope_yarn_log_mul for MLA models) then
        // multiply by the attn_factor read from metadata (default 1.0).
        cfg_.rope_attn_factor = 1.0f;
        if (cfg_.rope_ext_factor != 0.0f) {
            auto get_mscale = [](float scale, float mscale) {
                return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
            };
            const float factor = 1.0f / cfg_.rope_freq_scale;
            float mscale = 1.0f;
            const float mscale_all_dims = cfg_.rope_yarn_log_mul;
            if (mscale_all_dims != 0.0f && mscale_all_dims != 1.0f) mscale = mscale_all_dims;
            float attn_f = (mscale_all_dims != 0.0f)
                ? get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims)
                : get_mscale(factor, 1.0f);
            attn_f *= 1.0f / (1.0f + 0.1f * logf(factor));
            cfg_.rope_attn_factor = attn_f * rope_attn_factor_meta;
        }
    }

    const bool ffn_ok = cfg_.n_expert > 0 ? (cfg_.n_ff_expert > 0 && cfg_.n_expert_used > 0) : cfg_.n_ff > 0;
    if (cfg_.n_vocab == 0 || cfg_.n_layers == 0 || cfg_.n_embd == 0 ||
        cfg_.n_head == 0 || cfg_.n_head_kv == 0 || cfg_.head_dim == 0 ||
        cfg_.n_rot == 0 || !ffn_ok) {
        return false;
    }
    if (cfg_.n_expert > 0 && experts_ == nullptr) return false;

    const uint32_t kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    if (!load_embd(rd)) return false;
    if (!load_norms(rd)) return false;
    if (kv_dim == 0) return false;

    // Cache pesi layer: attiva solo se l'ingombro reale in RAM entra in una
    // frazione conservativa (60%) del budget RAM pianificato, per lasciare
    // margine a KV cache, embeddings e overhead del processo.
    // ram_budget_mb==0 (piano non configurato) disabilita la cache.
    const uint32_t q_dim = cfg_.n_head * cfg_.head_dim;
    // I tensori Q4_0/Q4_K restano quantizzati in RAM (vedi load_matrix):
    // stimare sempre 4 byte/peso come per il float sovrastimerebbe di
    // parecchio l'ingombro reale e disabiliterebbe la cache anche quando
    // ci starebbe comodamente. Si sonda il formato reale di un tensore
    // rappresentativo (attn_q del layer 0) e si usa il suo rapporto
    // byte/peso effettivo al posto di sizeof(float) fisso.
    double bytes_per_weight = (double) sizeof(float);
    {
        std::vector<uint8_t> probe_raw;
        int probe_type = 0;
        uint64_t probe_ne0 = 0, probe_rows = 0;
        if (rd.read_tensor_raw("blk.0.attn_q.weight", probe_raw, probe_type, probe_ne0, probe_rows) &&
            probe_ne0 > 0 && probe_rows > 0) {
            bytes_per_weight = (double) probe_raw.size() / ((double) probe_ne0 * (double) probe_rows);
        }
    }

    // Per i layer MoE non si carica wff_gate/up/down (sostituiti dal
    // router, piccolo: n_expert*n_embd) e i pesi esperto vivono nella
    // cache LRU separata di ExpertStore, non in layer_cache_: non li si
    // conta qui.
    const uint64_t ffn_term = cfg_.n_expert > 0
        ? (uint64_t) cfg_.n_expert * cfg_.n_embd
        : (uint64_t) cfg_.n_ff * cfg_.n_embd * 3;
    const uint64_t per_layer_weights =
        (uint64_t) q_dim * cfg_.n_embd                 // wq
        + (uint64_t) kv_dim * cfg_.n_embd * 2           // wk + wv
        + (uint64_t) cfg_.n_embd * q_dim                // wo
        + ffn_term;
    const uint64_t norm_floats = (uint64_t) cfg_.n_embd * 2; // attn_norm + ffn_norm, sempre float
    const uint64_t total_bytes = cfg_.n_layers *
        ((uint64_t) (per_layer_weights * bytes_per_weight) + norm_floats * sizeof(float));
    const uint64_t budget_bytes = ram_budget_mb * 1024ULL * 1024ULL;
    cache_enabled_ = budget_bytes > 0 && total_bytes <= (budget_bytes * 6 / 10);
    layer_cache_.clear();
    if (cache_enabled_) layer_cache_.resize(cfg_.n_layers);

    cache_capacity_ = 0;
    cache_cols_ = 0;
    grow_cache(512);
    return true;
}

void DenseForward::reset_cache() {
    cache_cols_ = 0;
}

bool DenseForward::load_embd(ModelReader& rd) {
    if (!load_matrix(rd, "token_embd.weight", cfg_.n_vocab, cfg_.n_embd, tok_embd_)) return false;

    // Posizione assoluta (gpt2, mpt opzionale): letta in modo difensivo,
    // vuota se il tensore non c'e' (nessun impatto sulle altre architetture).
    pos_embd_.clear();
    if (n_ctx_train_ > 0) {
        rd.read_tensor("position_embd.weight", pos_embd_);
        if (pos_embd_.size() != (size_t) n_ctx_train_ * cfg_.n_embd) pos_embd_.clear();
    }

    // Norm iniziale sugli embedding (bloom, DenseQuirks::embd_norm).
    tok_norm_.clear();
    tok_norm_b_.clear();
    if (quirks_.embd_norm) {
        if (!rd.read_tensor("token_embd_norm.weight", tok_norm_) || tok_norm_.size() != cfg_.n_embd) return false;
        if (quirks_.layer_norm) {
            rd.read_tensor("token_embd_norm.bias", tok_norm_b_);
            if (tok_norm_b_.size() != cfg_.n_embd) tok_norm_b_.clear();
        }
    }
    return true;
}

namespace {
int matvec_quant_type(MatVecFormat f) {
    switch (f) {
        case MatVecFormat::Q4_0: return DESIREEIA_QTYPE_Q4_0;
        case MatVecFormat::Q4_1: return DESIREEIA_QTYPE_Q4_1;
        case MatVecFormat::Q5_0: return DESIREEIA_QTYPE_Q5_0;
        case MatVecFormat::Q5_1: return DESIREEIA_QTYPE_Q5_1;
        case MatVecFormat::Q2_K: return DESIREEIA_QTYPE_Q2_K;
        case MatVecFormat::Q3_K: return DESIREEIA_QTYPE_Q3_K;
        case MatVecFormat::Q8_K: return DESIREEIA_QTYPE_Q8_K;
        case MatVecFormat::Q4_K: return DESIREEIA_QTYPE_Q4_K;
        case MatVecFormat::Q5_K: return DESIREEIA_QTYPE_Q5_K;
        case MatVecFormat::Q6_K: return DESIREEIA_QTYPE_Q6_K;
        case MatVecFormat::Q8_0: return DESIREEIA_QTYPE_Q8_0;
        default: return -1;
    }
}
}

bool DenseForward::embed_row(uint32_t idx, float* out) const {
    if (tok_embd_.format == MatVecFormat::Float) {
        const size_t off = (size_t) idx * cfg_.n_embd;
        if (off + cfg_.n_embd > tok_embd_.f.size()) return false;
        std::memcpy(out, tok_embd_.f.data() + off, cfg_.n_embd * sizeof(float));
        return true;
    }
    const int qtype = matvec_quant_type(tok_embd_.format);
    if (qtype < 0) return false;
    const size_t row_bytes = desireeia_row_size(qtype, (int64_t) cfg_.n_embd);
    if (row_bytes == 0) return false;
    const size_t off = (size_t) idx * row_bytes;
    if (off + row_bytes > tok_embd_.raw.size()) return false;
    return desireeia_dequantize_row(qtype, tok_embd_.raw.data() + off, out, (int64_t) cfg_.n_embd) == 0;
}

bool DenseForward::load_norms(ModelReader& rd) {
    out_norm_.clear();
    out_norm_b_.clear();
    if (!rd.read_tensor("output_norm.weight", out_norm_)) return false;
    if (quirks_.layer_norm) {
        rd.read_tensor("output_norm.bias", out_norm_b_);
        if (out_norm_b_.size() != cfg_.n_embd) out_norm_b_.clear();
    }
    return out_norm_.size() == cfg_.n_embd;
}

// Mappa (quant_type, cols) -> MatVecFormat diretto, se esiste un kernel
// quantizzato dedicato per quel formato/larghezza. Estratta da load_matrix
// per essere riusata anche da load_qkv_fused (falcon, DenseQuirks::fused_qkv)
// senza duplicare — E RISCHIARE DI DISALLINEARE — l'elenco dei formati
// supportati: stesso identico ordine/condizioni di prima, comportamento
// invariato per tutte le architetture gia' funzionanti.
static bool quant_format_for(int quant_type, uint32_t cols, MatVecFormat& fmt) {
    if (cols % 32 == 0 && quant_type == DESIREEIA_QTYPE_Q4_0) { fmt = MatVecFormat::Q4_0; return true; }
    if (cols % 256 == 0 && quant_type == DESIREEIA_QTYPE_Q4_K) { fmt = MatVecFormat::Q4_K; return true; }
    if (cols % 256 == 0 && quant_type == DESIREEIA_QTYPE_Q6_K) { fmt = MatVecFormat::Q6_K; return true; }
    if (cols % 32 == 0 && quant_type == DESIREEIA_QTYPE_Q8_0) { fmt = MatVecFormat::Q8_0; return true; }
    if (cols % 256 == 0 && quant_type == DESIREEIA_QTYPE_Q5_K) { fmt = MatVecFormat::Q5_K; return true; }
    if (cols % 32 == 0 && quant_type == DESIREEIA_QTYPE_Q4_1) { fmt = MatVecFormat::Q4_1; return true; }
    if (cols % 32 == 0 && quant_type == DESIREEIA_QTYPE_Q5_0) { fmt = MatVecFormat::Q5_0; return true; }
    if (cols % 32 == 0 && quant_type == DESIREEIA_QTYPE_Q5_1) { fmt = MatVecFormat::Q5_1; return true; }
    if (cols % 256 == 0 && quant_type == DESIREEIA_QTYPE_Q2_K) { fmt = MatVecFormat::Q2_K; return true; }
    if (cols % 256 == 0 && quant_type == DESIREEIA_QTYPE_Q3_K) { fmt = MatVecFormat::Q3_K; return true; }
    if (cols % 256 == 0 && quant_type == DESIREEIA_QTYPE_Q8_K) { fmt = MatVecFormat::Q8_K; return true; }
    return false;
}

// Opt-in: rewrite Q6_K tensors as Q4_K while loading them.
//
// Decode is bandwidth-bound — every weight is read once per token and the
// arithmetic per byte is tiny — so time spent is close to proportional to
// bytes moved. Q6_K costs 6.5625 bits per weight against Q4_K's 4.5, so a
// tensor that moves to Q4_K takes about 31% less time to stream.
//
// This is OFF by default because it is a real quality trade, not a free win,
// and one the model's author already considered: a Q4_K_M file deliberately
// keeps the output projection and some FFN tensors at Q6_K precisely because
// they are the ones that suffer most from 4-bit. Turning this on trades some
// of that back for speed. It exists so the trade can be measured on a real
// model rather than argued about.
//
// DESIREEIA_REQUANT_Q6K=1 to enable.
static bool requantize_q6k_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("DESIREEIA_REQUANT_Q6K");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

// Q6_K bytes in, Q4_K bytes out, one row at a time. Returns false and leaves
// `raw` untouched on any shape it cannot handle, so the caller keeps the
// original tensor rather than a half-converted one.
static bool requantize_q6k_to_q4k(std::vector<uint8_t>& raw, uint32_t rows, uint32_t cols) {
    if (cols == 0 || cols % QK_K != 0 || rows == 0) return false;
    const size_t n_super   = cols / QK_K;
    const size_t src_row   = n_super * sizeof(block_q6_K);
    const size_t dst_row   = n_super * sizeof(block_q4_K);
    if (raw.size() != src_row * rows) return false;

    std::vector<uint8_t> out(dst_row * rows);
    std::atomic<bool> ok{true};

    // Parallel across rows: the least-squares fit costs real work per weight,
    // and the tensor this matters for is the output projection — 671M weights
    // on the model this was built against. Serially that is seconds of extra
    // load time; spread over the pool it is a fraction of one. Rows are
    // independent and write to disjoint output, so there is nothing to guard.
    parallel_rows(rows, [&](size_t r0, size_t r1) {
        std::vector<float> row((size_t) cols);
        for (size_t r = r0; r < r1; ++r) {
            if (desireeia_dequantize_row(DESIREEIA_QTYPE_Q6_K, raw.data() + r * src_row,
                                         row.data(), (int64_t) cols) != DESIREEIA_OK) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
            quantize_row_q4_K(row.data(),
                              reinterpret_cast<block_q4_K*>(out.data() + r * dst_row),
                              (int64_t) cols);
        }
    });

    if (!ok.load(std::memory_order_relaxed)) return false;
    raw.swap(out);
    return true;
}

bool DenseForward::load_matrix(ModelReader& rd, const std::string& name, uint32_t rows, uint32_t cols, MatVec& out) {
    out.raw.clear();
    out.f.clear();
    out.format = MatVecFormat::Float;

    std::vector<uint8_t> raw;
    int quant_type = 0;
    uint64_t ne0 = 0, tensor_rows = 0;
    if (rd.read_tensor_raw(name, raw, quant_type, ne0, tensor_rows) &&
        ne0 == cols && tensor_rows == rows) {
        MatVecFormat fmt;
        if (quant_format_for(quant_type, cols, fmt)) {
            if (fmt == MatVecFormat::Q6_K && requantize_q6k_enabled() &&
                requantize_q6k_to_q4k(raw, rows, cols)) {
                fmt = MatVecFormat::Q4_K;
            }
            out.raw = std::move(raw);
            out.format = fmt;
            return true;
        }
    }

    // Formato non gestito da un kernel diretto (o read_tensor_raw non
    // supportato dal reader): fallback sul path dequantizzato in float,
    // sempre corretto per qualunque formato quant letto dal reader.
    if (!rd.read_tensor(name, out.f) || out.f.size() != (size_t) rows * cols) return false;
    return true;
}

// falcon (DenseQuirks::fused_qkv): a single attn_qkv.weight tensor, rows
// [0,q_dim)=Q, [q_dim,q_dim+kv_dim)=K, [q_dim+kv_dim,q_dim+2*kv_dim)=V.
// When the format has a direct quantized kernel, each row occupies
// exactly raw.size()/total_rows bytes (no quantization block ever spans a
// row boundary), so splitting by row is a plain byte slice — using the
// same format logic as load_matrix/quant_format_for.
bool DenseForward::load_qkv_fused(ModelReader& rd, const std::string& name,
                                   uint32_t q_dim, uint32_t kv_dim, uint32_t cols,
                                   MatVec& wq, MatVec& wk, MatVec& wv) {
    wq = MatVec{}; wk = MatVec{}; wv = MatVec{};
    const uint32_t total_rows = q_dim + 2 * kv_dim;

    std::vector<uint8_t> raw;
    int quant_type = 0;
    uint64_t ne0 = 0, tensor_rows = 0;
    if (rd.read_tensor_raw(name, raw, quant_type, ne0, tensor_rows) &&
        ne0 == cols && tensor_rows == total_rows) {
        MatVecFormat fmt;
        if (quant_format_for(quant_type, cols, fmt) && total_rows > 0 && raw.size() % total_rows == 0) {
            const size_t row_bytes = raw.size() / total_rows;
            wq.raw.assign(raw.begin(), raw.begin() + (size_t) q_dim * row_bytes);
            wq.format = fmt;
            wk.raw.assign(raw.begin() + (size_t) q_dim * row_bytes,
                          raw.begin() + (size_t) (q_dim + kv_dim) * row_bytes);
            wk.format = fmt;
            wv.raw.assign(raw.begin() + (size_t) (q_dim + kv_dim) * row_bytes, raw.end());
            wv.format = fmt;
            return true;
        }
    }

    std::vector<float> flat;
    if (!rd.read_tensor(name, flat) || flat.size() != (size_t) total_rows * cols) return false;
    wq.f.assign(flat.begin(), flat.begin() + (size_t) q_dim * cols);
    wk.f.assign(flat.begin() + (size_t) q_dim * cols, flat.begin() + (size_t) (q_dim + kv_dim) * cols);
    wv.f.assign(flat.begin() + (size_t) (q_dim + kv_dim) * cols, flat.end());
    wq.format = wk.format = wv.format = MatVecFormat::Float;
    return true;
}

bool DenseForward::load_layer_data(ModelReader& rd, uint32_t il, LayerWeights& w) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%u.", il);
    const std::string p = buf;

    // no_pre_norm (olmo2, exaone4): questi tensori non esistono affatto nel
    // GGUF (solo i post-norm sandwich li sostituiscono, letti sotto), quindi
    // non vanno richiesti come obbligatori.
    if (quirks_.no_pre_norm) {
        w.attn_norm.clear();
        w.ffn_norm.clear();
    } else if (quirks_.parallel_residual) {
        // cohere2: un solo norm (attn_norm) alimenta SIA l'attenzione SIA
        // la FFN — ffn_norm.weight non esiste affatto nel GGUF.
        if (!rd.read_tensor(p + "attn_norm.weight", w.attn_norm) || w.attn_norm.size() != cfg_.n_embd) return false;
        w.ffn_norm.clear();
    } else {
        if (!rd.read_tensor(p + "attn_norm.weight", w.attn_norm) || w.attn_norm.size() != cfg_.n_embd) return false;
        if (!rd.read_tensor(p + "ffn_norm.weight", w.ffn_norm) || w.ffn_norm.size() != cfg_.n_embd) return false;
    }

    // Bias della norma: solo le architetture LayerNorm (stablelm, orion,
    // ecc.) lo hanno. Letto anche se assente non fa danno (read_tensor
    // fallisce, il vettore resta vuoto -> nullptr in norm_vec), ma si
    // risparmia la lettura inutile sulle architetture RMSNorm.
    w.attn_norm_b.clear();
    w.ffn_norm_b.clear();
    if (quirks_.layer_norm) {
        rd.read_tensor(p + "attn_norm.bias", w.attn_norm_b);
        rd.read_tensor(p + "ffn_norm.bias", w.ffn_norm_b);
        if (w.attn_norm_b.size() != cfg_.n_embd) w.attn_norm_b.clear();
        if (w.ffn_norm_b.size() != cfg_.n_embd) w.ffn_norm_b.clear();
    }

    // The Q projection width (n_head*head_dim) can differ from n_embd
    // (e.g. gemma3: the declared head_dim isn't n_embd/n_head) — never
    // assume it's square like the simpler architectures.
    const uint32_t q_dim = cfg_.n_head * cfg_.head_dim;
    const uint32_t kv_dim = cfg_.n_head_kv * cfg_.head_dim;

    if (quirks_.mla) {
        // Vedi la nota estesa su DenseQuirks::mla e mla_attn_layer(): tensori
        // specifici MLA, sostituiscono interamente wq/wk/wv/attn_{q,k}.bias.
        w.wq = MatVec{}; w.wk = MatVec{}; w.wv = MatVec{};
        w.bq.clear(); w.bk.clear(); w.bv.clear(); w.bo.clear();

        if (cfg_.q_lora_rank > 0) {
            if (!load_matrix(rd, p + "attn_q_a.weight", cfg_.q_lora_rank, cfg_.n_embd, w.wq_a)) return false;
            if (!rd.read_tensor(p + "attn_q_a_norm.weight", w.attn_q_a_norm) || w.attn_q_a_norm.size() != cfg_.q_lora_rank) return false;
            if (!load_matrix(rd, p + "attn_q_b.weight", cfg_.n_head * (cfg_.n_embd_head_qk_nope + cfg_.n_embd_head_qk_rope), cfg_.q_lora_rank, w.wq_b)) return false;
            w.wq_lite = MatVec{};
        } else {
            if (!load_matrix(rd, p + "attn_q.weight", cfg_.n_head * (cfg_.n_embd_head_qk_nope + cfg_.n_embd_head_qk_rope), cfg_.n_embd, w.wq_lite)) return false;
            w.wq_a = MatVec{}; w.wq_b = MatVec{};
            w.attn_q_a_norm.clear();
        }

        if (!load_matrix(rd, p + "attn_kv_a_mqa.weight", cfg_.kv_lora_rank + cfg_.n_embd_head_qk_rope, cfg_.n_embd, w.wkv_a_mqa)) return false;
        if (!rd.read_tensor(p + "attn_kv_a_norm.weight", w.attn_kv_a_norm) || w.attn_kv_a_norm.size() != cfg_.kv_lora_rank) return false;

        // wk_b/wv_b: un blocco contiguo per testa (vedi la nota sul layout
        // in dense_forward.h). Letti come float piatto e poi suddivisi:
        // niente kernel quantizzato dedicato per queste matrici, sempre
        // corrette (fallback float), dimensione modesta rispetto al resto.
        {
            std::vector<float> wk_b_flat;
            if (!rd.read_tensor(p + "attn_k_b.weight", wk_b_flat) ||
                wk_b_flat.size() != (size_t) cfg_.n_embd_head_qk_nope * cfg_.kv_lora_rank * cfg_.n_head) return false;
            w.wk_b_h.assign(cfg_.n_head, MatVec{});
            const size_t chunk = (size_t) cfg_.n_embd_head_qk_nope * cfg_.kv_lora_rank;
            for (uint32_t h = 0; h < cfg_.n_head; ++h) {
                w.wk_b_h[h].f.assign(wk_b_flat.begin() + h * chunk, wk_b_flat.begin() + (h + 1) * chunk);
                w.wk_b_h[h].format = MatVecFormat::Float;
            }
        }
        {
            std::vector<float> wv_b_flat;
            if (!rd.read_tensor(p + "attn_v_b.weight", wv_b_flat) ||
                wv_b_flat.size() != (size_t) cfg_.kv_lora_rank * cfg_.n_embd_head_v_mla * cfg_.n_head) return false;
            w.wv_b_h.assign(cfg_.n_head, MatVec{});
            const size_t chunk = (size_t) cfg_.kv_lora_rank * cfg_.n_embd_head_v_mla;
            for (uint32_t h = 0; h < cfg_.n_head; ++h) {
                w.wv_b_h[h].f.assign(wv_b_flat.begin() + h * chunk, wv_b_flat.begin() + (h + 1) * chunk);
                w.wv_b_h[h].format = MatVecFormat::Float;
            }
        }

        const uint32_t mla_out_dim = cfg_.n_head * cfg_.n_embd_head_v_mla;
        if (!load_matrix(rd, p + "attn_output.weight", cfg_.n_embd, mla_out_dim, w.wo)) return false;
    } else if (quirks_.fused_qkv) {
        if (!load_qkv_fused(rd, p + "attn_qkv.weight", q_dim, kv_dim, cfg_.n_embd, w.wq, w.wk, w.wv)) return false;
        if (!load_matrix(rd, p + "attn_output.weight", cfg_.n_embd, q_dim, w.wo)) return false;
        w.bo.clear();
        rd.read_tensor(p + "attn_output.bias", w.bo);
        if (w.bo.size() != cfg_.n_embd) w.bo.clear();

        w.bq.clear(); w.bk.clear(); w.bv.clear();
        if (quirks_.qkv_bias) {
            // gpt2/bloom/mpt: un solo bias fuso (attn_qkv.bias), stesso
            // ordine [Q|K|V] del tensore peso fuso — split per intervallo.
            std::vector<float> bqkv;
            rd.read_tensor(p + "attn_qkv.bias", bqkv);
            if (bqkv.size() == (size_t) q_dim + 2 * kv_dim) {
                w.bq.assign(bqkv.begin(), bqkv.begin() + q_dim);
                w.bk.assign(bqkv.begin() + q_dim, bqkv.begin() + q_dim + kv_dim);
                w.bv.assign(bqkv.begin() + q_dim + kv_dim, bqkv.end());
            } else {
                w.bq.assign(q_dim, 0.0f);
                w.bk.assign(kv_dim, 0.0f);
                w.bv.assign(kv_dim, 0.0f);
            }
        } // falcon (qkv_bias=false): nessun bias, resta vuoto
    } else {
        if (!load_matrix(rd, p + "attn_q.weight", q_dim, cfg_.n_embd, w.wq)) return false;
        if (!load_matrix(rd, p + "attn_k.weight", kv_dim, cfg_.n_embd, w.wk)) return false;
        if (!load_matrix(rd, p + "attn_v.weight", kv_dim, cfg_.n_embd, w.wv)) return false;
        if (!load_matrix(rd, p + "attn_output.weight", cfg_.n_embd, q_dim, w.wo)) return false;
        w.bo.clear();
        rd.read_tensor(p + "attn_output.bias", w.bo);
        if (w.bo.size() != cfg_.n_embd) w.bo.clear();

        if (quirks_.qkv_bias) {
            w.bq.clear(); w.bk.clear(); w.bv.clear();
            rd.read_tensor(p + "attn_q.bias", w.bq);
            rd.read_tensor(p + "attn_k.bias", w.bk);
            rd.read_tensor(p + "attn_v.bias", w.bv);
            if (w.bq.size() != q_dim) w.bq.assign(q_dim, 0.0f);
            if (w.bk.size() != kv_dim) w.bk.assign(kv_dim, 0.0f);
            if (w.bv.size() != kv_dim) w.bv.assign(kv_dim, 0.0f);
        } else {
            w.bq.clear(); w.bk.clear(); w.bv.clear();
        }
    }

    // Secondo norm opzionale per l'attenzione (falcon-40B, vedi la nota su
    // LayerWeights::attn_norm2): assente ovunque tranne quel checkpoint
    // specifico, letto in modo difensivo.
    w.attn_norm2.clear(); w.attn_norm2_b.clear();
    if (quirks_.parallel_residual) {
        rd.read_tensor(p + "attn_norm_2.weight", w.attn_norm2);
        if (w.attn_norm2.size() != cfg_.n_embd) w.attn_norm2.clear();
        if (!w.attn_norm2.empty() && quirks_.layer_norm) {
            rd.read_tensor(p + "attn_norm_2.bias", w.attn_norm2_b);
            if (w.attn_norm2_b.size() != cfg_.n_embd) w.attn_norm2_b.clear();
        }
    }

    // Layer "dense-lead" (deepseek2/3, vedi DenseConfig::n_layer_dense_lead):
    // sotto quella soglia FFN gated classica anche se n_expert>0 globalmente.
    const bool layer_is_moe = cfg_.n_expert > 0 && il >= cfg_.n_layer_dense_lead;
    if (layer_is_moe) {
        // Layer MoE: router al posto del FFN denso, gli esperti veri
        // vengono presi da ExpertStore per-token in step() (dipendono dal
        // routing, non caricabili qui una volta per layer).
        w.wff_gate = MatVec{}; w.wff_up = MatVec{}; w.wff_down = MatVec{};
        if (!load_matrix(rd, p + "ffn_gate_inp.weight", cfg_.n_expert, cfg_.n_embd, w.router)) return false;
        w.router_bias.clear();
        if (quirks_.mla) {
            rd.read_tensor(p + "exp_probs_b.bias", w.router_bias);
            if (w.router_bias.size() != cfg_.n_expert) w.router_bias.clear();
        }
        w.ffn_gate_shexp = MatVec{}; w.ffn_up_shexp = MatVec{}; w.ffn_down_shexp = MatVec{};
        if (quirks_.mla && cfg_.n_expert_shared > 0) {
            const uint32_t n_ff_sh = cfg_.n_ff_expert * cfg_.n_expert_shared;
            if (!load_matrix(rd, p + "ffn_gate_shexp.weight", n_ff_sh, cfg_.n_embd, w.ffn_gate_shexp)) return false;
            if (!load_matrix(rd, p + "ffn_up_shexp.weight", n_ff_sh, cfg_.n_embd, w.ffn_up_shexp)) return false;
            if (!load_matrix(rd, p + "ffn_down_shexp.weight", cfg_.n_embd, n_ff_sh, w.ffn_down_shexp)) return false;
        }
    } else {
        w.router = MatVec{};
        w.router_bias.clear();
        w.ffn_gate_shexp = MatVec{}; w.ffn_up_shexp = MatVec{}; w.ffn_down_shexp = MatVec{};
        if (quirks_.ffn_gated) {
            if (!load_matrix(rd, p + "ffn_gate.weight", cfg_.n_ff, cfg_.n_embd, w.wff_gate)) return false;
        } else {
            w.wff_gate = MatVec{};
        }
        if (!load_matrix(rd, p + "ffn_up.weight", cfg_.n_ff, cfg_.n_embd, w.wff_up)) return false;
        if (!load_matrix(rd, p + "ffn_down.weight", cfg_.n_embd, cfg_.n_ff, w.wff_down)) return false;

        w.ffn_up_b.clear();
        w.ffn_down_b.clear();
        if (!quirks_.ffn_gated) {
            rd.read_tensor(p + "ffn_up.bias", w.ffn_up_b);
            rd.read_tensor(p + "ffn_down.bias", w.ffn_down_b);
            if (w.ffn_up_b.size() != cfg_.n_ff) w.ffn_up_b.clear();
            if (w.ffn_down_b.size() != cfg_.n_embd) w.ffn_down_b.clear();
        }
    }

    if (quirks_.qk_norm) {
        // olmo2 (qk_norm_full_width): norm applied over the entire Q/K
        // vector (width q_dim/kv_dim) BEFORE the per-head reshape — the
        // tensor width is n_embd/n_head_kv*n_embd_head, not head_dim.
        const uint32_t q_norm_dim = quirks_.qk_norm_full_width ? q_dim : cfg_.head_dim;
        const uint32_t k_norm_dim = quirks_.qk_norm_full_width ? kv_dim : cfg_.head_dim;
        if (!rd.read_tensor(p + "attn_q_norm.weight", w.q_norm) || w.q_norm.size() != q_norm_dim) return false;
        if (!rd.read_tensor(p + "attn_k_norm.weight", w.k_norm) || w.k_norm.size() != k_norm_dim) return false;
    } else {
        w.q_norm.clear(); w.k_norm.clear();
    }
    if (quirks_.sandwich_norm) {
        if (!rd.read_tensor(p + "post_attention_norm.weight", w.post_attn_norm) || w.post_attn_norm.size() != cfg_.n_embd) return false;
        if (!rd.read_tensor(p + "post_ffw_norm.weight", w.post_ffn_norm) || w.post_ffn_norm.size() != cfg_.n_embd) return false;
    } else {
        w.post_attn_norm.clear(); w.post_ffn_norm.clear();
    }
    return true;
}

const LayerWeights* DenseForward::get_layer(ModelReader& rd, uint32_t il) {
    if (cache_enabled_) {
        LayerWeights& w = layer_cache_[il];
        // Sentinella "layer non ancora caricato": wq e' sempre presente in
        // OGNI architettura (attn_norm invece resta vuoto per le
        // architetture no_pre_norm come olmo2/exaone4, dove non
        // ricaricherebbe mai la cache se usato come sentinella qui). MLA
        // (deepseek2/3) non usa affatto wq (usa wq_a/wq_b o wq_lite): serve
        // un secondo controllo su wkv_a_mqa, sempre presente li'.
        if (w.wq.empty() && w.wkv_a_mqa.empty()) {
            if (!load_layer_data(rd, il, w)) return nullptr;
        }
        return &w;
    }
    if (!load_layer_data(rd, il, scratch_)) return nullptr;
    return &scratch_;
}

void DenseForward::grow_cache(size_t needed) {
    if (needed <= cache_capacity_) return;
    const uint32_t kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    const size_t cols = cache_capacity_ > 0 ? cache_capacity_ : 1;
    const size_t new_cols = std::max(needed, cols * 2);

    std::vector<float> nk(cfg_.n_layers * new_cols * kv_dim, 0.0f);
    std::vector<float> nv(cfg_.n_layers * new_cols * kv_dim, 0.0f);
    if (cache_capacity_ > 0) {
        // BUG CORRETTO (scoperto durante il lavoro MLA, 2026-09-08): una
        // singola memcpy sull'intero buffer preserva solo il layer 0 quando
        // lo stride per-layer cambia (vecchio stride = cache_capacity_*kv_dim,
        // nuovo = new_cols*kv_dim): i layer successivi finivano copiati con
        // l'offset sbagliato ogni volta che grow_cache veniva chiamata con
        // cache_capacity_ gia' > 0 (cioe' oltre una singola crescita da 0).
        // Serve una copia layer per layer, ciascuno al proprio nuovo offset.
        for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
            std::memcpy(nk.data() + (size_t) l * new_cols * kv_dim,
                        k_cache_.data() + (size_t) l * cache_capacity_ * kv_dim,
                        cache_capacity_ * kv_dim * sizeof(float));
            std::memcpy(nv.data() + (size_t) l * new_cols * kv_dim,
                        v_cache_.data() + (size_t) l * cache_capacity_ * kv_dim,
                        cache_capacity_ * kv_dim * sizeof(float));
        }
    }
    k_cache_.swap(nk);
    v_cache_.swap(nv);
    cache_capacity_ = new_cols;
}

// MLA (DeepSeek2 and siblings): compressed-rank "absorbed" attention, see
// the derivation in the comment on DenseQuirks::mla in arch_tags.h. Only
// the absorbed form is implemented — see the note on ArchKind::DeepSeek2.
//
// K/V cache layout: cfg_.head_dim was set to kv_lora_rank+rope and
// cfg_.n_head_kv to 1 in open() (multi-query: a single entry shared
// across every head), so k_cache_ already has exactly the right shape to
// hold, per position, [kv_cmpr(kv_lora_rank) | k_pe_roped(rope)]. V
// doesn't need a separate cache: Vcur == kv_cmpr, i.e. the first
// kv_lora_rank floats of that SAME k_cache_ row (v_cache_ stays unused
// for MLA layers).
void DenseForward::mla_attn_layer(const LayerWeights* lw, uint32_t l, uint32_t pos,
                                   const float* xnp, float rope_th, float rope_sc,
                                   float ext_factor, float attn_factor, float corr_lo, float corr_hi,
                                   float kq_scale, float* proj_out) {
    const uint32_t n_embd = cfg_.n_embd;
    const uint32_t n_head = cfg_.n_head;
    const uint32_t kv_lora = cfg_.kv_lora_rank;
    const uint32_t rope_w = cfg_.n_embd_head_qk_rope;
    const uint32_t nope_w = cfg_.n_embd_head_qk_nope;
    const uint32_t v_mla = cfg_.n_embd_head_v_mla;
    const uint32_t comp_w = kv_lora + rope_w; // larghezza "compressa" condivisa da Q e K
    const uint32_t k_mla = nope_w + rope_w;

    // --- Q: proiezione (LoRA o diretta) -> [n_head * k_mla] ---
    std::vector<float> q(n_head * (size_t) k_mla);
    if (cfg_.q_lora_rank > 0) {
        std::vector<float> qa(cfg_.q_lora_rank);
        matvec(lw->wq_a, cfg_.q_lora_rank, n_embd, xnp, qa.data());
        rms_norm_vec(qa.data(), lw->attn_q_a_norm.data(), qa.data(), cfg_.q_lora_rank, cfg_.rms_eps);
        matvec(lw->wq_b, n_head * (size_t) k_mla, cfg_.q_lora_rank, qa.data(), q.data());
    } else {
        matvec(lw->wq_lite, n_head * (size_t) k_mla, n_embd, xnp, q.data());
    }

    // --- KV compresso + k_pe (condiviso da tutte le teste, MQA) ---
    std::vector<float> kv_cmpr_pe(kv_lora + rope_w);
    matvec(lw->wkv_a_mqa, kv_lora + rope_w, n_embd, xnp, kv_cmpr_pe.data());

    float* cache_row = k_cache_.data() + ((size_t) l * cache_capacity_ + pos) * comp_w;
    rms_norm_vec(kv_cmpr_pe.data(), lw->attn_kv_a_norm.data(), cache_row, kv_lora, cfg_.rms_eps);

    std::vector<float> rope_cache_kv(rope_w);
    rope_cache_init(rope_cache_kv, rope_w, pos, rope_th, rope_sc, ext_factor, attn_factor, corr_lo, corr_hi);
    std::memcpy(cache_row + kv_lora, kv_cmpr_pe.data() + kv_lora, rope_w * sizeof(float));
    rope_neox_cached(cache_row + kv_lora, rope_w, rope_cache_kv.data());

    // --- per-testa: assorbimento di q_nope, RoPE su q_pe, punteggi, softmax, de-assorbimento ---
    std::vector<float> attn_concat((size_t) n_head * v_mla);
    std::vector<float> scores((size_t) pos + 1);
    std::vector<float> qcur(comp_w);
    std::vector<float> attn_raw(kv_lora);
    for (uint32_t h = 0; h < n_head; ++h) {
        const float* qh = q.data() + (size_t) h * k_mla;
        // Assorbimento: q_nope_absorbed[j] = sum_i wk_b[i,j,h] * q_nope[i],
        // j in [0,kv_lora) — vedi la derivazione dal layout ggml nella nota
        // su LayerWeights::wk_b_h in dense_forward.h.
        matvec(lw->wk_b_h[h], kv_lora, nope_w, qh, qcur.data());
        std::memcpy(qcur.data() + kv_lora, qh + nope_w, rope_w * sizeof(float));
        rope_neox_cached(qcur.data() + kv_lora, rope_w, rope_cache_kv.data());

        const float* cache_base = k_cache_.data() + (size_t) l * cache_capacity_ * comp_w;
        for (uint32_t cc = 0; cc <= pos; ++cc) {
            scores[cc] = dot_f32(qcur.data(), cache_base + (size_t) cc * comp_w, comp_w) * kq_scale;
        }
        softmax_inplace(scores.data(), (size_t) pos + 1);

        std::fill(attn_raw.begin(), attn_raw.end(), 0.0f);
        for (uint32_t cc = 0; cc <= pos; ++cc) {
            axpy_f32(attn_raw.data(), cache_base + (size_t) cc * comp_w, scores[cc], kv_lora);
        }
        // De-assorbimento: out[j] = sum_i wv_b[i,j,h] * attn_raw[i], j in [0,v_mla).
        matvec(lw->wv_b_h[h], v_mla, kv_lora, attn_raw.data(), attn_concat.data() + (size_t) h * v_mla);
    }

    matvec(lw->wo, n_embd, (size_t) n_head * v_mla, attn_concat.data(), proj_out);
}

bool DenseForward::step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
                        std::vector<float>& last_logits, std::vector<float>* all_logits) {
    if (n_tokens == 0 || tokens == nullptr) return false;
    if (tok_embd_.empty()) return false;

    const uint32_t n_embd  = cfg_.n_embd;
    const uint32_t n_layers = cfg_.n_layers;
    const uint32_t n_head  = cfg_.n_head;
    const uint32_t q_dim   = cfg_.n_head * cfg_.head_dim;
    const uint32_t kv_dim  = cfg_.n_head_kv * cfg_.head_dim;
    const float  rms_eps  = quirks_.layer_norm ? cfg_.norm_eps : cfg_.rms_eps;
    // rope_th/rope_sc sono ora per-layer (vedi DenseConfig::layer_rope_*):
    // vengono ricalcolati dentro il loop sui layer, non piu' una volta
    // sola qui — gemma3 alterna base RoPE fra layer locali e globali.
    const uint32_t col0 = (uint32_t) cache_cols_;

    if (col0 + n_tokens > cache_capacity_) {
        grow_cache(col0 + n_tokens);
    }

    std::vector<float> x((size_t) n_tokens * n_embd);
    for (size_t p = 0; p < n_tokens; ++p) {
        const int32_t t = tokens[p];
        if (t < 0 || (uint32_t) t >= cfg_.n_vocab) return false;
        if (!embed_row((uint32_t) t, x.data() + p * n_embd)) return false;
        if (quirks_.embd_scale_sqrt) {
            const float s = sqrtf((float) n_embd);
            float* xp = x.data() + p * n_embd;
            for (uint32_t i = 0; i < n_embd; ++i) xp[i] *= s;
        }
        // Posizione assoluta (gpt2, mpt opzionale): sommata UNA volta
        // all'embedding del token, prima del primo layer.
        if (!pos_embd_.empty()) {
            const uint32_t pos = col0 + (uint32_t) p;
            if (pos < n_ctx_train_) {
                float* xp = x.data() + p * n_embd;
                const float* pe = pos_embd_.data() + (size_t) pos * n_embd;
                for (uint32_t i = 0; i < n_embd; ++i) xp[i] += pe[i];
            }
        }
    }
    // Norm iniziale sugli embedding (bloom, DenseQuirks::embd_norm): UNA
    // volta sola, prima del primo layer — non per-layer.
    if (quirks_.embd_norm) {
        for (size_t p = 0; p < n_tokens; ++p) {
            float* xp = x.data() + p * n_embd;
            norm_vec(true, xp, tok_norm_.data(),
                    tok_norm_b_.empty() ? nullptr : tok_norm_b_.data(),
                    xp, n_embd, rms_eps);
        }
    }

    std::vector<float> xn((size_t) n_tokens * n_embd);
    // Solo se lw->attn_norm2 e' presente (falcon-40B, vedi la nota su
    // LayerWeights::attn_norm2): input di Q/K/V separato da quello di FFN.
    std::vector<float> xn_attn;
    std::vector<float> xnp_attn(n_embd);
    std::vector<float> q(q_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    // Buffer riusati dal path "quantizzazione condivisa" (vedi
    // quantize_shared_q8k sopra): dichiarati una volta per riusare la
    // capacita' allocata invece di riallocare a ogni layer/token.
    std::vector<int8_t> qk_xq;
    std::vector<float> qk_dscale;
    std::vector<int32_t> qk_xsum;
    std::vector<float> rope_cache;  // cos/sin interleavati, uno per layer
    std::vector<float> attn_out(q_dim);
    const uint32_t ffn_buf_n = std::max(cfg_.n_ff, cfg_.n_ff_expert);
    std::vector<float> ffn((size_t) ffn_buf_n);
    std::vector<float> ffn_gate((size_t) ffn_buf_n);
    std::vector<float> scores;
    std::vector<float> proj(n_embd);
    std::vector<float> fout(n_embd);
    std::vector<float> hn(n_embd);

    // Buffer per il ramo MoE (usati solo se cfg_.n_expert > 0).
    std::vector<float> router_logits(cfg_.n_expert);
    std::vector<float> expert_out(n_embd);
    std::vector<float> egate, eup, edown;

    // Buffer per il ramo MLA (usati solo se quirks_.mla).
    std::vector<float> mla_xnp(n_embd);
    std::vector<float> mla_proj(n_embd);
    std::vector<float> mla_fnorm(n_embd);
    std::vector<float> mla_fout(n_embd);
    std::vector<float> mla_shexp_gate, mla_shexp_up;
    if (quirks_.mla && cfg_.n_expert_shared > 0) {
        mla_shexp_gate.resize((size_t) cfg_.n_ff_expert * cfg_.n_expert_shared);
        mla_shexp_up.resize((size_t) cfg_.n_ff_expert * cfg_.n_expert_shared);
    }
    // Precalcolo dei parametri YaRN/kq_scale MLA (costanti per l'intero
    // modello: deepseek non alterna base RoPE per layer come gemma3).
    // Vedi la nota estesa in DenseConfig::layer_rope_corr_dims e la
    // derivazione di kq_scale nel commento su open() (mscale/attn_factor_org).
    float mla_corr_lo = 0.0f, mla_corr_hi = 0.0f, mla_kq_scale = 0.0f;
    if (quirks_.mla) {
        cfg_.layer_rope_corr_dims(0, mla_corr_lo, mla_corr_hi);
        const float attn_factor_org = cfg_.rope_attn_factor * (1.0f + 0.1f * logf(1.0f / cfg_.rope_freq_scale));
        const float mscale = attn_factor_org * (1.0f + 0.1f * cfg_.rope_yarn_log_mul * logf(1.0f / cfg_.rope_freq_scale));
        mla_kq_scale = mscale * mscale / sqrtf((float) (cfg_.n_embd_head_qk_nope + cfg_.n_embd_head_qk_rope));
    }

    // Buffer per il path batched (Fase 8, usato solo quando n_tokens > 1,
    // cioe' in prefill: il decode a 1 token resta sul path per-token sotto,
    // gia' misurato e validato, per non introdurre rischio/overhead sul
    // caso critico per i tok/s). Le proiezioni Q/K/V/wo e la FFN densa
    // vengono calcolate con UNA chiamata batched per l'intero layer invece
    // di n_tokens chiamate separate: i byte grezzi del peso vengono decodi-
    // ficati una sola volta e riusati per tutte le colonne (vedi
    // matmul_qX_k_batch in core/matmul.cpp). L'attenzione resta per-token
    // (dipendenza sequenziale dalla KV cache/maschera causale), ma quella
    // non e' il collo di bottiglia dominante per matrici grandi.
    std::vector<float> q_all, k_all, v_all, attn_out_all, proj_all, ffn_xn_all,
                        ffn_all, ffn_gate_all, fout_all;
    std::vector<float> rope_cache_b;  // cos/sin del percorso batch
    if (n_tokens > 1) {
        q_all.resize((size_t) n_tokens * q_dim);
        k_all.resize((size_t) n_tokens * kv_dim);
        v_all.resize((size_t) n_tokens * kv_dim);
        attn_out_all.resize((size_t) n_tokens * q_dim);
        proj_all.resize((size_t) n_tokens * n_embd);
        ffn_xn_all.resize((size_t) n_tokens * n_embd);
        if (cfg_.n_expert == 0) {
            ffn_all.resize((size_t) n_tokens * cfg_.n_ff);
            ffn_gate_all.resize((size_t) n_tokens * cfg_.n_ff);
            fout_all.resize((size_t) n_tokens * n_embd);
        }
    }

    for (uint32_t l = 0; l < n_layers; ++l) {
        const LayerWeights* lw = get_layer(rd, l);
        if (!lw) return false;

        // RoPE per-layer: i layer locali (sliding window) e quelli globali
        // usano base/scala diverse — vedi la nota in DenseConfig.
        const float rope_th = cfg_.layer_rope_theta(l);
        const float rope_sc = cfg_.layer_rope_scale(l);

        if (quirks_.mla) {
            // Un solo percorso per-token, riusato sia per il prefill
            // (n_tokens>1) sia per il decode: MLA non ha un equivalente
            // "batched" qui (ogni token richiede comunque le due matvec
            // per-testa di assorbimento/de-assorbimento, vedi
            // mla_attn_layer), quindi non c'e' un matmul grande da
            // batchare come nel path denso/GQA classico.
            for (size_t p = 0; p < n_tokens; ++p) {
                const uint32_t pos = col0 + (uint32_t) p;
                const float* xp = x.data() + p * n_embd;

                norm_vec(false, xp, lw->attn_norm.data(), nullptr, mla_xnp.data(), n_embd, rms_eps);

                mla_attn_layer(lw, l, pos, mla_xnp.data(), rope_th, rope_sc,
                               cfg_.rope_ext_factor, cfg_.rope_attn_factor, mla_corr_lo, mla_corr_hi,
                               mla_kq_scale, mla_proj.data());

                for (uint32_t i = 0; i < n_embd; ++i) mla_proj[i] += xp[i];
                norm_vec(false, mla_proj.data(), lw->ffn_norm.data(), nullptr, mla_fnorm.data(), n_embd, rms_eps);

                const bool layer_is_moe = cfg_.n_expert > 0 && l >= cfg_.n_layer_dense_lead;
                if (layer_is_moe) {
                    matvec(lw->router, cfg_.n_expert, n_embd, mla_fnorm.data(), router_logits.data());
                    auto selected = moe_route_ex(router_logits, cfg_.n_expert_used, cfg_.moe_norm_w, cfg_.moe_w_scale,
                                                  cfg_.moe_sigmoid_gate ? MoeGatingFunc::Sigmoid : MoeGatingFunc::Softmax,
                                                  lw->router_bias.empty() ? nullptr : &lw->router_bias);
                    std::fill(expert_out.begin(), expert_out.end(), 0.0f);
                    for (const auto& sel : selected) {
                        const uint32_t eidx = sel.first;
                        const float weight = sel.second;
                        if (!experts_->fetch(l, eidx, ExpertPart::Gate, egate)) continue;
                        if (!experts_->fetch(l, eidx, ExpertPart::Up, eup)) continue;
                        if (!experts_->fetch(l, eidx, ExpertPart::Down, edown)) continue;
                        if (egate.size() != (size_t) cfg_.n_ff_expert * n_embd ||
                            eup.size() != (size_t) cfg_.n_ff_expert * n_embd ||
                            edown.size() != (size_t) n_embd * cfg_.n_ff_expert) continue;

                        matmul_f32(eup.data(), cfg_.n_ff_expert, n_embd, mla_fnorm.data(), ffn.data());
                        matmul_f32(egate.data(), cfg_.n_ff_expert, n_embd, mla_fnorm.data(), ffn_gate.data());
                        for (uint32_t i = 0; i < cfg_.n_ff_expert; ++i) ffn[i] = silu(ffn_gate[i]) * ffn[i];
                        matmul_f32(edown.data(), n_embd, cfg_.n_ff_expert, ffn.data(), mla_fout.data());
                        for (uint32_t i = 0; i < n_embd; ++i) expert_out[i] += weight * mla_fout[i];
                    }
                    mla_fout = expert_out;

                    // Shared expert: every token ALSO passes through this
                    // fixed block, whose output is ADDED (not averaged)
                    // to the routed MoE output.
                    if (cfg_.n_expert_shared > 0) {
                        const uint32_t n_ff_sh = cfg_.n_ff_expert * cfg_.n_expert_shared;
                        matvec(lw->ffn_up_shexp, n_ff_sh, n_embd, mla_fnorm.data(), mla_shexp_up.data());
                        matvec(lw->ffn_gate_shexp, n_ff_sh, n_embd, mla_fnorm.data(), mla_shexp_gate.data());
                        for (uint32_t i = 0; i < n_ff_sh; ++i) mla_shexp_up[i] = silu(mla_shexp_gate[i]) * mla_shexp_up[i];
                        std::vector<float> shexp_out(n_embd);
                        matvec(lw->ffn_down_shexp, n_embd, n_ff_sh, mla_shexp_up.data(), shexp_out.data());
                        for (uint32_t i = 0; i < n_embd; ++i) mla_fout[i] += shexp_out[i];
                    }
                } else {
                    // Layer "dense-lead": FFN gated classica (SiLU), stessi
                    // tensori/formula gia' usati dalle altre architetture.
                    matvec(lw->wff_up, cfg_.n_ff, n_embd, mla_fnorm.data(), ffn.data());
                    matvec(lw->wff_gate, cfg_.n_ff, n_embd, mla_fnorm.data(), ffn_gate.data());
                    for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffn[i] = silu(ffn_gate[i]) * ffn[i];
                    matvec(lw->wff_down, n_embd, cfg_.n_ff, ffn.data(), mla_fout.data());
                }

                float* xdst = x.data() + p * n_embd;
                for (uint32_t i = 0; i < n_embd; ++i) xdst[i] = mla_fout[i] + mla_proj[i];
            }
            continue;
        }

        if (n_tokens > 1) {
            for (size_t p = 0; p < n_tokens; ++p) {
                if (quirks_.no_pre_norm) {
                    std::memcpy(xn.data() + p * n_embd, x.data() + p * n_embd, n_embd * sizeof(float));
                } else {
                    norm_vec(quirks_.layer_norm, x.data() + p * n_embd, lw->attn_norm.data(),
                            lw->attn_norm_b.empty() ? nullptr : lw->attn_norm_b.data(),
                            xn.data() + p * n_embd, n_embd, rms_eps);
                }
            }
            const float* attn_in_all = xn.data();
            if (!lw->attn_norm2.empty()) {
                xn_attn.resize((size_t) n_tokens * n_embd);
                for (size_t p = 0; p < n_tokens; ++p) {
                    norm_vec(quirks_.layer_norm, x.data() + p * n_embd, lw->attn_norm2.data(),
                            lw->attn_norm2_b.empty() ? nullptr : lw->attn_norm2_b.data(),
                            xn_attn.data() + p * n_embd, n_embd, rms_eps);
                }
                attn_in_all = xn_attn.data();
            }
            matvec_batch(lw->wq, q_dim, n_embd, attn_in_all, n_tokens, q_all.data());
            matvec_batch(lw->wk, kv_dim, n_embd, attn_in_all, n_tokens, k_all.data());
            matvec_batch(lw->wv, kv_dim, n_embd, attn_in_all, n_tokens, v_all.data());

            const float inv_d = 1.0f / sqrtf((float) cfg_.head_dim);
            const uint32_t heads_per_kv = n_head / cfg_.n_head_kv;
            for (size_t p = 0; p < n_tokens; ++p) {
                const uint32_t pos = col0 + (uint32_t) p;
                float* qp = q_all.data() + p * q_dim;
                float* kp = k_all.data() + p * kv_dim;
                float* vp = v_all.data() + p * kv_dim;

                if (quirks_.qkv_bias) {
                    add_bias(qp, lw->bq.empty() ? nullptr : lw->bq.data(), q_dim);
                    add_bias(kp, lw->bk.empty() ? nullptr : lw->bk.data(), kv_dim);
                    add_bias(vp, lw->bv.empty() ? nullptr : lw->bv.data(), kv_dim);
                }
                if (cfg_.clamp_kqv > 0.0f) {
                    for (uint32_t i = 0; i < q_dim; ++i) qp[i] = std::max(-cfg_.clamp_kqv, std::min(cfg_.clamp_kqv, qp[i]));
                    for (uint32_t i = 0; i < kv_dim; ++i) kp[i] = std::max(-cfg_.clamp_kqv, std::min(cfg_.clamp_kqv, kp[i]));
                    for (uint32_t i = 0; i < kv_dim; ++i) vp[i] = std::max(-cfg_.clamp_kqv, std::min(cfg_.clamp_kqv, vp[i]));
                }
                if (quirks_.qk_norm) {
                    if (quirks_.qk_norm_full_width) {
                        rms_norm_vec(qp, lw->q_norm.data(), qp, q_dim, rms_eps);
                        rms_norm_vec(kp, lw->k_norm.data(), kp, kv_dim, rms_eps);
                    } else {
                        for (uint32_t h = 0; h < n_head; ++h) {
                            float* qh = qp + (size_t) h * cfg_.head_dim;
                            rms_norm_vec(qh, lw->q_norm.data(), qh, cfg_.head_dim, rms_eps);
                        }
                        for (uint32_t h = 0; h < cfg_.n_head_kv; ++h) {
                            float* kh = kp + (size_t) h * cfg_.head_dim;
                            rms_norm_vec(kh, lw->k_norm.data(), kh, cfg_.head_dim, rms_eps);
                        }
                    }
                }
                // Stessa cache cos/sin del percorso di decode: dipende solo
                // da posizione e parametri RoPE del layer, non dalla testa.
                // rope_only_swa (cohere2): NoPE sui layer globali, nessuna
                // chiamata rope — Qcur/Kcur restano invariati.
                if (!quirks_.no_rope && (!quirks_.rope_only_swa || cfg_.layer_is_swa(l))) {
                    rope_cache_init(rope_cache_b, cfg_.n_rot, pos, rope_th, rope_sc);
                    for (uint32_t h = 0; h < n_head; ++h) {
                        rope_neox_cached(qp + (size_t) h * cfg_.head_dim, cfg_.n_rot, rope_cache_b.data());
                    }
                    for (uint32_t h = 0; h < cfg_.n_head_kv; ++h) {
                        rope_neox_cached(kp + (size_t) h * cfg_.head_dim, cfg_.n_rot, rope_cache_b.data());
                    }
                }

                float* kc = k_cache_.data() + ((size_t) l * cache_capacity_ + pos) * kv_dim;
                float* vc = v_cache_.data() + ((size_t) l * cache_capacity_ + pos) * kv_dim;
                std::memcpy(kc, kp, kv_dim * sizeof(float));
                std::memcpy(vc, vp, kv_dim * sizeof(float));

                // Stessa mascheratura SWA del percorso di decode (vedi la
                // nota estesa li'): restringe l'intervallo invece di
                // aggiungere una maschera additiva, quindi riduce il lavoro
                // sui layer locali anziche' aggiungerne.
                const bool is_swa = cfg_.n_swa > 0 && cfg_.layer_is_swa(l);
                const uint32_t cc_start = is_swa
                    ? ((pos + 1 > cfg_.n_swa) ? (pos + 1 - cfg_.n_swa) : 0)
                    : 0;
                float* aout = attn_out_all.data() + p * q_dim;
                parallel_units(n_head, [&](size_t h_begin, size_t h_end) {
                for (uint32_t h = (uint32_t) h_begin; h < (uint32_t) h_end; ++h) {
                    const uint32_t hkv = h / heads_per_kv;
                    const float* qh = qp + (size_t) h * cfg_.head_dim;
                    const float* kb = k_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
                    const float* vb = v_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
                    // Ogni testa scrive nella propria fetta locale di scores
                    // (buffer allocato per-testa qui, non condiviso come nel
                    // percorso di decode): il batch path processa un token
                    // alla volta ma le teste dentro un token restano
                    // indipendenti fra loro.
                    std::vector<float> scores_h_local((size_t) pos + 1, 0.0f);
                    float* scores_h = scores_h_local.data();
                    if (quirks_.alibi) {
                        const float slope = alibi_slope(h, n_head, cfg_.max_alibi_bias);
                        for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                            scores_h[cc] = dot_f32(qh, kb + (size_t) cc * kv_dim, cfg_.head_dim) * inv_d
                                         - slope * (float) (pos - cc);
                        }
                    } else {
                        for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                            scores_h[cc] = dot_f32(qh, kb + (size_t) cc * kv_dim, cfg_.head_dim) * inv_d;
                        }
                    }
                    softmax_inplace(scores_h + cc_start, (size_t) pos + 1 - cc_start);
                    float* oh = aout + (size_t) h * cfg_.head_dim;
                    for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] = 0.0f;
                    for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                        axpy_f32(oh, vb + (size_t) cc * kv_dim, scores_h[cc], cfg_.head_dim);
                    }
                }
                });
            }

            matvec_batch(lw->wo, n_embd, q_dim, attn_out_all.data(), n_tokens, proj_all.data());
            for (size_t p = 0; p < n_tokens; ++p) {
                float* projp = proj_all.data() + p * n_embd;
                const float* xp = x.data() + p * n_embd;
                if (!lw->bo.empty()) {
                    for (uint32_t i = 0; i < n_embd; ++i) projp[i] += lw->bo[i];
                }
                if (quirks_.sandwich_norm) {
                    rms_norm_vec(projp, lw->post_attn_norm.data(), projp, n_embd, rms_eps);
                }
                if (quirks_.parallel_residual) {
                    // cohere2: proj_all resta SOLO l'uscita di attenzione (il
                    // residuo di input si somma alla fine insieme all'uscita
                    // FFN, non qui). ffn_xn_all riusa lo stesso xn gia'
                    // calcolato per Q/K/V — niente ffn_norm separata.
                    std::memcpy(ffn_xn_all.data() + p * n_embd, xn.data() + p * n_embd, n_embd * sizeof(float));
                } else {
                    for (uint32_t i = 0; i < n_embd; ++i) projp[i] += xp[i];
                    if (quirks_.no_pre_norm) {
                        std::memcpy(ffn_xn_all.data() + p * n_embd, projp, n_embd * sizeof(float));
                    } else {
                        norm_vec(quirks_.layer_norm, projp, lw->ffn_norm.data(),
                                lw->ffn_norm_b.empty() ? nullptr : lw->ffn_norm_b.data(),
                                ffn_xn_all.data() + p * n_embd, n_embd, rms_eps);
                    }
                }
            }

            if (cfg_.n_expert > 0) {
                // Ramo MoE: gating per-token (dipende dal router per ogni
                // posizione), resta iterato token per token come prima.
                for (size_t p = 0; p < n_tokens; ++p) {
                    float* xnp = ffn_xn_all.data() + p * n_embd;
                    float* projp = proj_all.data() + p * n_embd;
                    matvec(lw->router, cfg_.n_expert, n_embd, xnp, router_logits.data());
                    auto selected = moe_route(router_logits, cfg_.n_expert_used, cfg_.moe_norm_w, cfg_.moe_w_scale);
                    std::fill(expert_out.begin(), expert_out.end(), 0.0f);
                    for (const auto& sel : selected) {
                        const uint32_t eidx = sel.first;
                        const float weight = sel.second;
                        if (!experts_->fetch(l, eidx, ExpertPart::Gate, egate)) continue;
                        if (!experts_->fetch(l, eidx, ExpertPart::Up, eup)) continue;
                        if (!experts_->fetch(l, eidx, ExpertPart::Down, edown)) continue;
                        if (egate.size() != (size_t) cfg_.n_ff_expert * n_embd ||
                            eup.size() != (size_t) cfg_.n_ff_expert * n_embd ||
                            edown.size() != (size_t) n_embd * cfg_.n_ff_expert) continue;
                        matmul_f32(eup.data(), cfg_.n_ff_expert, n_embd, xnp, ffn.data());
                        matmul_f32(egate.data(), cfg_.n_ff_expert, n_embd, xnp, ffn_gate.data());
                        if (quirks_.gelu_tanh) {
                            for (uint32_t i = 0; i < cfg_.n_ff_expert; ++i) ffn[i] = gelu_tanh(ffn_gate[i]) * ffn[i];
                        } else {
                            for (uint32_t i = 0; i < cfg_.n_ff_expert; ++i) ffn[i] = silu(ffn_gate[i]) * ffn[i];
                        }
                        matmul_f32(edown.data(), n_embd, cfg_.n_ff_expert, ffn.data(), fout.data());
                        for (uint32_t i = 0; i < n_embd; ++i) expert_out[i] += weight * fout[i];
                    }
                    if (quirks_.sandwich_norm) {
                        rms_norm_vec(expert_out.data(), lw->post_ffn_norm.data(), expert_out.data(), n_embd, rms_eps);
                    }
                    float* xn2 = x.data() + p * n_embd;
                    if (quirks_.parallel_residual) {
                        const float* xp = x.data() + p * n_embd; // letto PRIMA di scrivere xn2 (stesso buffer)
                        for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = expert_out[i] + projp[i] + xp[i];
                    } else {
                        for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = expert_out[i] + projp[i];
                    }
                }
            } else if (quirks_.ffn_gated) {
                matvec_batch(lw->wff_up, cfg_.n_ff, n_embd, ffn_xn_all.data(), n_tokens, ffn_all.data());
                matvec_batch(lw->wff_gate, cfg_.n_ff, n_embd, ffn_xn_all.data(), n_tokens, ffn_gate_all.data());
                for (size_t p = 0; p < n_tokens; ++p) {
                    float* ffnp = ffn_all.data() + p * cfg_.n_ff;
                    float* ffngp = ffn_gate_all.data() + p * cfg_.n_ff;
                    if (quirks_.gelu_tanh) {
                        for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffnp[i] = gelu_tanh(ffngp[i]) * ffnp[i];
                    } else {
                        for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffnp[i] = silu(ffngp[i]) * ffnp[i];
                    }
                }
                matvec_batch(lw->wff_down, n_embd, cfg_.n_ff, ffn_all.data(), n_tokens, fout_all.data());
                for (size_t p = 0; p < n_tokens; ++p) {
                    float* foutp = fout_all.data() + p * n_embd;
                    const float* projp = proj_all.data() + p * n_embd;
                    if (quirks_.sandwich_norm) {
                        rms_norm_vec(foutp, lw->post_ffn_norm.data(), foutp, n_embd, rms_eps);
                    }
                    float* xn2 = x.data() + p * n_embd;
                    if (quirks_.parallel_residual) {
                        // xn2 e xp puntano alla STESSA cella (x[p]): leggere
                        // xn2[i] e scriverci nella stessa espressione e'
                        // sicuro (stesso indice su entrambi i lati, come il
                        // pattern equivalente nel ramo MoE sopra).
                        for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = foutp[i] + projp[i] + xn2[i];
                    } else {
                        for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = foutp[i] + projp[i];
                    }
                }
            } else {
                // FFN non-gated: singola proiezione up [+bias] -> attivazione
                // -> down [+bias]. Nessun ffn_gate, nessun prodotto gate*up.
                matvec_batch(lw->wff_up, cfg_.n_ff, n_embd, ffn_xn_all.data(), n_tokens, ffn_all.data());
                const bool has_up_b = !lw->ffn_up_b.empty();
                for (size_t p = 0; p < n_tokens; ++p) {
                    float* ffnp = ffn_all.data() + p * cfg_.n_ff;
                    if (has_up_b) for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffnp[i] += lw->ffn_up_b[i];
                    if (quirks_.ffn_act == DenseQuirks::PlainFfnAct::ReluSqr) {
                        for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffnp[i] = relu_sqr(ffnp[i]);
                    } else {
                        for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffnp[i] = gelu_tanh(ffnp[i]);
                    }
                }
                matvec_batch(lw->wff_down, n_embd, cfg_.n_ff, ffn_all.data(), n_tokens, fout_all.data());
                const bool has_down_b = !lw->ffn_down_b.empty();
                for (size_t p = 0; p < n_tokens; ++p) {
                    float* foutp = fout_all.data() + p * n_embd;
                    if (has_down_b) for (uint32_t i = 0; i < n_embd; ++i) foutp[i] += lw->ffn_down_b[i];
                    const float* projp = proj_all.data() + p * n_embd;
                    if (quirks_.sandwich_norm) {
                        rms_norm_vec(foutp, lw->post_ffn_norm.data(), foutp, n_embd, rms_eps);
                    }
                    float* xn2 = x.data() + p * n_embd;
                    for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = foutp[i] + projp[i];
                }
            }
            continue;
        }

        for (size_t p = 0; p < n_tokens; ++p) {
            const uint32_t pos = col0 + (uint32_t) p;
            const float* xp = x.data() + p * n_embd;
            float* xnp = xn.data() + p * n_embd;

            { ScopedTimer t(profile_counters().ns_ser_norm);
              if (quirks_.no_pre_norm) {
                  std::memcpy(xnp, xp, n_embd * sizeof(float));
              } else {
                  norm_vec(quirks_.layer_norm, xp, lw->attn_norm.data(),
                          lw->attn_norm_b.empty() ? nullptr : lw->attn_norm_b.data(),
                          xnp, n_embd, rms_eps);
              } }

            const float* attn_in = xnp;
            if (!lw->attn_norm2.empty()) {
                norm_vec(quirks_.layer_norm, xp, lw->attn_norm2.data(),
                        lw->attn_norm2_b.empty() ? nullptr : lw->attn_norm2_b.data(),
                        xnp_attn.data(), n_embd, rms_eps);
                attn_in = xnp_attn.data();
            }
            quantize_shared_q8k(attn_in, n_embd, qk_xq, qk_dscale, qk_xsum);
            const SharedMatvec qkv[3] = {
                {&lw->wq, q_dim,  n_embd, q.data()},
                {&lw->wk, kv_dim, n_embd, k.data()},
                {&lw->wv, kv_dim, n_embd, v.data()},
            };
            matvec_shared_group(qkv, 3, attn_in, qk_xq, qk_dscale, qk_xsum);
            if (quirks_.qkv_bias) {
                add_bias(q.data(), lw->bq.empty() ? nullptr : lw->bq.data(), q_dim);
                add_bias(k.data(), lw->bk.empty() ? nullptr : lw->bk.data(), kv_dim);
                add_bias(v.data(), lw->bv.empty() ? nullptr : lw->bv.data(), kv_dim);
            }
            if (cfg_.clamp_kqv > 0.0f) {
                for (uint32_t i = 0; i < q_dim; ++i) q[i] = std::max(-cfg_.clamp_kqv, std::min(cfg_.clamp_kqv, q[i]));
                for (uint32_t i = 0; i < kv_dim; ++i) k[i] = std::max(-cfg_.clamp_kqv, std::min(cfg_.clamp_kqv, k[i]));
                for (uint32_t i = 0; i < kv_dim; ++i) v[i] = std::max(-cfg_.clamp_kqv, std::min(cfg_.clamp_kqv, v[i]));
            }

            if (quirks_.qk_norm) {
              ScopedTimer t(profile_counters().ns_ser_norm);
                if (quirks_.qk_norm_full_width) {
                    rms_norm_vec(q.data(), lw->q_norm.data(), q.data(), q_dim, rms_eps);
                    rms_norm_vec(k.data(), lw->k_norm.data(), k.data(), kv_dim, rms_eps);
                } else {
                    for (uint32_t h = 0; h < n_head; ++h) {
                        float* qh = q.data() + (size_t) h * cfg_.head_dim;
                        rms_norm_vec(qh, lw->q_norm.data(), qh, cfg_.head_dim, rms_eps);
                    }
                    for (uint32_t h = 0; h < cfg_.n_head_kv; ++h) {
                        float* kh = k.data() + (size_t) h * cfg_.head_dim;
                        rms_norm_vec(kh, lw->k_norm.data(), kh, cfg_.head_dim, rms_eps);
                    }
                }
            }

            { ScopedTimer t(profile_counters().ns_ser_rope);
            // Una sola costruzione della cache per layer, poi riusata da
            // tutte le teste Q e K (vedi rope_cache_init). rope_only_swa
            // (cohere2): NoPE sui layer globali, nessuna chiamata rope.
            if (!quirks_.rope_only_swa || cfg_.layer_is_swa(l)) {
                rope_cache_init(rope_cache, cfg_.n_rot, pos, rope_th, rope_sc);
                for (uint32_t h = 0; h < n_head; ++h) {
                    rope_neox_cached(q.data() + (size_t) h * cfg_.head_dim, cfg_.n_rot, rope_cache.data());
                }
                for (uint32_t h = 0; h < cfg_.n_head_kv; ++h) {
                    rope_neox_cached(k.data() + (size_t) h * cfg_.head_dim, cfg_.n_rot, rope_cache.data());
                }
            }

            float* kc = k_cache_.data() + ((size_t) l * cache_capacity_ + pos) * kv_dim;
            float* vc = v_cache_.data() + ((size_t) l * cache_capacity_ + pos) * kv_dim;
            std::memcpy(kc, k.data(), kv_dim * sizeof(float));
            std::memcpy(vc, v.data(), kv_dim * sizeof(float));
            }

            {
            ScopedTimer attn_timer(profile_counters().ns_ser_attn);
            const float inv_d = 1.0f / sqrtf((float) cfg_.head_dim);
            const uint32_t heads_per_kv = n_head / cfg_.n_head_kv;
            // Le teste sono indipendenti fra loro: leggono la stessa KV cache
            // e scrivono blocchi separati di attn_out. Parallelizzarle e' la
            // voce piu' importante a contesto realistico — col profiler,
            // 269 ms su 672 (il 40% del decode) erano attenzione seriale con
            // 562 token di contesto, mentre con un prompt da 12 token il
            // costo era invisibile. Ogni testa ha la propria fetta di
            // `scores`, altrimenti i thread se la sovrascriverebbero.
            const size_t score_stride = (size_t) pos + 1;
            // Local attention window (sliding-window, gemma3). Local
            // layers must mask out keys where pos-cc >= n_swa: beyond
            // that distance the output would be wrong on contexts longer
            // than n_swa. Instead of applying an additive mask, this
            // directly narrows the [cc_start, pos] range, which is
            // equivalent and additionally REDUCES the work on local
            // layers instead of adding to it (never a performance
            // regression).
            const bool is_swa = cfg_.n_swa > 0 && cfg_.layer_is_swa(l);
            const uint32_t cc_start = is_swa
                ? ((pos + 1 > cfg_.n_swa) ? (pos + 1 - cfg_.n_swa) : 0)
                : 0;
            scores.assign((size_t) n_head * score_stride, 0.0f);
            parallel_units(n_head, [&](size_t h_begin, size_t h_end) {
            for (uint32_t h = (uint32_t) h_begin; h < (uint32_t) h_end; ++h) {
                float* scores_h = scores.data() + (size_t) h * score_stride;
                const uint32_t hkv = h / heads_per_kv;
                const float* qh = q.data() + (size_t) h * cfg_.head_dim;
                const float* kb = k_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
                const float* vb = v_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
                // Kept in float and vectorized, not a `double`
                // accumulator: double entirely blocked vectorization of
                // the innermost attention dot product, and the scores go
                // through a softmax right afterward anyway.
                if (quirks_.alibi) {
                    const float slope = alibi_slope(h, n_head, cfg_.max_alibi_bias);
                    for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                        scores_h[cc] = dot_f32(qh, kb + (size_t) cc * kv_dim, cfg_.head_dim) * inv_d
                                     - slope * (float) (pos - cc);
                    }
                } else {
                    for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                        scores_h[cc] = dot_f32(qh, kb + (size_t) cc * kv_dim, cfg_.head_dim) * inv_d;
                    }
                }
                // Softmax solo sulla sottofinestra [cc_start, pos]: le
                // posizioni prima di cc_start non partecipano (equivalente a
                // un punteggio -inf), e non vengono nemmeno lette dal ciclo
                // di accumulo di V sotto.
                softmax_inplace(scores_h + cc_start, score_stride - cc_start);
                float* oh = attn_out.data() + (size_t) h * cfg_.head_dim;
                // ORDINE DEI CICLI INVERTITO (2026-09-07). Prima il ciclo
                // esterno era su d e quello interno su cc:
                //     for d: for cc: acc += scores[cc] * vb[cc*kv_dim + d];
                // cioe' l'accesso piu' interno saltava di kv_dim float (4 KB
                // con kv_dim=1024) a ogni iterazione: OGNI accesso cadeva su
                // una cache line diversa, e nessuna delle linee caricate
                // veniva riusata prima di essere sfrattata. Contati ~1,5
                // milioni di accessi con miss per token, sufficienti da soli
                // a spiegare i ~17 ms di tratto seriale misurati dal profiler.
                // Con l'ordine giusto il ciclo interno percorre una riga
                // contigua di V, quindi ogni cache line serve 16 valori ed e'
                // anche vettorizzabile.
                for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] = 0.0f;
                for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                    axpy_f32(oh, vb + (size_t) cc * kv_dim, scores_h[cc], cfg_.head_dim);
                }
            }
            });
            }

            matvec(lw->wo, n_embd, q_dim, attn_out.data(), proj.data());
            { ScopedTimer t(profile_counters().ns_ser_norm);
            if (!lw->bo.empty()) {
                for (uint32_t i = 0; i < n_embd; ++i) proj[i] += lw->bo[i];
            }
            if (quirks_.sandwich_norm) {
                rms_norm_vec(proj.data(), lw->post_attn_norm.data(), proj.data(), n_embd, rms_eps);
            }
            if (quirks_.parallel_residual) {
                // cohere2: proj resta SOLO l'uscita di attenzione (il
                // residuo si somma alla fine insieme all'uscita FFN); xnp
                // resta quello gia' calcolato sopra per Q/K/V (stesso norm
                // condiviso), nessuna ffn_norm separata.
            } else {
                for (uint32_t i = 0; i < n_embd; ++i) proj[i] += xp[i];

                if (quirks_.no_pre_norm) {
                    std::memcpy(xnp, proj.data(), n_embd * sizeof(float));
                } else {
                    norm_vec(quirks_.layer_norm, proj.data(), lw->ffn_norm.data(),
                            lw->ffn_norm_b.empty() ? nullptr : lw->ffn_norm_b.data(),
                            xnp, n_embd, rms_eps);
                }
            } }

            if (cfg_.n_expert > 0) {
                // Router: logit per esperto, poi softmax + top-k (vedi
                // models/moe_route.h). Combinazione pesata delle uscite
                // FFN dei soli esperti selezionati, presi da ExpertStore
                // (dequantizzati; nessun kernel quantizzato per gli
                // esperti ancora, gap noto).
                matvec(lw->router, cfg_.n_expert, n_embd, xnp, router_logits.data());
                auto selected = moe_route(router_logits, cfg_.n_expert_used, cfg_.moe_norm_w, cfg_.moe_w_scale);

                std::fill(expert_out.begin(), expert_out.end(), 0.0f);
                for (const auto& sel : selected) {
                    const uint32_t eidx = sel.first;
                    const float weight = sel.second;
                    if (!experts_->fetch(l, eidx, ExpertPart::Gate, egate)) continue;
                    if (!experts_->fetch(l, eidx, ExpertPart::Up, eup)) continue;
                    if (!experts_->fetch(l, eidx, ExpertPart::Down, edown)) continue;
                    if (egate.size() != (size_t) cfg_.n_ff_expert * n_embd ||
                        eup.size() != (size_t) cfg_.n_ff_expert * n_embd ||
                        edown.size() != (size_t) n_embd * cfg_.n_ff_expert) continue;

                    matmul_f32(eup.data(), cfg_.n_ff_expert, n_embd, xnp, ffn.data());
                    matmul_f32(egate.data(), cfg_.n_ff_expert, n_embd, xnp, ffn_gate.data());
                    if (quirks_.gelu_tanh) {
                        for (uint32_t i = 0; i < cfg_.n_ff_expert; ++i) ffn[i] = gelu_tanh(ffn_gate[i]) * ffn[i];
                    } else {
                        for (uint32_t i = 0; i < cfg_.n_ff_expert; ++i) ffn[i] = silu(ffn_gate[i]) * ffn[i];
                    }
                    matmul_f32(edown.data(), n_embd, cfg_.n_ff_expert, ffn.data(), fout.data());
                    for (uint32_t i = 0; i < n_embd; ++i) expert_out[i] += weight * fout[i];
                }
                fout = expert_out;
            } else if (quirks_.ffn_gated) {
                quantize_shared_q8k(xnp, n_embd, qk_xq, qk_dscale, qk_xsum);
                const SharedMatvec gu[2] = {
                    {&lw->wff_up,   cfg_.n_ff, n_embd, ffn.data()},
                    {&lw->wff_gate, cfg_.n_ff, n_embd, ffn_gate.data()},
                };
                matvec_shared_group(gu, 2, xnp, qk_xq, qk_dscale, qk_xsum);
                { ScopedTimer t(profile_counters().ns_ser_act);
                if (quirks_.gelu_tanh) {
                    geglu_inplace(ffn.data(), ffn_gate.data(), cfg_.n_ff);
                } else {
                    for (uint32_t i = 0; i < cfg_.n_ff; ++i) {
                        ffn[i] = silu(ffn_gate[i]) * ffn[i];
                    }
                } }
                matvec(lw->wff_down, n_embd, cfg_.n_ff, ffn.data(), fout.data());
            } else {
                // FFN non-gated: singola proiezione, nessuna condivisione di
                // attivazione quantizzata con un secondo matmul (non c'e' un
                // secondo matmul), quindi matvec semplice invece di
                // matvec_shared come nel percorso gated.
                matvec(lw->wff_up, cfg_.n_ff, n_embd, xnp, ffn.data());
                { ScopedTimer t(profile_counters().ns_ser_act);
                if (!lw->ffn_up_b.empty()) {
                    for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffn[i] += lw->ffn_up_b[i];
                }
                if (quirks_.ffn_act == DenseQuirks::PlainFfnAct::ReluSqr) {
                    for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffn[i] = relu_sqr(ffn[i]);
                } else {
                    for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffn[i] = gelu_tanh(ffn[i]);
                } }
                matvec(lw->wff_down, n_embd, cfg_.n_ff, ffn.data(), fout.data());
                if (!lw->ffn_down_b.empty()) {
                    for (uint32_t i = 0; i < n_embd; ++i) fout[i] += lw->ffn_down_b[i];
                }
            }

            { ScopedTimer t(profile_counters().ns_ser_norm);
            if (quirks_.sandwich_norm) {
                rms_norm_vec(fout.data(), lw->post_ffn_norm.data(), fout.data(), n_embd, rms_eps);
            }
            float* xn2 = x.data() + p * n_embd;
            if (quirks_.parallel_residual) {
                // xn2 e xp puntano alla stessa cella x[p]: leggere xp[i]
                // (== xn2[i], il residuo di input originale) e scrivere
                // xn2[i] nella stessa espressione e' sicuro, stesso indice.
                for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = fout[i] + proj[i] + xp[i];
            } else {
                for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = fout[i] + proj[i];
            } }
        }
    }

    cache_cols_ = col0 + n_tokens;

    const float* xlast = x.data() + (size_t) (n_tokens - 1) * n_embd;
    norm_vec(quirks_.layer_norm, xlast, out_norm_.data(),
            out_norm_b_.empty() ? nullptr : out_norm_b_.data(),
            hn.data(), n_embd, rms_eps);

    last_logits.assign(cfg_.n_vocab, 0.0f);
    matvec(tok_embd_, cfg_.n_vocab, n_embd, hn.data(), last_logits.data());

    if (all_logits) {
        if (n_tokens == 1) {
            *all_logits = last_logits;
        } else {
            std::vector<float> hn_all((size_t) n_tokens * n_embd);
            for (size_t p = 0; p < n_tokens; ++p) {
                norm_vec(quirks_.layer_norm, x.data() + p * n_embd, out_norm_.data(),
                        out_norm_b_.empty() ? nullptr : out_norm_b_.data(),
                        hn_all.data() + p * n_embd, n_embd, rms_eps);
            }
            all_logits->assign((size_t) n_tokens * cfg_.n_vocab, 0.0f);
            matvec_batch(tok_embd_, cfg_.n_vocab, n_embd, hn_all.data(), n_tokens, all_logits->data());
            // L'ultima posizione e' identica a last_logits gia' calcolato
            // sopra (stesso hn): copio invece di ricalcolare per coerenza
            // bit-esatta col path a singola colonna gia' validato.
            std::memcpy(all_logits->data() + (n_tokens - 1) * cfg_.n_vocab,
                        last_logits.data(), cfg_.n_vocab * sizeof(float));
        }
    }

    if (cfg_.logit_scale != 0.0f) {
        for (float& v : last_logits) v *= cfg_.logit_scale;
        if (all_logits) for (float& v : *all_logits) v *= cfg_.logit_scale;
    }
    return true;
}

int32_t DenseForward::argmax(const std::vector<float>& v) {
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] > v[best]) best = (int32_t) i;
    }
    return best;
}

}
