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
    return _mm256_maddubs_epi16(ax, sy); // 16x int16, somma a coppie adiacenti
}
static inline __m256i sum_i16_pairs_i32_avx2(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, x); // 8x int32
}

// Riduzione orizzontale di un accumulatore float a 8 corsie. Fatta con
// shuffle invece che con store su array + 8 somme scalari: quest'ultima
// forma costringe a un round-trip in memoria (store-to-load forwarding)
// proprio sul valore appena calcolato.
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

// Come dot8_avx2 ma per n=32 fisso e SENZA la riduzione orizzontale finale
// (ritorna il vettore di 8 somme parziali int32, non ancora sommate in uno
// scalare). Usato quando il chiamante deve accumulare molti dot dello
// stesso ordine di grandezza in fila (es. gli 8 sotto-blocchi da 32 di un
// super-blocco Q4_K): fare la riduzione orizzontale (store in memoria + 8
// somme scalari) una volta sola per riga invece che una volta per
// sotto-blocco evita ~64 riduzioni ridondanti per riga su un modello
// tipico (n_super=8, 8 sotto-dot/super-blocco). La combinazione con la
// scala per-blocco avviene comunque in virgola mobile dopo la conversione
// (cvtepi32_ps), quindi il risultato resta numericamente equivalente
// (a meno di riordino in virgola mobile, come per il tiling a 2 righe).
static __m256i dot8_avx2_i32(const int8_t* a, const int8_t* b) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
    return sum_i16_pairs_i32_avx2(mul_add_i8_pairs_avx2(va, vb));
}

// Dot int8x16 dedicato (SSE4.1/SSSE3, sempre disponibile quando __AVX2__ e'
// definito). dot8_avx2 richiede n>=32 per entrare nel suo loop AVX2 (sotto
// soglia ricade tutta sullo scalare del tail loop): matmul_q6_k lavora
// pero' su sotto-blocchi da esattamente 16 elementi (scala per sotto-
// blocco Q6_K), quindi con dot8_avx2 quei dot NON erano mai vettorizzati.
// Scoperto misurando col profiler (core/profile.h), non ipotizzato. Stessa
// tecnica maddubs di dot8_avx2 sopra, a 128 bit.
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

// Come dot8_16 ma senza la riduzione orizzontale finale (vedi
// dot8_avx2_i32: stesso principio, applicato alla dimensione a 16 usata
// da Q6_K). matmul_q6_k fa 16 di questi dot per super-blocco (4 quadranti
// x 2 meta'): differire la riduzione a una volta per riga invece che una
// volta per dot evita altrettante riduzioni ridondanti.
static __m128i dot8_16_i32(const int8_t* a, const int8_t* b) {
    const __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    const __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
    return sum_i16_pairs_i32_sse(mul_add_i8_pairs_sse(va, vb));
}

// Nota per sviluppi futuri: e' stata provata una versione AVX2 esplicita
// dell'estrazione nibble sopra (shift a 16 bit + AND 0x0F, stesso trick di
// dot8_avx2) sia per Q4_0 che Q4_K. Misurata: nessun guadagno reale (~6%
// piu' lenta su 5 run ripetuti), verosimilmente perche' il loop scalare
// e' gia' ben auto-vettorizzato da -O3 e gli intrinsics manuali aggiungono
// solo latenza di store/reload. Rimossa, non tenuta: vedi
// docs/engine_gap_analysis.md per i dettagli della misura.
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

// Equivalenti NEON di dot8_avx2/dot8_16/dot8_16_i32 (definiti sopra per
// AVX2): stessa firma, stessa semantica, cosi' i ~15 call site sparsi nel
// file possono aggiungere un ramo `#elif defined(__ARM_NEON)` che chiama
// questi invece di duplicare la logica di estrazione nibble ogni volta.
//
// dot8_neon lavora a passi di 16 (un registro NEON), non 32 come dot8_avx2
// (che ne processa 2 in un colpo, essendo AVX2 a 256 bit): la coda
// scalare finale copre sia il caso n non multiplo di 16 sia, quando serve,
// gli ultimi 16 elementi di un blocco da 32.
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

// Equivalente NEON di dot8_16: n=16 fisso, CON riduzione a scalare.
static inline float dot8_16_neon(const int8_t* a, const int8_t* b) {
    const int32x4_t sum = dot_i8x16_neon(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b));
    return (float) vaddvq_s32(sum);
}

// n=16 fisso, senza riduzione orizzontale (equivalente di dot8_16_i32):
// usato dove il chiamante accumula molti dot da 16 elementi in fila prima
// di convertire una sola volta in float (stessa idea di dot8_avx2_i32/
// dot8_16_i32, qui a 128 bit — che e' gia' la larghezza nativa NEON, quindi
// non serve una variante "larga" separata come su AVX2).
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
// Stessa funzione di quant.cpp (non esposta in quant.h): estrae scala e
// minimo a 6 bit dal blocco scales[12] di un super-blocco Q4_K/Q5_K.
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
        // Tiling a 2 righe (stessa tecnica di matmul_q4_k/matmul_q6_k):
        // nessun nibble da spacchettare qui (Q8_0 e' gia' int8), quindi il
        // guadagno e' solo dare 2 catene di dot indipendenti per iterazione.
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
        // Costo: i byte del peso vengono riletti una volta per blocco di
        // 4 token invece che una sola volta per l'intera riga (traffico
        // di lettura pesi diviso per RN=4 invece che per n_tok) â€” ma
        // restano quasi certamente in L1/L2 fra un blocco di token e il
        // successivo (righe piccole, poche decine di byte), quindi il
        // costo reale e' basso rispetto al guadagno di avere accumulatori
        // davvero in registro. Misurato prima di tenerlo (regola di
        // sviluppo): vedi docs/engine_gap_analysis.md.
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
    // Sotto-blocchi Q2_K sono da 16 (non 32 come l'attivazione): xscale/xq
    // restano block-32, ma il termine "min" va sommato su range da 16, per
    // cui serve la somma delle attivazioni quantizzate su 16 elementi (non
    // sui 32 usati da Q4_K/Q5_K). Nota: essendo un sotto-range di un blocco
    // da 32 con la STESSA scala xscale (il blocco Q8_0 dell'attivazione e'
    // piu' largo del sotto-blocco Q2_K), non serve una scala diversa,
    // solo una somma piu' fine.
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
// Il punto non e' risparmiare sulla quantizzazione â€” e' che, con la scala
// costante dentro il super-blocco, il kernel puo' accumulare TUTTI e 8 i
// sotto-blocchi nel dominio intero e fare una sola conversione/
// moltiplicazione float ogni 256 pesi invece di otto. Con la scala per-32
// questo e' impossibile, perche' ogni sotto-blocco va riportato in float
// prima di poter essere sommato agli altri.
//
// La replica mantiene compatibili i kernel non ancora convertiti (Q5_K,
// Q6_K), che continuano a leggere xscale[sub] senza sapere che ora e'
// costante a tratti: nessuna doppia quantizzazione, nessuna firma cambiata.
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
        // Versione vettoriale. La scalare costava ~19 cicli per valore
        // (misurati 3,5 ms per token su ~550000 valori, la voce seriale piu'
        // pesante dopo l'attivazione): lrintf scalare e' lento e il
        // compilatore non puo' vettorizzare da solo per via dei clamp.
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
            // I 32 valori del sotto-blocco in quattro vettori, convertiti a
            // int32 con arrotondamento al pari (come lrintf in modalita'
            // predefinita), poi impacchettati a int8 con saturazione.
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 +  0), vid));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 +  8), vid));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 + 16), vid));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xs + sb * 32 + 24), vid));

            // La somma del sotto-blocco si ricava qui in int32, senza un
            // secondo passaggio di lettura sui byte gia' scritti.
            const __m256i vsum = _mm256_add_epi32(_mm256_add_epi32(i0, i1),
                                                  _mm256_add_epi32(i2, i3));

            // packs satura a [-128,127]; il -128 non si presenta perche'
            // |x*id| <= 127 per costruzione di id.
            __m256i p01 = _mm256_packs_epi32(i0, i1);   // corsie: [i0.lo i1.lo | i0.hi i1.hi]
            __m256i p23 = _mm256_packs_epi32(i2, i3);
            __m256i p   = _mm256_packs_epi16(p01, p23);
            // packs lavora per corsie da 128 bit: rimette in ordine i gruppi.
            p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(qs + sb * 32), p);

            const size_t idx = s * (QK_K / 32) + (size_t) sb;
            xscale32[idx] = d;
            xsum32[idx] = hsum_epi32_avx2(vsum);
        }
#elif defined(__ARM_NEON)
        // Stessa logica della versione AVX2 sopra, con le stesse garanzie:
        // `id` e' calcolato in modo che |x*id| <= 127 per costruzione,
        // quindi la somma dei valori PRIMA della saturazione a int8
        // coincide esattamente con la somma dei valori FINALI — si somma
        // quindi il vettore int32 pre-narrow, non serve un secondo giro
        // sui byte scritti.
        //
        // vcvtnq_s32_f32/vmaxvq_f32 sono intrinseche AArch64 (non ARMv7):
        // coerente col resto del file, che gia' assume aarch64 (vaddvq_s32
        // in dot_i8x16_neon) — il vincolo del progetto e' macOS Apple
        // Silicon e Linux ARM, entrambi aarch64.
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
            // Le 32 attivazioni del sotto-blocco si convertono in due meta'
            // da 16 (4 vettori float32x4 ciascuna), poi si restringono in
            // cascata: int32 -> int16 (vqmovn, satura) -> int8 (vqmovn,
            // satura di nuovo — la doppia saturazione e' innocua perche'
            // per costruzione i valori non escono mai da [-127,127]).
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
// Nucleo di matmul_q4_k, estratto per essere richiamabile sia col path
// normale (quantizza l'attivazione al suo interno) sia col path
// "pre-quantizzato" matmul_q4_k_pq (Fase "elimina ri-quantizzazione
// ridondante", 2026-09-07): wq/wk/wo/ffn_gate/ffn_up di un modello Q4_K_M
// leggono TUTTI dalla stessa attivazione (rispettivamente l'uscita di
// attn_norm/ffn_norm), ma essendo formati diversi (Q4_K/Q6_K) chiamavano
// ciascuno la propria matmul_qX_k che riquantizzava l'attivazione da
// capo â€” fino a 5 volte la stessa quantizzazione per layer. Dato che
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

int matmul_q4_k_core(const uint8_t* q4k_data, size_t rows, size_t cols,
                      const int8_t* xq, const float* xscale, const int32_t* xsum, float* y) {
    const size_t n_super = cols / QK_K;
    const size_t row_bytes = n_super * sizeof(block_q4_K);

    auto row_super_dot = [&](const block_q4_K& blk, size_t sub0) -> float {
        const float d = desireeia_fp16_to_fp32(blk.d);
        const float dmin = desireeia_fp16_to_fp32(blk.dmin);
        const uint8_t* q = blk.qs;

        // FASE SCALARE, tutta in testa e fuori dal loop vettoriale.
        // Prima le 8 scale/minimi venivano spacchettati DENTRO il loop, due
        // per gruppo: get_scale_min_k4 e' pieno di rami e manipolazione di
        // bit, e intercalarlo alle istruzioni SIMD teneva le unita'
        // vettoriali ferme ad aspettare. Contate ~20 operazioni scalari
        // ogni 32 byte di pesi, contro ~16 vettoriali: il loop era in
        // realta' limitato dalla parte scalare, non dal calcolo utile.
        // Separandole, il motore out-of-order puo' sovrapporre le due fasi.
        // Qui si precalcola direttamente il prodotto finale d*scala*xscale,
        // cosi' nel loop resta solo un broadcast da memoria.
        // L'attivazione e' quantizzata Q8_K: una sola scala per super-blocco,
        // replicata nelle 8 caselle per-32 (vedi quantize_act_q8k_rep). E'
        // questo che permette l'accumulazione intera piu' sotto.
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
        // DUE accumulatori indipendenti, non uno. Con un accumulatore solo
        // i due add_ps di ogni gruppo formano una catena di dipendenze
        // seriale: ogni add deve attendere il precedente (~4 cicli di
        // latenza), e il loop diventa latency-bound invece che
        // throughput-bound. Misurato: giravamo al 24% della banda di
        // memoria disponibile a QUALSIASI numero di thread â€” sintomo di
        // stallo per dipendenza, non di saturazione della memoria.
        // I due sotto-blocchi A e B sono indipendenti, quindi possono
        // accumulare in parallelo e sommarsi solo alla fine.
        //
        // Gli accumulatori sono INTERI, non float: la scala a 6 bit del peso
        // viene applicata nel dominio intero con madd_epi16 (stessa tecnica
        // gia' usata in q6k_group128_avx2), quindi tutti e 8 i sotto-blocchi
        // si sommano fra loro senza mai passare in virgola mobile. Resta una
        // sola cvtepi32_ps + moltiplicazione ogni 256 pesi invece di otto.
        // Nessun overflow: maddubs sta in +-3810, per la scala (<=63) fa
        // 240030, sommato su 8 sotto-blocchi resta ben dentro int32.
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
            // I 32 byte impacchettati contengono 64 pesi: i nibble bassi
            // sono il sotto-blocco A, quelli alti il sotto-blocco B. Si
            // spacchettano IN REGISTRO e si danno subito in pasto a
            // maddubs, senza mai passare da un array di stack: e' questa
            // la differenza col tentativo descritto a inizio file (che
            // spacchettava in registro ma poi ristoccava in wA[]/wB[] e
            // ricaricava, annullando il guadagno con lo store-to-load).
            //
            // In piu' qui si sfrutta un fatto specifico di Q4_K che il
            // percorso generico dot8_avx2_i32 non puo' sfruttare: i pesi
            // sono UNSIGNED 0..15 (il minimo e' gestito a parte da dmin),
            // e maddubs vuole esattamente un operando unsigned e uno
            // signed. Quindi entrano diretti, senza la coppia sign_epi8
            // del trucco per operandi entrambi signed. Nessun overflow:
            // il massimo per coppia e' 2*15*127 = 3810, dentro int16.
            const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
            const __m256i wA = _mm256_and_si256(packed, _mm256_set1_epi8(0x0F));
            const __m256i wB = _mm256_and_si256(_mm256_srli_epi16(packed, 4), _mm256_set1_epi8(0x0F));
            const __m256i vxA = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xA));
            const __m256i vxB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xB));
            // La scala del peso entra qui, in int16, invece che dopo in float.
            sumiA = _mm256_add_epi32(sumiA, _mm256_madd_epi16(
                _mm256_set1_epi16((short) sc8[is + 0]), _mm256_maddubs_epi16(wA, vxA)));
            sumiB = _mm256_add_epi32(sumiB, _mm256_madd_epi16(
                _mm256_set1_epi16((short) sc8[is + 1]), _mm256_maddubs_epi16(wB, vxB)));
#elif defined(__ARM_NEON)
            // 32 byte impacchettati = 64 pesi (32 per A nei nibble bassi, 32
            // per B negli alti), letti in due meta' da 16 byte perche' i
            // registri NEON sono a 128 bit contro i 256 di AVX2. I pesi
            // 0..15 restano validi come int8 con segno senza bisogno del
            // trucco unsigned/signed di maddubs (il bit di segno non e' mai
            // impostato per un nibble).
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
        // UNA sola conversione + moltiplicazione per super-blocco.
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
    // REVERT DI CORRETTEZZA (2026-09-07): il tile 2x4 con quantizzazione
    // Q8_K (vedi la nota in matmul_q4_k_core) produceva output corrotto
    // su un modello reale. Finche' non si riscrive il tile con la
    // quantizzazione fine (Q8_0 per sotto-blocco da 32), si ricade sulla
    // versione a singola colonna gia' corretta, riga per riga. Piu' lento
    // (nessun riuso del peso decodificato fra colonne), ma corretto.
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
    // Stesso revert di matmul_q4_k_batch sopra, stessa motivazione.
    if (n_tok == 0) return DESIREEIA_OK;
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    for (size_t tk = 0; tk < n_tok; ++tk) {
        int rc = matmul_q6_k(q6k_data, rows, cols, x + tk * cols, y + tk * rows);
        if (rc != DESIREEIA_OK) return rc;
    }
    return DESIREEIA_OK;
}

// REVERT DI CORRETTEZZA (2026-09-07): stessa motivazione di matmul_q4_k
// sopra, quantizzazione tornata a Q8_0 per sotto-blocco da 32.
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
            // Pesi Q5_K: 5 bit (0..31), sicuri come int8 con segno (mai
            // impostato il bit 7). Stesso schema scalare-accumulate di
            // Q4_K/Q6_K NEON: riduzione a vaddvq_s32, moltiplicazione per
            // la scala, accumulo in float (qui non nel dominio intero
            // perche' l'attivazione Q5_K resta a scala per-32, non Q8_K
            // per-256 come Q4_K/Q6_K — vedi il commento REVERT DI
            // CORRETTEZZA in testa a questa funzione).
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
    // Stesso revert di matmul_q4_k_batch, stessa motivazione: ricade sulla
    // versione a singola colonna, gia' corretta.
    if (n_tok == 0) return DESIREEIA_OK;
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    for (size_t tk = 0; tk < n_tok; ++tk) {
        int rc = matmul_q5_k(q5k_data, rows, cols, x + tk * cols, y + tk * rows);
        if (rc != DESIREEIA_OK) return rc;
    }
    return DESIREEIA_OK;
}

namespace {
// REVERT DI CORRETTEZZA (2026-09-07): vedi la nota completa in
// matmul_q4_k_core sopra. Tornati alla quantizzazione per sotto-blocco da
// 16 (quantize_q8_0 dell'attivazione, poi usata a granularita' 32 con
// indicizzazione /16 per Q6_K) invece del singolo scale Q8_K per 256.
#if defined(__AVX2__)
// Accumula il contributo di un gruppo di 128 pesi Q6_K (il gruppo naturale
// del formato: 64 byte di nibble bassi + 32 byte di bit alti + 8 scale).
//
// Tre differenze rispetto alla versione precedente, tutte misurate:
//  1. Larghezza piena. Prima si lavorava a 16 elementi (SSE, 128 bit)
//     perche' la scala Q6_K e' per-16: meta' della larghezza vettoriale
//     disponibile buttata su ogni singolo prodotto.
//  2. Le scale per-16 sono applicate NEL DOMINIO INTERO con madd_epi16.
//     I 32 byte di un vettore AVX2 si dividono in due corsie da 128 bit,
//     e i primi 16 pesi finiscono esattamente nelle prime 8 lane int16:
//     basta quindi un vettore di scale con sc[2q] nella corsia bassa e
//     sc[2q+1] in quella alta. Cosi' resta UNA moltiplicazione float ogni
//     32 pesi invece di due ogni 16.
//  3. L'offset -32 dei pesi non viene applicato byte per byte (cosa che
//     renderebbe i pesi signed e costringerebbe al trucco sign_epi8):
//     si tiene w unsigned 0..63 e si sottrae 32*x nel dominio int16.
//     Nessuna saturazione: maddubs(w,x) sta in [-16128, 16002],
//     maddubs(32,x) in [-8128, 8128], la differenza in [-24256, 24130].
// L'accumulatore e' INTERO: l'attivazione e' Q8_K (una scala per super-blocco,
// vedi quantize_act_q8k_rep), quindi tutti i gruppi si sommano fra loro
// nel dominio intero e la conversione in float avviene una sola volta per
// super-blocco invece di una ogni 32 pesi.
static inline __m256i q6k_group128_avx2(const uint8_t* ql, const uint8_t* qh,
                                        const int8_t* sc,
                                        const int8_t* x, __m256i acc) {
    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i m3  = _mm256_set1_epi8(0x03);
    const __m256i c32 = _mm256_set1_epi8(32);
    const __m256i qlL = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
    const __m256i qlH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
    const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));

    // Gli shift sono a 16 bit (non esiste lo shift per byte in AVX2): i bit
    // possono migrare fra i due byte della parola, ma l'AND successivo con
    // m3/m4 ripulisce, quindi il risultato per byte resta esatto.
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
// Equivalente NEON di q6k_group128_avx2 sopra, ma strutturato come il
// kernel Q4_K NEON: accumulo scalare int32 (scala moltiplicata subito dopo
// la riduzione orizzontale), non accumulo vettoriale con scala nel dominio
// SIMD. Motivazione identica: su NEON (128 bit, 4 corsie int32)
// vaddvq_s32 e' una singola istruzione, quindi non c'e' nulla da
// guadagnare a rimandare la riduzione.
//
// Differenza dal ramo AVX2: la' l'offset -32 veniva sottratto con un
// secondo maddubs (perche' maddubs vuole un operando unsigned, e i pesi
// dopo -32 diventerebbero signed). NEON non ha questo vincolo — vmull_s8
// accetta due operandi signed direttamente — quindi qui si sottrae 32 UNA
// volta con vsubq_s8, prima del dot-product, non due maddubs per gruppo.
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

    // Bit alti dei quattro gruppi, gia' posizionati nel nibble alto (<<4)
    // cosi' l'OR con il nibble basso di ql ricompone il valore a 6 bit.
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

int matmul_q6_k_core(const uint8_t* q6k_data, size_t rows, size_t cols,
                      const int8_t* xq, const float* xscale, float* y) {
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
        // Una sola conversione in float per super-blocco.
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
                    // Le due righe condividono le stesse attivazioni: il
                    // costo di quantizzazione e i load di x sono ammortizzati
                    // su entrambe (e' il motivo per cui il tiling a 2 righe
                    // resta, mentre a 4 era peggiorativo â€” vedi le note sul
                    // tiling in docs/engine_gap_analysis.md).
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
                // d cambia a ogni super-blocco, quindi l'accumulatore intero
                // va riversato in float qui e azzerato: e' comunque UNA
                // conversione ogni 256 pesi invece di una ogni 32.
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
        // Q8_K: il kernel Q6_K accumula ora in interi e assume la scala
        // costante dentro il super-blocco (vedi q6k_group128_avx2).
        quantize_act_q8k_rep(x, cols, xq, xscale, unused_xsum);
    }
    return matmul_q6_k_core(q6k_data, rows, cols, xq.data(), xscale.data(), y);
}

int matmul_q6_k_pq(const uint8_t* q6k_data, size_t rows, size_t cols,
                    const int8_t* xq, const float* xscale, float* y) {
    if (cols == 0 || cols % QK_K != 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    return matmul_q6_k_core(q6k_data, rows, cols, xq, xscale, y);
}

}
