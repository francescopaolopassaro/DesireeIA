// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "dense_forward.h"
#include "moe_route.h"
#include "../core/profile.h"
#include "../quant/quant.h"
#include "../kv/kv_quant.h"
#include "../core/mem_lock.h"
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

#ifdef DESIREEIA_CUDA_ENABLED
// Active backend for the current process, set exactly once by
// DenseForward::open() (cfg_.backend, from plan.backend). matvec/matvec_batch
// are free functions with no access to `this`/cfg_ (dispatch shared by
// every layer, not a per-instance method), so a file-level value is the
// simplest way to have them see the backend without changing the signature
// of every existing call site (dozens of them). Safe under
// single-model-per-process, which is the engine's current use case; if in
// the future multiple models with different backends share a process,
// this will need revisiting (a note for a future session, not a problem today).
static int32_t g_active_backend = DESIREEIA_BACKEND_CPU;
#endif

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
// Vectorized exp(x). Classic scheme: bring the exponential to base 2,
// split off the integer part k (which becomes directly the float's exponent
// field) from the remainder r in [-0.5, 0.5], and evaluate 2^r with a
// polynomial. Relative error ~1e-7 in the useful range, well beyond what's
// needed here.
static inline __m256 exp_ps_avx2(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    // Saturation: beyond these limits the result is 0 or infinity anyway,
    // and this only exists to keep k from overflowing the exponent field.
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(87.0f));

    const __m256 t = _mm256_mul_ps(x, log2e);
    const __m256 k = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 r = _mm256_sub_ps(t, k);

    // 2^r over [-0.5, 0.5], degree-5 polynomial (coefficients from the
    // ln(2)^n/n! series).
    __m256 p = _mm256_set1_ps(1.3327544e-3f);
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(9.6181292e-3f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(1.0f));

    // 2^k built by writing k+127 into the exponent field.
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
    // The sum of squares stays in double (it's a reduction over n_embd
    // values and numerical stability matters here), but with four vector
    // accumulators instead of a serial scalar chain.
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

// Picks between rms_norm_vec and layer_norm_vec based on the current
// architecture's quirk: a single dispatch point, so the ~15 call sites
// scattered across the forward path don't need to know which normalization
// the loaded model uses. `b` is ignored when using RMSNorm (which in this
// engine never has a bias).
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

static inline float dot_f32(const float* a, const float* b, size_t n);

// SIMD float32 counterpart to matmul_f32's scalar double accumulation.
//
// matmul_f32 stays as it is: it is the generic fallback for any tensor
// format without a quantized kernel, used across every architecture here,
// and its double accumulation is deliberate precision headroom for a path
// meant to be rare. MLA's absorbed attention breaks that assumption — the
// K/V up-projection halves (wk_b_h/wv_b_h) are genuinely float (no on-disk
// quantized form survives the transpose the "absorb" trick needs), but they
// run on EVERY head, EVERY layer, EVERY token, not rarely. Measured on a
// real model: this one fallback was 43% of total decode time. Reusing the
// same float32 SIMD accumulation already trusted for the attention score dot
// product (dot_f32, right below) fixes that without touching matmul_f32
// itself — nothing that already depends on its double-precision behavior is
// affected.
static void matmul_f32_fast(const float* w, size_t r, size_t c, const float* x, float* y) {
    ScopedTimer t(profile_counters().ns_f32_compute);
    parallel_rows(r, [&](size_t j0, size_t j1) {
        for (size_t j = j0; j < j1; ++j) {
            y[j] = dot_f32(w + j * c, x, c);
        }
    });
    profile_counters().calls_f32.fetch_add(1, std::memory_order_relaxed);
}

// Adds every active LoRA delta of `m` into y, on top of whatever the base
// matmul already wrote there:
//   y += scale * B @ (A @ x)     for each loaded adapter that targets m
// A/B are always float and rank-sized (small), so this is a pair of tiny
// dense matmuls, negligible next to the base matmul regardless of its
// quant format. No-op (one empty() check) when nothing is loaded, which
// is the overwhelming common case — this must stay cheap since it runs on
// the same hot path as every quantized matmul in the engine.
static void apply_lora(const MatVec& m, size_t r, size_t c, const float* x, float* y) {
    if (m.lora.empty()) return;
    std::vector<float> mid;
    for (const auto& lw : m.lora) {
        if (lw.rank == 0) continue;
        mid.assign(lw.rank, 0.0f);
        for (size_t k = 0; k < lw.rank; ++k) {
            const float* arow = lw.a.data() + k * c;
            double acc = 0.0;
            for (size_t i = 0; i < c; ++i) acc += (double) arow[i] * x[i];
            mid[k] = (float) acc;
        }
        for (size_t j = 0; j < r; ++j) {
            const float* brow = lw.b.data() + j * lw.rank;
            double acc = 0.0;
            for (size_t k = 0; k < lw.rank; ++k) acc += (double) brow[k] * mid[k];
            y[j] += lw.scale * (float) acc;
        }
    }
}

// Same as apply_lora, but over n_tok columns of activation (see matvec_batch).
static void apply_lora_batch(const MatVec& m, size_t r, size_t c, const float* x, size_t n_tok, float* y) {
    if (m.lora.empty()) return;
    for (size_t tk = 0; tk < n_tok; ++tk) {
        apply_lora(m, r, c, x + tk * c, y + tk * r);
    }
}


#ifdef DESIREEIA_CUDA_ENABLED
// Maps a weight format onto the CUDA backend's format id, or 0 when the
// device has no kernel for it (in which case the CPU path handles it).
static int cuda_fmt_of(MatVecFormat f) {
    switch (f) {
        case MatVecFormat::Q4_0: return DESIREEIA_CUDA_FMT_Q4_0;
        case MatVecFormat::Q4_1: return DESIREEIA_CUDA_FMT_Q4_1;
        case MatVecFormat::Q5_0: return DESIREEIA_CUDA_FMT_Q5_0;
        case MatVecFormat::Q5_1: return DESIREEIA_CUDA_FMT_Q5_1;
        case MatVecFormat::Q4_K: return DESIREEIA_CUDA_FMT_Q4_K;
        case MatVecFormat::Q5_K: return DESIREEIA_CUDA_FMT_Q5_K;
        case MatVecFormat::Q6_K: return DESIREEIA_CUDA_FMT_Q6_K;
        default: return 0;
    }
}
#endif

#ifdef DESIREEIA_CUDA_ENABLED
// A matrix can take part in the fused layer episode when its weights are
// resident on device in a format the backend has a kernel for. Q8_0 has
// its own split representation (cuda_fmt_of returns 0 for it), everything
// else keeps its native packed blocks.
static bool cuda_layer_ready(const MatVec& m) {
    if (!m.cuda_qs) return false;
    return m.format == MatVecFormat::Q8_0 || cuda_fmt_of(m.format) != 0;
}
#endif
// Dispatcher: uses the direct quantized kernel if the tensor is Q4_0 on
// disk, otherwise falls back to the classic float matmul on dequantized data.
static void matvec(const MatVec& m, size_t r, size_t c, const float* x, float* y) {
#ifdef DESIREEIA_CUDA_ENABLED
    // Phase 1 (docs/CUDAPiano.md): only Q8_0 has a device kernel, and only
    // when the plan has actually chosen the CUDA backend. Every other
    // format/backend falls back to the unchanged CPU path below. Nothing
    // changes when DESIREEIA_ENABLE_CUDA is OFF in CMake (the macro itself
    // isn't defined, so this block doesn't even compile).
    // Every quantized format with a device kernel: weights stay in their
    // native packed layout in VRAM and are decoded by the kernel.
    if (g_active_backend == DESIREEIA_BACKEND_CUDA && m.cuda_qs &&
        m.format != MatVecFormat::Q8_0) {
        const int kfmt = cuda_fmt_of(m.format);
        if (kfmt && matmul_kquant_cuda_resident(kfmt, m.cuda_qs.get(), r, c, x, y) == DESIREEIA_OK) {
            apply_lora(m, r, c, x, y);
            return;
        }
    }
    if (g_active_backend == DESIREEIA_BACKEND_CUDA && m.format == MatVecFormat::Q8_0) {
        // Weights already on device (persistent cache, see load_matrix): no
        // upload on every call, just the small activation. Otherwise
        // falls back to the per-call path (slower but correct).
        // The return value matters: a failed launch leaves y untouched,
        // and silently returning would feed garbage into the rest of the
        // forward pass (exactly how one sticky CUDA error turned into
        // degenerate text instead of an error). On failure fall through to
        // the CPU kernels below.
        const int rc = (m.cuda_qs && m.cuda_scale)
            ? matmul_q8_0_cuda_resident(m.cuda_qs.get(), m.cuda_scale.get(), r, c, x, y)
            : matmul_q8_0_cuda(m.raw.data(), r, c, x, y);
        if (rc == DESIREEIA_OK) {
            apply_lora(m, r, c, x, y);
            return;
        }
    }
#endif
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
    apply_lora(m, r, c, x, y);
}

// Same dispatch as matvec above, but over a bare pointer instead of a MatVec.
//
// Needed to compute one expert out of a stacked *_exps tensor: the experts sit
// in it as contiguous row ranges and no quantization block ever straddles a
// row, so an expert is just an offset into the quantized bytes. Copying it out
// into its own MatVec first would defeat the point.
static void matvec_raw(MatVecFormat fmt, const uint8_t* data, size_t r, size_t c,
                       const float* x, float* y) {
    switch (fmt) {
        case MatVecFormat::Q4_0: matmul_q4_0(data, r, c, x, y); break;
        case MatVecFormat::Q4_1: matmul_q4_1(data, r, c, x, y); break;
        case MatVecFormat::Q5_0: matmul_q5_0(data, r, c, x, y); break;
        case MatVecFormat::Q5_1: matmul_q5_1(data, r, c, x, y); break;
        case MatVecFormat::Q2_K: matmul_q2_k(data, r, c, x, y); break;
        case MatVecFormat::Q3_K: matmul_q3_k(data, r, c, x, y); break;
        case MatVecFormat::Q4_K: matmul_q4_k(data, r, c, x, y); break;
        case MatVecFormat::Q5_K: matmul_q5_k(data, r, c, x, y); break;
        case MatVecFormat::Q6_K: matmul_q6_k(data, r, c, x, y); break;
        case MatVecFormat::Q8_0: matmul_q8_0(data, r, c, x, y); break;
        case MatVecFormat::Q8_K: matmul_q8_k(data, r, c, x, y); break;
        default: break;   // Float has no raw form; callers check first.
    }
}

// Bytes one row of `m` occupies, given how many rows it holds in total.
// Uniform across rows for every quantized format here, which is what makes
// slicing an expert out of a stacked tensor a plain offset.
static size_t row_bytes_of(const MatVec& m, size_t total_rows) {
    if (total_rows == 0 || m.raw.empty()) return 0;
    if (m.raw.size() % total_rows != 0) return 0;
    return m.raw.size() / total_rows;
}

// Same as matvec but for n_tok columns of activation in a single call
// (Phase 8): x and y are n_tok contiguous blocks of c/r elements. For the
// Float fallback (rare: only non-quantized tensors) there is no dedicated
// batched matmul_f32 yet, so it simply calls matmul_f32 per column: no
// regression (same cost as before), the batching gain only applies to the
// quantized kernels.
static void matvec_batch(const MatVec& m, size_t r, size_t c, const float* x, size_t n_tok, float* y) {
#ifdef DESIREEIA_CUDA_ENABLED
    // Same backend gate as matvec above. This still calls the per-token
    // kernel one column at a time (a true batched kernel, reading each
    // weight once for ALL columns, is the real fix and is separate); what
    // it does do is stop restricting that to Q8_0. Every K-quant used to
    // fall through to the CPU branch below, which meant a Q4_K_M model —
    // i.e. the common case — ran its ENTIRE prefill on CPU while decode
    // ran on the GPU. Measured on gemma-2b: prefill was 30 tok/s against
    // 58 tok/s decode, slower per token than decode despite prefill being
    // the part that can actually be batched.
    if (g_active_backend == DESIREEIA_BACKEND_CUDA) {
        if (m.format == MatVecFormat::Q8_0 && m.cuda_qs && m.cuda_scale) {
            for (size_t tk = 0; tk < n_tok; ++tk) {
                matmul_q8_0_cuda_resident(m.cuda_qs.get(), m.cuda_scale.get(), r, c, x + tk * c, y + tk * r);
            }
            apply_lora_batch(m, r, c, x, n_tok, y);
            return;
        }
        const int cuda_fmt = cuda_fmt_of(m.format);
        if (cuda_fmt != 0 && m.cuda_qs) {
            // One episode for every column, each weight read once for all
            // of them. Only some formats have a batched kernel; the rest
            // fall through to the per-column loop below.
            if (n_tok > 1 &&
                matmul_kquant_cuda_batch(cuda_fmt, m.cuda_qs.get(), r, c, x, n_tok, y) == DESIREEIA_OK) {
                apply_lora_batch(m, r, c, x, n_tok, y);
                return;
            }
            bool all_ok = true;
            for (size_t tk = 0; tk < n_tok && all_ok; ++tk) {
                all_ok = matmul_kquant_cuda_resident(cuda_fmt, m.cuda_qs.get(), r, c,
                                                     x + tk * c, y + tk * r) == DESIREEIA_OK;
            }
            if (all_ok) {
                apply_lora_batch(m, r, c, x, n_tok, y);
                return;
            }
            // Any failure falls through to the CPU path below, which
            // recomputes every column from scratch: correct either way.
        }
    }
#endif
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
    apply_lora_batch(m, r, c, x, n_tok, y);
}

// "Eliminate redundant re-quantization" phase (2026-09-07): wq/wk/wv (and
// separately wff_gate/wff_up) all read from the SAME activation
// (respectively the output of attn_norm and of ffn_norm), but since they
// are often different formats (e.g. gemma3 Q4_K_M: wq/wk Q4_K, wv Q6_K)
// each matmul_qX_k used to re-quantize it from scratch internally — up to
// 3-5 times the same quantization per layer. Quantizing it once here
// (Q8_0-style, per 32-wide sub-block — see the CORRECTNESS REVERT in
// matmul.cpp: the single-scale-per-256 Q8_K scheme degraded precision too
// much on a real model) and passing the result to the "_pq" variants
// eliminates the redundancy for Q4_K/Q6_K without losing precision (the
// other formats, rarer in a K-quant model, fall back to the normal matvec,
// no regression).
static void quantize_shared_q8k(const float* x, size_t cols, std::vector<int8_t>& xq,
                                 std::vector<float>& xscale, std::vector<int32_t>& xsum) {
    ScopedTimer t(profile_counters().ns_quantize_act);
    // Q8_K (one scale every 256) instead of Q8_0 (one every 32): this is
    // what enables the integer accumulation inside matmul_q4_k_core. The
    // scale stays replicated into the per-32 slots, so the Q5_K/Q6_K
    // kernels, which read xscale[sub], keep working unmodified.
    quantize_act_q8k_rep(x, cols, xq, xscale, xsum);
}

static void matvec_shared(const MatVec& m, size_t r, size_t c, const float* x,
                           const std::vector<int8_t>& xq, const std::vector<float>& xscale,
                           const std::vector<int32_t>& xsum, float* y) {
    switch (m.format) {
        case MatVecFormat::Q4_K:
            matmul_q4_k_pq(m.raw.data(), r, c, xq.data(), xscale.data(), xsum.data(), y);
            apply_lora(m, r, c, x, y);
            break;
        case MatVecFormat::Q6_K:
            matmul_q6_k_pq(m.raw.data(), r, c, xq.data(), xscale.data(), y);
            apply_lora(m, r, c, x, y);
            break;
        default: matvec(m, r, c, x, y); break; // matvec() already applies LoRA
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
#ifdef DESIREEIA_CUDA_ENABLED
    // All Q8_0 with weights already on device: a single synchronization
    // for the whole group instead of one per matrix (see CudaQ80Job). The
    // group's matrices share the activation by construction, so it too is
    // quantized and loaded only once.
    if (g_active_backend == DESIREEIA_BACKEND_CUDA && n >= 2 && n <= 4) {
        CudaQ80Job cjobs[4];
        bool all_resident = true;
        const size_t cols = items[0].c;
        for (size_t i = 0; i < n && all_resident; ++i) {
            const MatVec& m = *items[i].m;
            all_resident = (m.format == MatVecFormat::Q8_0) && m.cuda_qs && m.cuda_scale &&
                            items[i].c == cols;
            if (all_resident) {
                cjobs[i].d_qs = m.cuda_qs.get();
                cjobs[i].d_scale = m.cuda_scale.get();
                cjobs[i].rows = items[i].r;
                cjobs[i].y = items[i].y;
            }
        }
        if (all_resident &&
            matmul_q8_0_cuda_resident_group(cjobs, n, cols, x) == DESIREEIA_OK) {
            for (size_t i = 0; i < n; ++i) {
                apply_lora(*items[i].m, items[i].r, items[i].c, x, items[i].y);
            }
            return;
        }
    }
#endif
    FusedPqJob jobs[4];
    bool fusable = (n >= 2 && n <= 4);
    for (size_t i = 0; fusable && i < n; ++i) {
        fusable = fused_job_for(items[i], xq, xscale, xsum, jobs[i]);
    }
    if (fusable && fused_pq_supported(jobs, n) &&
        matmul_fused_pq(jobs, n) == DESIREEIA_OK) {
        for (size_t i = 0; i < n; ++i) {
            apply_lora(*items[i].m, items[i].r, items[i].c, x, items[i].y);
        }
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

// freq_scale: linear scaling of positions (inverted rope.scaling.factor).
// 1.0 = no scaling. See the note on DenseConfig.
// Cache of cos/sin for one position and one (theta_base, freq_scale) pair.
// Interleaved: cache[2j] = cos, cache[2j+1] = sin.
//
// Previously every call to rope_neox computed, for EVERY pair of dimensions,
// one powf plus one sinf and one cosf — and it was invoked once per head:
// with 8 Q heads + 4 K heads across 34 layers that's 408 invocations per
// token, i.e. ~52000 powf calls and just as many identical repeated sincos.
// But the result doesn't depend on the head: it depends only on position and
// the layer's RoPE parameters.
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

// Same rotation as rope_neox_cached, but pairing CONSECUTIVE dimensions
// (2j, 2j+1) on input instead of split halves (j, j+half) — MLA's own RoPE
// convention for its rope-carrying slice (DeepSeek2 and siblings), distinct
// from the split-half convention every other architecture here declares.
// Output is still split-half: v[j] gets the "cos" component, v[j+half] gets
// the "sin" component, matching what mla_attn_layer reads back from qcur and
// from the K cache row afterward.
//
// NOT safe to do in place directly: consecutive-pair reads and split-half
// writes hit overlapping indices partway through (e.g. n_rot=8: the write at
// j=0 lands on v[4], which the read at j=2 needs) — a later iteration would
// read a value an earlier one already overwrote. The read side goes through
// a small copy so every read sees the original input regardless of order.
// Numeric tracing, off unless DESIREEIA_TRACE names a file. Writes the range
// of a vector so a NaN or a blow-up can be located by stage instead of
// guessed at; native stdout/stderr from this DLL does not reliably reach the
// host process, so it goes to a file.
static void trace_msg(const char* msg) {
    static const char* path = std::getenv("DESIREEIA_TRACE");
    if (!path) return;
    std::FILE* f = std::fopen(path, "a");
    if (!f) return;
    std::fprintf(f, "%s\n", msg);
    std::fclose(f);
}

static void trace_vec(const char* tag, uint32_t l, uint32_t pos, const float* v, size_t n) {
    static const char* path = std::getenv("DESIREEIA_TRACE");
    if (!path || n == 0) return;
    float lo = v[0], hi = v[0];
    size_t nan_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(v[i]) || std::isinf(v[i])) { nan_count++; continue; }
        if (v[i] < lo || std::isnan(lo)) lo = v[i];
        if (v[i] > hi || std::isnan(hi)) hi = v[i];
    }
    std::FILE* f = std::fopen(path, "a");
    if (!f) return;
    std::fprintf(f, "l=%u pos=%u %-14s n=%zu min=%.5g max=%.5g bad=%zu\n",
                 l, pos, tag, n, (double) lo, (double) hi, nan_count);
    std::fclose(f);
}

// Which pairing MLA's rope slice uses. The two conventions differ in which
// dimensions get rotated together, and a model only works with the one it was
// trained (and converted) for — with the wrong one every value stays finite
// and plausible while the output is quietly wrong, so this is worth being
// able to flip while establishing which a given file wants.
static bool mla_rope_split_half() {
    static const bool v = std::getenv("DESIREEIA_MLA_ROPE_NEOX") != nullptr;
    return v;
}

static void rope_mla_cached(float* v, size_t n_rot, const float* cache) {
    if (mla_rope_split_half()) { rope_neox_cached(v, n_rot, cache); return; }
    // n_rot is qk_rope_head_dim: 64 on every MLA checkpoint seen so far. 256
    // leaves headroom for a future variant without needing a heap allocation
    // on this per-head, per-token path; a config past that is refused rather
    // than silently truncated.
    if (n_rot == 0 || n_rot > 256) return;
    const size_t half = n_rot / 2;
    float in[256];
    std::memcpy(in, v, n_rot * sizeof(float));
    for (size_t j = 0; j < half; ++j) {
        const float cos_t = cache[2 * j + 0];
        const float sin_t = cache[2 * j + 1];
        const float a = in[2 * j + 0];
        const float b = in[2 * j + 1];
        v[j]        = a * cos_t - b * sin_t;
        v[j + half] = b * cos_t + a * sin_t;
    }
}

// Dot product over contiguous vectors (used for the attention QK scores,
// where head_dim is a multiple of 8).
static inline float dot_f32(const float* a, const float* b, size_t n) {
    size_t i = 0;
    float acc = 0.0f;
#if defined(__AVX2__)
    // Four accumulators: they break the dependency chain between successive
    // additions, which would otherwise impose ~4 cycles of latency per step.
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

// y[i] += s * x[i]  (weighted accumulation of V in attention).
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

// Test-only entry point: apply_lora() above is translation-unit-local
// (anonymous namespace), so tests/selftest.cpp — which links against this
// object file but can't see anonymous-namespace symbols by name — reaches
// the same code through this thin wrapper instead of duplicating the math.
void lora_apply_delta_for_test(const MatVec& m, size_t r, size_t c, const float* x, float* y) {
    apply_lora(m, r, c, x, y);
}

bool DenseForward::open(ModelReader& rd, const ModelMeta& meta, ArchKind arch, uint64_t ram_budget_mb,
                         ExpertStore* experts, bool kv_quantized, int32_t backend) {
    const std::string arch_tag = meta.arch.empty() ? "gemma" : meta.arch;
    const std::string kp = arch_tag + ".";
    quirks_ = quirks_for(arch);
    experts_ = experts;
    cfg_.backend = backend;
#ifdef DESIREEIA_CUDA_ENABLED
    g_active_backend = backend;
#endif
    // Never for MLA: see the note on kv_quantized_ in the header. head_dim
    // is repurposed there to the compressed kv_lora_rank+rope width, not a
    // classic per-head size, so kv_q_row_bytes_ is only meaningful (and only
    // computed) for the non-MLA case anyway.
    kv_quantized_ = kv_quantized && !quirks_.mla;

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

        // The same key can hold either a period (an integer, handled above)
        // or an explicit per-layer flag array. The array form is not a
        // repeating pattern at all, so no period can stand in for it: when
        // it's there it takes over from the formula entirely.
        std::vector<uint32_t> swa_flags;
        if (rd.meta_u32_array(kp + "attention.sliding_window_pattern", swa_flags) &&
            swa_flags.size() == cfg_.n_layers) {
            cfg_.swa_mask.assign(cfg_.n_layers, 0);
            for (uint32_t i = 0; i < cfg_.n_layers; ++i) {
                cfg_.swa_mask[i] = swa_flags[i] != 0 ? 1 : 0;
            }
        }
        // Sliding-window layers can rotate a different number of dimensions
        // than the rest (see DenseConfig::n_rot_swa).
        if (rd.meta_u32(kp + "rope.dimension_count_swa", v32) && v32 > 0) {
            cfg_.n_rot_swa = v32;
        }
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
        // Whether the top-k routing weights get renormalized to sum to 1.
        //
        // Wrong either way is not a crash, it is a silent scale error on the
        // whole MoE branch: with 6 of 64 experts the selected softmax weights
        // sum to well under 1, so renormalizing when the model did not
        // inflates every expert's contribution several-fold, and the model
        // degenerates into repetition while every intermediate value still
        // looks perfectly reasonable.
        //
        // Newer files state it outright; older ones don't carry the key at
        // all, so the default has to follow the family. The MLA family's
        // first generation does NOT renormalize (and neither does one other
        // family here); the later generation does, and those files are
        // exactly the ones that ship the key.
        cfg_.moe_norm_w = (arch_tag != "qwen2moe") && !quirks_.mla;
        uint32_t norm_w_meta = 0;
        if (rd.meta_u32(kp + "expert_weights_norm", norm_w_meta)) {
            cfg_.moe_norm_w = (norm_w_meta != 0);
        }
        cfg_.moe_w_scale = 1.0f;
        if (rd.meta_f32(kp + "expert_weights_scale", f32)) cfg_.moe_w_scale = f32;
    }

    // MLA (deepseek2/3): replaces head_dim/n_rot/q_dim with the compressed
    // space's dimensions. See the extended note in arch_tags.h on
    // DenseQuirks::mla and mla_attn_layer() further below for the derivation.
    if (quirks_.mla) {
        if (rd.meta_u32(kp + "attention.q_lora_rank", v32)) cfg_.q_lora_rank = v32;
        if (rd.meta_u32(kp + "attention.kv_lora_rank", v32)) cfg_.kv_lora_rank = v32;
        uint32_t k_mla = cfg_.head_dim; // fallback: attention.key_length already read above
        rd.meta_u32(kp + "attention.key_length_mla", k_mla);
        // v_head_dim genuinely differs from k_head_dim for MLA (nope+rope vs
        // just v), so it needs its own source, not a copy of k_mla. Some
        // GGUF exports use the "_mla"-suffixed key; the ones actually seen in
        // real files use the plain "attention.value_length" instead — read
        // there first, `_mla` as an override if a file does carry it, k_mla
        // only as a last resort if neither is present.
        uint32_t v_mla = k_mla;
        rd.meta_u32(kp + "attention.value_length", v_mla);
        rd.meta_u32(kp + "attention.value_length_mla", v_mla);
        cfg_.n_embd_head_v_mla = v_mla;
        cfg_.n_embd_head_qk_rope = cfg_.n_rot; // rope.dimension_count, already read above
        cfg_.n_embd_head_qk_nope = k_mla > cfg_.n_rot ? k_mla - cfg_.n_rot : 0;
        // head_dim/n_rot/n_head_kv are reused as the "compressed width" for
        // the generic K/V cache (see grow_cache): after absorption, Q and K
        // both live in a space of dimension kv_lora_rank+rope, shared by
        // EVERY head (MQA) — n_head_kv=1.
        cfg_.head_dim = cfg_.kv_lora_rank + cfg_.n_embd_head_qk_rope;
        cfg_.n_head_kv = 1;

        if (rd.meta_u32(kp + "leading_dense_block_count", v32)) cfg_.n_layer_dense_lead = v32;
        if (rd.meta_u32(kp + "expert_shared_count", v32)) cfg_.n_expert_shared = v32;
        if (rd.meta_u32(kp + "expert_gating_func", v32)) cfg_.moe_sigmoid_gate = (v32 == 2);

        // YaRN. rope_ext_factor==0 (default) fully disables the YaRN
        // branch in rope_cache_init: if the model doesn't declare
        // rope.scaling.type=="yarn" the RoPE stays the classic one.
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

    // Layer weight cache: active only if the actual RAM footprint fits
    // within a conservative fraction (60%) of the planned RAM budget, to
    // leave margin for the KV cache, embeddings and process overhead.
    // ram_budget_mb==0 (plan not configured) disables the cache.
    const uint32_t q_dim = cfg_.n_head * cfg_.head_dim;
    // Q4_0/Q4_K tensors stay quantized in RAM (see load_matrix): always
    // estimating 4 bytes/weight as if float would substantially overestimate
    // the actual footprint and would disable the cache even when it would
    // comfortably fit. The actual format of a representative tensor
    // (layer 0's attn_q) is probed and its real bytes/weight ratio is used
    // instead of a fixed sizeof(float).
    //
    // The probe tries several tensor names because not every architecture
    // has attn_q: the ones that fuse Q/K/V into a single tensor carry
    // attn_qkv instead. Probing only attn_q there found nothing, silently
    // fell back to 4 bytes/weight, overestimated a quantized model by
    // roughly 7x and switched the layer cache OFF — which costs a re-read
    // of every layer's weights on every token, and shows up as a
    // catastrophic decode rate rather than as an error.
    double bytes_per_weight = (double) sizeof(float);
    {
        static const char* kProbeNames[] = {
            "blk.0.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_down.weight",
        };
        std::vector<uint8_t> probe_raw;
        int probe_type = 0;
        uint64_t probe_ne0 = 0, probe_rows = 0;
        for (const char* name : kProbeNames) {
            if (rd.read_tensor_raw(name, probe_raw, probe_type, probe_ne0, probe_rows) &&
                probe_ne0 > 0 && probe_rows > 0) {
                bytes_per_weight = (double) probe_raw.size() / ((double) probe_ne0 * (double) probe_rows);
                break;
            }
        }
    }

    // A MoE layer has no dense wff_gate/up/down: it has the router (small,
    // n_expert*n_embd) plus every expert of the layer, held stacked and
    // quantized in the layer itself.
    //
    // Those experts have to be counted. They dominate a MoE layer — on a 64
    // expert model they are hundreds of MiB against a router of a few — and
    // they used to live in ExpertStore's own LRU, outside this cache, which is
    // why the old term ignored them. Leaving that term in place after moving
    // them here would have the cache believe a layer costs a rounding error
    // and happily hold every one of them.
    const uint64_t ffn_term = cfg_.n_expert > 0
        ? (uint64_t) cfg_.n_expert * cfg_.n_embd
          + (uint64_t) cfg_.n_expert * (uint64_t) cfg_.n_ff_expert * cfg_.n_embd * 3
        : (uint64_t) cfg_.n_ff * cfg_.n_embd * 3;
    const uint64_t per_layer_weights =
        (uint64_t) q_dim * cfg_.n_embd                 // wq
        + (uint64_t) kv_dim * cfg_.n_embd * 2           // wk + wv
        + (uint64_t) cfg_.n_embd * q_dim                // wo
        + ffn_term;
    const uint64_t norm_floats = (uint64_t) cfg_.n_embd * 2; // attn_norm + ffn_norm, always float
    const uint64_t total_bytes = cfg_.n_layers *
        ((uint64_t) (per_layer_weights * bytes_per_weight) + norm_floats * sizeof(float));
    const uint64_t budget_bytes = ram_budget_mb * 1024ULL * 1024ULL;
    cache_enabled_ = budget_bytes > 0 && total_bytes <= (budget_bytes * 6 / 10);

    // Stacking the experts into the layer is the fast arrangement, but only
    // while the layer stays resident. Without the layer cache, a MoE layer is
    // rebuilt on every token, and rebuilding it means re-reading ALL of its
    // experts — hundreds of MiB — to use six of them. The per-expert path is
    // slower per expert but only ever touches the experts actually routed to,
    // so it is the right one when the layers cannot stay in memory.
    stack_experts_ = cfg_.n_expert > 0 && cache_enabled_;
    // Diagnostic escape hatch, see the note on this member in the header.
    if (std::getenv("DESIREEIA_NO_STACK_EXPERTS")) stack_experts_ = false;

    layer_cache_.clear();
    if (cache_enabled_) layer_cache_.resize(cfg_.n_layers);

    if (kv_quantized_) {
        // Q8_0 needs 32-wide sub-blocks; every head_dim in practice (64,
        // 80, 96, 128...) is already a multiple of 32, but a checkpoint
        // that somehow isn't falls back to the float cache rather than
        // quantizing a partial block silently.
        if (cfg_.head_dim == 0 || cfg_.head_dim % 32 != 0) kv_quantized_ = false;
        else kv_q_row_bytes_ = kv_quant_row_bytes(cfg_.head_dim);
    }

    cache_capacity_ = 0;
    cache_cols_ = 0;
    grow_cache(512);

    // Opt-in, same reasoning as every other new capability this session:
    // proven in isolation, not yet run for hours against real memory
    // pressure, so it doesn't get to default on just because failing safely
    // costs nothing. DESIREEIA_MLOCK=1 to enable.
    //
    // Locks only the token embedding and (when present) the separate output
    // head — the single largest tensors that are simple to reach here as one
    // contiguous buffer each, and loaded once, never resized again. NOT a
    // guarantee that the whole model stays resident: the per-layer weights
    // (layer_cache_, when the weight cache is on) are dozens of separate
    // vectors per layer and are not locked by this — that would need walking
    // every LayerWeights member across every layer, real additional work
    // left for a follow-up rather than claimed here. See
    // docs/performance_plan.md item 4.
    if (std::getenv("DESIREEIA_MLOCK")) {
        auto lock_matvec = [this](const MatVec& m) {
            bool ok = false;
            if (!m.raw.empty()) ok = mlock_range(m.raw.data(), m.raw.size());
            else if (!m.f.empty()) ok = mlock_range(m.f.data(), m.f.size() * sizeof(float));
            mlocked_ = mlocked_ || ok;
        };
        lock_matvec(tok_embd_);
        lock_matvec(out_head_);
    }

    return true;
}

DenseForward::~DenseForward() {
#ifdef DESIREEIA_CUDA_ENABLED
    // CUDA stream + scratch buffers shared by the process: freed here
    // because another model loaded afterward will want its own (different
    // shapes). The device-resident weights of individual MatVecs free
    // themselves (shared_ptr, see MatVec::cuda_qs).
    if (g_active_backend == DESIREEIA_BACKEND_CUDA) cuda_backend_shutdown();
#endif
    if (!mlocked_) return;
    auto unlock_matvec = [](const MatVec& m) {
        if (!m.raw.empty()) munlock_range(m.raw.data(), m.raw.size());
        else if (!m.f.empty()) munlock_range(m.f.data(), m.f.size() * sizeof(float));
    };
    unlock_matvec(tok_embd_);
    unlock_matvec(out_head_);
}

void DenseForward::reset_cache() {
    cache_cols_ = 0;
}

bool DenseForward::load_embd(ModelReader& rd) {
    // The embedding and the output head live for the whole session (loaded
    // once here, never reloaded), so they're always valid candidates for
    // VRAM residency — independent of cache_enabled_, which concerns only
    // the layer cache and at this point in open() hasn't even been
    // computed yet.
    loading_persistent_ = true;
    struct PersistGuard {
        bool& flag;
        ~PersistGuard() { flag = false; }
    } guard{loading_persistent_};

    if (!load_matrix(rd, "token_embd.weight", cfg_.n_vocab, cfg_.n_embd, tok_embd_)) return false;

    // A separate output head, if this model has one. Read defensively: most
    // architectures here tie it to the embedding and ship no such tensor, and
    // those must keep using tok_embd_ exactly as before. See the note on
    // out_head_ for why getting this wrong is invisible rather than fatal.
    out_head_ = MatVec{};
    {
        MatVec head;
        const std::string err_before = last_fail_;
        if (load_matrix(rd, "output.weight", cfg_.n_vocab, cfg_.n_embd, head) && !head.empty()) {
            out_head_ = std::move(head);
        }
        last_fail_ = err_before; // absence is normal, not a failure worth reporting
    }

    // Absolute position (gpt2, mpt optional): read defensively, empty if
    // the tensor isn't there (no impact on the other architectures).
    pos_embd_.clear();
    if (n_ctx_train_ > 0) {
        rd.read_tensor("position_embd.weight", pos_embd_);
        if (pos_embd_.size() != (size_t) n_ctx_train_ * cfg_.n_embd) pos_embd_.clear();
    }

    // Initial norm applied to the embeddings (bloom, DenseQuirks::embd_norm).
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

// Maps (quant_type, cols) -> a direct MatVecFormat, if a dedicated
// quantized kernel exists for that format/width. Extracted out of
// load_matrix so it can also be reused by load_qkv_fused (falcon,
// DenseQuirks::fused_qkv) without duplicating — AND RISKING DRIFT IN —
// the list of supported formats: the exact same order/conditions as
// before, unchanged behavior for every architecture already working.
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

void DenseForward::maybe_prefetch_experts(const LayerWeights& lw, uint32_t layer,
                                          const std::vector<std::pair<uint32_t, float>>& selected) {
    if (!experts_ || lw.exps_stacked || selected.empty()) return;
    std::vector<uint32_t> idxs;
    idxs.reserve(selected.size());
    for (const auto& sel : selected) idxs.push_back(sel.first);
    experts_->prefetch_async(layer, idxs);
}

bool DenseForward::expert_ffn(const LayerWeights& lw, uint32_t layer, uint32_t eidx,
                              const float* xin, bool gelu_act,
                              std::vector<float>& ffn, std::vector<float>& ffn_gate,
                              std::vector<float>& out,
                              std::vector<float>& egate, std::vector<float>& eup,
                              std::vector<float>& edown) {
    const uint32_t n_embd = cfg_.n_embd;
    const uint32_t n_ff_e = cfg_.n_ff_expert;

    auto activate = [&]() {
        if (gelu_act) {
            for (uint32_t i = 0; i < n_ff_e; ++i) ffn[i] = gelu_tanh(ffn_gate[i]) * ffn[i];
        } else {
            for (uint32_t i = 0; i < n_ff_e; ++i) ffn[i] = silu(ffn_gate[i]) * ffn[i];
        }
    };

    if (lw.exps_stacked) {
        const size_t gu_rows = (size_t) n_ff_e * cfg_.n_expert;
        const size_t d_rows  = (size_t) n_embd * cfg_.n_expert;
        const size_t rb_gate = row_bytes_of(lw.wexp_gate, gu_rows);
        const size_t rb_up   = row_bytes_of(lw.wexp_up,   gu_rows);
        const size_t rb_down = row_bytes_of(lw.wexp_down, d_rows);

        if (rb_gate != 0 && rb_up != 0 && rb_down != 0 && eidx < cfg_.n_expert) {
            const size_t off_gu = (size_t) eidx * n_ff_e;
            const size_t off_d  = (size_t) eidx * n_embd;
            matvec_raw(lw.wexp_up.format,   lw.wexp_up.raw.data()   + off_gu * rb_up,
                       n_ff_e, n_embd, xin, ffn.data());
            matvec_raw(lw.wexp_gate.format, lw.wexp_gate.raw.data() + off_gu * rb_gate,
                       n_ff_e, n_embd, xin, ffn_gate.data());
            activate();
            matvec_raw(lw.wexp_down.format, lw.wexp_down.raw.data() + off_d * rb_down,
                       n_embd, n_ff_e, ffn.data(), out.data());
            return true;
        }
    }

    // Fallback: one dequantized expert at a time, through ExpertStore.
    if (!experts_) return false;
    if (!experts_->fetch(layer, eidx, ExpertPart::Gate, egate)) return false;
    if (!experts_->fetch(layer, eidx, ExpertPart::Up,   eup))   return false;
    if (!experts_->fetch(layer, eidx, ExpertPart::Down, edown)) return false;
    if (egate.size() != (size_t) n_ff_e * n_embd ||
        eup.size()   != (size_t) n_ff_e * n_embd ||
        edown.size() != (size_t) n_embd * n_ff_e) {
        return false;
    }

    trace_vec("exp_gate_w", layer, eidx, egate.data(), egate.size());
    trace_vec("exp_up_w", layer, eidx, eup.data(), eup.size());
    trace_vec("exp_down_w", layer, eidx, edown.data(), edown.size());

    matmul_f32(eup.data(),   n_ff_e, n_embd, xin, ffn.data());
    matmul_f32(egate.data(), n_ff_e, n_embd, xin, ffn_gate.data());
    trace_vec("exp_xin", layer, eidx, xin, n_embd);
    trace_vec("exp_up", layer, eidx, ffn.data(), n_ff_e);
    trace_vec("exp_gate", layer, eidx, ffn_gate.data(), n_ff_e);
    activate();
    trace_vec("exp_act", layer, eidx, ffn.data(), n_ff_e);
    matmul_f32(edown.data(), n_embd, n_ff_e, ffn.data(), out.data());
    trace_vec("exp_out", layer, eidx, out.data(), n_embd);
    return true;
}

#ifdef DESIREEIA_CUDA_ENABLED
// Makes a quantized matrix device-resident, once, when it belongs to the
// persistent weight cache. Shared by load_matrix and by the fused-QKV
// loader: the latter builds its three matrices by slicing one tensor's
// bytes, so it never went through load_matrix and its Q/K/V therefore
// stayed host-only — which silently disqualified every fused-QKV
// architecture from the whole-layer device path.
static void cuda_make_resident(MatVec& out, uint32_t rows, uint32_t cols, bool persistent) {
    const int cuda_fmt = cuda_fmt_of(out.format);
    if (!persistent || g_active_backend != DESIREEIA_BACKEND_CUDA) return;
    if (out.format != MatVecFormat::Q8_0 && cuda_fmt == 0) return;
    void* d_qs = nullptr;
    void* d_scale = nullptr;
    const bool ok = (out.format == MatVecFormat::Q8_0)
        ? matmul_q8_0_cuda_upload_weights(out.raw.data(), rows, cols, &d_qs, &d_scale)
        : cuda_upload_weights(cuda_fmt, out.raw.data(), rows, cols, &d_qs, &d_scale);
    if (!ok) return; // out of VRAM: stays host-only, still correct, just slower
    out.cuda_qs = std::shared_ptr<void>(d_qs, cuda_free_device);
    if (d_scale) out.cuda_scale = std::shared_ptr<void>(d_scale, cuda_free_device);
}
#endif

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
#ifdef DESIREEIA_CUDA_ENABLED
            // Upload to device ONCE, only for tensors that enter the
            // persistent weight cache (otherwise they'd be reloaded from
            // disk and reloaded to VRAM on every step anyway — no gain,
            // see the note on MatVec::cuda_qs). g_active_backend is the
            // same variable matvec/matvec_batch read for dispatch, set by
            // DenseForward::open() from plan.backend.
            cuda_make_resident(out, rows, cols, loading_persistent_);
#endif
            attach_lora_deltas(name, rows, cols, out);
            return true;
        }
    }

    // Format not handled by a direct kernel (or read_tensor_raw not
    // supported by the reader): falls back to the dequantized float path,
    // always correct for whatever quant format the reader reads.
    //
    // The float check below only compares the TOTAL element count, so a
    // tensor stored with rows and columns the other way round slips through
    // it and then gets read as if it were the expected shape — every value
    // finite, every result wrong. The raw path above does check the shape, so
    // whenever a tensor lands here for a reason OTHER than an unsupported
    // quantization, that is worth reporting rather than silently accepting.
    if (rd.read_tensor_raw(name, raw, quant_type, ne0, tensor_rows) &&
        (ne0 != cols || tensor_rows != rows)) {
        char msg[224];
        std::snprintf(msg, sizeof(msg),
                      "shape mismatch on %s: file has ne0=%llu rows=%llu, expected cols=%u rows=%u",
                      name.c_str(), (unsigned long long) ne0, (unsigned long long) tensor_rows,
                      cols, rows);
        last_fail_ = msg;
        trace_msg(msg);
    }

    if (!rd.read_tensor(name, out.f) || out.f.size() != (size_t) rows * cols) return false;
    attach_lora_deltas(name, rows, cols, out);
    return true;
}

void DenseForward::attach_lora_deltas(const std::string& name, uint32_t rows, uint32_t cols, MatVec& out) {
    if (lora_adapters_.empty()) return;
    for (auto& ad : lora_adapters_) {
        std::vector<float> a, b;
        if (!ad.reader->read_tensor(name + ".lora_a", a)) continue;
        if (!ad.reader->read_tensor(name + ".lora_b", b)) continue;
        if (cols == 0 || a.empty() || a.size() % cols != 0) continue;
        const uint32_t rank = (uint32_t) (a.size() / cols);
        if (rank == 0 || b.size() != (size_t) rows * rank) continue;

        LoraWeight lw;
        lw.a = std::move(a);
        lw.b = std::move(b);
        lw.rank = rank;
        // Classic PEFT alpha/rank scaling times the caller-supplied scale.
        lw.scale = ad.alpha > 0.0f ? ad.user_scale * ad.alpha / (float) rank : ad.user_scale;
        out.lora.push_back(std::move(lw));
    }
}

bool DenseForward::load_lora(const std::string& lora_gguf_path, float scale, std::string& err) {
    std::unique_ptr<ModelReader> reader(make_gguf_reader());
    ModelMeta meta;
    if (!reader->open(lora_gguf_path, meta)) {
        err = "failed to open LoRA adapter file: " + lora_gguf_path;
        return false;
    }
    std::string adapter_type;
    if (reader->meta_str("adapter.type", adapter_type) && adapter_type != "lora") {
        err = "not a LoRA adapter file (adapter.type=" + adapter_type + "): " + lora_gguf_path;
        return false;
    }

    LoraAdapter ad;
    ad.reader = std::move(reader);
    ad.reader->meta_f32("adapter.lora.alpha", ad.alpha);
    ad.user_scale = scale;
    lora_adapters_.push_back(std::move(ad));

    // layer_cache_ is populated LAZILY (see get_layer()'s sentinel check on
    // w.wq/w.wkv_a_mqa): the common case (adapter loaded right after the
    // model, before the first generation call) means every entry is still
    // untouched and will pick this adapter up naturally on first access —
    // nothing to do. For a layer already lazily loaded (generation already
    // happened on this model), reset it back to that same "not loaded yet"
    // state so the next get_layer() call re-reads it, this time attaching
    // the new adapter. Streaming models (cache disabled) need nothing
    // extra either way: load_matrix() re-runs on every step regardless.
    if (cache_enabled_) {
        for (auto& lw : layer_cache_) {
            if (!lw.wq.empty() || !lw.wkv_a_mqa.empty()) lw = LayerWeights{};
        }
    }
    return true;
}

void DenseForward::maybe_prefetch_predicted(uint32_t layer, const std::vector<uint32_t>& idxs) {
    if (!experts_ || stack_experts_ || idxs.empty() || layer >= cfg_.n_layers) return;
    experts_->prefetch_async(layer, idxs);
}

void DenseForward::build_prerouter_feature(const float* hidden,
                                           const std::vector<std::pair<uint32_t, float>>& selected,
                                           std::vector<float>& out) const {
    const uint32_t n_embd = cfg_.n_embd;
    const uint32_t n_expert = cfg_.n_expert;
    out.assign((size_t) n_embd + 2 * (size_t) n_expert, 0.0f);
    std::copy(hidden, hidden + n_embd, out.begin());
    for (const auto& sel : selected) {
        if (sel.first < n_expert) {
            out[n_embd + sel.first] = 1.0f;
            out[n_embd + n_expert + sel.first] = 1.0f;
        }
    }
}

void DenseForward::maybe_predict_next_layer(uint32_t l, const float* hidden,
                                            const std::vector<std::pair<uint32_t, float>>& selected) {
    if (l + 1 >= cfg_.n_layers) return;

    std::vector<uint32_t> predicted;
    auto it = prerouter_heads_.find(l);
    if (it != prerouter_heads_.end()) {
        std::vector<float> feat;
        build_prerouter_feature(hidden, selected, feat);
        prerouter_predict(it->second, feat.data(), feat.size(), cfg_.n_expert_used, predicted);
    } else if (prerouter_heuristic_) {
        predicted.reserve(selected.size());
        for (const auto& sel : selected) predicted.push_back(sel.first);
    } else {
        return;
    }
    maybe_prefetch_predicted(l + 1, predicted);
}

bool DenseForward::load_prerouter(const std::string& path, std::string& err) {
    if (cfg_.n_expert == 0) {
        err = "model has no MoE experts (n_expert=0); prerouter has nothing to predict";
        return false;
    }
    std::unique_ptr<ModelReader> reader(make_gguf_reader());
    ModelMeta meta;
    if (!reader->open(path, meta)) {
        err = "failed to open prerouter file: " + path;
        return false;
    }

    // Tensor naming: "prerouter.<owner_layer>.fc1.weight" / ".fc2.weight" /
    // ".linear_init.weight" — this engine's own convention, NOT the same
    // layout as any external reference implementation's format (there is
    // no byte-compatible external file to load here; see README).
    const uint32_t n_expert = cfg_.n_expert;
    const size_t feat_dim = (size_t) cfg_.n_embd + 2 * (size_t) n_expert;
    uint32_t loaded = 0;
    for (uint32_t owner = 0; owner < cfg_.n_layers; ++owner) {
        const std::string p = "prerouter." + std::to_string(owner) + ".";
        std::vector<float> fc1, fc2, lin;
        if (!reader->read_tensor(p + "fc1.weight", fc1)) continue;
        if (!reader->read_tensor(p + "fc2.weight", fc2)) continue;
        if (!reader->read_tensor(p + "linear_init.weight", lin)) continue;
        if (feat_dim == 0 || fc1.empty() || fc1.size() % feat_dim != 0) continue;
        const uint32_t prerouter_hidden = (uint32_t) (fc1.size() / feat_dim);
        if (fc2.size() != (size_t) n_expert * prerouter_hidden) continue;
        if (lin.size() != (size_t) n_expert * feat_dim) continue;

        PrerouterHead head;
        head.fc1 = std::move(fc1);
        head.fc2 = std::move(fc2);
        head.lin = std::move(lin);
        head.hidden = cfg_.n_embd;
        head.prerouter_hidden = prerouter_hidden;
        head.n_expert = n_expert;
        prerouter_heads_[owner] = std::move(head);
        ++loaded;
    }
    if (loaded == 0) {
        err = "no valid prerouter heads found (expected tensors named "
              "prerouter.<N>.fc1/fc2/linear_init.weight): " + path;
        return false;
    }
    return true;
}

void DenseForward::clear_prerouter() {
    prerouter_heads_.clear();
}

void DenseForward::set_prerouter_heuristic(bool on) {
    prerouter_heuristic_ = on;
}

void DenseForward::clear_lora() {
    lora_adapters_.clear();
    // Same reasoning as load_lora(): force already-loaded layers to reload
    // on next access, this time with no adapter to attach.
    if (cache_enabled_) {
        for (auto& lw : layer_cache_) {
            if (!lw.wq.empty() || !lw.wkv_a_mqa.empty()) lw = LayerWeights{};
        }
    }
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
            // Same opt-in narrowing load_matrix does. Applied to the WHOLE
            // tensor before the split: rows never interact, so converting
            // first and slicing the narrower rows afterwards is equivalent
            // to converting each slice.
            if (fmt == MatVecFormat::Q6_K && requantize_q6k_enabled() &&
                requantize_q6k_to_q4k(raw, total_rows, cols)) {
                fmt = MatVecFormat::Q4_K;
            }
            const size_t row_bytes = raw.size() / total_rows;
            wq.raw.assign(raw.begin(), raw.begin() + (size_t) q_dim * row_bytes);
            wq.format = fmt;
            wk.raw.assign(raw.begin() + (size_t) q_dim * row_bytes,
                          raw.begin() + (size_t) (q_dim + kv_dim) * row_bytes);
            wk.format = fmt;
            wv.raw.assign(raw.begin() + (size_t) (q_dim + kv_dim) * row_bytes, raw.end());
            wv.format = fmt;
#ifdef DESIREEIA_CUDA_ENABLED
            cuda_make_resident(wq, q_dim, cols, loading_persistent_);
            cuda_make_resident(wk, kv_dim, cols, loading_persistent_);
            cuda_make_resident(wv, kv_dim, cols, loading_persistent_);
#endif
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
    // A layer's weights are persistent only when the weight cache is
    // active; otherwise they end up in scratch_ and are re-read from disk
    // on every step (nothing to reuse on VRAM). See loading_persistent_.
    loading_persistent_ = cache_enabled_;
    struct PersistGuard {
        bool& flag;
        ~PersistGuard() { flag = false; }
    } guard{loading_persistent_};

    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%u.", il);
    const std::string p = buf;

    // no_pre_norm (olmo2, exaone4): these tensors don't exist at all in the
    // GGUF (only the sandwich post-norms replace them, read below), so
    // they must not be requested as mandatory.
    if (quirks_.no_pre_norm) {
        w.attn_norm.clear();
        w.ffn_norm.clear();
    } else if (quirks_.parallel_residual) {
        // cohere2: a single norm (attn_norm) feeds BOTH the attention AND
        // the FFN — ffn_norm.weight doesn't exist at all in the GGUF.
        if (!rd.read_tensor(p + "attn_norm.weight", w.attn_norm) || w.attn_norm.size() != cfg_.n_embd) return false;
        w.ffn_norm.clear();
    } else {
        if (!rd.read_tensor(p + "attn_norm.weight", w.attn_norm) || w.attn_norm.size() != cfg_.n_embd) { last_fail_ = "attn_norm.weight il=" + std::to_string(il); return false; }
        if (!rd.read_tensor(p + "ffn_norm.weight", w.ffn_norm) || w.ffn_norm.size() != cfg_.n_embd) { last_fail_ = "ffn_norm.weight il=" + std::to_string(il); return false; }
    }

    // Norm bias: only LayerNorm architectures (stablelm, orion, etc.) have
    // it. Reading it even when absent does no harm (read_tensor fails, the
    // vector stays empty -> nullptr in norm_vec), but this skips the
    // pointless read on RMSNorm architectures.
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
        // See the extended note on DenseQuirks::mla and mla_attn_layer():
        // MLA-specific tensors, entirely replace wq/wk/wv/attn_{q,k}.bias.
        w.wq = MatVec{}; w.wk = MatVec{}; w.wv = MatVec{};
        w.bq.clear(); w.bk.clear(); w.bv.clear(); w.bo.clear();

        if (cfg_.q_lora_rank > 0) {
            if (!load_matrix(rd, p + "attn_q_a.weight", cfg_.q_lora_rank, cfg_.n_embd, w.wq_a)) { last_fail_ = "attn_q_a.weight il=" + std::to_string(il); return false; }
            if (!rd.read_tensor(p + "attn_q_a_norm.weight", w.attn_q_a_norm) || w.attn_q_a_norm.size() != cfg_.q_lora_rank) { last_fail_ = "attn_q_a_norm.weight il=" + std::to_string(il) + " got=" + std::to_string(w.attn_q_a_norm.size()) + " want=" + std::to_string(cfg_.q_lora_rank); return false; }
            if (!load_matrix(rd, p + "attn_q_b.weight", cfg_.n_head * (cfg_.n_embd_head_qk_nope + cfg_.n_embd_head_qk_rope), cfg_.q_lora_rank, w.wq_b)) { last_fail_ = "attn_q_b.weight il=" + std::to_string(il); return false; }
            w.wq_lite = MatVec{};
        } else {
            if (!load_matrix(rd, p + "attn_q.weight", cfg_.n_head * (cfg_.n_embd_head_qk_nope + cfg_.n_embd_head_qk_rope), cfg_.n_embd, w.wq_lite)) { last_fail_ = "attn_q.weight(lite) il=" + std::to_string(il); return false; }
            w.wq_a = MatVec{}; w.wq_b = MatVec{};
            w.attn_q_a_norm.clear();
        }

        if (!load_matrix(rd, p + "attn_kv_a_mqa.weight", cfg_.kv_lora_rank + cfg_.n_embd_head_qk_rope, cfg_.n_embd, w.wkv_a_mqa)) { last_fail_ = "attn_kv_a_mqa.weight il=" + std::to_string(il); return false; }
        if (!rd.read_tensor(p + "attn_kv_a_norm.weight", w.attn_kv_a_norm) || w.attn_kv_a_norm.size() != cfg_.kv_lora_rank) { last_fail_ = "attn_kv_a_norm.weight il=" + std::to_string(il) + " got=" + std::to_string(w.attn_kv_a_norm.size()) + " want=" + std::to_string(cfg_.kv_lora_rank); return false; }

        // wk_b/wv_b: derived from the compressed KV up-projection, per head.
        // Never quantized here (float fallback): modest size next to the
        // rest, and the "absorbed" MLA trick needs the K half transposed
        // relative to how the projection is naturally stored, which a raw
        // quantized kernel can't do for us.
        //
        // Two GGUF conventions exist for this projection, tried in order:
        //
        //  1. Split (attn_k_b.weight / attn_v_b.weight): each already in the
        //     exact per-head shape this engine wants. No transform needed.
        //
        //  2. Combined (attn_kv_b.weight, shape [kv_lora_rank, n_head *
        //     (qk_nope_head_dim + v_head_dim)] on disk): the up-projection as
        //     the model actually learned it, before any splitting for
        //     inference convenience. Row r belongs to head r / (nope+v); the
        //     first nope_head_dim rows of that block are the K half, the
        //     remaining v_head_dim are the V half.
        //
        //     The V half copies straight across: mla_attn_layer wants
        //     wv_b_h[h] as [v_head_dim rows x kv_lora_rank cols], which is
        //     exactly what those rows already are.
        //
        //     The K half needs transposing: mla_attn_layer wants wk_b_h[h] as
        //     [kv_lora_rank rows x nope_head_dim cols] (it right-multiplies a
        //     q_nope vector to fold the K up-projection into the query — the
        //     "absorption" the format is named for), but on disk those rows
        //     are [nope_head_dim rows x kv_lora_rank cols]. Same 65536
        //     floats, transposed layout.
        const uint32_t nope_dim = cfg_.n_embd_head_qk_nope;
        const uint32_t v_dim    = cfg_.n_embd_head_v_mla;
        const uint32_t kv_lora  = cfg_.kv_lora_rank;

        std::vector<float> wk_b_flat, wv_b_flat;
        const bool have_split =
            rd.read_tensor(p + "attn_k_b.weight", wk_b_flat) &&
            wk_b_flat.size() == (size_t) nope_dim * kv_lora * cfg_.n_head &&
            rd.read_tensor(p + "attn_v_b.weight", wv_b_flat) &&
            wv_b_flat.size() == (size_t) kv_lora * v_dim * cfg_.n_head;

        if (!have_split) {
            std::vector<float> combined;
            const size_t want = (size_t) kv_lora * cfg_.n_head * (nope_dim + v_dim);
            if (!rd.read_tensor(p + "attn_kv_b.weight", combined) || combined.size() != want) {
                last_fail_ = "attn_kv_b.weight il=" + std::to_string(il) + " got=" + std::to_string(combined.size()) +
                            " want=" + std::to_string(want) + " nope=" + std::to_string(nope_dim) +
                            " v=" + std::to_string(v_dim) + " kv_lora=" + std::to_string(kv_lora) +
                            " n_head=" + std::to_string(cfg_.n_head);
                return false;
            }
            const size_t block = (size_t) (nope_dim + v_dim) * kv_lora; // one head's rows, on disk
            wk_b_flat.assign((size_t) nope_dim * kv_lora * cfg_.n_head, 0.0f);
            wv_b_flat.assign((size_t) kv_lora * v_dim * cfg_.n_head, 0.0f);
            for (uint32_t h = 0; h < cfg_.n_head; ++h) {
                const float* head_base = combined.data() + (size_t) h * block;
                // K half: transpose [nope_dim rows x kv_lora cols] -> the
                // [kv_lora rows x nope_dim cols] this engine expects.
                float* kdst = wk_b_flat.data() + (size_t) h * nope_dim * kv_lora;
                for (uint32_t i = 0; i < nope_dim; ++i) {
                    const float* row = head_base + (size_t) i * kv_lora;
                    for (uint32_t j = 0; j < kv_lora; ++j) {
                        kdst[(size_t) j * nope_dim + i] = row[j];
                    }
                }
                // V half: already [v_dim rows x kv_lora cols] — a straight copy.
                const float* vsrc = head_base + (size_t) nope_dim * kv_lora;
                float* vdst = wv_b_flat.data() + (size_t) h * v_dim * kv_lora;
                std::memcpy(vdst, vsrc, (size_t) v_dim * kv_lora * sizeof(float));
            }
        }

        {
            w.wk_b_h.assign(cfg_.n_head, MatVec{});
            const size_t chunk = (size_t) nope_dim * kv_lora;
            for (uint32_t h = 0; h < cfg_.n_head; ++h) {
                w.wk_b_h[h].f.assign(wk_b_flat.begin() + h * chunk, wk_b_flat.begin() + (h + 1) * chunk);
                w.wk_b_h[h].format = MatVecFormat::Float;
            }
        }
        {
            w.wv_b_h.assign(cfg_.n_head, MatVec{});
            const size_t chunk = (size_t) kv_lora * v_dim;
            for (uint32_t h = 0; h < cfg_.n_head; ++h) {
                w.wv_b_h[h].f.assign(wv_b_flat.begin() + h * chunk, wv_b_flat.begin() + (h + 1) * chunk);
                w.wv_b_h[h].format = MatVecFormat::Float;
            }
        }

        const uint32_t mla_out_dim = cfg_.n_head * cfg_.n_embd_head_v_mla;
        if (!load_matrix(rd, p + "attn_output.weight", cfg_.n_embd, mla_out_dim, w.wo)) { last_fail_ = "attn_output.weight(mla) il=" + std::to_string(il); return false; }
    } else if (quirks_.fused_qkv) {
        if (!load_qkv_fused(rd, p + "attn_qkv.weight", q_dim, kv_dim, cfg_.n_embd, w.wq, w.wk, w.wv)) return false;
        if (!load_matrix(rd, p + "attn_output.weight", cfg_.n_embd, q_dim, w.wo)) return false;
        w.bo.clear();
        rd.read_tensor(p + "attn_output.bias", w.bo);
        if (w.bo.size() != cfg_.n_embd) w.bo.clear();

        w.bq.clear(); w.bk.clear(); w.bv.clear();
        if (quirks_.qkv_bias) {
            // gpt2/bloom/mpt: a single fused bias (attn_qkv.bias), same
            // [Q|K|V] order as the fused weight tensor — split by range.
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
        } // falcon (qkv_bias=false): no bias, stays empty

        // Per-head output gate: one output row per head (see
        // DenseQuirks::attn_gate).
        if (quirks_.attn_gate) {
            if (!load_matrix(rd, p + "attn_gate.weight", cfg_.n_head, cfg_.n_embd, w.attn_gate)) {
                last_fail_ = "attn_gate.weight il=" + std::to_string(il);
                return false;
            }
        }
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

    // Optional second norm for the attention (falcon-40B, see the note on
    // LayerWeights::attn_norm2): absent everywhere except that specific
    // checkpoint, read defensively.
    w.attn_norm2.clear(); w.attn_norm2_b.clear();
    if (quirks_.parallel_residual) {
        rd.read_tensor(p + "attn_norm_2.weight", w.attn_norm2);
        if (w.attn_norm2.size() != cfg_.n_embd) w.attn_norm2.clear();
        if (!w.attn_norm2.empty() && quirks_.layer_norm) {
            rd.read_tensor(p + "attn_norm_2.bias", w.attn_norm2_b);
            if (w.attn_norm2_b.size() != cfg_.n_embd) w.attn_norm2_b.clear();
        }
    }

    // "Dense-lead" layer (deepseek2/3, see DenseConfig::n_layer_dense_lead):
    // below that threshold, classic gated FFN even though n_expert>0 globally.
    const bool layer_is_moe = cfg_.n_expert > 0 && il >= cfg_.n_layer_dense_lead;
    if (layer_is_moe) {
        // MoE layer: router instead of the dense FFN, the actual experts
        // are fetched from ExpertStore per-token in step() (they depend on
        // routing, not loadable here once per layer).
        w.wff_gate = MatVec{}; w.wff_up = MatVec{}; w.wff_down = MatVec{};
        if (!load_matrix(rd, p + "ffn_gate_inp.weight", cfg_.n_expert, cfg_.n_embd, w.router)) { last_fail_ = "ffn_gate_inp.weight il=" + std::to_string(il); return false; }

        // Every expert of this layer in one quantized tensor, loaded once,
        // instead of one dequantized expert per token through ExpertStore.
        // The three have to arrive together: a half-stacked layer would mean
        // deciding per matrix which path to take, for no benefit.
        w.wexp_gate = MatVec{}; w.wexp_up = MatVec{}; w.wexp_down = MatVec{};
        w.exps_stacked = false;
        if (stack_experts_ && cfg_.n_ff_expert > 0) {
            const uint32_t stacked_rows = cfg_.n_ff_expert * cfg_.n_expert;
            const uint32_t down_rows    = cfg_.n_embd * cfg_.n_expert;
            if (load_matrix(rd, p + "ffn_gate_exps.weight", stacked_rows, cfg_.n_embd, w.wexp_gate) &&
                load_matrix(rd, p + "ffn_up_exps.weight",   stacked_rows, cfg_.n_embd, w.wexp_up) &&
                load_matrix(rd, p + "ffn_down_exps.weight", down_rows, cfg_.n_ff_expert, w.wexp_down) &&
                w.wexp_gate.format != MatVecFormat::Float &&
                w.wexp_up.format   != MatVecFormat::Float &&
                w.wexp_down.format != MatVecFormat::Float) {
                w.exps_stacked = true;
            } else {
                // Anything unexpected: drop it all and let the per-expert
                // path run, rather than carrying a partial state into decode.
                w.wexp_gate = MatVec{}; w.wexp_up = MatVec{}; w.wexp_down = MatVec{};
            }
        }
        w.router_bias.clear();
        if (quirks_.mla) {
            rd.read_tensor(p + "exp_probs_b.bias", w.router_bias);
            if (w.router_bias.size() != cfg_.n_expert) w.router_bias.clear();
        }
        w.ffn_gate_shexp = MatVec{}; w.ffn_up_shexp = MatVec{}; w.ffn_down_shexp = MatVec{};
        if (quirks_.mla && cfg_.n_expert_shared > 0) {
            const uint32_t n_ff_sh = cfg_.n_ff_expert * cfg_.n_expert_shared;
            if (!load_matrix(rd, p + "ffn_gate_shexp.weight", n_ff_sh, cfg_.n_embd, w.ffn_gate_shexp)) { last_fail_ = "ffn_gate_shexp.weight il=" + std::to_string(il); return false; }
            if (!load_matrix(rd, p + "ffn_up_shexp.weight", n_ff_sh, cfg_.n_embd, w.ffn_up_shexp)) { last_fail_ = "ffn_up_shexp.weight il=" + std::to_string(il); return false; }
            if (!load_matrix(rd, p + "ffn_down_shexp.weight", cfg_.n_embd, n_ff_sh, w.ffn_down_shexp)) { last_fail_ = "ffn_down_shexp.weight il=" + std::to_string(il); return false; }
        }
    } else {
        w.router = MatVec{};
        w.router_bias.clear();
        w.ffn_gate_shexp = MatVec{}; w.ffn_up_shexp = MatVec{}; w.ffn_down_shexp = MatVec{};
        if (quirks_.ffn_gated) {
            if (!load_matrix(rd, p + "ffn_gate.weight", cfg_.n_ff, cfg_.n_embd, w.wff_gate)) { last_fail_ = "ffn_gate.weight(dense-lead) il=" + std::to_string(il) + " n_ff=" + std::to_string(cfg_.n_ff); return false; }
        } else {
            w.wff_gate = MatVec{};
        }
        if (!load_matrix(rd, p + "ffn_up.weight", cfg_.n_ff, cfg_.n_embd, w.wff_up)) { last_fail_ = "ffn_up.weight(dense-lead) il=" + std::to_string(il) + " n_ff=" + std::to_string(cfg_.n_ff); return false; }
        if (!load_matrix(rd, p + "ffn_down.weight", cfg_.n_embd, cfg_.n_ff, w.wff_down)) { last_fail_ = "ffn_down.weight(dense-lead) il=" + std::to_string(il) + " n_ff=" + std::to_string(cfg_.n_ff); return false; }

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
        // "Layer not loaded yet" sentinel: wq is always present in EVERY
        // architecture (attn_norm, on the other hand, stays empty for
        // no_pre_norm architectures like olmo2/exaone4, where it would
        // never refill the cache if used as the sentinel here). MLA
        // (deepseek2/3) doesn't use wq at all (uses wq_a/wq_b or wq_lite):
        // a second check on wkv_a_mqa is needed, always present there.
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

    // kv_quantized_ is a whole-model decision (never set for MLA layers, see
    // the note on the member): only one of the two storages is ever grown,
    // not both — no point paying for an allocation that stays unused.
    if (kv_quantized_) {
        const size_t row_bytes = (size_t) cfg_.n_head_kv * kv_q_row_bytes_;
#ifdef DESIREEIA_CUDA_ENABLED
        // The whole-layer fused path writes the KV cache ONLY on device and
        // then skips write_kv_cache, so by now the host copy is stale and
        // the device one is authoritative. Pull it down BEFORE re-laying
        // it out below, otherwise the re-upload at the end of this function
        // overwrites every position generated so far with stale bytes.
        //
        // This is what made long generations collapse: output stayed
        // perfect up to the first growth (the cache starts at 512
        // positions), then attention read a wiped cache for everything
        // before that point and never recovered.
        if (cuda_kv_ready_ && cache_capacity_ > 0 && g_active_backend == DESIREEIA_BACKEND_CUDA) {
            cuda_kv_cache_download(k_cache_q_.data(), v_cache_q_.data(), k_cache_q_.size());
        }
#endif
        std::vector<uint8_t> nk(cfg_.n_layers * new_cols * row_bytes, 0);
        std::vector<uint8_t> nv(cfg_.n_layers * new_cols * row_bytes, 0);
        if (cache_capacity_ > 0) {
            // Same per-layer copy as the float path below, and for the same
            // reason: a flat memcpy only preserves layer 0 once the
            // per-layer stride changes between the old and new capacity.
            for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
                std::memcpy(nk.data() + (size_t) l * new_cols * row_bytes,
                            k_cache_q_.data() + (size_t) l * cache_capacity_ * row_bytes,
                            cache_capacity_ * row_bytes);
                std::memcpy(nv.data() + (size_t) l * new_cols * row_bytes,
                            v_cache_q_.data() + (size_t) l * cache_capacity_ * row_bytes,
                            cache_capacity_ * row_bytes);
            }
        }
        k_cache_q_.swap(nk);
        v_cache_q_.swap(nv);
        cache_capacity_ = new_cols;
#ifdef DESIREEIA_CUDA_ENABLED
        if (g_active_backend == DESIREEIA_BACKEND_CUDA) {
            const size_t bytes = k_cache_q_.size();
            cuda_kv_ready_ = cuda_kv_cache_reserve(bytes) &&
                              cuda_kv_cache_upload(k_cache_q_.data(), v_cache_q_.data(), bytes);
        }
#endif
        return;
    }

    std::vector<float> nk(cfg_.n_layers * new_cols * kv_dim, 0.0f);
    std::vector<float> nv(cfg_.n_layers * new_cols * kv_dim, 0.0f);
    if (cache_capacity_ > 0) {
        // FIXED BUG (discovered during the MLA work, 2026-09-08): a single
        // memcpy over the whole buffer only preserves layer 0 when the
        // per-layer stride changes (old stride = cache_capacity_*kv_dim,
        // new = new_cols*kv_dim): subsequent layers ended up copied at the
        // wrong offset every time grow_cache was called with
        // cache_capacity_ already > 0 (i.e. beyond a single growth from 0).
        // A layer-by-layer copy is needed, each to its own new offset.
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

#ifdef DESIREEIA_CUDA_ENABLED
    // The device copy follows the host copy's growth and reloads its valid
    // content: after a growth the per-layer offsets change, so a partial
    // mirror would be wrong. If the allocation fails (insufficient VRAM)
    // the mirror stays disabled and attention keeps running on the CPU
    // path — never a half-and-half state.
    if (g_active_backend == DESIREEIA_BACKEND_CUDA) {
        const size_t bytes = k_cache_.size() * sizeof(float);
        cuda_kv_ready_ = cuda_kv_cache_reserve(bytes) &&
                          cuda_kv_cache_upload(k_cache_.data(), v_cache_.data(), bytes);
    }
#endif
}

void DenseForward::write_kv_cache(uint32_t l, uint32_t pos, const float* k, const float* v) {
    const uint32_t kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    if (!kv_quantized_) {
        const size_t elem_off = ((size_t) l * cache_capacity_ + pos) * kv_dim;
        float* kc = k_cache_.data() + elem_off;
        float* vc = v_cache_.data() + elem_off;
        std::memcpy(kc, k, kv_dim * sizeof(float));
        std::memcpy(vc, v, kv_dim * sizeof(float));
#ifdef DESIREEIA_CUDA_ENABLED
        // Mirrors the same position onto the device copy. Here because
        // this is the ONLY place the float KV cache is ever written: every
        // path (batch prefill, decode, CPU fallback) goes through here, so
        // the two copies can never diverge.
        if (cuda_kv_ready_) {
            cuda_kv_cache_write(elem_off * sizeof(float), k, v, kv_dim * sizeof(float));
        }
#endif
        return;
    }
    // One Q8_0 row per kv-head: each head_dim-wide slice gets its own
    // sub-block scales, matching the read side's per-head addressing
    // (kv_dot_q8_0/kv_axpy_q8_0 are called once per head, never across the
    // whole kv_dim-wide position at once — see the note on kv_dot_q8_0 for
    // why a single wide row wouldn't fit how GQA/MHA actually read this).
    const size_t row_bytes = kv_q_row_bytes_;
    const size_t pos_bytes = (size_t) cfg_.n_head_kv * row_bytes;
    uint8_t* kc = k_cache_q_.data() + (size_t) l * cache_capacity_ * pos_bytes + (size_t) pos * pos_bytes;
    uint8_t* vc = v_cache_q_.data() + (size_t) l * cache_capacity_ * pos_bytes + (size_t) pos * pos_bytes;
    for (uint32_t h = 0; h < cfg_.n_head_kv; ++h) {
        kv_quantize_row(k + (size_t) h * cfg_.head_dim, cfg_.head_dim, kc + (size_t) h * row_bytes);
        kv_quantize_row(v + (size_t) h * cfg_.head_dim, cfg_.head_dim, vc + (size_t) h * row_bytes);
    }
#ifdef DESIREEIA_CUDA_ENABLED
    // Mirror the freshly written quantized position onto the device copy,
    // for the same reason as the float branch above: this is the only
    // place the KV cache is ever written.
    if (cuda_kv_ready_) {
        const size_t byte_off = (size_t) l * cache_capacity_ * pos_bytes + (size_t) pos * pos_bytes;
        cuda_kv_cache_write(byte_off, kc, vc, pos_bytes);
    }
#endif
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
    const uint32_t comp_w = kv_lora + rope_w; // "compressed" width shared by Q and K
    const uint32_t k_mla = nope_w + rope_w;

    trace_vec("enter/xnp", l, pos, xnp, n_embd);

    // --- Q: projection (LoRA or direct) -> [n_head * k_mla] ---
    std::vector<float> q(n_head * (size_t) k_mla);
    if (cfg_.q_lora_rank > 0) {
        std::vector<float> qa(cfg_.q_lora_rank);
        matvec(lw->wq_a, cfg_.q_lora_rank, n_embd, xnp, qa.data());
        rms_norm_vec(qa.data(), lw->attn_q_a_norm.data(), qa.data(), cfg_.q_lora_rank, cfg_.rms_eps);
        matvec(lw->wq_b, n_head * (size_t) k_mla, cfg_.q_lora_rank, qa.data(), q.data());
    } else {
        matvec(lw->wq_lite, n_head * (size_t) k_mla, n_embd, xnp, q.data());
    }

    // --- compressed KV + k_pe (shared by every head, MQA) ---
    std::vector<float> kv_cmpr_pe(kv_lora + rope_w);
    matvec(lw->wkv_a_mqa, kv_lora + rope_w, n_embd, xnp, kv_cmpr_pe.data());

    trace_vec("xnp", l, pos, xnp, n_embd);
    trace_vec("q", l, pos, q.data(), q.size());
    trace_vec("kv_cmpr_pe", l, pos, kv_cmpr_pe.data(), kv_cmpr_pe.size());

    float* cache_row = k_cache_.data() + ((size_t) l * cache_capacity_ + pos) * comp_w;
    rms_norm_vec(kv_cmpr_pe.data(), lw->attn_kv_a_norm.data(), cache_row, kv_lora, cfg_.rms_eps);
    trace_vec("latent", l, pos, cache_row, kv_lora);

    std::vector<float> rope_cache_kv(rope_w);
    rope_cache_init(rope_cache_kv, rope_w, pos, rope_th, rope_sc, ext_factor, attn_factor, corr_lo, corr_hi);
    std::memcpy(cache_row + kv_lora, kv_cmpr_pe.data() + kv_lora, rope_w * sizeof(float));
    rope_mla_cached(cache_row + kv_lora, rope_w, rope_cache_kv.data());

    // --- per-head: q_nope absorption, RoPE on q_pe, scores, softmax, de-absorption ---
    std::vector<float> attn_concat((size_t) n_head * v_mla);
    std::vector<float> scores((size_t) pos + 1);
    std::vector<float> qcur(comp_w);
    std::vector<float> attn_raw(kv_lora);
    for (uint32_t h = 0; h < n_head; ++h) {
        const float* qh = q.data() + (size_t) h * k_mla;
        // Absorption: q_nope_absorbed[j] = sum_i wk_b[i,j,h] * q_nope[i],
        // j in [0,kv_lora) — see the derivation from the ggml layout in the
        // note on LayerWeights::wk_b_h in dense_forward.h.
        // Not matvec(): wk_b_h is always Float format (see the note on
        // out_head_/wk_b_h construction), and this call happens on every
        // head of every layer of every token — the fast SIMD path, not the
        // generic scalar-double fallback matvec() would otherwise pick.
        matmul_f32_fast(lw->wk_b_h[h].f.data(), kv_lora, nope_w, qh, qcur.data());
        std::memcpy(qcur.data() + kv_lora, qh + nope_w, rope_w * sizeof(float));
        rope_mla_cached(qcur.data() + kv_lora, rope_w, rope_cache_kv.data());
        if (h == 0) trace_vec("qcur", l, pos, qcur.data(), comp_w);

        const float* cache_base = k_cache_.data() + (size_t) l * cache_capacity_ * comp_w;
        for (uint32_t cc = 0; cc <= pos; ++cc) {
            scores[cc] = dot_f32(qcur.data(), cache_base + (size_t) cc * comp_w, comp_w) * kq_scale;
        }
        if (h == 0) trace_vec("scores_raw", l, pos, scores.data(), (size_t) pos + 1);
        softmax_inplace(scores.data(), (size_t) pos + 1);

        std::fill(attn_raw.begin(), attn_raw.end(), 0.0f);
        for (uint32_t cc = 0; cc <= pos; ++cc) {
            axpy_f32(attn_raw.data(), cache_base + (size_t) cc * comp_w, scores[cc], kv_lora);
        }
        // De-absorption: out[j] = sum_i wv_b[i,j,h] * attn_raw[i], j in [0,v_mla).
        matmul_f32_fast(lw->wv_b_h[h].f.data(), v_mla, kv_lora, attn_raw.data(),
                        attn_concat.data() + (size_t) h * v_mla);
        if (h == 0) trace_vec("attn_raw", l, pos, attn_raw.data(), kv_lora);
    }
    trace_vec("attn_concat", l, pos, attn_concat.data(), attn_concat.size());

    matvec(lw->wo, n_embd, (size_t) n_head * v_mla, attn_concat.data(), proj_out);
    trace_vec("proj_out", l, pos, proj_out, n_embd);
}

bool DenseForward::step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
                        std::vector<float>& last_logits, std::vector<float>* all_logits) {
    last_fail_.clear();
    if (n_tokens == 0 || tokens == nullptr) { last_fail_ = "bad args to step"; return false; }
    if (tok_embd_.empty()) { last_fail_ = "tok_embd_ empty"; return false; }

    const uint32_t n_embd  = cfg_.n_embd;
    const uint32_t n_layers = cfg_.n_layers;
    const uint32_t n_head  = cfg_.n_head;
    const uint32_t q_dim   = cfg_.n_head * cfg_.head_dim;
    const uint32_t kv_dim  = cfg_.n_head_kv * cfg_.head_dim;
    const float  rms_eps  = quirks_.layer_norm ? cfg_.norm_eps : cfg_.rms_eps;
    // rope_th/rope_sc are now per-layer (see DenseConfig::layer_rope_*):
    // they get recomputed inside the layer loop, no longer just once
    // here — gemma3 alternates the RoPE base between local and global layers.
    const uint32_t col0 = (uint32_t) cache_cols_;

    if (col0 + n_tokens > cache_capacity_) {
        grow_cache(col0 + n_tokens);
    }

    std::vector<float> x((size_t) n_tokens * n_embd);
    for (size_t p = 0; p < n_tokens; ++p) {
        const int32_t t = tokens[p];
        if (t < 0 || (uint32_t) t >= cfg_.n_vocab) { last_fail_ = "token out of range t=" + std::to_string(t) + " n_vocab=" + std::to_string(cfg_.n_vocab); return false; }
        bool overridden = false;
        if (!embd_override_.empty() && t == embd_override_token_ &&
            embd_override_cursor_ + n_embd <= embd_override_.size()) {
            std::memcpy(x.data() + p * n_embd,
                        embd_override_.data() + embd_override_cursor_,
                        n_embd * sizeof(float));
            embd_override_cursor_ += n_embd;
            overridden = true;
        }
        if (!overridden) {
            if (!embed_row((uint32_t) t, x.data() + p * n_embd)) { last_fail_ = "embed_row failed t=" + std::to_string(t); return false; }
        }
        if (quirks_.embd_scale_sqrt) {
            const float s = sqrtf((float) n_embd);
            float* xp = x.data() + p * n_embd;
            for (uint32_t i = 0; i < n_embd; ++i) xp[i] *= s;
        }
        // Absolute position (gpt2, mpt optional): added ONCE to the token
        // embedding, before the first layer.
        if (!pos_embd_.empty()) {
            const uint32_t pos = col0 + (uint32_t) p;
            if (pos < n_ctx_train_) {
                float* xp = x.data() + p * n_embd;
                const float* pe = pos_embd_.data() + (size_t) pos * n_embd;
                for (uint32_t i = 0; i < n_embd; ++i) xp[i] += pe[i];
            }
        }
    }
    // Initial norm applied to the embeddings (bloom, DenseQuirks::embd_norm):
    // ONCE only, before the first layer — not per-layer.
    if (quirks_.embd_norm) {
        for (size_t p = 0; p < n_tokens; ++p) {
            float* xp = x.data() + p * n_embd;
            norm_vec(true, xp, tok_norm_.data(),
                    tok_norm_b_.empty() ? nullptr : tok_norm_b_.data(),
                    xp, n_embd, rms_eps);
        }
    }

    std::vector<float> xn((size_t) n_tokens * n_embd);
    // Only if lw->attn_norm2 is present (falcon-40B, see the note on
    // LayerWeights::attn_norm2): Q/K/V input separate from the FFN's.
    std::vector<float> xn_attn;
    std::vector<float> xnp_attn(n_embd);
    std::vector<float> q(q_dim);
    std::vector<float> k(kv_dim);
    std::vector<float> v(kv_dim);
    // Buffers reused by the "shared quantization" path (see
    // quantize_shared_q8k above): declared once to reuse the allocated
    // capacity instead of reallocating on every layer/token.
    std::vector<int8_t> qk_xq;
    std::vector<float> qk_dscale;
    std::vector<int32_t> qk_xsum;
    std::vector<float> gate_vec;    // DenseQuirks::attn_gate, one entry per head
    std::vector<float> rope_cache;  // interleaved cos/sin, one per layer
    std::vector<float> attn_out(q_dim);
    const uint32_t ffn_buf_n = std::max(cfg_.n_ff, cfg_.n_ff_expert);
    std::vector<float> ffn((size_t) ffn_buf_n);
    std::vector<float> ffn_gate((size_t) ffn_buf_n);
    std::vector<float> scores;
    std::vector<float> proj(n_embd);
    std::vector<float> fout(n_embd);
    std::vector<float> hn(n_embd);

    // Buffers for the MoE branch (used only if cfg_.n_expert > 0).
    std::vector<float> router_logits(cfg_.n_expert);
    std::vector<float> expert_out(n_embd);
    std::vector<float> egate, eup, edown;

    // Buffers for the MLA branch (used only if quirks_.mla).
    std::vector<float> mla_xnp(n_embd);
    std::vector<float> mla_proj(n_embd);
    std::vector<float> mla_fnorm(n_embd);
    std::vector<float> mla_fout(n_embd);
    std::vector<float> mla_shexp_gate, mla_shexp_up;
    if (quirks_.mla && cfg_.n_expert_shared > 0) {
        mla_shexp_gate.resize((size_t) cfg_.n_ff_expert * cfg_.n_expert_shared);
        mla_shexp_up.resize((size_t) cfg_.n_ff_expert * cfg_.n_expert_shared);
    }
    // Precompute the MLA YaRN/kq_scale parameters (constant for the whole
    // model: deepseek doesn't alternate the RoPE base per layer like gemma3).
    // See the extended note in DenseConfig::layer_rope_corr_dims and the
    // derivation of kq_scale in the comment on open() (mscale/attn_factor_org).
    float mla_corr_lo = 0.0f, mla_corr_hi = 0.0f, mla_kq_scale = 0.0f;
    if (quirks_.mla) {
        cfg_.layer_rope_corr_dims(0, mla_corr_lo, mla_corr_hi);
        // The plain 1/sqrt(d) scale, unconditionally — matches a real
        // reference implementation's attn_scale exactly and never changes
        // with YaRN.
        //
        // A YaRN attn_factor correction belongs on the ROPE-carrying
        // dimensions ONLY, and rope_cache_init already applies it there
        // (scaling cos/sin, which is passed cfg_.rope_attn_factor below):
        // both q's roped slice and the cached k's roped slice pick it up
        // independently, so their dot product already carries it squared.
        // Multiplying it into kq_scale here on top of that would apply it a
        // second time — and to the "nope" portion of the score too, which
        // never goes through rope_cache_init and has no business being
        // rescaled by a rope correction at all. Measured: with this file's
        // factor=40 the double count alone made the pre-softmax score ~3x
        // too large, collapsing softmax into a near one-hot distribution and
        // degenerating decoding into repeating a handful of dominant tokens.
        mla_kq_scale = 1.0f / sqrtf((float) (cfg_.n_embd_head_qk_nope + cfg_.n_embd_head_qk_rope));
        // Diagnostic escape hatch: isolate the YaRN scale/ramp contribution
        // from the base absorbed-attention math while tracking down a
        // real-model MLA bug. Not a normal user knob.
        if (std::getenv("DESIREEIA_MLA_NO_YARN")) {
            mla_kq_scale = 1.0f / sqrtf((float) (cfg_.n_embd_head_qk_nope + cfg_.n_embd_head_qk_rope));
            cfg_.rope_ext_factor = 0.0f;
            cfg_.rope_attn_factor = 1.0f;
            // freq_scale has to go back to 1 as well: leaving it at 1/factor
            // is not "YaRN off", it is plain RoPE rotating `factor` times too
            // slowly — a different corruption, which made an earlier run of
            // this same experiment worthless.
            cfg_.rope_freq_scale = 1.0f;
        }
    }

    // Buffers for the batched path (Phase 8, used only when n_tokens > 1,
    // i.e. in prefill: single-token decode stays on the per-token path
    // below, already measured and validated, so as not to introduce
    // risk/overhead on the critical path for tok/s). The Q/K/V/wo
    // projections and the dense FFN are computed with ONE batched call for
    // the whole layer instead of n_tokens separate calls: the weight's raw
    // bytes get decoded only once and reused across every column (see
    // matmul_qX_k_batch in core/matmul.cpp). Attention stays per-token
    // (sequential dependency on the KV cache/causal mask), but that isn't
    // the dominant bottleneck for large matrices.
    std::vector<float> q_all, k_all, v_all, attn_out_all, proj_all, ffn_xn_all,
                        ffn_all, ffn_gate_all, fout_all;
    std::vector<float> gate_all;      // DenseQuirks::attn_gate, n_head per token
    std::vector<float> rope_cache_b;  // cos/sin for the batch path
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
        if (!lw) { if (last_fail_.empty()) last_fail_ = "get_layer null l=" + std::to_string(l); return false; }

        // Per-layer RoPE: local (sliding window) layers and global ones
        // use different base/scale — see the note in DenseConfig.
        const float rope_th = cfg_.layer_rope_theta(l);
        const float rope_sc = cfg_.layer_rope_scale(l);

        if (quirks_.mla) {
            // A single per-token path, reused for both prefill
            // (n_tokens>1) and decode: MLA has no "batched" equivalent
            // here (every token still needs the two per-head
            // absorption/de-absorption matvecs, see mla_attn_layer), so
            // there is no large matmul to batch like in the classic
            // dense/GQA path.
            for (size_t p = 0; p < n_tokens; ++p) {
                const uint32_t pos = col0 + (uint32_t) p;
                const float* xp = x.data() + p * n_embd;

                norm_vec(false, xp, lw->attn_norm.data(), nullptr, mla_xnp.data(), n_embd, rms_eps);

                mla_attn_layer(lw, l, pos, mla_xnp.data(), rope_th, rope_sc,
                               cfg_.rope_ext_factor, cfg_.rope_attn_factor, mla_corr_lo, mla_corr_hi,
                               mla_kq_scale, mla_proj.data());

                for (uint32_t i = 0; i < n_embd; ++i) mla_proj[i] += xp[i];
                if (p == 0) trace_vec("resid_attn", l, pos, mla_proj.data(), n_embd);
                norm_vec(false, mla_proj.data(), lw->ffn_norm.data(), nullptr, mla_fnorm.data(), n_embd, rms_eps);

                const bool layer_is_moe = cfg_.n_expert > 0 && l >= cfg_.n_layer_dense_lead;
                if (layer_is_moe) {
                    matvec(lw->router, cfg_.n_expert, n_embd, mla_fnorm.data(), router_logits.data());
                    auto selected = moe_route_ex(router_logits, cfg_.n_expert_used, cfg_.moe_norm_w, cfg_.moe_w_scale,
                                                  cfg_.moe_sigmoid_gate ? MoeGatingFunc::Sigmoid : MoeGatingFunc::Softmax,
                                                  lw->router_bias.empty() ? nullptr : &lw->router_bias);
                    maybe_prefetch_experts(*lw, l, selected);
                    maybe_predict_next_layer(l, mla_fnorm.data(), selected);
                    std::fill(expert_out.begin(), expert_out.end(), 0.0f);
                    static const bool skip_routed = std::getenv("DESIREEIA_MOE_SHARED_ONLY") != nullptr;
                    if (!skip_routed)
                    for (const auto& sel : selected) {
                        const uint32_t eidx = sel.first;
                        const float weight = sel.second;
                        if (!expert_ffn(*lw, l, eidx, mla_fnorm.data(), /*gelu_act=*/false,
                                        ffn, ffn_gate, mla_fout, egate, eup, edown)) {
                            continue;
                        }
                        for (uint32_t i = 0; i < n_embd; ++i) expert_out[i] += weight * mla_fout[i];
                    }
                    mla_fout = expert_out;
                    if (p == 0) {
                        trace_vec("moe_routed", l, pos, expert_out.data(), n_embd);
                        trace_vec("moe_fnorm", l, pos, mla_fnorm.data(), n_embd);
                        char m2[160];
                        std::snprintf(m2, sizeof(m2), "l=%u pos=%u sel=%zu w0=%.5g w1=%.5g",
                                      l, pos, selected.size(),
                                      selected.empty() ? 0.0 : (double) selected[0].second,
                                      selected.size() > 1 ? (double) selected[1].second : 0.0);
                        trace_msg(m2);
                    }

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
                        if (p == 0) trace_vec("moe_shared", l, pos, shexp_out.data(), n_embd);
                        for (uint32_t i = 0; i < n_embd; ++i) mla_fout[i] += shexp_out[i];
                    }
                } else {
                    // "Dense-lead" layer: classic gated FFN (SiLU), the
                    // same tensors/formula already used by the other
                    // architectures.
                    matvec(lw->wff_up, cfg_.n_ff, n_embd, mla_fnorm.data(), ffn.data());
                    matvec(lw->wff_gate, cfg_.n_ff, n_embd, mla_fnorm.data(), ffn_gate.data());
                    if (p == 0) {
                        trace_vec("dense_xin", l, pos, mla_fnorm.data(), n_embd);
                        trace_vec("dense_up", l, pos, ffn.data(), cfg_.n_ff);
                        trace_vec("dense_gate", l, pos, ffn_gate.data(), cfg_.n_ff);
                    }
                    for (uint32_t i = 0; i < cfg_.n_ff; ++i) ffn[i] = silu(ffn_gate[i]) * ffn[i];
                    matvec(lw->wff_down, n_embd, cfg_.n_ff, ffn.data(), mla_fout.data());
                }

                float* xdst = x.data() + p * n_embd;
                for (uint32_t i = 0; i < n_embd; ++i) xdst[i] = mla_fout[i] + mla_proj[i];
                if (p == 0) {
                    trace_vec("ffn_out", l, pos, mla_fout.data(), n_embd);
                    trace_vec("resid_end", l, pos, xdst, n_embd);
                }
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
                // Same cos/sin cache as the decode path: depends only on
                // position and the layer's RoPE parameters, not on the head.
                // rope_only_swa (cohere2): NoPE on global layers, no rope
                // call — Qcur/Kcur stay unchanged.
                if (!quirks_.no_rope && (!quirks_.rope_only_swa || cfg_.layer_is_swa(l))) {
                    const uint32_t n_rot_l = cfg_.layer_n_rot(l);
                    rope_cache_init(rope_cache_b, n_rot_l, pos, rope_th, rope_sc);
                    for (uint32_t h = 0; h < n_head; ++h) {
                        rope_neox_cached(qp + (size_t) h * cfg_.head_dim, n_rot_l, rope_cache_b.data());
                    }
                    for (uint32_t h = 0; h < cfg_.n_head_kv; ++h) {
                        rope_neox_cached(kp + (size_t) h * cfg_.head_dim, n_rot_l, rope_cache_b.data());
                    }
                }

                write_kv_cache(l, pos, kp, vp);
            }

            // Attention for the whole batch in ONE dispatch over every
            // (token, head) pair.
            //
            // It used to sit inside the token loop above, which dispatched
            // the thread pool once per token per layer with only n_head
            // units of work each — for a 600-token prompt on a 36-layer
            // model that is ~21k dispatches of 16 units apiece, most of
            // the pool idle every time, and it allocated a fresh scores
            // vector per head per token per layer (~350k heap allocations
            // for one prefill). Every (token, head) pair is independent
            // and reads only KV positions written by the loop above, which
            // has now fully completed, so one dispatch over the product is
            // the same arithmetic with the work actually spread out.
            {
            const size_t attn_units = n_tokens * (size_t) n_head;
            parallel_units(attn_units, [&](size_t u_begin, size_t u_end) {
                // One scratch buffer per worker range, not per unit.
                std::vector<float> scores_buf((size_t) col0 + n_tokens, 0.0f);
                for (size_t u = u_begin; u < u_end; ++u) {
                    const size_t p = u / (size_t) n_head;
                    const uint32_t h = (uint32_t) (u % (size_t) n_head);
                    const uint32_t pos = col0 + (uint32_t) p;
                    const float* qp = q_all.data() + p * q_dim;
                    const bool is_swa = cfg_.n_swa > 0 && cfg_.layer_is_swa(l);
                    const uint32_t cc_start = is_swa
                        ? ((pos + 1 > cfg_.n_swa) ? (pos + 1 - cfg_.n_swa) : 0)
                        : 0;
                    float* aout = attn_out_all.data() + p * q_dim;
                    const uint32_t hkv = h / heads_per_kv;
                    const float* qh = qp + (size_t) h * cfg_.head_dim;
                    float* scores_h = scores_buf.data();
                    float* oh = aout + (size_t) h * cfg_.head_dim;
                    if (kv_quantized_) {
                        const size_t row_bytes = kv_q_row_bytes_;
                        const size_t pos_bytes = (size_t) cfg_.n_head_kv * row_bytes;
                        const uint8_t* kb = k_cache_q_.data() + (size_t) l * cache_capacity_ * pos_bytes
                                           + (size_t) hkv * row_bytes;
                        const uint8_t* vb = v_cache_q_.data() + (size_t) l * cache_capacity_ * pos_bytes
                                           + (size_t) hkv * row_bytes;
                        if (quirks_.alibi) {
                            const float slope = alibi_slope(h, n_head, cfg_.max_alibi_bias);
                            for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                                scores_h[cc] = kv_dot_q8_0(qh, kb + (size_t) cc * pos_bytes, cfg_.head_dim) * inv_d
                                             - slope * (float) (pos - cc);
                            }
                        } else {
                            for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                                scores_h[cc] = kv_dot_q8_0(qh, kb + (size_t) cc * pos_bytes, cfg_.head_dim) * inv_d;
                            }
                        }
                        softmax_inplace(scores_h + cc_start, (size_t) pos + 1 - cc_start);
                        for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] = 0.0f;
                        for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                            kv_axpy_q8_0(oh, vb + (size_t) cc * pos_bytes, cfg_.head_dim, scores_h[cc]);
                        }
                    } else {
                        const float* kb = k_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
                        const float* vb = v_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
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
                        for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] = 0.0f;
                        for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                            axpy_f32(oh, vb + (size_t) cc * kv_dim, scores_h[cc], cfg_.head_dim);
                        }
                    }
                }
            });
            }

            // Per-head output gate (see DenseQuirks::attn_gate): computed
            // per token from the same normalized input that fed Q/K/V.
            if (quirks_.attn_gate) {
                gate_all.resize((size_t) n_tokens * n_head);
                matvec_batch(lw->attn_gate, n_head, n_embd, attn_in_all, n_tokens,
                             gate_all.data());
                for (size_t p = 0; p < n_tokens; ++p) {
                    float* aout = attn_out_all.data() + p * q_dim;
                    const float* gp = gate_all.data() + p * n_head;
                    for (uint32_t h = 0; h < n_head; ++h) {
                        const float g = 1.0f / (1.0f + std::exp(-gp[h]));
                        float* oh = aout + (size_t) h * cfg_.head_dim;
                        for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] *= g;
                    }
                }
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
                    // cohere2: proj_all stays ONLY the attention output
                    // (the input residual gets added at the end together
                    // with the FFN output, not here). ffn_xn_all reuses the
                    // same xn already computed for Q/K/V — no separate
                    // ffn_norm.
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
                // MoE branch: per-token gating (depends on the router for
                // every position), stays iterated token by token as before.
                for (size_t p = 0; p < n_tokens; ++p) {
                    float* xnp = ffn_xn_all.data() + p * n_embd;
                    float* projp = proj_all.data() + p * n_embd;
                    matvec(lw->router, cfg_.n_expert, n_embd, xnp, router_logits.data());
                    auto selected = moe_route(router_logits, cfg_.n_expert_used, cfg_.moe_norm_w, cfg_.moe_w_scale);
                    maybe_prefetch_experts(*lw, l, selected);
                    maybe_predict_next_layer(l, xnp, selected);
                    std::fill(expert_out.begin(), expert_out.end(), 0.0f);
                    for (const auto& sel : selected) {
                        const uint32_t eidx = sel.first;
                        const float weight = sel.second;
                        if (!expert_ffn(*lw, l, eidx, xnp, quirks_.gelu_tanh,
                                        ffn, ffn_gate, fout, egate, eup, edown)) {
                            continue;
                        }
                        for (uint32_t i = 0; i < n_embd; ++i) expert_out[i] += weight * fout[i];
                    }
                    if (quirks_.sandwich_norm) {
                        rms_norm_vec(expert_out.data(), lw->post_ffn_norm.data(), expert_out.data(), n_embd, rms_eps);
                    }
                    float* xn2 = x.data() + p * n_embd;
                    if (quirks_.parallel_residual) {
                        const float* xp = x.data() + p * n_embd; // read BEFORE writing xn2 (same buffer)
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
                        // xn2 and xp point at the SAME cell (x[p]): reading
                        // xn2[i] and writing it in the same expression is
                        // safe (same index on both sides, like the
                        // equivalent pattern in the MoE branch above).
                        for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = foutp[i] + projp[i] + xn2[i];
                    } else {
                        for (uint32_t i = 0; i < n_embd; ++i) xn2[i] = foutp[i] + projp[i];
                    }
                }
            } else {
                // Non-gated FFN: a single up projection [+bias] -> activation
                // -> down [+bias]. No ffn_gate, no gate*up product.
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

#ifdef DESIREEIA_CUDA_ENABLED
            // Whole layer in one device episode (one captured graph per
            // layer). Everything the plain dense path does — norms,
            // Q/K/V, RoPE, KV write, attention, output projection, both
            // residuals and the gated FFN — runs on the GPU, so the
            // layer's intermediates never touch the host. Any quirk
            // outside this shape falls through to the code below, which
            // stays the definition of record.
            static const bool layer_fused_enabled =
                std::getenv("DESIREEIA_CUDA_NO_LAYER") == nullptr;
            if (layer_fused_enabled &&
                g_active_backend == DESIREEIA_BACKEND_CUDA && cuda_kv_ready_ &&
                kv_quantized_ && quirks_.ffn_gated && cfg_.n_expert == 0 &&
                !quirks_.alibi && !quirks_.mla &&
                !(quirks_.qk_norm && quirks_.qk_norm_full_width) &&
                !quirks_.no_pre_norm && !quirks_.layer_norm &&
                !quirks_.parallel_residual &&
                (!quirks_.attn_gate || cuda_layer_ready(lw->attn_gate)) &&
                cfg_.clamp_kqv <= 0.0f &&
                lw->attn_norm2.empty() && lw->attn_norm_b.empty() &&
                lw->ffn_norm_b.empty() && lw->bo.empty() && lw->ffn_down_b.empty() &&
                cuda_layer_ready(lw->wq) && cuda_layer_ready(lw->wk) &&
                cuda_layer_ready(lw->wv) && cuda_layer_ready(lw->wo) &&
                cuda_layer_ready(lw->wff_gate) && cuda_layer_ready(lw->wff_up) &&
                cuda_layer_ready(lw->wff_down) &&
                lw->wq.lora.empty() && lw->wk.lora.empty() && lw->wv.lora.empty() &&
                lw->wo.lora.empty() && lw->wff_gate.lora.empty() &&
                lw->wff_up.lora.empty() && lw->wff_down.lora.empty()) {

                const bool do_rope = !quirks_.rope_only_swa || cfg_.layer_is_swa(l);
                // With no sliding-window pattern every layer shares the
                // same RoPE parameters and window, so the table is built
                // and sent once per token rather than 36 times.
                const bool uniform_layers = cfg_.swa_pattern == 0 && cfg_.swa_mask.empty();
                const bool send_dyn = (l == 0) || !uniform_layers;
                if (do_rope && send_dyn) {
                    rope_cache_init(rope_cache, cfg_.layer_n_rot(l), pos, rope_th, rope_sc);
                }
                const size_t pos_bytes = (size_t) cfg_.n_head_kv * kv_q_row_bytes_;
                const bool swa_here = cfg_.n_swa > 0 && cfg_.layer_is_swa(l);

                CudaLayerArgs la{};
                la.wq_qs = lw->wq.cuda_qs.get();       la.wq_scale = lw->wq.cuda_scale.get();
                la.wk_qs = lw->wk.cuda_qs.get();       la.wk_scale = lw->wk.cuda_scale.get();
                la.wv_qs = lw->wv.cuda_qs.get();       la.wv_scale = lw->wv.cuda_scale.get();
                la.wo_qs = lw->wo.cuda_qs.get();       la.wo_scale = lw->wo.cuda_scale.get();
                la.wgate_qs = lw->wff_gate.cuda_qs.get(); la.wgate_scale = lw->wff_gate.cuda_scale.get();
                la.wup_qs = lw->wff_up.cuda_qs.get();  la.wup_scale = lw->wff_up.cuda_scale.get();
                la.wdown_qs = lw->wff_down.cuda_qs.get(); la.wdown_scale = lw->wff_down.cuda_scale.get();
                la.bq = (quirks_.qkv_bias && !lw->bq.empty()) ? lw->bq.data() : nullptr;
                la.bk = (quirks_.qkv_bias && !lw->bk.empty()) ? lw->bk.data() : nullptr;
                la.bv = (quirks_.qkv_bias && !lw->bv.empty()) ? lw->bv.data() : nullptr;
                la.attn_norm_w = lw->attn_norm.data();
                la.ffn_norm_w = lw->ffn_norm.data();
                la.q_norm_w = (quirks_.qk_norm && !lw->q_norm.empty()) ? lw->q_norm.data() : nullptr;
                la.k_norm_w = (quirks_.qk_norm && !lw->k_norm.empty()) ? lw->k_norm.data() : nullptr;
                la.post_attn_norm_w = (quirks_.sandwich_norm && !lw->post_attn_norm.empty())
                    ? lw->post_attn_norm.data() : nullptr;
                la.post_ffn_norm_w = (quirks_.sandwich_norm && !lw->post_ffn_norm.empty())
                    ? lw->post_ffn_norm.data() : nullptr;
                la.x = xp;
                la.rope_cache = do_rope ? rope_cache.data() : nullptr;
                la.x_out = x.data() + p * n_embd;   // in place, as the CPU path does
                la.n_embd = n_embd; la.q_dim = q_dim; la.kv_dim = kv_dim; la.n_ff = cfg_.n_ff;
                la.n_head = n_head; la.n_head_kv = cfg_.n_head_kv;
                la.heads_per_kv = n_head / cfg_.n_head_kv;
                la.head_dim = cfg_.head_dim; la.n_rot = cfg_.layer_n_rot(l);
                la.kv_layer_off = (size_t) l * cache_capacity_ * pos_bytes;
                la.cc_start = swa_here
                    ? ((pos + 1 > cfg_.n_swa) ? (pos + 1 - cfg_.n_swa) : 0) : 0;
                la.pos = pos;
                la.rms_eps = rms_eps;
                la.act_gelu = quirks_.gelu_tanh ? 1 : 0;
                la.fmt_q = cuda_fmt_of(lw->wq.format);
                la.fmt_k = cuda_fmt_of(lw->wk.format);
                la.fmt_v = cuda_fmt_of(lw->wv.format);
                la.fmt_o = cuda_fmt_of(lw->wo.format);
                la.fmt_gate = cuda_fmt_of(lw->wff_gate.format);
                la.fmt_up = cuda_fmt_of(lw->wff_up.format);
                la.fmt_down = cuda_fmt_of(lw->wff_down.format);
                if (quirks_.attn_gate) {
                    la.wag_qs = lw->attn_gate.cuda_qs.get();
                    la.wag_scale = lw->attn_gate.cuda_scale.get();
                    la.fmt_ag = cuda_fmt_of(lw->attn_gate.format);
                }
                // x crosses the whole stack on device: uploaded once before
                // the first layer, read back once after the last.
                la.upload_x = (l == 0) ? 1 : 0;
                la.download_x = (l + 1 == cfg_.n_layers) ? 1 : 0;
                la.upload_dyn = send_dyn ? 1 : 0;
                if (cuda_layer_forward(la) == DESIREEIA_OK) continue;
            }
#endif
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
#ifdef DESIREEIA_CUDA_ENABLED
            // Whole attention block in one device episode. Only the plain
            // path qualifies; every quirk below is handled by the CPU code
            // that follows, which stays the definition of record.
            // ON by default: measured faster than running the projections,
            // the attention and the output projection as separate episodes
            // (37.0 vs 33.8 tok/s), with identical output. Set
            // DESIREEIA_CUDA_NO_FUSED to fall back to the split path.
            static const bool fused_attn_enabled =
                std::getenv("DESIREEIA_CUDA_NO_FUSED") == nullptr;
            bool cuda_block_done = false;
            if (fused_attn_enabled &&
                g_active_backend == DESIREEIA_BACKEND_CUDA && cuda_kv_ready_ &&
                kv_quantized_ && !quirks_.alibi && !quirks_.qk_norm &&
                !quirks_.mla && !quirks_.attn_gate &&
                cfg_.clamp_kqv <= 0.0f && cfg_.n_expert == 0 &&
                lw->wq.format == MatVecFormat::Q8_0 && lw->wq.cuda_qs &&
                lw->wk.format == MatVecFormat::Q8_0 && lw->wk.cuda_qs &&
                lw->wv.format == MatVecFormat::Q8_0 && lw->wv.cuda_qs &&
                lw->wo.format == MatVecFormat::Q8_0 && lw->wo.cuda_qs &&
                lw->wq.lora.empty() && lw->wk.lora.empty() &&
                lw->wv.lora.empty() && lw->wo.lora.empty() && lw->bo.empty()) {

                const bool do_rope = !quirks_.rope_only_swa || cfg_.layer_is_swa(l);
                // With no sliding-window pattern every layer shares the
                // same RoPE parameters and window, so the table is built
                // and sent once per token rather than 36 times.
                const bool uniform_layers = cfg_.swa_pattern == 0 && cfg_.swa_mask.empty();
                const bool send_dyn = (l == 0) || !uniform_layers;
                if (do_rope && send_dyn) {
                    rope_cache_init(rope_cache, cfg_.layer_n_rot(l), pos, rope_th, rope_sc);
                }

                const size_t pos_bytes = (size_t) cfg_.n_head_kv * kv_q_row_bytes_;
                const size_t layer_off = (size_t) l * cache_capacity_ * pos_bytes;
                const bool swa_here = cfg_.n_swa > 0 && cfg_.layer_is_swa(l);
                CudaQkvAttnArgs ca{};
                ca.wq_qs = lw->wq.cuda_qs.get();   ca.wq_scale = lw->wq.cuda_scale.get();
                ca.wk_qs = lw->wk.cuda_qs.get();   ca.wk_scale = lw->wk.cuda_scale.get();
                ca.wv_qs = lw->wv.cuda_qs.get();   ca.wv_scale = lw->wv.cuda_scale.get();
                ca.wo_qs = lw->wo.cuda_qs.get();   ca.wo_scale = lw->wo.cuda_scale.get();
                ca.bq = (quirks_.qkv_bias && !lw->bq.empty()) ? lw->bq.data() : nullptr;
                ca.bk = (quirks_.qkv_bias && !lw->bk.empty()) ? lw->bk.data() : nullptr;
                ca.bv = (quirks_.qkv_bias && !lw->bv.empty()) ? lw->bv.data() : nullptr;
                ca.attn_in = attn_in;
                ca.rope_cache = do_rope ? rope_cache.data() : nullptr;
                ca.proj = proj.data();
                ca.host_k_row = k_cache_q_.data() + layer_off + (size_t) pos * pos_bytes;
                ca.host_v_row = v_cache_q_.data() + layer_off + (size_t) pos * pos_bytes;
                ca.n_embd = n_embd; ca.q_dim = q_dim; ca.kv_dim = kv_dim;
                ca.n_head = n_head; ca.n_head_kv = cfg_.n_head_kv;
                ca.heads_per_kv = n_head / cfg_.n_head_kv;
                ca.head_dim = cfg_.head_dim; ca.n_rot = cfg_.layer_n_rot(l);
                ca.kv_layer_off = layer_off;
                ca.cc_start = swa_here
                    ? ((pos + 1 > cfg_.n_swa) ? (pos + 1 - cfg_.n_swa) : 0) : 0;
                ca.pos = pos;
                // DESIREEIA_CUDA_VERIFY=1 runs the fused episode and then
                // the CPU path on the same inputs and reports the largest
                // divergence in proj, layer by layer. Kept because the
                // first version of this episode produced correct output
                // for one token and then degenerated: a end-to-end text
                // check cannot say WHICH stage drifted, this can.
                static const bool verify =
                    std::getenv("DESIREEIA_CUDA_VERIFY") != nullptr;
                std::vector<float> proj_gpu;
                cuda_block_done = (cuda_qkv_attention_out(ca) == DESIREEIA_OK);
                if (verify && cuda_block_done) {
                    proj_gpu.assign(proj.begin(), proj.begin() + n_embd);
                    cuda_block_done = false;  // fall through and redo on CPU
                    verify_proj_gpu_ = std::move(proj_gpu);
                    verify_layer_ = l;
                }
            }
            // When the episode ran, proj is ready and the KV cache (device
            // plus its host mirror) is written: the whole CPU block below
            // is skipped.
            if (!cuda_block_done) {
#endif
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
            // A single cache build per layer, then reused by every Q and K
            // head (see rope_cache_init). rope_only_swa (cohere2): NoPE on
            // global layers, no rope call.
            if (!quirks_.rope_only_swa || cfg_.layer_is_swa(l)) {
                const uint32_t n_rot_l = cfg_.layer_n_rot(l);
                rope_cache_init(rope_cache, n_rot_l, pos, rope_th, rope_sc);
                for (uint32_t h = 0; h < n_head; ++h) {
                    rope_neox_cached(q.data() + (size_t) h * cfg_.head_dim, n_rot_l, rope_cache.data());
                }
                for (uint32_t h = 0; h < cfg_.n_head_kv; ++h) {
                    rope_neox_cached(k.data() + (size_t) h * cfg_.head_dim, n_rot_l, rope_cache.data());
                }
            }

            write_kv_cache(l, pos, k.data(), v.data());
            }

            // True when attention and the output projection were run on
            // device in a single shot (see below): in that case proj is
            // already ready and wo's matvec must be skipped.
            bool cuda_attn_done = false;
            {
            ScopedTimer attn_timer(profile_counters().ns_ser_attn);
            const float inv_d = 1.0f / sqrtf((float) cfg_.head_dim);
            const uint32_t heads_per_kv = n_head / cfg_.n_head_kv;
            // Heads are independent of each other: they read the same KV
            // cache and write separate attn_out blocks. Parallelizing them
            // is the single largest item at realistic context — per the
            // profiler, 269 ms out of 672 (40% of decode) were serial
            // attention with 562 tokens of context, while with a 12-token
            // prompt the cost was invisible. Each head has its own slice of
            // `scores`, otherwise the threads would overwrite each other.
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
#ifdef DESIREEIA_CUDA_ENABLED
            // Attention + output projection on device, in a single episode.
            // Conditions: float KV cache (the quantized variant has a
            // different layout, not ported yet), no ALiBi (adds a
            // per-position term), resident wo weights and no LoRA. Outside
            // of these, proceeds with the CPU path below, which remains the
            // reference definition.
            const size_t kv_layer_off = kv_quantized_
                ? (size_t) l * cache_capacity_ * ((size_t) cfg_.n_head_kv * kv_q_row_bytes_)
                : ((size_t) l * cache_capacity_) * kv_dim * sizeof(float);
            if (g_active_backend == DESIREEIA_BACKEND_CUDA && cuda_kv_ready_ &&
                !quirks_.alibi &&
                lw->wo.format == MatVecFormat::Q8_0 && lw->wo.cuda_qs &&
                lw->wo.lora.empty() && lw->bo.empty() &&
                cuda_attention_out(
                    lw->wo.cuda_qs.get(), lw->wo.cuda_scale.get(), q.data(),
                    n_head, heads_per_kv, cfg_.head_dim, kv_dim,
                    kv_layer_off, cc_start, pos,
                    n_embd, q_dim, proj.data(),
                    kv_quantized_ ? 1 : 0, kv_q_row_bytes_) == DESIREEIA_OK) {
                cuda_attn_done = true;
            }
            if (!cuda_attn_done) {
#endif
            scores.assign((size_t) n_head * score_stride, 0.0f);
            parallel_units(n_head, [&](size_t h_begin, size_t h_end) {
            for (uint32_t h = (uint32_t) h_begin; h < (uint32_t) h_end; ++h) {
                float* scores_h = scores.data() + (size_t) h * score_stride;
                const uint32_t hkv = h / heads_per_kv;
                const float* qh = q.data() + (size_t) h * cfg_.head_dim;
                float* oh = attn_out.data() + (size_t) h * cfg_.head_dim;
                // Kept in float and vectorized, not a `double`
                // accumulator: double entirely blocked vectorization of
                // the innermost attention dot product, and the scores go
                // through a softmax right afterward anyway.
                //
                // LOOP ORDER REVERSED (2026-09-07) in the V accumulation.
                // Before, the outer loop was over d and the inner over cc:
                //     for d: for cc: acc += scores[cc] * vb[cc*kv_dim + d];
                // i.e. the innermost access jumped by kv_dim floats (4 KB
                // with kv_dim=1024) on every iteration: EVERY access landed
                // on a different cache line, and none of the loaded lines
                // got reused before being evicted. ~1.5 million miss
                // accesses per token counted, enough on their own to
                // account for the ~17 ms of serial time measured by the
                // profiler. With the right order the inner loop walks a
                // contiguous row of V, so each cache line serves 16 values
                // and it's also vectorizable. Applies to both branches below.
                if (kv_quantized_) {
                    const size_t row_bytes = kv_q_row_bytes_;
                    const size_t pos_bytes = (size_t) cfg_.n_head_kv * row_bytes;
                    const uint8_t* kb = k_cache_q_.data() + (size_t) l * cache_capacity_ * pos_bytes
                                       + (size_t) hkv * row_bytes;
                    const uint8_t* vb = v_cache_q_.data() + (size_t) l * cache_capacity_ * pos_bytes
                                       + (size_t) hkv * row_bytes;
                    if (quirks_.alibi) {
                        const float slope = alibi_slope(h, n_head, cfg_.max_alibi_bias);
                        for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                            scores_h[cc] = kv_dot_q8_0(qh, kb + (size_t) cc * pos_bytes, cfg_.head_dim) * inv_d
                                         - slope * (float) (pos - cc);
                        }
                    } else {
                        for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                            scores_h[cc] = kv_dot_q8_0(qh, kb + (size_t) cc * pos_bytes, cfg_.head_dim) * inv_d;
                        }
                    }
                    softmax_inplace(scores_h + cc_start, score_stride - cc_start);
                    for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] = 0.0f;
                    for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                        kv_axpy_q8_0(oh, vb + (size_t) cc * pos_bytes, cfg_.head_dim, scores_h[cc]);
                    }
                } else {
                    const float* kb = k_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
                    const float* vb = v_cache_.data() + ((size_t) l * cache_capacity_) * kv_dim + (size_t) hkv * cfg_.head_dim;
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
                    // Softmax only over the sub-window [cc_start, pos]:
                    // positions before cc_start don't participate
                    // (equivalent to a -inf score), and aren't even read by
                    // the V accumulation loop below.
                    softmax_inplace(scores_h + cc_start, score_stride - cc_start);
                    for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] = 0.0f;
                    for (uint32_t cc = cc_start; cc <= pos; ++cc) {
                        axpy_f32(oh, vb + (size_t) cc * kv_dim, scores_h[cc], cfg_.head_dim);
                    }
                }
            }
            });
#ifdef DESIREEIA_CUDA_ENABLED
            }
#endif
            }

            // Per-head output gate, between the attention and the output
            // projection (see DenseQuirks::attn_gate). It reads the SAME
            // normalized input that fed Q/K/V, not the attention output.
            if (quirks_.attn_gate && !cuda_attn_done) {
                gate_vec.resize(n_head);
                matvec(lw->attn_gate, n_head, n_embd, attn_in, gate_vec.data());
                for (uint32_t h = 0; h < n_head; ++h) {
                    const float g = 1.0f / (1.0f + std::exp(-gate_vec[h]));
                    float* oh = attn_out.data() + (size_t) h * cfg_.head_dim;
                    for (uint32_t d = 0; d < cfg_.head_dim; ++d) oh[d] *= g;
                }
            }

            // On the CUDA path proj was already produced together with the
            // attention (its output never came back to the host).
            if (!cuda_attn_done) {
                matvec(lw->wo, n_embd, q_dim, attn_out.data(), proj.data());
            }
#ifdef DESIREEIA_CUDA_ENABLED
            }  // end of the CPU attention block skipped by the fused episode
            if (!verify_proj_gpu_.empty() && verify_layer_ == l) {
                float worst = 0.0f;
                size_t worst_i = 0;
                for (uint32_t i = 0; i < n_embd; ++i) {
                    const float d = std::fabs(verify_proj_gpu_[i] - proj[i]);
                    if (d > worst) { worst = d; worst_i = i; }
                }
                std::fprintf(stderr,
                    "[cuda-verify] layer=%u pos=%u worst|gpu-cpu|=%.6f at %zu (gpu=%.6f cpu=%.6f)\n",
                    l, pos, worst, worst_i, verify_proj_gpu_[worst_i], proj[worst_i]);
                verify_proj_gpu_.clear();
            }
#endif
            { ScopedTimer t(profile_counters().ns_ser_norm);
            if (!lw->bo.empty()) {
                for (uint32_t i = 0; i < n_embd; ++i) proj[i] += lw->bo[i];
            }
            if (quirks_.sandwich_norm) {
                rms_norm_vec(proj.data(), lw->post_attn_norm.data(), proj.data(), n_embd, rms_eps);
            }
            if (quirks_.parallel_residual) {
                // cohere2: proj stays ONLY the attention output (the
                // residual gets added at the end together with the FFN
                // output); xnp stays the one already computed above for
                // Q/K/V (same shared norm), no separate ffn_norm.
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
                // Router: one logit per expert, then softmax + top-k (see
                // models/moe_route.h). Weighted combination of the FFN
                // outputs of only the selected experts, fetched from
                // ExpertStore (dequantized; no quantized kernel for the
                // experts yet, a known gap).
                matvec(lw->router, cfg_.n_expert, n_embd, xnp, router_logits.data());
                auto selected = moe_route(router_logits, cfg_.n_expert_used, cfg_.moe_norm_w, cfg_.moe_w_scale);
                maybe_prefetch_experts(*lw, l, selected);
                maybe_predict_next_layer(l, xnp, selected);

                std::fill(expert_out.begin(), expert_out.end(), 0.0f);
                for (const auto& sel : selected) {
                    const uint32_t eidx = sel.first;
                    const float weight = sel.second;
                    if (!expert_ffn(*lw, l, eidx, xnp, quirks_.gelu_tanh,
                                    ffn, ffn_gate, fout, egate, eup, edown)) {
                        continue;
                    }
                    for (uint32_t i = 0; i < n_embd; ++i) expert_out[i] += weight * fout[i];
                }
                fout = expert_out;
            } else if (quirks_.ffn_gated) {
#ifdef DESIREEIA_CUDA_ENABLED
                // Entire FFN block on device when the three matrices are
                // Q8_0-resident: avoids bringing gate and up back to the
                // host (n_ff floats each) just to apply the activation and
                // re-quantize the intermediate. Same math as the CPU
                // branch below; if any condition doesn't hold, proceeds
                // with the normal path.
                if (g_active_backend == DESIREEIA_BACKEND_CUDA &&
                    lw->wff_up.format == MatVecFormat::Q8_0 && lw->wff_up.cuda_qs &&
                    lw->wff_gate.format == MatVecFormat::Q8_0 && lw->wff_gate.cuda_qs &&
                    lw->wff_down.format == MatVecFormat::Q8_0 && lw->wff_down.cuda_qs &&
                    lw->wff_up.lora.empty() && lw->wff_gate.lora.empty() &&
                    lw->wff_down.lora.empty() &&
                    matmul_q8_0_cuda_ffn_gated(
                        lw->wff_up.cuda_qs.get(), lw->wff_up.cuda_scale.get(),
                        lw->wff_gate.cuda_qs.get(), lw->wff_gate.cuda_scale.get(),
                        lw->wff_down.cuda_qs.get(), lw->wff_down.cuda_scale.get(),
                        cfg_.n_ff, n_embd, xnp, fout.data(),
                        quirks_.gelu_tanh ? 1 : 0) == DESIREEIA_OK) {
                    // fout already ready: skip the entire CPU block below.
                } else {
#endif
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
#ifdef DESIREEIA_CUDA_ENABLED
                }
#endif
            } else {
                // Non-gated FFN: a single projection, no sharing of a
                // quantized activation with a second matmul (there is no
                // second matmul), so a plain matvec instead of
                // matvec_shared like in the gated path.
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
                // xn2 and xp point at the same cell x[p]: reading xp[i]
                // (== xn2[i], the original input residual) and writing
                // xn2[i] in the same expression is safe, same index.
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

    // The model's own output head when it has one, the tied embedding when it
    // does not — see the note on out_head_.
    const MatVec& head = out_head_.empty() ? tok_embd_ : out_head_;

    last_logits.assign(cfg_.n_vocab, 0.0f);
    matvec(head, cfg_.n_vocab, n_embd, hn.data(), last_logits.data());

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
            matvec_batch(head, cfg_.n_vocab, n_embd, hn_all.data(), n_tokens, all_logits->data());
            // The last position is identical to last_logits already
            // computed above (same hn): copied instead of recomputed for
            // bit-exact consistency with the already-validated
            // single-column path.
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
