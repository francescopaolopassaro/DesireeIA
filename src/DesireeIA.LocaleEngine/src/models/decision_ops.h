// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_DECISION_OPS_H
#define DESIREEIA_DECISION_OPS_H

#include "core/engine.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace desireeia {

// A weight matrix kept in half precision, row-major [rows, cols] (the same
// layout as the checkpoint: one output feature per row).
struct HalfMat {
    std::vector<uint16_t> data;
    size_t rows = 0;
    size_t cols = 0;
    bool empty() const { return data.empty(); }
};

// Minimal safetensors file reader for the decision-model backend. It is
// independent from the engine-wide safetensors ModelReader on purpose: that
// reader always widens tensors to float32 (1.7 GB for a 420M-parameter
// encoder), while this one keeps matrices in half precision.
class DESIREEIA_INTERNAL StFile {
public:
    bool open(const std::string& path);
    bool has(const std::string& name) const;
    // Tensor shape, empty if absent.
    std::vector<uint64_t> shape(const std::string& name) const;
    // Reads a tensor as float32 (F32, F16 and BF16 sources are accepted).
    bool read_f32(const std::string& name, std::vector<float>& out) const;
    // Reads a 2-D tensor as half precision (F32/BF16 sources are rounded).
    bool read_half(const std::string& name, HalfMat& out) const;
    // Reads one row of a 2-D tensor as float32.
    bool read_row_f32(const std::string& name, size_t row, float* out, size_t cols) const;
    size_t tensor_count() const { return tensors_.size(); }

private:
    struct Entry {
        std::string dtype;
        std::vector<uint64_t> shape;
        uint64_t begin = 0;
        uint64_t end = 0;
    };
    bool read_raw(const Entry& e, uint64_t off, uint64_t len, void* dst) const;

    std::string path_;
    // Kept open between reads: the encoder fetches one embedding row per token.
    // A StFile is used by one thread at a time (the engine serializes calls).
    mutable std::ifstream stream_;
    uint64_t data_base_ = 0;
    std::unordered_map<std::string, Entry> tensors_;
};

// Small vector helpers shared by the encoder and the head attention.
inline float dot_f32(const float* a, const float* b, size_t n) {
    size_t i = 0;
    float s = 0.0f;
#if defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc0 = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    s = _mm_cvtss_f32(lo);
#elif defined(__aarch64__)
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    for (; i + 4 <= n; i += 4) acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    s = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// y[j] += w * x[j]
inline void axpy_f32(float* y, float w, const float* x, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256 wv = _mm256_set1_ps(w);
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
#elif defined(__aarch64__)
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, vfmaq_n_f32(vld1q_f32(y + i), vld1q_f32(x + i), w));
#endif
    for (; i < n; ++i) y[i] += w * x[i];
}

// Y[n, o] = sum_i X[n, i] * W[o, i] + bias[o]   (bias may be null)
// X is [n, in], W is [out, in] in half precision, Y is [n, out].
DESIREEIA_INTERNAL void linear_f16(const float* X, size_t n, size_t in, const HalfMat& W,
                                   const float* bias, float* Y);

// Row-wise LayerNorm over `dim` features: (x - mean) / sqrt(var + eps) * w + b.
// bias may be null (bias-free norm).
DESIREEIA_INTERNAL void layer_norm(const float* X, size_t n, size_t dim, const float* w,
                                   const float* b, float eps, float* Y);

DESIREEIA_INTERNAL float gelu_exact(float x);
DESIREEIA_INTERNAL void softmax_inplace(float* x, size_t n);

}

#endif
