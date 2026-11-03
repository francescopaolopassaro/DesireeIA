// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "decision_ops.h"
#include "decision_json.h"
#include "core/thread_pool.h"
#include "quant/quant.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace desireeia {

namespace {

// Round-to-nearest-even float32 -> IEEE half.
uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) {
        return (uint16_t) (sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) half++;
        return (uint16_t) (sign | half);
    }
    uint32_t half = ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return (uint16_t) (sign | half);
}

float bf16_to_f32(uint16_t v) {
    const uint32_t x = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void half_row_to_f32(const uint16_t* src, float* dst, size_t n) {
    size_t i = 0;
#if defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(h));
    }
#endif
    for (; i < n; ++i) dst[i] = desireeia_fp16_to_fp32(src[i]);
}

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
#endif

}

// ---- StFile ----------------------------------------------------------------

bool StFile::open(const std::string& path) {
    tensors_.clear();
    path_ = path;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t header_len = 0;
    if (!f.read(reinterpret_cast<char*>(&header_len), sizeof header_len)) return false;
    if (header_len == 0 || header_len > (1ull << 30)) return false;
    std::string header((size_t) header_len, '\0');
    if (!f.read(header.data(), (std::streamsize) header_len)) return false;

    djson::Value root;
    djson::Parser parser(header);
    if (!parser.parse(root) || !root.is_obj()) return false;
    data_base_ = sizeof header_len + header_len;

    for (const auto& kv : root.obj) {
        if (kv.first == "__metadata__" || !kv.second.is_obj()) continue;
        Entry e;
        const djson::Value* dt = kv.second.find("dtype");
        const djson::Value* sh = kv.second.find("shape");
        const djson::Value* of = kv.second.find("data_offsets");
        if (!dt || !sh || !of || !of->is_arr() || of->arr.size() != 2) continue;
        e.dtype = dt->str;
        for (const auto& d : sh->arr) e.shape.push_back((uint64_t) d.num);
        e.begin = (uint64_t) of->arr[0].num;
        e.end = (uint64_t) of->arr[1].num;
        tensors_[kv.first] = std::move(e);
    }
    return !tensors_.empty();
}

bool StFile::has(const std::string& name) const { return tensors_.count(name) != 0; }

std::vector<uint64_t> StFile::shape(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? std::vector<uint64_t>() : it->second.shape;
}

bool StFile::read_raw(const Entry& e, uint64_t off, uint64_t len, void* dst) const {
    if (off + len > e.end - e.begin) return false;
    if (!stream_.is_open()) {
        stream_.open(path_, std::ios::binary);
        if (!stream_) return false;
    }
    stream_.clear();
    stream_.seekg((std::streamoff) (data_base_ + e.begin + off), std::ios::beg);
    return (bool) stream_.read(reinterpret_cast<char*>(dst), (std::streamsize) len);
}

namespace {
size_t dtype_size(const std::string& dt) {
    if (dt == "F32") return 4;
    if (dt == "F16" || dt == "BF16") return 2;
    return 0;
}
uint64_t numel(const std::vector<uint64_t>& s) {
    uint64_t n = 1;
    for (uint64_t d : s) n *= d;
    return n;
}
}

bool StFile::read_f32(const std::string& name, std::vector<float>& out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;
    const Entry& e = it->second;
    const size_t es = dtype_size(e.dtype);
    if (es == 0) return false;
    const uint64_t n = numel(e.shape);
    out.resize((size_t) n);
    if (e.dtype == "F32") return read_raw(e, 0, n * 4, out.data());
    std::vector<uint16_t> raw((size_t) n);
    if (!read_raw(e, 0, n * 2, raw.data())) return false;
    if (e.dtype == "F16") half_row_to_f32(raw.data(), out.data(), (size_t) n);
    else for (size_t i = 0; i < (size_t) n; ++i) out[i] = bf16_to_f32(raw[i]);
    return true;
}

bool StFile::read_half(const std::string& name, HalfMat& out) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;
    const Entry& e = it->second;
    if (e.shape.size() != 2 || dtype_size(e.dtype) == 0) return false;
    out.rows = (size_t) e.shape[0];
    out.cols = (size_t) e.shape[1];
    const uint64_t n = numel(e.shape);
    out.data.resize((size_t) n);
    if (e.dtype == "F16") return read_raw(e, 0, n * 2, out.data.data());
    std::vector<float> tmp;
    if (!read_f32(name, tmp)) return false;
    for (size_t i = 0; i < (size_t) n; ++i) out.data[i] = f32_to_f16(tmp[i]);
    return true;
}

bool StFile::read_row_f32(const std::string& name, size_t row, float* out, size_t cols) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return false;
    const Entry& e = it->second;
    const size_t es = dtype_size(e.dtype);
    if (es == 0 || e.shape.size() != 2 || e.shape[1] != cols || row >= e.shape[0]) return false;
    const uint64_t off = (uint64_t) row * cols * es;
    if (e.dtype == "F32") return read_raw(e, off, cols * 4, out);
    std::vector<uint16_t> raw(cols);
    if (!read_raw(e, off, cols * 2, raw.data())) return false;
    if (e.dtype == "F16") half_row_to_f32(raw.data(), out, cols);
    else for (size_t i = 0; i < cols; ++i) out[i] = bf16_to_f32(raw[i]);
    return true;
}

// ---- kernels ---------------------------------------------------------------

void linear_f16(const float* X, size_t n, size_t in, const HalfMat& W, const float* bias, float* Y) {
    const size_t out = W.rows;
    // Tokens are processed in tiles that stay resident in the L2 cache while
    // every weight row of the chunk sweeps over them; four weight rows are
    // widened at a time and combined with two tokens per step, so each loaded
    // activation feeds four multiply-adds. Reading the whole activation
    // matrix once per weight row made this memory-bound.
    constexpr size_t TILE = 128;
    ThreadPool::global().parallel_for(out, [&](size_t a, size_t b) {
        std::vector<float> wbuf(4 * in);
        for (size_t t0 = 0; t0 < n; t0 += TILE) {
            const size_t t1 = std::min(n, t0 + TILE);
            for (size_t o = a; o < b; o += 4) {
                const size_t rows = std::min<size_t>(4, b - o);
                for (size_t r = 0; r < rows; ++r) half_row_to_f32(W.data.data() + (o + r) * in, wbuf.data() + r * in, in);
                float bo[4] = { 0.f, 0.f, 0.f, 0.f };
                if (bias) for (size_t r = 0; r < rows; ++r) bo[r] = bias[o + r];
                size_t t = t0;
#if defined(__AVX2__)
                if (rows == 4) {
                    const float* w0 = wbuf.data();
                    const float* w1 = w0 + in;
                    const float* w2 = w1 + in;
                    const float* w3 = w2 + in;
                    for (; t + 2 <= t1; t += 2) {
                        const float* x0 = X + t * in;
                        const float* x1 = x0 + in;
                        __m256 a00 = _mm256_setzero_ps(), a01 = a00, a10 = a00, a11 = a00;
                        __m256 a20 = a00, a21 = a00, a30 = a00, a31 = a00;
                        size_t i = 0;
                        for (; i + 8 <= in; i += 8) {
                            const __m256 xv0 = _mm256_loadu_ps(x0 + i);
                            const __m256 xv1 = _mm256_loadu_ps(x1 + i);
                            __m256 w = _mm256_loadu_ps(w0 + i);
                            a00 = _mm256_fmadd_ps(xv0, w, a00); a01 = _mm256_fmadd_ps(xv1, w, a01);
                            w = _mm256_loadu_ps(w1 + i);
                            a10 = _mm256_fmadd_ps(xv0, w, a10); a11 = _mm256_fmadd_ps(xv1, w, a11);
                            w = _mm256_loadu_ps(w2 + i);
                            a20 = _mm256_fmadd_ps(xv0, w, a20); a21 = _mm256_fmadd_ps(xv1, w, a21);
                            w = _mm256_loadu_ps(w3 + i);
                            a30 = _mm256_fmadd_ps(xv0, w, a30); a31 = _mm256_fmadd_ps(xv1, w, a31);
                        }
                        float s[4][2] = { { hsum256(a00), hsum256(a01) }, { hsum256(a10), hsum256(a11) },
                                          { hsum256(a20), hsum256(a21) }, { hsum256(a30), hsum256(a31) } };
                        for (; i < in; ++i) {
                            s[0][0] += x0[i] * w0[i]; s[0][1] += x1[i] * w0[i];
                            s[1][0] += x0[i] * w1[i]; s[1][1] += x1[i] * w1[i];
                            s[2][0] += x0[i] * w2[i]; s[2][1] += x1[i] * w2[i];
                            s[3][0] += x0[i] * w3[i]; s[3][1] += x1[i] * w3[i];
                        }
                        for (size_t r = 0; r < 4; ++r) {
                            Y[t * out + o + r] = s[r][0] + bo[r];
                            Y[(t + 1) * out + o + r] = s[r][1] + bo[r];
                        }
                    }
                }
#endif
                for (; t < t1; ++t) {
                    const float* x = X + t * in;
                    for (size_t r = 0; r < rows; ++r) Y[t * out + o + r] = dot_f32(x, wbuf.data() + r * in, in) + bo[r];
                }
            }
        }
    });
}

void layer_norm(const float* X, size_t n, size_t dim, const float* w, const float* b, float eps, float* Y) {
    for (size_t t = 0; t < n; ++t) {
        const float* x = X + t * dim;
        float* y = Y + t * dim;
        double mean = 0.0;
        for (size_t i = 0; i < dim; ++i) mean += x[i];
        mean /= (double) dim;
        double var = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            const double d = x[i] - mean;
            var += d * d;
        }
        var /= (double) dim;
        const float inv = (float) (1.0 / std::sqrt(var + (double) eps));
        const float m = (float) mean;
        for (size_t i = 0; i < dim; ++i) {
            float v = (x[i] - m) * inv * w[i];
            if (b) v += b[i];
            y[i] = v;
        }
    }
}

float gelu_exact(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

void softmax_inplace(float* x, size_t n) {
    if (n == 0) return;
    float mx = x[0];
    for (size_t i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = (float) (1.0 / sum);
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

}
