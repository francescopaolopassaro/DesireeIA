// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "kv_quant.h"
#include "../quant/quant.h"
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define DESIREEIA_KV_HAS_AVX2 1
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace desireeia {

size_t kv_quant_row_bytes(size_t dim) {
    const size_t nblocks = (dim + 31) / 32;
    return nblocks * sizeof(block_q8_0);
}

// Rounding/clamping convention matches quantize_q8_0 in core/quant.cpp
// exactly (scale = amax/127, round-to-nearest, clamp to [-127,127]): the
// codebase's one Q8_0 recipe, not a second one invented for this file.
void kv_quantize_row(const float* x, size_t dim, uint8_t* out) {
    const size_t nblocks = (dim + 31) / 32;
    auto* blocks = reinterpret_cast<block_q8_0*>(out);

    for (size_t b = 0; b < nblocks; ++b) {
        const size_t start = b * 32;
        const size_t len = (dim - start) < 32 ? (dim - start) : 32;

        float amax = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            const float a = std::fabs(x[start + i]);
            if (a > amax) amax = a;
        }
        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        blocks[b].d = desireeia_fp32_to_fp16(d);
        for (size_t i = 0; i < len; ++i) {
            int32_t v = (int32_t) std::lrintf(x[start + i] * id);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            blocks[b].qs[i] = (int8_t) v;
        }
        for (size_t i = len; i < 32; ++i) blocks[b].qs[i] = 0;
    }
}

float kv_dot_q8_0(const float* query, const uint8_t* row, size_t dim) {
    const size_t nblocks = (dim + 31) / 32;
    const auto* blocks = reinterpret_cast<const block_q8_0*>(row);
    float acc = 0.0f;

    for (size_t b = 0; b < nblocks; ++b) {
        const size_t start = b * 32;
        const size_t len = (dim - start) < 32 ? (dim - start) : 32;
        const float d = desireeia_fp16_to_fp32(blocks[b].d);
        const int8_t* qs = blocks[b].qs;
        const float* qp = query + start;

        float block_dot = 0.0f;
        size_t i = 0;
#if defined(DESIREEIA_KV_HAS_AVX2)
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= len; i += 8) {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + i));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            __m256 qf = _mm256_cvtepi32_ps(q32);
            vsum = _mm256_fmadd_ps(qf, _mm256_loadu_ps(qp + i), vsum);
        }
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(vsum), _mm256_extractf128_ps(vsum, 1));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 0x55));
        block_dot = _mm_cvtss_f32(h);
#elif defined(__ARM_NEON)
        float32x4_t vsum0 = vdupq_n_f32(0), vsum1 = vdupq_n_f32(0);
        for (; i + 8 <= len; i += 8) {
            int8x8_t q8 = vld1_s8(qs + i);
            int16x8_t q16 = vmovl_s8(q8);
            float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            vsum0 = vfmaq_f32(vsum0, qf0, vld1q_f32(qp + i));
            vsum1 = vfmaq_f32(vsum1, qf1, vld1q_f32(qp + i + 4));
        }
        block_dot = vaddvq_f32(vaddq_f32(vsum0, vsum1));
#endif
        for (; i < len; ++i) block_dot += (float) qs[i] * qp[i];
        acc += d * block_dot;
    }
    return acc;
}

void kv_axpy_q8_0(float* y, const uint8_t* row, size_t dim, float weight) {
    const size_t nblocks = (dim + 31) / 32;
    const auto* blocks = reinterpret_cast<const block_q8_0*>(row);

    for (size_t b = 0; b < nblocks; ++b) {
        const size_t start = b * 32;
        const size_t len = (dim - start) < 32 ? (dim - start) : 32;
        const float scale = weight * desireeia_fp16_to_fp32(blocks[b].d);
        const int8_t* qs = blocks[b].qs;
        float* yb = y + start;

        size_t i = 0;
#if defined(DESIREEIA_KV_HAS_AVX2)
        const __m256 vscale = _mm256_set1_ps(scale);
        for (; i + 8 <= len; i += 8) {
            // int8 -> int32 -> float32, then FMA into the (already float)
            // accumulator: same widen-then-convert path the quantized
            // matmul kernels use for their activations.
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + i));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            __m256 qf = _mm256_cvtepi32_ps(q32);
            __m256 acc = _mm256_loadu_ps(yb + i);
            acc = _mm256_fmadd_ps(vscale, qf, acc);
            _mm256_storeu_ps(yb + i, acc);
        }
#elif defined(__ARM_NEON)
        const float32x4_t vscale = vdupq_n_f32(scale);
        for (; i + 8 <= len; i += 8) {
            int8x8_t q8 = vld1_s8(qs + i);
            int16x8_t q16 = vmovl_s8(q8);
            float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            vst1q_f32(yb + i,     vfmaq_f32(vld1q_f32(yb + i),     vscale, qf0));
            vst1q_f32(yb + i + 4, vfmaq_f32(vld1q_f32(yb + i + 4), vscale, qf1));
        }
#endif
        for (; i < len; ++i) yb[i] += scale * (float) qs[i];
    }
}

}
