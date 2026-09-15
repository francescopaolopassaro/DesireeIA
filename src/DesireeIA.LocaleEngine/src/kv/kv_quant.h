// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_KV_QUANT_H
#define DESIREEIA_KV_QUANT_H

#include <cstddef>
#include <cstdint>

namespace desireeia {

// Q8_0 quantization for one KV cache row (one token's key or value vector).
//
// Reuses the exact on-disk block_q8_0 layout weight tensors already use (see
// quant/quant.h): `dim` floats become ceil(dim/32) blocks, each 32 int8
// values plus one fp16 scale. That reuse is what lets the K-side read path
// below call the existing, already-verified matmul_q8_0 kernel directly on
// cache rows instead of needing a parallel kernel of its own.
//
// A row costs 34 bytes per 32 floats here (32 int8 + 2-byte fp16 scale)
// against 128 bytes for float32 — roughly 3.76x smaller, in the range the
// project's KV-compression flag has always claimed (50-75% reduction) but,
// until this file, never actually delivered: see docs/performance_plan.md
// item 2.

// Bytes one quantized row of `dim` floats occupies.
size_t kv_quant_row_bytes(size_t dim);

// Encodes `dim` floats into a packed Q8_0 row at `out` (must be at least
// kv_quant_row_bytes(dim) bytes). `dim` must be a multiple of 32 — true for
// every head_dim/kv_lora_rank+rope width this engine produces; the caller
// checks once at cache-allocation time rather than on every token.
void kv_quantize_row(const float* x, size_t dim, uint8_t* out);

// dot(query, dequant(row)), for i in [0, dim). The K-cache scoring step of
// attention.
//
// This is NOT routed through the existing matmul_q8_0 kernel: that kernel
// assumes its rows sit back-to-back in memory (row r at data + r*row_bytes),
// which is true for the MQA-shaped caches (a single kv-head, so every cached
// position IS one contiguous row) but not for GQA/MHA — there several
// kv-heads interleave within each cached position, so one head's rows across
// positions are strided, not contiguous. This kernel takes an explicit
// stride so it works for both, keeping the one-row-at-a-time loop shape the
// non-MLA attention path already has (unavoidable there anyway: ALiBi and
// sliding-window masking both need to touch each position's score
// individually).
float kv_dot_q8_0(const float* query, const uint8_t* row, size_t dim);

// y[i] += weight * dequant(row)[i], for i in [0, dim). The V-cache
// accumulation step of attention: `row` is one cached (quantized) value
// vector, `weight` its softmax score. No existing kernel does this
// (matmul kernels produce a dot product, not a scaled row copy), so this one
// is new — but it is the AVX2 techniques already proven elsewhere in this
// engine (axpy_f32 in models/dense_forward.cpp), applied to a dequantized
// source instead of a float one.
void kv_axpy_q8_0(float* y, const uint8_t* row, size_t dim, float weight);

}

#endif
