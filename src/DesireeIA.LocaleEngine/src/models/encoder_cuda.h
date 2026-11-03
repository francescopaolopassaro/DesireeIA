// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_ENCODER_CUDA_H
#define DESIREEIA_ENCODER_CUDA_H

#include "core/engine.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

// One encoder layer as the GPU needs it: host pointers to half-precision
// weight matrices (row-major, one output feature per row) and float norms.
struct CudaEncoderLayer {
    const uint16_t* wqkv = nullptr;       // [3d, d]
    const uint16_t* wo = nullptr;         // [d, d]
    const uint16_t* wi = nullptr;         // [2*ff, d]
    const uint16_t* mlp_wo = nullptr;     // [d, ff]
    const float* attn_norm = nullptr;     // [d], null for the first layer (identity)
    const float* mlp_norm = nullptr;      // [d]
    bool is_global = false;               // full attention (true) or sliding window (false)
};

struct CudaEncoderDesc {
    uint32_t n_vocab = 0, d = 0, heads = 0, ff = 0, local_window = 128;
    float eps = 1e-5f, theta_global = 160000.0f, theta_local = 10000.0f;
    const uint16_t* tok_emb = nullptr;    // [n_vocab, d]
    const float* emb_norm = nullptr;      // [d]
    const float* final_norm = nullptr;    // [d]
    std::vector<CudaEncoderLayer> layers;
};

// The decision head that follows the encoder: pre-norm transformer layers
// (LayerNorm with bias, fused QKV projection with bias, full attention, ReLU
// feed-forward) plus the per-question-type embedding added to the encoder output.
struct CudaHeadLayer {
    const uint16_t* in_proj = nullptr;    // [3d, d]
    const uint16_t* out_proj = nullptr;   // [d, d]
    const uint16_t* lin1 = nullptr;       // [ffn, d]
    const uint16_t* lin2 = nullptr;       // [d, ffn]
    const float* in_proj_b = nullptr;     // [3d]
    const float* out_proj_b = nullptr;    // [d]
    const float* lin1_b = nullptr;        // [ffn]
    const float* lin2_b = nullptr;        // [d]
    const float *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;   // [d]
};

struct CudaHeadDesc {
    uint32_t ffn = 0;                     // feed-forward width of the head layers
    const float* type_emb = nullptr;      // [3, d]
    std::vector<CudaHeadLayer> layers;
};

// The ModernBERT-style encoder on an NVIDIA GPU: half-precision tensor-core
// matrix products with float accumulation, float residual stream. It is an
// accelerator for ModernBertEncoder, never a requirement: every failure
// (no device, not enough memory, an input too long for the workspace) is
// reported as `false` and the caller runs the CPU path instead. Built only
// with the CUDA backend (DESIREEIA_CUDA_ENABLED).
class DESIREEIA_INTERNAL ModernBertCuda {
public:
    ModernBertCuda();
    ~ModernBertCuda();
    ModernBertCuda(const ModernBertCuda&) = delete;
    ModernBertCuda& operator=(const ModernBertCuda&) = delete;

    // True when a usable CUDA device exists.
    static bool device_available();
    // Uploads the weights. False (with err) if there is no device or the
    // weights do not fit in its free memory.
    bool init(const CudaEncoderDesc& desc, std::string& err);
    // Last hidden state of every position: n * d floats. False on any failure.
    bool forward(const int32_t* ids, size_t n, std::vector<float>& out);
    // Uploads the decision head so that forward_head() can run encoder and head
    // without the activations ever leaving the GPU.
    bool set_head(const CudaHeadDesc& head, std::string& err);
    // Encoder, then type embedding `qtype`, then the head layers; returns the hidden
    // state of the requested positions only (n_rows * d floats).
    bool forward_head(const int32_t* ids, size_t n, int qtype, const size_t* rows, size_t n_rows,
                      std::vector<float>& out);
    // Longest input the workspace is sized for.
    static size_t max_tokens() { return 2048; }

private:
    struct Impl;
    Impl* p_;
};

}

#endif
