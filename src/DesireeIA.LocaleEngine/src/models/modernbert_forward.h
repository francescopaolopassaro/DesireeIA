// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_MODERNBERT_FORWARD_H
#define DESIREEIA_MODERNBERT_FORWARD_H

#include "decision_ops.h"
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

// Encoder configuration, read from the checkpoint's encoder/config.json.
struct ModernBertConfig {
    uint32_t n_vocab = 0;
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_ff = 0;            // width of ONE of the two gated halves
    uint32_t local_window = 128;  // sliding-window span (total, both sides)
    float eps = 1e-5f;
    float theta_global = 160000.0f;
    float theta_local = 10000.0f;
    std::vector<uint8_t> is_global; // per layer: full attention (1) or sliding window (0)
};

// Encoder-only transformer with rotary positions, gated-GELU feed-forward,
// bias-free LayerNorm, no absolute position table and attention that
// alternates between full-sequence layers and sliding-window layers.
// DELIBERATELY separate from BertForward (plain absolute-position BERT, kept
// untouched): the two share nothing but the idea of a bidirectional encoder.
//
// Weights stay in half precision; each matmul widens a weight row once and
// reuses it for every token of the sequence. No state persists between
// forward() calls.
class ModernBertCuda;
struct CudaHeadDesc;

class DESIREEIA_INTERNAL ModernBertEncoder {
public:
    ModernBertEncoder() = default;
    ~ModernBertEncoder();
    ModernBertEncoder(const ModernBertEncoder&) = delete;
    ModernBertEncoder& operator=(const ModernBertEncoder&) = delete;

    // `prefix` is the tensor-name prefix of the encoder inside the file
    // (e.g. "encoder." for a checkpoint that bundles encoder and head).
    bool load(const StFile& st, const std::string& prefix, const ModernBertConfig& cfg,
              std::string& err);
    // Runs the encoder over `n` token ids and fills out with n * n_embd
    // floats (the last hidden state of every position).
    bool forward(const int32_t* ids, size_t n, std::vector<float>& out) const;
    const ModernBertConfig& config() const { return cfg_; }

    // Moves the encoder passes to an NVIDIA GPU (tensor cores). Optional: when
    // it fails (err says why: no device, not enough memory, a build without
    // CUDA) the CPU path stays in use; forward() also falls back to the CPU by
    // itself for an input the GPU workspace cannot take.
    bool enable_gpu(std::string& err);
    bool gpu_enabled() const { return gpu_ != nullptr; }
    // Decision head on the GPU as well (optional, needs enable_gpu() first). When it is
    // ready, gpu_forward_head() returns the post-head hidden state of `rows` positions.
    bool gpu_set_head(const CudaHeadDesc& head, std::string& err);
    bool gpu_head_ready() const { return gpu_head_; }
    bool gpu_forward_head(const int32_t* ids, size_t n, int qtype, const size_t* rows, size_t n_rows,
                          std::vector<float>& out) const;

    // Parses an encoder config.json (HF layout, both the newer
    // rope_parameters form and the older *_rope_theta form).
    static bool parse_config(const std::string& json_text, ModernBertConfig& cfg, std::string& err);

private:
    struct Layer {
        std::vector<float> attn_norm; // empty for layer 0 (identity)
        std::vector<float> mlp_norm;
        HalfMat wqkv, wo, wi, mlp_wo;
    };

    ModernBertConfig cfg_;
    const StFile* st_ = nullptr;
    std::string tok_name_;
    std::vector<float> emb_norm_;
    std::vector<float> final_norm_;
    std::vector<Layer> layers_;
    ModernBertCuda* gpu_ = nullptr;
    bool gpu_head_ = false;
};

}

#endif
