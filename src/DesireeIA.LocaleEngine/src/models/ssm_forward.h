// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_SSM_FORWARD_H
#define DESIREEIA_SSM_FORWARD_H

#include "core/engine.h"
#include "core/forward_iface.h"
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

// Mamba2 (state-space model) configuration. This class is DELIBERATELY
// separate from DenseConfig/DenseForward (see core/forward_iface.h): an
// SSM model has no attention, no position-indexed KV-cache, and no
// separate FFN block — each layer is a single "norm -> recurrent mixer
// -> residual" block, nothing after it. Fitting this inside DenseForward
// would have meant nesting a second kind of state (recurrent, overwritten
// every token) inside a class designed for append-only positional
// caching — riskier than writing a new class, and the two implementations
// share almost nothing anyway.
struct SsmConfig {
    uint32_t n_vocab = 0;
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    float rms_eps = 1e-6f;

    uint32_t d_conv = 0;   // causal convolution kernel width (typically 4)
    uint32_t d_inner = 0;  // "inner" channel width (expansion, typically 2*n_embd)
    uint32_t d_state = 0;  // SSM state size per channel (typically 16-128)
    uint32_t n_head = 0;   // for mamba2: hparams.ssm_dt_rank reused as the head count
    uint32_t n_group = 1;  // B/C groups shared across heads (mamba2; 1 for mamba1)

    uint32_t head_dim() const { return n_head > 0 ? d_inner / n_head : 0; }
    uint32_t conv_dim() const { return d_inner + 2 * n_group * d_state; }
    uint32_t d_in_proj() const { return 2 * d_inner + 2 * n_group * d_state + n_head; } // z + xBC + dt
};

struct SsmLayerWeights {
    std::vector<float> attn_norm;    // dim n_embd
    std::vector<float> ssm_in;       // [d_in_proj, n_embd] row = d_in_proj row, n_embd columns
    std::vector<float> ssm_conv1d;   // [conv_dim, d_conv] row = channel, d_conv columns (oldest tap first)
    std::vector<float> ssm_conv1d_b; // dim conv_dim
    std::vector<float> ssm_dt_b;     // dim n_head
    std::vector<float> ssm_a;        // dim n_head (per-head scalar, mamba2)
    std::vector<float> ssm_d;        // dim n_head (per-head scalar, mamba2)
    std::vector<float> ssm_norm;     // dim d_inner (optional: n_group blocks of d_inner/n_group, concatenated)
    std::vector<float> ssm_out;      // [n_embd, d_inner] row = n_embd row, d_inner columns
};

// Forward path for Mamba2. Covers ONLY the "pure" variant (norm -> mixer ->
// residual at every layer, no attention): hybrid variants (falcon-h1,
// jamba, granite-hybrid, nemotron-h — which alternate SSM layers and
// classic attention layers within the same model) are NOT covered, they
// would also require the dense forward path in the same model: a known,
// documented gap. Mamba1 (per-state A instead of a per-head scalar) and
// the RWKV families (a different, non-SSM recurrence) remain out of
// scope: known gaps.
class SsmForward : public IForwardEngine {
public:
    bool open(ModelReader& rd, const ModelMeta& meta, uint64_t ram_budget_mb);
    void reset_cache() override;
    bool step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
              std::vector<float>& last_logits, std::vector<float>* all_logits = nullptr) override;
    uint64_t kv_bytes() const override;
    bool weight_cache_enabled() const override { return cache_enabled_; }
    const SsmConfig& config() const { return cfg_; }

private:
    bool load_embd(ModelReader& rd);
    bool load_norms(ModelReader& rd);
    bool load_layer_data(ModelReader& rd, uint32_t il, SsmLayerWeights& w);
    const SsmLayerWeights* get_layer(ModelReader& rd, uint32_t il);

    // Runs ONE token through all layers, updating the recurrent state in
    // place. If want_logits is true, writes the logits into logits_out
    // (otherwise leaves it empty, for the intermediate tokens of a batch
    // when all_logits is not requested: saves the model's largest matvec,
    // lm_head, when the caller doesn't want it).
    void step_token(ModelReader& rd, int32_t token, bool want_logits, std::vector<float>& logits_out);

    SsmConfig cfg_;
    std::vector<float> tok_embd_; // [n_vocab, n_embd], always float (see note in ssm_forward.cpp)
    std::vector<float> output_;   // [n_vocab, n_embd], may be the same buffer as tok_embd_ (tied)
    std::vector<float> out_norm_;

    bool cache_enabled_ = false;
    std::vector<SsmLayerWeights> layer_cache_;
    SsmLayerWeights scratch_;

    // Recurrent state: OVERWRITTEN at every token (not append-only like
    // the dense engine's KV-cache) — this is the property that makes SSMs
    // constant-memory, independent of context length.
    std::vector<float> conv_state_; // [n_layers, conv_dim, d_conv-1]
    std::vector<float> ssm_state_;  // [n_layers, n_head, head_dim, d_state]
    size_t n_tokens_seen_ = 0;
};

}

#endif
