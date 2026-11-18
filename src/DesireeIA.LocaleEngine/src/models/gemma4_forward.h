// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_GEMMA4_FORWARD_H
#define DESIREEIA_GEMMA4_FORWARD_H

#include "core/engine.h"
#include "core/forward_iface.h"
#include "models/dense_forward.h"
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

// Forward pass of the Gemma 4 family, in its own file so nothing of the shared dense path changes: per-layer head sizes (the local
// and the global layers differ), a second embedding per layer (projected, normalised and added through a gate after every block),
// KV shared between layers (the last layers own no KV and read the one of an earlier layer of the same kind), a norm without weights
// on V, no scaling of the attention scores, partial/proportional RoPE on the global layers, soft capping of the final logits.
// The matrices go through the same quantized (and GPU) products as the dense path.
struct Gemma4Config {
    uint32_t n_vocab = 0, n_layers = 0, n_embd = 0, n_head = 0, n_embd_per_layer = 0;
    uint32_t head_dim_global = 0, head_dim_swa = 0, n_rot_global = 0, n_rot_swa = 0;
    uint32_t n_swa = 0, n_kv_from_start = 0, n_ctx_train = 0;
    float eps = 1e-6f, rope_base_global = 1000000.0f, rope_base_swa = 10000.0f, softcap = 0.0f;
    std::vector<uint32_t> n_head_kv, n_ff;
    std::vector<uint8_t> is_swa;
    std::vector<int32_t> kv_source; // layer whose KV this layer uses (itself when it owns one)
    uint32_t head_dim(uint32_t l) const { return is_swa[l] ? head_dim_swa : head_dim_global; }
    bool owns_kv(uint32_t l) const { return kv_source[l] == (int32_t) l; }
};

struct Gemma4Layer {
    std::vector<float> attn_norm, q_norm, k_norm, post_attn_norm, ffn_norm, post_ffn_norm, ple_post_norm;
    MatVec wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down, ple_gate, ple_proj;
    float out_scale = 1.0f;
    bool has_v = false;
};

class Gemma4Forward : public IForwardEngine {
public:
    ~Gemma4Forward() override;
    bool open(ModelReader& rd, const ModelMeta& meta, int32_t backend);
    void reset_cache() override;
    bool step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
              std::vector<float>& last_logits, std::vector<float>* all_logits = nullptr) override;
    uint64_t kv_bytes() const override;
    bool weight_cache_enabled() const override { return true; }
    uint32_t trained_context_length() const override { return cfg_.n_ctx_train; }
    const std::string& last_fail() const override { return last_fail_; }
    void set_hooks(const StepHooks* hooks) override { hooks_ = hooks; }
    bool was_cancelled() const override { return cancelled_; }

private:
    void step_token(int32_t token, bool want_logits, std::vector<float>& logits);
    void rope(float* v, uint32_t hd, uint32_t n_rot, float base, const float* factors, uint32_t pos) const;

    Gemma4Config cfg_;
    std::vector<Gemma4Layer> layers_;
    MatVec tok_embd_, output_, ple_tok_embd_, ple_model_proj_;
    bool tied_ = true;
    bool dev_ = false;        // the whole token runs on the device (see open())
    uint32_t dev_cap_ = 0;
    bool build_device(int32_t backend);
    std::vector<float> out_norm_, ple_proj_norm_, rope_freqs_;
    std::vector<std::vector<float>> k_cache_, v_cache_; // per layer that owns a KV: [pos][n_head_kv * head_dim]
    size_t n_seen_ = 0;
    std::string last_fail_;
    const StepHooks* hooks_ = nullptr;
    bool cancelled_ = false;
};

}

#endif
