#ifndef DESIREEIA_BERT_FORWARD_H
#define DESIREEIA_BERT_FORWARD_H

#include "core/engine.h"
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

// BERT (encoder-only) configuration. This class is DELIBERATELY separate
// from DenseConfig/DenseForward and from SsmConfig/SsmForward (see
// core/forward_iface.h and the note on ArchKind::Bert): a BERT encoder has
// nothing in common with a causal decoder — BIDIRECTIONAL attention (no
// causal mask, every position sees every other position), no KV-cache (the
// whole sequence is processed in a single pass, never incrementally),
// POST-norm topology (norm AFTER the residual, not before), and the
// output is a per-token embedding — there are no vocabulary logits, no
// sampling, no "next token" at all.
//
// Covers only the plain variant (absolute position, no RoPE). Related
// variants with optional RoPE on some layers and a gated GEGLU FFN, or
// with a still-different structure, aren't covered: known, documented gap
// — no sample file for those variants was available to verify against.
struct BertConfig {
    uint32_t n_vocab = 0;
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_ff = 0;
    uint32_t n_ctx_train = 0;
    float eps = 1e-12f; // typical BERT default, overridden by attention.layer_norm_eps
};

struct BertLayerWeights {
    std::vector<float> wq, wk, wv, wo; // kept as plain float, see the note in bert_forward.cpp
    std::vector<float> bq, bk, bv, bo;
    std::vector<float> attn_out_norm, attn_out_norm_b;
    std::vector<float> ffn_up, ffn_down;
    std::vector<float> ffn_up_b, ffn_down_b;
    std::vector<float> layer_out_norm, layer_out_norm_b;
};

// Forward path for BERT (encoder-only). Processes the whole sequence in a
// single pass (encode()), no state persists between calls: unlike
// DenseForward/SsmForward there's no "continuation" — each call is
// independent, which matches how it's actually used (a sentence embedding
// doesn't depend on earlier calls).
class BertForward {
public:
    bool open(ModelReader& rd, const ModelMeta& meta, uint64_t ram_budget_mb);
    // Fills out_embd with n_tokens*n_embd floats (a PER-TOKEN embedding,
    // not pooled — any pooling/normalization downstream is the
    // application's choice, not the engine's).
    bool encode(ModelReader& rd, const int32_t* tokens, size_t n_tokens, std::vector<float>& out_embd);
    const BertConfig& config() const { return cfg_; }

private:
    bool load_layer(ModelReader& rd, uint32_t il, BertLayerWeights& w);

    BertConfig cfg_;
    std::vector<float> tok_embd_;      // [n_vocab, n_embd]
    std::vector<float> type_embd_row0_; // [n_embd] — only row 0 ("segment A", the only one ever used here)
    std::vector<float> pos_embd_;      // [n_ctx_train, n_embd]
    std::vector<float> tok_norm_, tok_norm_b_;
    std::vector<BertLayerWeights> layers_; // always cached: BERT models are typically small (<1GB)
};

}

#endif
