// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_ARCH_TAGS_H
#define DESIREEIA_ARCH_TAGS_H

#include "core/engine.h"
#include <string>

namespace desireeia {

// Model architectures recognized by the dense forward path, identified by
// the "general.architecture" tag stored in the model file's metadata.
//
// WHAT CAN RUN HERE. The dense forward path implements one shape: input
// RMSNorm -> Q/K/V (optional bias) -> optional per-head RMSNorm on Q/K ->
// NEOX RoPE -> causal attention (grouped-query) -> output projection ->
// residual -> RMSNorm -> gated FFN (SiLU or GELU) -> residual. Every
// architecture listed below was checked one by one against its actual
// published implementation to confirm it fits this shape — never assumed
// from the name alone. Architectures that don't fit are deliberately left
// out (see the notes below): tagging one without the matching code would
// load it and silently produce wrong output, which is worse than a clean
// rejection.
enum class ArchKind {
    Unknown,
    Gemma,     // gemma v1: sqrt(embd) embedding scale, GELU
    Gemma2,    // like Gemma plus a "sandwich" norm after attention/FFN
               // (known gap: logit softcapping isn't implemented)
    Gemma3,    // like Gemma2 plus per-head QK-norm
    DenseGqa,  // the whole family of architecturally identical dense
               // transformers (see arch_tags.cpp for the exact tag list):
               // no quirks beyond SiLU + gated FFN
    Mistral,
    Qwen2,     // + bias on Q/K/V
    Qwen3,     // + per-head QK-norm, no bias
    SeedOss,   // + sandwich norm, no QK-norm
    StableLM,  // LayerNorm (with bias) instead of RMSNorm, otherwise identical
    Orion,     // like StableLM: LayerNorm with bias, sequential structure
    Starcoder2,// LayerNorm + non-gated GELU FFN (bias on norm and FFN),
               // uses RoPE like the rest of the dense family — confirmed
               // distinct from gpt2 (absolute position embeddings, no
               // RoPE), bloom and mpt (ALiBi, no RoPE): those three are
               // excluded because they need a positional mechanism this
               // path doesn't implement — including them here would give
               // silently wrong positions. codeshell uses RoPE the same
               // way starcoder2 does, identical shape, so it maps onto
               // the same ArchKind (see arch_tags.cpp).
    Nemotron,  // LayerNorm + non-gated RELU^2 FFN (bias on norm and FFN)
    Arcee,     // RMSNorm + non-gated RELU^2 FFN, no bias anywhere
    Olmo2,     // post-norm only (no norm before attention/FFN), full-width
               // RMS QK-norm (over the whole Q/K vector, before the
               // per-head reshape)
    Exaone4,   // like Olmo2 (post-norm only) but per-head QK-norm like
               // gemma3/qwen3 (confirmed: the Q/K norm runs after the
               // projection has already been reshaped into per-head form)
    DeepSeek2, // Multi-head Latent Attention plus Mixture-of-Experts with
               // a shared-expert branch. See DenseQuirks::mla below and
               // the extended note in dense_forward.cpp (the MLA branch
               // of step()). Covers the "absorbed" attention form used by
               // every modern export of this family; the older, unabsorbed
               // tensor layout used only by legacy files isn't implemented.
               // Also not implemented: expert-group routing (only needed
               // by the largest variant of this family), multi-token
               // prediction layers, and the alternate MHA-attention
               // variant used by some OCR-specialized checkpoints —
               // known, documented gaps.
    Cohere2,   // LayerNorm without bias, parallel attention/FFN topology,
               // RoPE only on local (sliding-window) layers, plus a final
               // logit scale. A sibling architecture with a similar
               // parallel topology but a fused QKV tensor isn't covered
               // by this ArchKind — see Falcon below.
    Falcon,    // parallel topology like Cohere2, but with QKV fused into
               // one tensor (DenseQuirks::fused_qkv) and LayerNorm WITH
               // bias. RoPE applies on every layer (unlike Cohere2, no
               // rope_only_swa). The optional second norm used by the
               // largest checkpoint in this family (feeding attention
               // separately from the norm that feeds the FFN) is read
               // defensively — present only there, absent everywhere
               // else — so no dedicated quirk is needed for it.
    Spark25,   // hybrid sliding-window attention: which layers are local is
               // declared as an explicit per-layer flag ARRAY rather than a
               // period, and local and global layers differ in BOTH the RoPE
               // base frequency AND the number of rotated dimensions (global
               // layers rotate only a prefix of each head). QKV is fused into
               // one tensor, the FFN is gated with GELU, and the attention
               // output is scaled per head by a sigmoid gate computed from
               // the same normalized input that feeds Q/K/V
               // (DenseQuirks::attn_gate). No QK-norm, no sandwich norm.
    Gpt2,      // absolute position embeddings (a position table added to
               // the token embeddings once, not per layer), QKV fused
               // WITH bias, LayerNorm with bias, non-gated GELU FFN with
               // bias, no RoPE at all.
    Bloom,     // like gpt2 but ALiBi instead of absolute position, plus a
               // one-time norm applied to the embeddings before the first
               // layer (tok_norm/tok_norm_b, DenseQuirks::embd_norm).
    Mpt,       // like bloom (ALiBi) but without the initial embedding
               // norm; the position table and biases are optional and
               // read defensively (most checkpoints in this family don't
               // have them). The optional full-width Q/K LayerNorm-with-
               // bias variant some checkpoints carry is NOT implemented:
               // known gap, rare in practice and flagged as fragile even
               // in its own reference tooling.
    Bert,      // pure encoder-only: the forward path is COMPLETELY
               // separate (models/bert_forward.h/.cpp), doesn't touch
               // DenseQuirks/DenseForward at all — this enum value only
               // exists so detect_arch()/ctx.cpp can recognize it.
               // Bidirectional attention (no causal mask), no KV-cache,
               // post-norm topology, output is a per-token embedding (no
               // logits, no sampling). Covers only the plain variant
               // (absolute position, no RoPE) — see the extended note in
               // bert_forward.h for the related variants left out.
    Mamba2,    // pure state-space model: no attention at all, forward
               // path COMPLETELY separate (models/ssm_forward.h/.cpp),
               // doesn't touch DenseQuirks/DenseForward at all — this
               // enum value only exists so detect_arch()/ctx.cpp can
               // recognize it. Covers only the pure variant (norm -> SSM
               // mixer -> residual per layer, no separate FFN block). The
               // hybrid variants that mix SSM layers and attention layers
               // in the same model aren't covered — they'd need both
               // forward paths active in one model. The earlier
               // generation of this family (per-state decay instead of a
               // per-head scalar) and the RWKV family (a different,
               // non-SSM recurrence) are also out of scope: known,
               // undocumented gaps.
};

DESIREEIA_INTERNAL ArchKind detect_arch(const std::string& tag);

struct DenseQuirks {
    bool embd_scale_sqrt = false; // scales embeddings by sqrt(n_embd)
    bool gelu_tanh = false;       // false -> SiLU
    bool qkv_bias = false;        // reads blk.N.attn_{q,k,v}.bias if present
    bool qk_norm = false;         // per-head RMSNorm on Q/K before RoPE
                                   // (blk.N.attn_{q,k}_norm.weight, dim head_dim)
    bool sandwich_norm = false;   // RMSNorm after attention/FFN, before the
                                   // residual add (blk.N.post_attention_norm.weight,
                                   // blk.N.post_ffw_norm.weight, dim n_embd)
    // Classic LayerNorm (mean + variance, then weight and BIAS) instead of
    // RMSNorm for attn_norm/ffn_norm/output_norm. Population variance (not
    // sample variance), scale = 1/sqrt(var+eps). The bias is read from
    // blk.N.attn_norm.bias / blk.N.ffn_norm.bias / output_norm.bias, all
    // optional (absent = 0, i.e. RMSNorm-like behavior with no offset).
    bool layer_norm = false;

    // Non-gated FFN: a single up projection -> activation -> down
    // projection, with no ffn_gate (left unused). The families that need
    // this pass an explicit "no gate" flag in their own graph construction.
    // The `true` default preserves every already-supported architecture in
    // the Llama/Gemma/Qwen-shaped family, which always had two separate
    // gate+up projections.
    bool ffn_gated = true;
    enum class PlainFfnAct { Gelu, ReluSqr };
    // Only used when ffn_gated == false.
    PlainFfnAct ffn_act = PlainFfnAct::Gelu;

    // No norm before attention or before the FFN (olmo2, exaone4): Q/K/V
    // are computed directly on the layer's raw input, and the FFN receives
    // the raw post-attention residual directly (no ffn_norm). The only
    // norms applied are the "sandwich" ones (post-attention/post-FFN,
    // already covered by sandwich_norm) right before each residual add.
    bool no_pre_norm = false;

    // Full-width QK-norm: RMSNorm applied ONCE over the entire Q vector
    // (width q_dim) / K vector (width kv_dim) before the per-head reshape,
    // instead of the per-head loop at head_dim width used elsewhere.
    // The tensor width for this variant is n_embd, not head_dim, which is
    // how it's distinguished from the per-head QK-norm architectures.
    bool qk_norm_full_width = false;

    // Multi-head Latent Attention (deepseek2 and its siblings): replaces
    // the standard Q/K/V and attention computation entirely with a
    // compressed-rank "absorbed" path. When true, every other Q/K/V-related
    // quirk above (qkv_bias, qk_norm, qk_norm_full_width) is ignored: MLA
    // has its own tensors (wq_a/wq_b or wq, wkv_a_mqa, wk_b, wv_b) with
    // their own intermediate norms (attn_q_a_norm, attn_kv_a_norm), handled
    // by a dedicated branch in dense_forward.cpp. RoPE applies only to the
    // "rope" portion of Q/K (n_embd_head_qk_rope, typically 64 dims), not
    // the full head.
    bool mla = false;

    // PARALLEL topology (cohere2, falcon): attention and FFN are BOTH
    // computed from the same norm output (no separate ffn_norm), then
    // summed together with the original residual in one three-term step
    // (x_new = attn_out + ffn_out + x_orig), instead of the sequential
    // chain (attention -> residual -> norm -> FFN -> residual) used by the
    // rest of the engine.
    bool parallel_residual = false;

    // RoPE applied ONLY on local (sliding-window) layers; no positional
    // encoding at all on global layers. Requires swa_pattern > 0 (see
    // DenseConfig::layer_is_swa) to have any effect.
    bool rope_only_swa = false;

    // QKV fused into a single tensor (falcon: attn_qkv.weight, shape
    // {n_embd, n_embd+2*n_embd_gqa}) instead of separate attn_{q,k,v}.weight
    // tensors. Split by ROW at load time (see load_layer_data): rows
    // [0,q_dim) = Q, [q_dim,q_dim+kv_dim) = K, [q_dim+kv_dim,q_dim+2*kv_dim)
    // = V — a row-contiguous layout that holds for both the float fallback
    // and the direct quantized formats (no quantization block ever spans a
    // row boundary).
    bool fused_qkv = false;

    // No RoPE at all (gpt2: absolute position; bloom/mpt: ALiBi). Q/K stay
    // unchanged after the projection step.
    bool no_rope = false;

    // ALiBi (bloom, mpt): an additive bias on the attention scores,
    // proportional to the query-key distance and a per-head slope, used
    // INSTEAD of RoPE. The exact formula and slope derivation are in the
    // comment on DenseConfig::max_alibi_bias.
    bool alibi = false;

    // A norm applied ONCE to the embeddings, before the first layer (not
    // per layer): bloom (tok_norm/tok_norm_b).
    bool embd_norm = false;

    // Per-head output gate on attention (blk.N.attn_gate.weight, shape
    // {n_embd, n_head}: ONE scalar per head, not per channel). The gate is
    // computed from the SAME normalized input that feeds Q/K/V — not from
    // the attention output and not from the raw residual — then passed
    // through a sigmoid, and each head's slice of the attention output is
    // multiplied by its own scalar before the output projection:
    //
    //   g    = sigmoid(W_gate * x_norm)          // one value per head
    //   a[h] = a[h] * g[h]                       // a[h] is head_dim wide
    //   out  = W_o * a
    //
    // Reading the gate from the wrong input, or broadcasting it per channel
    // instead of per head, both produce plausible-looking values and
    // degenerate text, so the source and the broadcast shape both matter.
    bool attn_gate = false;
};

DESIREEIA_INTERNAL DenseQuirks quirks_for(ArchKind kind);

// Tokenizer algorithms recognized from the "tokenizer.ggml.model" tag (or
// equivalent) declared in the format's metadata.
enum class TokenizerKind {
    Unknown,
    SpmUnigram, // SentencePiece Unigram with byte-fallback
    Bpe,        // BPE with a merge table
};

DESIREEIA_INTERNAL TokenizerKind detect_tokenizer(const std::string& tag);

}

#endif
