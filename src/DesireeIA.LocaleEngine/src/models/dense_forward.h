#ifndef DESIREEIA_DENSE_FORWARD_H
#define DESIREEIA_DENSE_FORWARD_H

#include "core/engine.h"
#include "core/arch_tags.h"
#include "core/forward_iface.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

struct DenseConfig {
    uint32_t n_vocab = 0;
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_rot = 0;
    uint32_t n_ff = 0;
    float rms_eps = 1e-6f;
    // LayerNorm epsilon (DenseQuirks::layer_norm), a separate metadata key
    // from rms_eps above: RMSNorm architectures never write this key, so
    // the default here never affects them.
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;

    // Per-layer RoPE (gemma3 and similar sliding-window architectures).
    // gemma3 alternates "local" (sliding-window) and "global" layers with
    // a DIFFERENT RoPE base frequency. The pattern is `is_swa[il] = (il %
    // swa_pattern < swa_pattern-1)`, i.e. with swa_pattern=6: layers 0-4
    // are local, layer 5 is global, and so on. Local layers use
    // rope_theta_swa (default 10000, "rope.freq_base_swa" key if present)
    // and freq_scale 1.0; global layers use rope_theta ("rope.freq_base"
    // key) and freq_scale = 1/"rope.scaling.factor". Using a single theta
    // for every layer, as an earlier version of this code did, got 5 out
    // of 6 layers wrong and produced completely degenerate output.
    // swa_pattern == 0 disables all of this (the classic single-theta
    // behavior, correct for architectures without sliding-window attention).
    uint32_t swa_pattern = 0;
    uint32_t n_swa = 0;              // local window width (0 = no masking)
    float rope_theta_swa = 10000.0f;
    float rope_freq_scale = 1.0f;      // global layers
    float rope_freq_scale_swa = 1.0f;  // local layers

    bool layer_is_swa(uint32_t il) const {
        return swa_pattern != 0 && (il % swa_pattern) < (swa_pattern - 1);
    }
    float layer_rope_theta(uint32_t il) const {
        return layer_is_swa(il) ? rope_theta_swa : rope_theta;
    }
    float layer_rope_scale(uint32_t il) const {
        return layer_is_swa(il) ? rope_freq_scale_swa : rope_freq_scale;
    }

    // YaRN (extended-context RoPE, used by DeepSeek2 and its siblings).
    // With rope_ext_factor == 0 (the default) the YaRN branch is
    // completely disabled and the formula collapses exactly onto the
    // classic one already used by every other architecture (theta =
    // freq_scale*pos*base^-2j/n, mscale = rope_attn_factor multiplied
    // onto cos/sin) — zero regression risk for models that don't set
    // these fields.
    float rope_ext_factor = 0.0f;
    float rope_attn_factor = 1.0f;
    float rope_beta_fast = 32.0f;
    float rope_beta_slow = 1.0f;
    uint32_t n_ctx_orig_yarn = 0; // the model's trained context length ("context_length" key)

    // dims[0]/dims[1]: the range (in dimension-pair index, not raw
    // dimensions) over which the YaRN ramp transitions from interpolation
    // to extrapolation.
    void layer_rope_corr_dims(uint32_t il, float& lo, float& hi) const {
        if (rope_ext_factor == 0.0f || n_ctx_orig_yarn == 0) { lo = 0.0f; hi = 0.0f; return; }
        const float base = layer_rope_theta(il);
        auto corr_dim = [&](float n_rot_beta) {
            return (float) n_rot * logf((float) n_ctx_orig_yarn / (n_rot_beta * 2.0f * 3.14159265358979323846f)) / (2.0f * logf(base));
        };
        float start = floorf(corr_dim(rope_beta_fast));
        float end = ceilf(corr_dim(rope_beta_slow));
        lo = std::max(0.0f, start);
        hi = std::min((float) n_rot - 1.0f, end);
    }

    // MoE (n_expert==0 -> classic dense layer, wff_*; n_expert>0 -> router
    // + ExpertStore, wff_* unused). See models/moe_route.h.
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_ff_expert = 0; // can differ from n_ff (expert_feed_forward_length)
    bool moe_norm_w = true;   // false for a couple of architectures that skip renormalization
    float moe_w_scale = 1.0f;
    // "Dense-lead" layers (DeepSeek2 and siblings): the first
    // n_layer_dense_lead layers use the classic gated FFN even though
    // n_expert>0 globally (the MoE router only kicks in from that layer
    // onward).
    uint32_t n_layer_dense_lead = 0;
    // Shared expert (DeepSeek2 and siblings): besides the n_expert_used
    // routed experts, EVERY token also passes through a fixed gated FFN
    // block of width n_ff_expert*n_expert_shared, whose output gets ADDED
    // (not averaged) to the routed MoE output.
    uint32_t n_expert_shared = 0;
    bool moe_sigmoid_gate = false; // true for the sigmoid-gated MoE variant
    bool moe_has_sel_bias = false; // blk.N.exp_probs_b.bias present

    // MLA (DeepSeek2 and siblings): compressed-rank attention. See the
    // extended note on DenseQuirks::mla in arch_tags.h for the formula
    // derivation.
    uint32_t q_lora_rank = 0;  // 0 -> "lite" variant, Q projected directly (wq)
    uint32_t kv_lora_rank = 0;
    uint32_t n_embd_head_qk_rope = 0; // == n_rot
    uint32_t n_embd_head_qk_nope = 0; // n_embd_head_k_mla - n_embd_head_qk_rope
    uint32_t n_embd_head_v_mla = 0;
    float rope_yarn_log_mul = 0.0f; // extra mscale correction, MLA-only

    // Final scale applied to the logits before returning them (cohere2
    // and other architectures with a logit-scale metadata key). 0.0f
    // (default) means no scaling, since a real scale of 0 would zero out
    // the logits and never makes sense in practice — a safe sentinel for
    // "not set".
    float logit_scale = 0.0f;

    // ALiBi (bloom, mpt): the maximum slope used to derive the per-head
    // slope. With n_head_log2 = 1 << floor(log2(n_head)):
    //   m0 = 2^(-max_alibi_bias/n_head_log2), m1 = 2^(-max_alibi_bias/2/n_head_log2)
    //   slope(h) = h < n_head_log2 ? m0^(h+1) : m1^(2*(h-n_head_log2)+1)
    // final score = dot*scale - slope(h)*(query_pos - key_pos), applied
    // only to positions already allowed by the causal mask.
    float max_alibi_bias = 0.0f;

    // Clamps Q/K/V right after the projection (+bias), before qk_norm/RoPE
    // (mpt: attention.clamp_kqv, almost always absent/0 = disabled).
    float clamp_kqv = 0.0f;
};

// A weight matrix, in one of two representations:
// - quantized: `raw` holds the Q4_0 bytes exactly as stored on disk (18
//   bytes per 32-weight block), consumed directly by matmul_q4_0 without
//   ever materializing a float version (roughly 7x less RAM and memory
//   bandwidth).
// - float: `f` is the dequantized version, consumed by matmul_f32.
// The choice is automatic per tensor based on its actual on-disk format
// (see DenseForward::load_matrix): not every tensor in a model is
// necessarily in the same format.
enum class MatVecFormat { Float, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_K };

struct MatVec {
    std::vector<float> f;
    std::vector<uint8_t> raw;
    MatVecFormat format = MatVecFormat::Float;
    bool empty() const { return f.empty() && raw.empty(); }
};

// A single layer's weights. Used both as a scratch buffer (weight cache
// disabled: reloaded/overwritten every step) and as a persistent cache
// slot (weight cache enabled: loaded once).
struct LayerWeights {
    std::vector<float> attn_norm;
    std::vector<float> ffn_norm;
    std::vector<float> attn_norm_b; // DenseQuirks::layer_norm, dim n_embd, optional
    std::vector<float> ffn_norm_b;  // DenseQuirks::layer_norm, dim n_embd, optional
    // Optional second norm for the attention input (a rare large
    // checkpoint variant, DenseQuirks::parallel_residual): when present,
    // attention uses this instead of attn_norm's output (which then feeds
    // ONLY the FFN). Absent almost everywhere: read defensively, dim n_embd.
    std::vector<float> attn_norm2;
    std::vector<float> attn_norm2_b;
    MatVec wq;
    MatVec wk;
    MatVec wv;
    MatVec wo;
    std::vector<float> bq;
    std::vector<float> bk;
    std::vector<float> bv;
    std::vector<float> bo; // attention output projection bias
                           // (attn_output.bias), dim n_embd, optional.
                           // Present in several LayerNorm-based
                           // architectures, absent elsewhere: always read
                           // defensively (see load_layer_data), no
                           // dedicated quirk needed.
    MatVec wff_gate;
    MatVec wff_up;
    MatVec wff_down;
    std::vector<float> ffn_up_b;   // DenseQuirks::ffn_gated==false, dim n_ff, optional
    std::vector<float> ffn_down_b; // DenseQuirks::ffn_gated==false, dim n_embd, optional
    std::vector<float> q_norm;        // DenseQuirks::qk_norm, dim head_dim
    std::vector<float> k_norm;        // DenseQuirks::qk_norm, dim head_dim
    std::vector<float> post_attn_norm; // DenseQuirks::sandwich_norm, dim n_embd
    std::vector<float> post_ffn_norm;  // DenseQuirks::sandwich_norm, dim n_embd
    MatVec router; // DenseConfig::n_expert>0, shape [n_expert, n_embd] (ffn_gate_inp)
    std::vector<float> router_bias; // DenseQuirks::moe_has_sel_bias, dim n_expert, optional

    // MoE experts, stacked. GGUF keeps every expert of a layer in one tensor
    // (blk.N.ffn_{gate,up,down}_exps.weight) with the experts laid out as
    // contiguous row ranges, so a single expert is a byte slice of it and
    // needs no read of its own.
    //
    // Kept here still quantized, for the same reason every other weight is:
    // the matmul kernels consume quantized bytes directly. The older path
    // fetched one expert at a time through ExpertStore, which reopened the
    // model file and dequantized the expert to float on every fetch — per
    // expert, per layer, per token, then ran the scalar float matmul on the
    // result. These stay empty when the stacked tensors aren't present, and
    // that older path still runs.
    MatVec wexp_gate;
    MatVec wexp_up;
    MatVec wexp_down;
    bool   exps_stacked = false;

    // MLA (DenseQuirks::mla).
    std::vector<float> attn_q_a_norm;  // dim q_lora_rank, only if q_lora_rank>0
    std::vector<float> attn_kv_a_norm; // dim kv_lora_rank
    MatVec wq_a; // n_embd -> q_lora_rank (only if q_lora_rank>0)
    MatVec wq_b; // q_lora_rank -> n_head*n_embd_head_k_mla (only if q_lora_rank>0)
    MatVec wq_lite; // n_embd -> n_head*n_embd_head_k_mla ("lite" variant, q_lora_rank==0)
    MatVec wkv_a_mqa; // n_embd -> kv_lora_rank + n_embd_head_qk_rope
    // wk_b/wv_b: one block per head, [n_embd_head_qk_nope|v_mla, kv_lora_rank]
    // each, kept as a vector of MatVec (one per head) since the
    // computation applies them as a batch of small per-head matmuls
    // rather than a single combined matrix.
    std::vector<MatVec> wk_b_h; // n_head elements, each [kv_lora_rank, n_embd_head_qk_nope]
    std::vector<MatVec> wv_b_h; // n_head elements, each [n_embd_head_v_mla, kv_lora_rank]

    // Shared-expert FFN (DeepSeek2 and siblings, DenseQuirks::mla + n_expert_shared>0).
    MatVec ffn_gate_shexp;
    MatVec ffn_up_shexp;
    MatVec ffn_down_shexp;
};

// Float forward path for dense transformer architectures (grouped-query
// attention, NEOX RoPE, RMSNorm, gated FFN). Differences between model
// families (embedding scale, SiLU/GELU-tanh activation, bias on q/k/v)
// are parameterized through DenseQuirks rather than needing one class per
// architecture. Weights are streamed from the reader layer by layer,
// dequantized on demand (or cached in RAM if the model fits the planned
// budget, see open()); embeddings and the KV cache stay resident.
class DenseForward : public IForwardEngine {
public:
    ~DenseForward() override;
    bool open(ModelReader& rd, const ModelMeta& meta, ArchKind arch, uint64_t ram_budget_mb,
              ExpertStore* experts, bool kv_quantized = false);
    void reset_cache() override;
    // If all_logits != nullptr, it's filled with n_tokens*n_vocab logits
    // (one per batch position, not just the last): used by speculative
    // decoding / prompt-lookup verification, which needs to compare the
    // model's prediction at EVERY position against the draft token that
    // follows it. The extra cost is only paid when requested (the output
    // projection is the model's largest matrix): a single batched call
    // instead of n_tokens separate ones.
    bool step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
              std::vector<float>& last_logits, std::vector<float>* all_logits = nullptr) override;
    const DenseConfig& config() const { return cfg_; }
    static int32_t argmax(const std::vector<float>& logits);

    // Truncates the KV cache to `new_cols` positions (used to discard
    // rejected drafts after speculative verification): bytes beyond
    // new_cols stay in the buffer but get overwritten by the next step()
    // calls, no explicit erasure needed.
    void truncate_cache(size_t new_cols) { if (new_cols <= cache_cols_) cache_cols_ = new_cols; }
    size_t cache_len() const { return cache_cols_; }

    // Bytes used by the resident K/V cache (tokens already processed, not
    // allocated capacity): used by engine_context_size.
    uint64_t kv_bytes() const override {
        if (kv_quantized_) {
            return (uint64_t) cache_cols_ * cfg_.n_layers * cfg_.n_head_kv * kv_q_row_bytes_ * 2;
        }
        return (uint64_t) cache_cols_ * cfg_.n_layers * cfg_.n_head_kv * cfg_.head_dim * 2 * sizeof(float);
    }
    bool weight_cache_enabled() const override { return cache_enabled_; }

    // Diagnostic only: which tensor/check made the last load_layer_data or
    // step() call fail. The ABI reports failures as a bare error code with no
    // detail, and finding the actual cause meant reading through native stdio
    // that doesn't reliably reach a caller across the DLL boundary (a mingw
    // CRT inside the DLL does not necessarily share the host process's
    // stdio buffering/handles) — this is the channel that does: the engine
    // already logs it through the caller-supplied LogFn on failure.
    const std::string& last_fail() const override { return last_fail_; }

private:
    mutable std::string last_fail_;
    bool load_embd(ModelReader& rd);
    bool load_norms(ModelReader& rd);
    bool load_layer_data(ModelReader& rd, uint32_t il, LayerWeights& w);
    bool load_matrix(ModelReader& rd, const std::string& name, uint32_t rows, uint32_t cols, MatVec& out);
    bool load_qkv_fused(ModelReader& rd, const std::string& name,
                         uint32_t q_dim, uint32_t kv_dim, uint32_t cols,
                         MatVec& wq, MatVec& wk, MatVec& wv);
    // Runs one MoE expert's gated FFN for the activation `xin`, writing the
    // expert's contribution (unweighted) into `out`.
    //
    // Prefers the stacked quantized tensors, where the expert is a byte slice
    // and the quantized kernels apply. Falls back to ExpertStore's
    // dequantized per-expert copies when a layer has no stacked tensors, so
    // checkpoints laid out differently keep working.
    //
    // ffn/ffn_gate are scratch of n_ff_expert; egate/eup/edown are scratch for
    // the fallback only. They are passed in rather than allocated here because
    // this runs once per selected expert per layer per token.
    bool expert_ffn(const LayerWeights& lw, uint32_t layer, uint32_t eidx,
                    const float* xin, bool gelu_act,
                    std::vector<float>& ffn, std::vector<float>& ffn_gate,
                    std::vector<float>& out,
                    std::vector<float>& egate, std::vector<float>& eup,
                    std::vector<float>& edown);
    const LayerWeights* get_layer(ModelReader& rd, uint32_t il);
    void grow_cache(size_t needed);

    // Stores one position's freshly computed K and V (kv_dim floats each,
    // n_head_kv heads back to back) into whichever cache is active — float
    // or, when kv_quantized_, one Q8_0 row per kv-head. Shared by the
    // prefill (batched) and decode write sites so the branch on
    // kv_quantized_ exists in exactly one place.
    void write_kv_cache(uint32_t l, uint32_t pos, const float* k, const float* v);

    // MLA (DenseQuirks::mla): computes compressed-rank attention for ONE
    // token (position pos of layer l) and writes the result, already
    // projected through wo, into proj_out (n_embd elements) — called
    // identically from both the batch and decode branches of step(). See
    // the extended note in the .cpp for the formula derivation.
    void mla_attn_layer(const LayerWeights* lw, uint32_t l, uint32_t pos,
                         const float* xnp, float rope_th, float rope_sc,
                         float ext_factor, float attn_factor, float corr_lo, float corr_hi,
                         float kq_scale, float* proj_out);
    // Extracts row `idx` (one embedding) from tok_embd_, dequantizing only
    // that row on the fly if the tensor stayed quantized in RAM (not the
    // whole tensor, which would stay multiple gigabytes in float for a
    // large vocabulary).
    bool embed_row(uint32_t idx, float* out) const;

    DenseConfig cfg_;
    DenseQuirks quirks_;
    ExpertStore* experts_ = nullptr; // not owned, see ctx.cpp
    MatVec tok_embd_;

    // The output head, when the model ships one of its own ("output.weight").
    //
    // Plenty of architectures tie it to the input embedding and store no such
    // tensor, and this used to assume that was always the case — projecting
    // the logits through tok_embd_ unconditionally. For a model that does
    // carry a separate head that is silently the wrong matrix: the logits
    // come out finite and normally distributed, so nothing fails, and the
    // model simply predicts confident nonsense. Empty means tied, and then
    // tok_embd_ really is the right matrix.
    MatVec out_head_;

    std::vector<float> out_norm_;
    std::vector<float> out_norm_b_; // DenseQuirks::layer_norm, optional

    // Absolute position (gpt2, mpt optionally): table [n_ctx_train,
    // n_embd], added to the token embedding BEFORE the first layer.
    // Empty if the "position_embd.weight" tensor doesn't exist in the
    // file (read defensively, not only for DenseQuirks::no_rope: mpt can
    // have it even while using ALiBi as its main mechanism).
    std::vector<float> pos_embd_;
    uint32_t n_ctx_train_ = 0;

    // Initial norm applied to the embeddings, ONCE before the first layer
    // (bloom, DenseQuirks::embd_norm).
    std::vector<float> tok_norm_;
    std::vector<float> tok_norm_b_;

    // Layer weight cache: when enabled (the dequantized model fits within
    // a conservative fraction of the planned RAM budget), each layer is
    // read/dequantized from disk once and reused on later steps.
    // Otherwise it falls back to the original behavior (re-dequantizes
    // scratch_ on every step, slower but with constant memory usage
    // independent of layer count).
    bool cache_enabled_ = false;
    // Whether MoE layers keep their experts stacked and quantized inside the
    // layer (fast, but only sound while layers stay resident) or go one
    // expert at a time through ExpertStore. Decided at load, from whether the
    // layer cache can hold the model — see the note where it is set.
    bool stack_experts_ = false;
    std::vector<LayerWeights> layer_cache_;
    LayerWeights scratch_;

    // KV cache: [layer][col][kv_dim]
    std::vector<float> k_cache_;
    std::vector<float> v_cache_;
    size_t cache_capacity_ = 0;
    size_t cache_cols_ = 0;

    // Q8_0-quantized KV cache, used instead of k_cache_/v_cache_ when
    // kv_quantized_ is set (see open()'s kv_quantized parameter). Roughly
    // 3.76x smaller than the float cache (34 bytes per 32-float sub-block vs
    // 128), which is real memory-bandwidth headroom on the path that reads
    // the cache once per token per position — see docs/performance_plan.md
    // item 2. Off (empty, k_cache_/v_cache_ used) unless the caller opts in:
    // this changes what decode actually computes, not just how it's stored,
    // and stays a deliberate choice rather than a silent default flip.
    //
    // Not used for MLA layers regardless of this flag: MLA's own compressed
    // latent cache (kv_lora_rank+rope wide, a few hundred floats) is already
    // far smaller than a classic per-head KV cache, so quantizing it on top
    // buys little, and MLA's absorbed-attention math was hard-won correctness
    // work this session — not something to put back at risk for a small gain.
    bool kv_quantized_ = false;
    std::vector<uint8_t> k_cache_q_;
    std::vector<uint8_t> v_cache_q_;
    // Whether tok_embd_/out_head_ were successfully locked resident (see
    // DESIREEIA_MLOCK in open()), so the destructor knows whether to unlock
    // them — a reload within the same process (Load/Dispose in the .NET
    // wrapper) must not leave a previous model's pages holding locked-memory
    // quota after that model is gone.
    bool mlocked_ = false;
    // Bytes one kv-head's quantized row occupies (kv_quant_row_bytes(head_dim)),
    // cached at open() time so the hot path never recomputes it.
    size_t kv_q_row_bytes_ = 0;
};

}

#endif
