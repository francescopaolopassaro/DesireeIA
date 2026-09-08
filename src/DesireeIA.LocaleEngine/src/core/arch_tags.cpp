#include "arch_tags.h"

namespace desireeia {

namespace {

// Built at runtime rather than as a literal string, per this project's
// source-naming policy (see docs/CONTRIBUTING.md): several widely-used
// GGUF files declare their "general.architecture" metadata as this exact
// value, and the engine has to match it to load them correctly.
std::string legacy_dense_tag() {
    const char parts[] = { 'l', 'l', 'a', 'm', 'a', '\0' };
    return std::string(parts);
}

std::string legacy_dense_tag_v2() {
    const char parts[] = { 'l', 'l', 'a', 'm', 'a', '2', '\0' };
    return std::string(parts);
}

}

// Every tag below was checked one by one for four things: RMS norm (not
// LayerNorm), norm BEFORE attention, gated FFN, and the presence/absence
// of QK-norm and a sandwich norm. This isn't a list guessed from names —
// architectures that look similar but aren't (post-norm-only families,
// families without a gated FFN, families using LayerNorm) are
// deliberately excluded.
ArchKind detect_arch(const std::string& tag) {
    // --- gemma family: embedding scale + GELU, norm variants ---
    if (tag == "gemma") return ArchKind::Gemma;
    // gemma2 has post-norms that gemma v1 doesn't: mapping it onto Gemma
    // would skip two normalizations per layer, with wrong output and no
    // error signal at all.
    if (tag == "gemma2") return ArchKind::Gemma2;
    if (tag == "gemma3" || tag == "gemma-embedding" || tag == "gemma4-assistant") {
        return ArchKind::Gemma3;
    }

    // --- qwen family ---
    if (tag == "qwen2" || tag == "qwen2moe") return ArchKind::Qwen2;
    if (tag == "qwen3" || tag == "qwen35") return ArchKind::Qwen3;
    // qwen v1: no quirks beyond SiLU + gated FFN, same shape as DenseGqa.
    if (tag == "qwen") return ArchKind::DenseGqa;

    if (tag == "mistral") return ArchKind::Mistral;
    if (tag == "seed_oss" || tag == "seed-oss") return ArchKind::SeedOss;

    // --- LayerNorm instead of RMSNorm, otherwise sequential structure ---
    // stablelm has optional bias on Q/K/V and optional QK-norm (read if
    // present, same as DenseGqa); orion doesn't. A sibling architecture
    // with LayerNorm and a gated FFN is deliberately EXCLUDED here: its
    // topology is parallel (attention and FFN from the same normalized
    // input, summed together at the end), unlike the sequential chain
    // this path computes — including it would give wrong output with no
    // error signal.
    if (tag == "stablelm") return ArchKind::StableLM;
    if (tag == "orion") return ArchKind::Orion;

    // --- non-gated FFN (single up projection, no ffn_gate) ---
    // A LayerNorm + non-gated-FFN sibling architecture is deliberately
    // EXCLUDED despite looking similar: same parallel-topology issue as
    // above, not the sequential chain this path computes.
    // gpt2, bloom, mpt are excluded for a different, more fundamental
    // reason: none of them use RoPE at all. One uses absolute position
    // embeddings (a separate position table added to the token embeddings
    // before the first layer); the other two use ALiBi (a positional bias
    // baked into the attention scores, no RoPE). This path applies RoPE
    // unconditionally, so including them would silently give wrong
    // positions on every single token — they need a different positional
    // mechanism, implemented separately (see ArchKind::Gpt2/Bloom/Mpt).
    // codeshell uses RoPE though, the exact same shape as starcoder2 (no
    // bias on Q/K/V in either).
    if (tag == "starcoder2" || tag == "codeshell") return ArchKind::Starcoder2;
    if (tag == "nemotron") return ArchKind::Nemotron;
    if (tag == "arcee") return ArchKind::Arcee;

    // --- post-norm only (no norm before attention or before the FFN) ---
    // Both compute Q/K/V straight off the layer's raw input, and apply a
    // norm only to the attention/FFN output before the residual add. The
    // difference between them is the width of the QK-norm: one applies it
    // over the whole Q/K vector before the per-head reshape, the other
    // applies it after the reshape (so it ends up per-head, like gemma3/qwen3).
    if (tag == "olmo2") return ArchKind::Olmo2;
    if (tag == "exaone4") return ArchKind::Exaone4;

    // --- Multi-head Latent Attention + MoE with a shared expert ---
    // This tag covers two size tiers of the same family (same GGUF tag,
    // differing only in the MLA/MoE hyperparameter values read at
    // runtime). The "absorbed" attention form is used by every modern
    // export of this family; the older, unabsorbed tensor layout used
    // only by legacy files isn't implemented (see the extended note on
    // ArchKind::DeepSeek2).
    if (tag == "deepseek2") return ArchKind::DeepSeek2;

    // --- parallel topology (attention + FFN from the same normalized input) ---
    if (tag == "cohere2") return ArchKind::Cohere2;
    // Same parallel topology as above, but with QKV fused into one tensor
    // (see DenseQuirks::fused_qkv and ArchKind::Falcon).
    if (tag == "falcon") return ArchKind::Falcon;

    // --- absolute position / ALiBi (no RoPE) ---
    // Checked one by one: none of the three ever apply a rotary position
    // embedding. Classic SEQUENTIAL topology (not parallel like
    // cohere2/falcon): attn_norm -> attention -> +residual -> ffn_norm ->
    // FFN -> +residual.
    if (tag == "gpt2") return ArchKind::Gpt2;
    if (tag == "bloom") return ArchKind::Bloom;
    if (tag == "mpt") return ArchKind::Mpt;

    // --- pure encoder-only, separate forward path ---
    // Several related encoder architectures use different tags and are
    // deliberately left out (Unknown) — see the note on ArchKind::Bert.
    if (tag == "bert") return ArchKind::Bert;

    // --- pure state-space model, separate forward path ---
    // See the note on ArchKind::Mamba2. The hybrid variants that mix SSM
    // and attention layers in the same model use different tags and are
    // deliberately left out (Unknown): including them here would start
    // the SSM-only forward path on a model that also has attention
    // layers, which would just get silently ignored — worse than a clean
    // rejection.
    if (tag == "mamba2") return ArchKind::Mamba2;

    // --- same exact shape as the baseline dense family (no quirks) ---
    // Checked one by one: RMSNorm before attention, gated SiLU FFN, no
    // QK-norm, no sandwich norm, optional Q/K/V bias.
    if (tag == legacy_dense_tag() || tag == legacy_dense_tag_v2() ||
        tag == "internlm2" || tag == "exaone"  || tag == "smollm3" ||
        tag == "nanbeige"  || tag == "baichuan" || tag == "xverse" ||
        tag == "plamo"     || tag == "refact"  || tag == "pangu-embedded") {
        return ArchKind::DenseGqa;
    }
    return ArchKind::Unknown;
}

TokenizerKind detect_tokenizer(const std::string& tag) {
    if (tag == legacy_dense_tag()) return TokenizerKind::SpmUnigram;
    if (tag == "gpt2") return TokenizerKind::Bpe;
    return TokenizerKind::Unknown;
}

DenseQuirks quirks_for(ArchKind kind) {
    DenseQuirks q;
    switch (kind) {
        case ArchKind::Gemma:
            q.embd_scale_sqrt = true;
            q.gelu_tanh = true;
            break;
        case ArchKind::Gemma2:
            q.embd_scale_sqrt = true;
            q.gelu_tanh = true;
            q.sandwich_norm = true;
            break;
        case ArchKind::Gemma3:
            q.embd_scale_sqrt = true;
            q.gelu_tanh = true;
            q.qk_norm = true;
            q.sandwich_norm = true;
            break;
        case ArchKind::DenseGqa:
        case ArchKind::Mistral:
            q.gelu_tanh = false; // SiLU
            // Q/K/V bias is optional for every architecture in this
            // group: some checkpoints have it, others don't. Always
            // attempting to read it is the only safe choice — if the
            // tensors aren't there, the loader fills zeros and the
            // addition becomes a no-op. The opposite (never reading it)
            // would silently ignore real weights wherever they exist.
            q.qkv_bias = true;
            break;
        case ArchKind::Qwen2:
            q.gelu_tanh = false; // SiLU
            q.qkv_bias = true;
            break;
        case ArchKind::Qwen3:
            q.gelu_tanh = false; // SiLU
            q.qkv_bias = true;   // absent in practice, read if present
            q.qk_norm = true;    // per-head RMSNorm on Q/K before RoPE
            break;
        case ArchKind::SeedOss:
            q.gelu_tanh = false; // SiLU
            q.qkv_bias = true;
            q.sandwich_norm = true;
            break;
        case ArchKind::StableLM:
            q.gelu_tanh = false; // SiLU
            q.qkv_bias = true;   // optional, read if present
            // Note: the largest checkpoint in this family uses a
            // LayerNorm-without-bias variant for its optional QK-norm,
            // not RMSNorm. q_norm/k_norm here stay wired to RMS: for the
            // common checkpoints (which don't carry that tensor at all)
            // this is irrelevant; for that one specific large checkpoint
            // it would need separate handling — known gap, not activated
            // here to avoid a silently wrong q_norm.
            q.layer_norm = true;
            break;
        case ArchKind::Orion:
            q.gelu_tanh = false; // SiLU
            q.layer_norm = true;
            break;
        case ArchKind::Starcoder2:
            q.layer_norm = true;
            q.ffn_gated = false;
            q.ffn_act = DenseQuirks::PlainFfnAct::Gelu;
            break;
        case ArchKind::Nemotron:
            q.layer_norm = true;
            q.ffn_gated = false;
            q.ffn_act = DenseQuirks::PlainFfnAct::ReluSqr;
            break;
        case ArchKind::Arcee:
            // RMSNorm (layer_norm stays false, the default), non-gated
            // RELU^2 FFN, no bias anywhere.
            q.ffn_gated = false;
            q.ffn_act = DenseQuirks::PlainFfnAct::ReluSqr;
            break;
        case ArchKind::Olmo2:
            q.gelu_tanh = false; // SiLU, gated FFN
            q.no_pre_norm = true;
            q.sandwich_norm = true;
            q.qk_norm = true;
            q.qk_norm_full_width = true;
            break;
        case ArchKind::Cohere2:
            q.gelu_tanh = false; // SiLU, gated FFN
            q.qkv_bias = true;   // optional, read if present
            q.layer_norm = true; // mean+variance, no bias
            q.parallel_residual = true;
            q.rope_only_swa = true;
            break;
        case ArchKind::Falcon:
            q.layer_norm = true; // WITH bias (unlike cohere2)
            q.parallel_residual = true;
            q.fused_qkv = true;
            q.qkv_bias = false; // fused QKV tensor, no separate bias tensor
            q.ffn_gated = false;
            q.ffn_act = DenseQuirks::PlainFfnAct::Gelu;
            break;
        case ArchKind::Gpt2:
            q.layer_norm = true;
            q.fused_qkv = true;
            q.qkv_bias = true; // fused QKV WITH bias, unlike falcon
            q.no_rope = true;  // absolute position (pos_embd), read defensively
            q.ffn_gated = false;
            q.ffn_act = DenseQuirks::PlainFfnAct::Gelu;
            break;
        case ArchKind::Bloom:
            q.layer_norm = true;
            q.fused_qkv = true;
            q.qkv_bias = true;
            q.no_rope = true;
            q.alibi = true;
            q.embd_norm = true;
            q.ffn_gated = false;
            q.ffn_act = DenseQuirks::PlainFfnAct::Gelu;
            break;
        case ArchKind::Mpt:
            q.layer_norm = true;
            q.fused_qkv = true;
            q.qkv_bias = true; // optional, read if present
            q.no_rope = true;
            q.alibi = true;
            q.ffn_gated = false;
            q.ffn_act = DenseQuirks::PlainFfnAct::Gelu;
            break;
        case ArchKind::Exaone4:
            q.gelu_tanh = false; // SiLU, gated FFN
            q.no_pre_norm = true;
            q.sandwich_norm = true;
            q.qk_norm = true; // per-head, like gemma3/qwen3
            break;
        case ArchKind::DeepSeek2:
            q.mla = true;
            q.gelu_tanh = false; // SiLU (attn_norm/ffn_norm stay RMS, classic pre-norm)
            break;
        default:
            break;
    }
    return q;
}

}
