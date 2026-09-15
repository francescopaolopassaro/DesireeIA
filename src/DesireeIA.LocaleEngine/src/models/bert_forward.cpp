// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "bert_forward.h"
#include "core/thread_pool.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace desireeia {

namespace {

// Deliberate simplification (first correctness pass, no real BERT GGUF
// available to test against): weights kept entirely in float, without
// DenseForward's direct quantized kernels. BERT embedding models are
// typically small (a few tens to a few hundred MB even unquantized): a
// known, documented gap, not the critical bottleneck it would be for a
// large generative model.
void matvec_f32(const float* w, size_t r, size_t c, const float* x, float* y) {
    ThreadPool::global().parallel_for(r, [&](size_t j0, size_t j1) {
        for (size_t j = j0; j < j1; ++j) {
            const float* wrow = w + j * c;
            double acc = 0.0;
            for (size_t i = 0; i < c; ++i) acc += (double) wrow[i] * x[i];
            y[j] = (float) acc;
        }
    });
}

// gelu_tanh: same formula used by DenseForward for the other non-gated
// architectures (starcoder2/gpt2/bloom/mpt/falcon) — the tanh
// approximation, not the exact erf-based form.
float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Classic LayerNorm (population mean + variance, then weight and bias),
// the same formula as dense_forward.cpp::layer_norm_vec.
void layer_norm_vec(const float* x, const float* w, const float* b, float* y, size_t n, float eps) {
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += x[i];
    mean /= (double) n;
    double var = 0.0;
    for (size_t i = 0; i < n; ++i) { const double d = x[i] - mean; var += d * d; }
    var /= (double) n;
    const float scale = 1.0f / sqrtf((float) var + eps);
    for (size_t i = 0; i < n; ++i) y[i] = ((x[i] - (float) mean) * scale) * w[i] + b[i];
}

}

bool BertForward::load_layer(ModelReader& rd, uint32_t il, BertLayerWeights& w) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%u.", il);
    const std::string p = buf;

    const uint32_t q_dim = cfg_.n_head * cfg_.head_dim;
    const uint32_t kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    const uint32_t n_embd = cfg_.n_embd;

    if (!rd.read_tensor(p + "attn_q.weight", w.wq) || w.wq.size() != (size_t) q_dim * n_embd) return false;
    if (!rd.read_tensor(p + "attn_k.weight", w.wk) || w.wk.size() != (size_t) kv_dim * n_embd) return false;
    if (!rd.read_tensor(p + "attn_v.weight", w.wv) || w.wv.size() != (size_t) kv_dim * n_embd) return false;
    if (!rd.read_tensor(p + "attn_output.weight", w.wo) || w.wo.size() != (size_t) n_embd * q_dim) return false;

    w.bq.clear(); w.bk.clear(); w.bv.clear(); w.bo.clear();
    rd.read_tensor(p + "attn_q.bias", w.bq);
    rd.read_tensor(p + "attn_k.bias", w.bk);
    rd.read_tensor(p + "attn_v.bias", w.bv);
    rd.read_tensor(p + "attn_output.bias", w.bo);
    if (w.bq.size() != q_dim) w.bq.assign(q_dim, 0.0f);
    if (w.bk.size() != kv_dim) w.bk.assign(kv_dim, 0.0f);
    if (w.bv.size() != kv_dim) w.bv.assign(kv_dim, 0.0f);
    if (w.bo.size() != n_embd) w.bo.assign(n_embd, 0.0f);

    if (!rd.read_tensor(p + "attn_output_norm.weight", w.attn_out_norm) || w.attn_out_norm.size() != n_embd) return false;
    if (!rd.read_tensor(p + "attn_output_norm.bias", w.attn_out_norm_b) || w.attn_out_norm_b.size() != n_embd) return false;

    if (!rd.read_tensor(p + "ffn_up.weight", w.ffn_up) || w.ffn_up.size() != (size_t) cfg_.n_ff * n_embd) return false;
    if (!rd.read_tensor(p + "ffn_down.weight", w.ffn_down) || w.ffn_down.size() != (size_t) n_embd * cfg_.n_ff) return false;
    w.ffn_up_b.clear(); w.ffn_down_b.clear();
    rd.read_tensor(p + "ffn_up.bias", w.ffn_up_b);
    rd.read_tensor(p + "ffn_down.bias", w.ffn_down_b);
    if (w.ffn_up_b.size() != cfg_.n_ff) w.ffn_up_b.assign(cfg_.n_ff, 0.0f);
    if (w.ffn_down_b.size() != n_embd) w.ffn_down_b.assign(n_embd, 0.0f);

    if (!rd.read_tensor(p + "layer_output_norm.weight", w.layer_out_norm) || w.layer_out_norm.size() != n_embd) return false;
    if (!rd.read_tensor(p + "layer_output_norm.bias", w.layer_out_norm_b) || w.layer_out_norm_b.size() != n_embd) return false;

    return true;
}

bool BertForward::open(ModelReader& rd, const ModelMeta& meta, uint64_t /*ram_budget_mb*/) {
    const std::string arch_tag = meta.arch;
    const std::string kp = arch_tag + ".";

    cfg_.n_vocab = meta.n_vocab;
    cfg_.n_layers = meta.n_layers;
    uint32_t v32 = 0;
    float f32 = 0.0f;
    if (rd.meta_u32(kp + "embedding_length", v32)) cfg_.n_embd = v32;
    if (rd.meta_u32(kp + "attention.head_count", v32)) cfg_.n_head = v32;
    cfg_.n_head_kv = cfg_.n_head;
    if (rd.meta_u32(kp + "attention.head_count_kv", v32)) cfg_.n_head_kv = v32;
    if (rd.meta_u32(kp + "attention.key_length", v32)) {
        cfg_.head_dim = v32;
    } else if (cfg_.n_embd > 0 && cfg_.n_head > 0) {
        cfg_.head_dim = cfg_.n_embd / cfg_.n_head;
    }
    if (rd.meta_u32(kp + "feed_forward_length", v32)) cfg_.n_ff = v32;
    if (rd.meta_u32(kp + "context_length", v32)) cfg_.n_ctx_train = v32;
    if (rd.meta_f32(kp + "attention.layer_norm_eps", f32)) cfg_.eps = f32;

    if (cfg_.n_vocab == 0 || cfg_.n_layers == 0 || cfg_.n_embd == 0 || cfg_.n_head == 0 ||
        cfg_.n_head_kv == 0 || cfg_.head_dim == 0 || cfg_.n_ff == 0 || cfg_.n_ctx_train == 0) {
        return false;
    }

    if (!rd.read_tensor("token_embd.weight", tok_embd_) ||
        tok_embd_.size() != (size_t) cfg_.n_vocab * cfg_.n_embd) return false;

    // token_types is optional: only row 0 is ever used (every token is
    // hardcoded to "segment A" — this engine's API has no second
    // segment). If absent, simply nothing gets added.
    type_embd_row0_.clear();
    {
        std::vector<float> type_embd_full;
        rd.read_tensor("token_types.weight", type_embd_full);
        if (type_embd_full.size() >= cfg_.n_embd) {
            type_embd_row0_.assign(type_embd_full.begin(), type_embd_full.begin() + cfg_.n_embd);
        }
    }

    if (!rd.read_tensor("position_embd.weight", pos_embd_) ||
        pos_embd_.size() != (size_t) cfg_.n_ctx_train * cfg_.n_embd) return false;

    if (!rd.read_tensor("token_embd_norm.weight", tok_norm_) || tok_norm_.size() != cfg_.n_embd) return false;
    if (!rd.read_tensor("token_embd_norm.bias", tok_norm_b_) || tok_norm_b_.size() != cfg_.n_embd) return false;

    layers_.assign(cfg_.n_layers, BertLayerWeights{});
    for (uint32_t il = 0; il < cfg_.n_layers; ++il) {
        if (!load_layer(rd, il, layers_[il])) return false;
    }

    return true;
}

bool BertForward::encode(ModelReader& /*rd*/, const int32_t* tokens, size_t n_tokens, std::vector<float>& out_embd) {
    if (n_tokens == 0 || tokens == nullptr || tok_embd_.empty()) return false;

    const uint32_t n_embd = cfg_.n_embd;
    const uint32_t n_head = cfg_.n_head;
    const uint32_t n_head_kv = cfg_.n_head_kv;
    const uint32_t head_dim = cfg_.head_dim;
    const uint32_t q_dim = n_head * head_dim;
    const uint32_t kv_dim = n_head_kv * head_dim;
    const uint32_t heads_per_kv = n_head / n_head_kv;
    const float inv_d = 1.0f / sqrtf((float) head_dim);

    std::vector<float> x((size_t) n_tokens * n_embd);
    for (size_t p = 0; p < n_tokens; ++p) {
        const int32_t t = tokens[p];
        if (t < 0 || (uint32_t) t >= cfg_.n_vocab) return false;
        float* xp = x.data() + p * n_embd;
        std::memcpy(xp, tok_embd_.data() + (size_t) t * n_embd, n_embd * sizeof(float));
        if (!type_embd_row0_.empty()) {
            for (uint32_t i = 0; i < n_embd; ++i) xp[i] += type_embd_row0_[i];
        }
        if (p < cfg_.n_ctx_train) {
            const float* pe = pos_embd_.data() + p * n_embd;
            for (uint32_t i = 0; i < n_embd; ++i) xp[i] += pe[i];
        }
    }
    // Initial norm on the embeddings, applied ONCE before the first layer.
    for (size_t p = 0; p < n_tokens; ++p) {
        float* xp = x.data() + p * n_embd;
        layer_norm_vec(xp, tok_norm_.data(), tok_norm_b_.data(), xp, n_embd, cfg_.eps);
    }

    std::vector<float> q((size_t) n_tokens * q_dim), k((size_t) n_tokens * kv_dim), v((size_t) n_tokens * kv_dim);
    std::vector<float> attn_out((size_t) n_tokens * q_dim);
    std::vector<float> proj((size_t) n_tokens * n_embd);
    std::vector<float> ffn((size_t) n_tokens * cfg_.n_ff);
    std::vector<float> fout((size_t) n_tokens * n_embd);
    std::vector<float> scores(n_tokens);

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const BertLayerWeights& lw = layers_[l];

        // No pre-norm: Q/K/V computed directly on the layer's raw input
        // (verified against bert.cpp: `cur = inpL;` with no build_norm
        // before attention — an entirely post-norm topology, different
        // both from classic pre-norm and from olmo2/exaone4's
        // no_pre_norm+sandwich).
        for (size_t p = 0; p < n_tokens; ++p) {
            const float* xp = x.data() + p * n_embd;
            matvec_f32(lw.wq.data(), q_dim, n_embd, xp, q.data() + p * q_dim);
            matvec_f32(lw.wk.data(), kv_dim, n_embd, xp, k.data() + p * kv_dim);
            matvec_f32(lw.wv.data(), kv_dim, n_embd, xp, v.data() + p * kv_dim);
            for (uint32_t i = 0; i < q_dim; ++i) q[p * q_dim + i] += lw.bq[i];
            for (uint32_t i = 0; i < kv_dim; ++i) k[p * kv_dim + i] += lw.bk[i];
            for (uint32_t i = 0; i < kv_dim; ++i) v[p * kv_dim + i] += lw.bv[i];
        }

        // BIDIRECTIONAL attention: every position sees ALL others, no
        // causal mask.
        for (uint32_t h = 0; h < n_head; ++h) {
            const uint32_t hkv = h / heads_per_kv;
            for (size_t qi = 0; qi < n_tokens; ++qi) {
                const float* qh = q.data() + qi * q_dim + (size_t) h * head_dim;
                for (size_t ki = 0; ki < n_tokens; ++ki) {
                    const float* kh = k.data() + ki * kv_dim + (size_t) hkv * head_dim;
                    double acc = 0.0;
                    for (uint32_t d = 0; d < head_dim; ++d) acc += (double) qh[d] * kh[d];
                    scores[ki] = (float) acc * inv_d;
                }
                float m = -INFINITY;
                for (size_t ki = 0; ki < n_tokens; ++ki) m = std::max(m, scores[ki]);
                double sum = 0.0;
                for (size_t ki = 0; ki < n_tokens; ++ki) { scores[ki] = expf(scores[ki] - m); sum += scores[ki]; }
                const float inv_sum = (float) (1.0 / sum);
                float* oh = attn_out.data() + qi * q_dim + (size_t) h * head_dim;
                for (uint32_t d = 0; d < head_dim; ++d) oh[d] = 0.0f;
                for (size_t ki = 0; ki < n_tokens; ++ki) {
                    const float wgt = scores[ki] * inv_sum;
                    const float* vh = v.data() + ki * kv_dim + (size_t) hkv * head_dim;
                    for (uint32_t d = 0; d < head_dim; ++d) oh[d] += wgt * vh[d];
                }
            }
        }

        for (size_t p = 0; p < n_tokens; ++p) {
            matvec_f32(lw.wo.data(), n_embd, q_dim, attn_out.data() + p * q_dim, proj.data() + p * n_embd);
            float* pr = proj.data() + p * n_embd;
            const float* xp = x.data() + p * n_embd;
            for (uint32_t i = 0; i < n_embd; ++i) pr[i] += lw.bo[i] + xp[i]; // +bias, then residual onto the layer's RAW input
            layer_norm_vec(pr, lw.attn_out_norm.data(), lw.attn_out_norm_b.data(), pr, n_embd, cfg_.eps);
        }

        for (size_t p = 0; p < n_tokens; ++p) {
            const float* pr = proj.data() + p * n_embd;
            float* fp = ffn.data() + p * cfg_.n_ff;
            matvec_f32(lw.ffn_up.data(), cfg_.n_ff, n_embd, pr, fp);
            for (uint32_t i = 0; i < cfg_.n_ff; ++i) fp[i] = gelu_tanh(fp[i] + lw.ffn_up_b[i]);
            float* fo = fout.data() + p * n_embd;
            matvec_f32(lw.ffn_down.data(), n_embd, cfg_.n_ff, fp, fo);
            for (uint32_t i = 0; i < n_embd; ++i) fo[i] += lw.ffn_down_b[i] + pr[i]; // residual onto the post-attn-norm value (ffn_inp)
            layer_norm_vec(fo, lw.layer_out_norm.data(), lw.layer_out_norm_b.data(), fo, n_embd, cfg_.eps);
        }

        std::memcpy(x.data(), fout.data(), (size_t) n_tokens * n_embd * sizeof(float));
    }

    out_embd = std::move(x);
    return true;
}

}
