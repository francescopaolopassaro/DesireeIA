// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "modernbert_forward.h"
#include "decision_json.h"
#include "core/thread_pool.h"
#include <algorithm>
#include <cmath>
#ifdef DESIREEIA_CUDA_ENABLED
#include "encoder_cuda.h"
#endif

namespace desireeia {

namespace {

double num_or(const djson::Value& o, const char* key, double dflt) {
    const djson::Value* v = o.find(key);
    return (v && v->is_num()) ? v->num : dflt;
}

// cos/sin tables for rotary embedding: [n][half]
void rope_tables(size_t n, size_t head_dim, float theta, std::vector<float>& cs, std::vector<float>& sn) {
    const size_t half = head_dim / 2;
    cs.resize(n * half);
    sn.resize(n * half);
    for (size_t j = 0; j < half; ++j) {
        const double inv = std::pow((double) theta, -2.0 * (double) j / (double) head_dim);
        for (size_t p = 0; p < n; ++p) {
            const double a = (double) p * inv;
            cs[p * half + j] = (float) std::cos(a);
            sn[p * half + j] = (float) std::sin(a);
        }
    }
}

// Rotates the q and k parts of every token in place. qkv is [n, 3*d].
void apply_rope(float* qkv, size_t n, size_t d, size_t n_head, const std::vector<float>& cs,
                const std::vector<float>& sn) {
    const size_t hd = d / n_head;
    const size_t half = hd / 2;
    for (size_t t = 0; t < n; ++t) {
        for (size_t part = 0; part < 2; ++part) {          // 0 = q, 1 = k
            float* base = qkv + t * 3 * d + part * d;
            for (size_t h = 0; h < n_head; ++h) {
                float* x = base + h * hd;
                for (size_t j = 0; j < half; ++j) {
                    const float c = cs[t * half + j];
                    const float s = sn[t * half + j];
                    const float x1 = x[j];
                    const float x2 = x[j + half];
                    x[j] = x1 * c - x2 * s;
                    x[j + half] = x2 * c + x1 * s;
                }
            }
        }
    }
}

}

bool ModernBertEncoder::parse_config(const std::string& json_text, ModernBertConfig& cfg, std::string& err) {
    djson::Value root;
    djson::Parser parser(json_text);
    if (!parser.parse(root) || !root.is_obj()) { err = "encoder config is not valid JSON"; return false; }

    cfg.n_vocab = (uint32_t) num_or(root, "vocab_size", 0);
    cfg.n_layers = (uint32_t) num_or(root, "num_hidden_layers", 0);
    cfg.n_embd = (uint32_t) num_or(root, "hidden_size", 0);
    cfg.n_head = (uint32_t) num_or(root, "num_attention_heads", 0);
    cfg.n_ff = (uint32_t) num_or(root, "intermediate_size", 0);
    cfg.local_window = (uint32_t) num_or(root, "local_attention", 128);
    cfg.eps = (float) num_or(root, "norm_eps", num_or(root, "layer_norm_eps", 1e-5));
    if (cfg.n_vocab == 0 || cfg.n_layers == 0 || cfg.n_embd == 0 || cfg.n_head == 0 || cfg.n_ff == 0 ||
        cfg.n_embd % cfg.n_head != 0 || (cfg.n_embd / cfg.n_head) % 2 != 0) {
        err = "encoder config is missing or has inconsistent dimensions";
        return false;
    }

    cfg.theta_global = (float) num_or(root, "global_rope_theta", 160000.0);
    cfg.theta_local = (float) num_or(root, "local_rope_theta", 10000.0);
    if (const djson::Value* rp = root.find("rope_parameters")) {
        if (const djson::Value* fa = rp->find("full_attention"))
            cfg.theta_global = (float) num_or(*fa, "rope_theta", cfg.theta_global);
        if (const djson::Value* sa = rp->find("sliding_attention"))
            cfg.theta_local = (float) num_or(*sa, "rope_theta", cfg.theta_local);
    }

    cfg.is_global.assign(cfg.n_layers, 0);
    const djson::Value* lt = root.find("layer_types");
    if (lt && lt->is_arr() && lt->arr.size() == cfg.n_layers) {
        for (uint32_t i = 0; i < cfg.n_layers; ++i) cfg.is_global[i] = lt->arr[i].str == "full_attention";
    } else {
        const uint32_t every = std::max<uint32_t>(1, (uint32_t) num_or(root, "global_attn_every_n_layers", 3));
        for (uint32_t i = 0; i < cfg.n_layers; ++i) cfg.is_global[i] = (i % every) == 0;
    }
    return true;
}

bool ModernBertEncoder::load(const StFile& st, const std::string& prefix, const ModernBertConfig& cfg,
                             std::string& err) {
    cfg_ = cfg;
    st_ = &st;
    tok_name_ = prefix + "embeddings.tok_embeddings.weight";
    const uint64_t d = cfg.n_embd;

    auto need = [&](const std::string& n) {
        if (!st.has(n)) { err = "missing tensor: " + n; return false; }
        return true;
    };
    auto vec = [&](const std::string& n, std::vector<float>& v) {
        if (!need(n) || !st.read_f32(n, v)) { if (err.empty()) err = "cannot read tensor: " + n; return false; }
        return true;
    };
    auto mat = [&](const std::string& n, HalfMat& m, uint64_t rows, uint64_t cols) {
        if (!need(n)) return false;
        const auto sh = st.shape(n);
        if (sh.size() != 2 || sh[0] != rows || sh[1] != cols) { err = "unexpected shape for " + n; return false; }
        if (!st.read_half(n, m)) { err = "cannot read tensor: " + n; return false; }
        return true;
    };

    if (!need(tok_name_)) return false;
    const auto tsh = st.shape(tok_name_);
    if (tsh.size() != 2 || tsh[1] != d) { err = "unexpected shape for " + tok_name_; return false; }
    if (!vec(prefix + "embeddings.norm.weight", emb_norm_)) return false;
    if (!vec(prefix + "final_norm.weight", final_norm_)) return false;

    layers_.assign(cfg.n_layers, Layer());
    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        const std::string p = prefix + "layers." + std::to_string(l) + ".";
        Layer& L = layers_[l];
        if (l > 0 && !vec(p + "attn_norm.weight", L.attn_norm)) return false;
        if (!vec(p + "mlp_norm.weight", L.mlp_norm)) return false;
        if (!mat(p + "attn.Wqkv.weight", L.wqkv, 3 * d, d)) return false;
        if (!mat(p + "attn.Wo.weight", L.wo, d, d)) return false;
        if (!mat(p + "mlp.Wi.weight", L.wi, 2ull * cfg.n_ff, d)) return false;
        if (!mat(p + "mlp.Wo.weight", L.mlp_wo, d, cfg.n_ff)) return false;
    }
    return true;
}

ModernBertEncoder::~ModernBertEncoder() {
#ifdef DESIREEIA_CUDA_ENABLED
    delete gpu_;
#endif
}

bool ModernBertEncoder::enable_gpu(std::string& err) {
#ifdef DESIREEIA_CUDA_ENABLED
    if (gpu_) return true;
    if (!st_) { err = "encoder is not loaded"; return false; }
    if (!ModernBertCuda::device_available()) { err = "no CUDA device"; return false; }
    HalfMat tok;
    if (!st_->read_half(tok_name_, tok)) { err = "cannot read the token table"; return false; }
    CudaEncoderDesc desc;
    desc.n_vocab = cfg_.n_vocab;
    desc.d = cfg_.n_embd;
    desc.heads = cfg_.n_head;
    desc.ff = cfg_.n_ff;
    desc.local_window = cfg_.local_window;
    desc.eps = cfg_.eps;
    desc.theta_global = cfg_.theta_global;
    desc.theta_local = cfg_.theta_local;
    desc.tok_emb = tok.data.data();
    desc.emb_norm = emb_norm_.data();
    desc.final_norm = final_norm_.data();
    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& L = layers_[l];
        CudaEncoderLayer cl;
        cl.wqkv = L.wqkv.data.data();
        cl.wo = L.wo.data.data();
        cl.wi = L.wi.data.data();
        cl.mlp_wo = L.mlp_wo.data.data();
        cl.attn_norm = L.attn_norm.empty() ? nullptr : L.attn_norm.data();
        cl.mlp_norm = L.mlp_norm.data();
        cl.is_global = cfg_.is_global[l] != 0;
        desc.layers.push_back(cl);
    }
    ModernBertCuda* g = new ModernBertCuda;
    if (!g->init(desc, err)) { delete g; return false; }
    gpu_ = g;
    return true;
#else
    err = "this build has no CUDA backend";
    return false;
#endif
}

bool ModernBertEncoder::gpu_set_head(const CudaHeadDesc& head, std::string& err) {
#ifdef DESIREEIA_CUDA_ENABLED
    if (!gpu_) { err = "the encoder is not on the GPU"; return false; }
    gpu_head_ = gpu_->set_head(head, err);
    return gpu_head_;
#else
    (void) head;
    err = "this build has no CUDA backend";
    return false;
#endif
}

bool ModernBertEncoder::gpu_forward_head(const int32_t* ids, size_t n, int qtype, const size_t* rows, size_t n_rows,
                                         std::vector<float>& out) const {
#ifdef DESIREEIA_CUDA_ENABLED
    if (!gpu_ || !gpu_head_ || n == 0 || n > ModernBertCuda::max_tokens()) return false;
    for (size_t t = 0; t < n; ++t) {
        if (ids[t] < 0 || (uint32_t) ids[t] >= cfg_.n_vocab) return false;
    }
    return gpu_->forward_head(ids, n, qtype, rows, n_rows, out);
#else
    (void) ids; (void) n; (void) qtype; (void) rows; (void) n_rows; (void) out;
    return false;
#endif
}

bool ModernBertEncoder::forward(const int32_t* ids, size_t n, std::vector<float>& out) const {
    if (!st_ || n == 0) return false;
#ifdef DESIREEIA_CUDA_ENABLED
    if (gpu_ && n <= ModernBertCuda::max_tokens()) {
        bool valid = true;
        for (size_t t = 0; t < n; ++t) valid = valid && ids[t] >= 0 && (uint32_t) ids[t] < cfg_.n_vocab;
        if (valid && gpu_->forward(ids, n, out)) return true;
        // any GPU failure: run the same pass on the CPU
    }
#endif
    const size_t d = cfg_.n_embd;
    const size_t H = cfg_.n_head;
    const size_t hd = d / H;
    const size_t ff = cfg_.n_ff;

    // embeddings + norm
    std::vector<float> x(n * d), tmp(n * d);
    for (size_t t = 0; t < n; ++t) {
        const int32_t id = ids[t];
        if (id < 0 || (uint32_t) id >= cfg_.n_vocab) return false;
        if (!st_->read_row_f32(tok_name_, (size_t) id, tmp.data() + t * d, d)) return false;
    }
    layer_norm(tmp.data(), n, d, emb_norm_.data(), nullptr, cfg_.eps, x.data());

    std::vector<float> cs_g, sn_g, cs_l, sn_l;
    rope_tables(n, hd, cfg_.theta_global, cs_g, sn_g);
    rope_tables(n, hd, cfg_.theta_local, cs_l, sn_l);

    std::vector<float> normed(n * d), qkv(n * 3 * d), attn(n * d), proj(n * d), wi(n * 2 * ff), act(n * ff);
    const float scale = 1.0f / std::sqrt((float) hd);
    const long half_win = (long) (cfg_.local_window / 2);

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const Layer& L = layers_[l];
        const bool global = cfg_.is_global[l] != 0;

        // ---- attention ----
        if (L.attn_norm.empty()) normed = x;
        else layer_norm(x.data(), n, d, L.attn_norm.data(), nullptr, cfg_.eps, normed.data());
        linear_f16(normed.data(), n, d, L.wqkv, nullptr, qkv.data());
        apply_rope(qkv.data(), n, d, H, global ? cs_g : cs_l, global ? sn_g : sn_l);

        ThreadPool::global().parallel_for(n * H, [&](size_t a, size_t b) {
            std::vector<float> sc(n);
            for (size_t idx = a; idx < b; ++idx) {
                const size_t t = idx / H;
                const size_t h = idx % H;
                const float* q = qkv.data() + t * 3 * d + h * hd;
                size_t lo = 0, hi = n;                       // key range [lo, hi)
                if (!global) {
                    lo = (long) t - half_win < 0 ? 0 : (size_t) ((long) t - half_win);
                    hi = std::min(n, (size_t) ((long) t + half_win + 1));
                }
                float mx = -1e30f;
                for (size_t k = lo; k < hi; ++k) {
                    const float* kk = qkv.data() + k * 3 * d + d + h * hd;
                    const float s = dot_f32(q, kk, hd) * scale;
                    sc[k - lo] = s;
                    mx = std::max(mx, s);
                }
                double sum = 0.0;
                for (size_t k = lo; k < hi; ++k) {
                    sc[k - lo] = std::exp(sc[k - lo] - mx);
                    sum += sc[k - lo];
                }
                const float inv = (float) (1.0 / sum);
                float* o = attn.data() + t * d + h * hd;
                for (size_t j = 0; j < hd; ++j) o[j] = 0.0f;
                for (size_t k = lo; k < hi; ++k) {
                    const float w = sc[k - lo] * inv;
                    axpy_f32(o, w, qkv.data() + k * 3 * d + 2 * d + h * hd, hd);
                }
            }
        });

        linear_f16(attn.data(), n, d, L.wo, nullptr, proj.data());
        for (size_t i = 0; i < n * d; ++i) x[i] += proj[i];

        // ---- feed-forward (gated GELU) ----
        layer_norm(x.data(), n, d, L.mlp_norm.data(), nullptr, cfg_.eps, normed.data());
        linear_f16(normed.data(), n, d, L.wi, nullptr, wi.data());
        ThreadPool::global().parallel_for(n, [&](size_t t0, size_t t1) {
            for (size_t t = t0; t < t1; ++t) {
                const float* row = wi.data() + t * 2 * ff;
                float* a = act.data() + t * ff;
                for (size_t j = 0; j < ff; ++j) a[j] = gelu_exact(row[j]) * row[ff + j];
            }
        });
        linear_f16(act.data(), n, ff, L.mlp_wo, nullptr, proj.data());
        for (size_t i = 0; i < n * d; ++i) x[i] += proj[i];
    }

    out.resize(n * d);
    layer_norm(x.data(), n, d, final_norm_.data(), nullptr, cfg_.eps, out.data());
    return true;
}

}
