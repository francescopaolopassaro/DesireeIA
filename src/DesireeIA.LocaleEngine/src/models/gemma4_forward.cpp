// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "gemma4_forward.h"
#include "core/thread_pool.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cstdlib>

namespace desireeia {

namespace {

struct Prof {
    bool on = std::getenv("DESIREEIA_G4_PROF") != nullptr;
    double t_wo = 0, t_gate = 0, t_up = 0, t_ple = 0, t_attn_mv = 0, t_attn = 0, t_ffn = 0, t_pl = 0, t_logits = 0;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double lap() { auto n = std::chrono::steady_clock::now(); double d = std::chrono::duration<double, std::milli>(n - t0).count(); t0 = n; return d; }
};

// y = x / rms(x) * w (w may be null: a norm without weights).
void rms_norm(const float* x, const float* w, float* y, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float s = 1.0f / sqrtf((float) (ss / (double) n) + eps);
    for (size_t i = 0; i < n; ++i) y[i] = x[i] * s * (w ? w[i] : 1.0f);
}

float gelu(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// A per-layer value that the file stores either as one number or as one per layer.
bool layer_values(ModelReader& rd, const std::string& key, uint32_t n, std::vector<uint32_t>& out) {
    std::vector<uint32_t> arr;
    if (rd.meta_u32_array(key, arr) && !arr.empty()) {
        out.assign(n, arr.back());
        for (uint32_t i = 0; i < n && i < arr.size(); ++i) out[i] = arr[i];
        return true;
    }
    uint32_t v = 0;
    if (rd.meta_u32(key, v)) { out.assign(n, v); return true; }
    return false;
}

}

bool Gemma4Forward::open(ModelReader& rd, const ModelMeta& meta, int32_t backend) {
    mv_set_backend(backend);
    const std::string kp = meta.arch + ".";
    Gemma4Config& c = cfg_;
    c.n_vocab = meta.n_vocab;
    c.n_layers = meta.n_layers;
    c.n_ctx_train = meta.n_ctx;
    uint32_t v = 0;
    float f = 0.0f;
    if (!rd.meta_u32(kp + "embedding_length", c.n_embd)) { last_fail_ = "embedding_length"; return false; }
    std::vector<uint32_t> heads;
    if (!layer_values(rd, kp + "attention.head_count", c.n_layers, heads)) { last_fail_ = "head_count"; return false; }
    c.n_head = heads[0];
    if (!layer_values(rd, kp + "attention.head_count_kv", c.n_layers, c.n_head_kv)) c.n_head_kv.assign(c.n_layers, c.n_head);
    if (!layer_values(rd, kp + "feed_forward_length", c.n_layers, c.n_ff)) { last_fail_ = "feed_forward_length"; return false; }
    if (rd.meta_u32(kp + "attention.key_length", v)) c.head_dim_global = v; else c.head_dim_global = c.n_embd / c.n_head;
    c.head_dim_swa = rd.meta_u32(kp + "attention.key_length_swa", v) ? v : c.head_dim_global;
    rd.meta_u32(kp + "attention.sliding_window", c.n_swa);
    std::vector<uint32_t> pat;
    c.is_swa.assign(c.n_layers, 0);
    if (rd.meta_u32_array(kp + "attention.sliding_window_pattern", pat))
        for (uint32_t i = 0; i < c.n_layers && i < pat.size(); ++i) c.is_swa[i] = pat[i] ? 1 : 0;
    uint32_t shared = 0;
    rd.meta_u32(kp + "attention.shared_kv_layers", shared);
    c.n_kv_from_start = c.n_layers > shared ? c.n_layers - shared : c.n_layers;
    c.kv_source.assign(c.n_layers, 0);
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        if (l < c.n_kv_from_start) c.kv_source[l] = (int32_t) l;
        else {
            const int32_t back = c.is_swa[l] ? 2 : 1;
            c.kv_source[l] = (int32_t) c.n_kv_from_start - back;
            if (c.kv_source[l] < 0) { last_fail_ = "shared kv layout"; return false; }
        }
    }
    rd.meta_u32(kp + "embedding_length_per_layer_input", c.n_embd_per_layer);
    if (rd.meta_f32(kp + "rope.freq_base", f)) c.rope_base_global = f;
    if (rd.meta_f32(kp + "rope.freq_base_swa", f)) c.rope_base_swa = f;
    c.n_rot_global = rd.meta_u32(kp + "rope.dimension_count", v) ? v : c.head_dim_global;
    c.n_rot_swa = rd.meta_u32(kp + "rope.dimension_count_swa", v) ? v : c.head_dim_swa;
    if (rd.meta_f32(kp + "final_logit_softcapping", f)) c.softcap = f;
    if (rd.meta_f32(kp + "attention.layer_norm_rms_epsilon", f)) c.eps = f;
    if (c.n_vocab == 0 || c.n_layers == 0 || c.n_embd == 0 || c.n_head == 0 || c.head_dim_global == 0) { last_fail_ = "hparams"; return false; }

    // The routed-experts variant (a router next to the dense FFN) is not implemented in this file yet.
    {
        std::vector<uint8_t> raw; int qt = 0; uint64_t a = 0, b = 0;
        if (rd.read_tensor_raw("blk.0.ffn_gate_inp.weight", raw, qt, a, b)) { last_fail_ = "gemma4 routed-experts layers are not supported yet"; return false; }
    }

    if (!mv_load(rd, "token_embd.weight", c.n_vocab, c.n_embd, tok_embd_, true)) { last_fail_ = "token_embd.weight"; return false; }
    tied_ = !mv_load(rd, "output.weight", c.n_vocab, c.n_embd, output_, true);
    if (!rd.read_tensor("output_norm.weight", out_norm_) || out_norm_.size() != c.n_embd) { last_fail_ = "output_norm.weight"; return false; }
    const uint32_t n_ple_all = c.n_embd_per_layer * c.n_layers;
    if (c.n_embd_per_layer > 0) {
        if (!mv_load(rd, "per_layer_token_embd.weight", c.n_vocab, n_ple_all, ple_tok_embd_, false)) { last_fail_ = "per_layer_token_embd.weight"; return false; }
        if (!mv_load(rd, "per_layer_model_proj.weight", n_ple_all, c.n_embd, ple_model_proj_, true)) { last_fail_ = "per_layer_model_proj.weight"; return false; }
        if (!rd.read_tensor("per_layer_proj_norm.weight", ple_proj_norm_) || ple_proj_norm_.size() != c.n_embd_per_layer) { last_fail_ = "per_layer_proj_norm.weight"; return false; }
    }
    rope_freqs_.clear();
    rd.read_tensor("rope_freqs.weight", rope_freqs_);

    layers_.assign(c.n_layers, Gemma4Layer{});
    k_cache_.assign(c.n_layers, {});
    v_cache_.assign(c.n_layers, {});
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        Gemma4Layer& w = layers_[l];
        char buf[32];
        std::snprintf(buf, sizeof(buf), "blk.%u.", l);
        const std::string p = buf;
        const uint32_t hd = c.head_dim(l), nh = c.n_head, nk = c.n_head_kv[l], ff = c.n_ff[l];
        auto fail = [&](const char* what) { last_fail_ = p + what; return false; };
        if (!rd.read_tensor(p + "attn_norm.weight", w.attn_norm) || w.attn_norm.size() != c.n_embd) return fail("attn_norm");
        if (!mv_load(rd, p + "attn_q.weight", nh * hd, c.n_embd, w.wq, true)) return fail("attn_q");
        if (!rd.read_tensor(p + "attn_q_norm.weight", w.q_norm) || w.q_norm.size() != hd) return fail("attn_q_norm");
        if (c.owns_kv(l)) {
            if (!mv_load(rd, p + "attn_k.weight", nk * hd, c.n_embd, w.wk, true)) return fail("attn_k");
            w.has_v = mv_load(rd, p + "attn_v.weight", nk * hd, c.n_embd, w.wv, true);
            if (!rd.read_tensor(p + "attn_k_norm.weight", w.k_norm) || w.k_norm.size() != hd) return fail("attn_k_norm");
        }
        if (!mv_load(rd, p + "attn_output.weight", c.n_embd, nh * hd, w.wo, true)) return fail("attn_output");
        if (!rd.read_tensor(p + "post_attention_norm.weight", w.post_attn_norm) || w.post_attn_norm.size() != c.n_embd) return fail("post_attention_norm");
        if (!rd.read_tensor(p + "ffn_norm.weight", w.ffn_norm) || w.ffn_norm.size() != c.n_embd) return fail("ffn_norm");
        if (!mv_load(rd, p + "ffn_gate.weight", ff, c.n_embd, w.ffn_gate, true)) return fail("ffn_gate");
        if (!mv_load(rd, p + "ffn_up.weight", ff, c.n_embd, w.ffn_up, true)) return fail("ffn_up");
        if (!mv_load(rd, p + "ffn_down.weight", c.n_embd, ff, w.ffn_down, true)) return fail("ffn_down");
        if (!rd.read_tensor(p + "post_ffw_norm.weight", w.post_ffn_norm) || w.post_ffn_norm.size() != c.n_embd) return fail("post_ffw_norm");
        std::vector<float> sc;
        if (rd.read_tensor(p + "layer_output_scale.weight", sc) && sc.size() == 1) w.out_scale = sc[0];
        if (c.n_embd_per_layer > 0) {
            if (!mv_load(rd, p + "inp_gate.weight", c.n_embd_per_layer, c.n_embd, w.ple_gate, true)) return fail("inp_gate");
            if (!mv_load(rd, p + "proj.weight", c.n_embd, c.n_embd_per_layer, w.ple_proj, true)) return fail("proj");
            if (!rd.read_tensor(p + "post_norm.weight", w.ple_post_norm) || w.ple_post_norm.size() != c.n_embd) return fail("post_norm");
        }
    }
    n_seen_ = 0;
    build_device(backend);
    return true;
}

Gemma4Forward::~Gemma4Forward() {
#ifdef DESIREEIA_CUDA_ENABLED
    if (dev_) g4_dev_free();
#endif
}

// The whole token on the GPU when every matrix is resident in a K-quant format; otherwise the host path below runs unchanged.
bool Gemma4Forward::build_device(int32_t backend) {
#ifdef DESIREEIA_CUDA_ENABLED
    dev_ = false;
    if (backend != DESIREEIA_BACKEND_CUDA || std::getenv("DESIREEIA_G4_HOST")) return false;
    const Gemma4Config& c = cfg_;
    if (c.n_head > 64) return false;
    uint32_t cap_global = c.n_ctx_train ? c.n_ctx_train : 8192;
    uint32_t want = 16384;
    if (const char* e = std::getenv("DESIREEIA_G4_CTX")) want = (uint32_t) std::max(512, std::atoi(e));
    cap_global = std::min(cap_global, want);
    uint32_t max_q = 0, max_ff = 0, max_kv = 0;
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        max_q = std::max(max_q, c.n_head * c.head_dim(l));
        max_kv = std::max(max_kv, c.n_head_kv[l] * c.head_dim(l));
        max_ff = std::max(max_ff, c.n_ff[l]);
    }
    uint32_t max_cols = std::max(std::max(c.n_embd, max_ff), std::max(max_q, std::max(c.n_embd_per_layer, 32u)));
    const MatVec& head_m = tied_ ? tok_embd_ : output_;
    const void* dw = nullptr;
    G4DevMat head;
    head.format = mv_cuda_format(head_m, &dw);
    head.d_w = dw; head.rows = c.n_vocab; head.cols = c.n_embd;
    if (!head.format) return false;
    if (!g4_dev_init(c.n_layers, c.n_embd, c.n_embd_per_layer, c.n_vocab, c.eps, c.softcap, cap_global, max_cols, max_q, max_ff, max_kv,
                     head, out_norm_.data(), rope_freqs_.empty() ? nullptr : rope_freqs_.data(), rope_freqs_.size())) return false;
    auto mat = [&](const MatVec& m, uint32_t rows, uint32_t cols, G4DevMat& out) {
        const void* p = nullptr;
        out.format = mv_cuda_format(m, &p);
        out.d_w = p; out.rows = rows; out.cols = cols;
        return out.format != 0;
    };
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        const Gemma4Layer& w = layers_[l];
        G4DevLayer L;
        const uint32_t hd = c.head_dim(l), nh = c.n_head, nk = c.n_head_kv[l], ff = c.n_ff[l], E = c.n_embd, P = c.n_embd_per_layer;
        L.hd = hd; L.nh = nh; L.nk = nk; L.ff = ff;
        L.swa = c.is_swa[l] != 0;
        L.owns = c.owns_kv(l);
        L.has_v = w.has_v;
        L.src = (uint32_t) c.kv_source[l];
        L.n_swa = c.n_swa;
        L.base = L.swa ? c.rope_base_swa : c.rope_base_global;
        L.use_factors = !L.swa && !rope_freqs_.empty();
        const uint32_t n_rot = L.swa ? c.n_rot_swa : c.n_rot_global;
        L.rot = L.use_factors ? (uint32_t) std::min<size_t>(rope_freqs_.size() * 2, hd) : std::min(n_rot, hd);
        L.cap = (L.swa && c.n_swa > 0) ? std::min(c.n_swa, cap_global) : cap_global;
        L.out_scale = w.out_scale;
        bool ok = mat(w.wq, nh * hd, E, L.wq) && mat(w.wo, E, nh * hd, L.wo) && mat(w.ffn_gate, ff, E, L.gate) &&
                  mat(w.ffn_up, ff, E, L.up) && mat(w.ffn_down, E, ff, L.down);
        if (L.owns) ok = ok && mat(w.wk, nk * hd, E, L.wk) && (!w.has_v || mat(w.wv, nk * hd, E, L.wv));
        if (P > 0) ok = ok && mat(w.ple_gate, P, E, L.ple_gate) && mat(w.ple_proj, E, P, L.ple_proj);
        if (!ok || !g4_dev_set_layer(l, L, w.attn_norm.data(), w.q_norm.data(), L.owns ? w.k_norm.data() : nullptr, w.post_attn_norm.data(),
                                     w.ffn_norm.data(), w.post_ffn_norm.data(), P > 0 ? w.ple_post_norm.data() : nullptr)) {
            g4_dev_free();
            return false;
        }
    }
    dev_ = true;
    dev_cap_ = cap_global;
    return true;
#else
    (void) backend;
    return false;
#endif
}

void Gemma4Forward::reset_cache() {
    for (auto& k : k_cache_) std::vector<float>().swap(k);
    for (auto& v : v_cache_) std::vector<float>().swap(v);
    n_seen_ = 0;
}

uint64_t Gemma4Forward::kv_bytes() const {
    uint64_t n = 0;
    for (size_t l = 0; l < k_cache_.size(); ++l) n += (k_cache_[l].size() + v_cache_[l].size()) * sizeof(float);
    return n;
}

// Rotation of the first n_rot values (split-half pairing). `factors` divides the frequencies (the global layers' proportional RoPE).
void Gemma4Forward::rope(float* x, uint32_t hd, uint32_t n_rot, float base, const float* factors, uint32_t pos) const {
    (void) hd;
    const uint32_t half = n_rot / 2;
    for (uint32_t i = 0; i < half; ++i) {
        float freq = powf(base, -2.0f * (float) i / (float) n_rot);
        if (factors) freq /= factors[i];
        const float th = (float) pos * freq, cs = cosf(th), sn = sinf(th);
        const float a = x[i], b = x[i + half];
        x[i] = a * cs - b * sn;
        x[i + half] = a * sn + b * cs;
    }
}

void Gemma4Forward::step_token(int32_t token, bool want_logits, std::vector<float>& logits) {
    const Gemma4Config& c = cfg_;
    const uint32_t E = c.n_embd, P = c.n_embd_per_layer;
    const uint32_t pos = (uint32_t) n_seen_;
    Prof pf; pf.lap();
    std::vector<float> x(E), h(E), tmp(E);
    mv_row(tok_embd_, E, (uint32_t) token, x.data());
    const float es = sqrtf((float) E);
    for (auto& v : x) v *= es;

    std::vector<float> ple; // [n_layers][P]
    if (P > 0) {
        const uint32_t N = P * c.n_layers;
        ple.assign(N, 0.0f);
        std::vector<float> tok(N), proj(N);
        mv_row(ple_tok_embd_, N, (uint32_t) token, tok.data());
        const float ts = sqrtf((float) P) / sqrtf(2.0f);
        mv_apply(ple_model_proj_, N, E, x.data(), proj.data());
        std::vector<float> nw(P);
        for (uint32_t i = 0; i < P; ++i) nw[i] = ple_proj_norm_[i] * (1.0f / sqrtf(2.0f));
        const float eps2 = c.eps * (float) E; // eps / (1/sqrt(E))^2
        for (uint32_t l = 0; l < c.n_layers; ++l) {
            rms_norm(proj.data() + (size_t) l * P, nw.data(), ple.data() + (size_t) l * P, P, eps2);
            for (uint32_t i = 0; i < P; ++i) ple[(size_t) l * P + i] += tok[(size_t) l * P + i] * ts;
        }
    }

#ifdef DESIREEIA_CUDA_ENABLED
    if (dev_) {
        if (want_logits) logits.assign(c.n_vocab, 0.0f);
        if (g4_dev_token(x.data(), ple.data(), pos, want_logits ? logits.data() : nullptr) != DESIREEIA_OK) cancelled_ = true;
        ++n_seen_;
        return;
    }
#endif
    if (pf.on) pf.t_ple = pf.lap();
    std::vector<float> q, k, v, att, ao(E), g, u;
    for (uint32_t l = 0; l < c.n_layers; ++l) {
        const Gemma4Layer& w = layers_[l];
        const uint32_t hd = c.head_dim(l), nh = c.n_head, nk = c.n_head_kv[l], ff = c.n_ff[l];
        const bool swa = c.is_swa[l] != 0;
        const uint32_t n_rot = swa ? c.n_rot_swa : c.n_rot_global;
        const float base = swa ? c.rope_base_swa : c.rope_base_global;
        const float* factors = (!swa && !rope_freqs_.empty()) ? rope_freqs_.data() : nullptr;
        const uint32_t rot = factors ? (uint32_t) std::min<size_t>(rope_freqs_.size() * 2, hd) : std::min(n_rot, hd);

        rms_norm(x.data(), w.attn_norm.data(), h.data(), E, c.eps);
        q.assign((size_t) nh * hd, 0.0f);
        const bool own = c.owns_kv(l);
        if (own) {
            k.assign((size_t) nk * hd, 0.0f);
            v.assign((size_t) nk * hd, 0.0f);
            if (w.has_v) {
                const MatVec* ms[3] = {&w.wq, &w.wk, &w.wv};
                const size_t rs[3] = {(size_t) nh * hd, (size_t) nk * hd, (size_t) nk * hd};
                float* ys[3] = {q.data(), k.data(), v.data()};
                mv_apply_group(ms, rs, ys, 3, E, h.data());
            } else {
                const MatVec* ms[2] = {&w.wq, &w.wk};
                const size_t rs[2] = {(size_t) nh * hd, (size_t) nk * hd};
                float* ys[2] = {q.data(), k.data()};
                mv_apply_group(ms, rs, ys, 2, E, h.data());
                v = k;
            }
        } else {
            mv_apply(w.wq, (size_t) nh * hd, E, h.data(), q.data());
        }
        for (uint32_t i = 0; i < nh; ++i) {
            float* qh = q.data() + (size_t) i * hd;
            rms_norm(qh, w.q_norm.data(), qh, hd, c.eps);
            rope(qh, hd, rot, base, factors, pos);
        }
        if (pf.on) pf.t_attn_mv += pf.lap();
        const uint32_t src = (uint32_t) c.kv_source[l];
        if (c.owns_kv(l)) {
            for (uint32_t i = 0; i < nk; ++i) {
                float* kh = k.data() + (size_t) i * hd;
                rms_norm(kh, w.k_norm.data(), kh, hd, c.eps);
                rope(kh, hd, rot, base, factors, pos);
                float* vh = v.data() + (size_t) i * hd;
                rms_norm(vh, nullptr, vh, hd, c.eps);
            }
            k_cache_[l].resize((size_t) (pos + 1) * nk * hd);
            v_cache_[l].resize((size_t) (pos + 1) * nk * hd);
            std::memcpy(k_cache_[l].data() + (size_t) pos * nk * hd, k.data(), (size_t) nk * hd * sizeof(float));
            std::memcpy(v_cache_[l].data() + (size_t) pos * nk * hd, v.data(), (size_t) nk * hd * sizeof(float));
        }
        const uint32_t nk_src = c.n_head_kv[src];
        const std::vector<float>& K = k_cache_[src];
        const std::vector<float>& V = v_cache_[src];
        const uint32_t lo = (swa && c.n_swa > 0 && pos + 1 > c.n_swa) ? pos + 1 - c.n_swa : 0;
        const uint32_t span = pos + 1 - lo, group = std::max(1u, nh / nk_src);
        att.assign((size_t) nh * hd, 0.0f);
        auto attn_fn = [&](size_t h0, size_t h1) {
            std::vector<float> sc(span);
            for (size_t hh = h0; hh < h1; ++hh) {
                const float* qh = q.data() + hh * hd;
                const size_t kvh = hh / group;
                float mx = -1e30f;
                for (uint32_t j = 0; j < span; ++j) {
                    const float* kj = K.data() + ((size_t) (lo + j) * nk_src + kvh) * hd;
                    double d = 0.0;
                    for (uint32_t t = 0; t < hd; ++t) d += (double) qh[t] * kj[t];
                    sc[j] = (float) d; // no 1/sqrt(d): the norms on Q and K carry the scale
                    mx = std::max(mx, sc[j]);
                }
                double den = 0.0;
                for (uint32_t j = 0; j < span; ++j) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
                float* oh = att.data() + hh * hd;
                for (uint32_t j = 0; j < span; ++j) {
                    const float wgt = (float) (sc[j] / den);
                    const float* vj = V.data() + ((size_t) (lo + j) * nk_src + kvh) * hd;
                    for (uint32_t t = 0; t < hd; ++t) oh[t] += wgt * vj[t];
                }
            }
        };
        if (span < 192) attn_fn(0, nh); else ThreadPool::global().parallel_for(nh, attn_fn);
        if (pf.on) pf.t_attn += pf.lap();
        mv_apply(w.wo, E, (size_t) nh * hd, att.data(), ao.data());
        rms_norm(ao.data(), w.post_attn_norm.data(), ao.data(), E, c.eps);
        for (uint32_t i = 0; i < E; ++i) x[i] += ao[i];

        rms_norm(x.data(), w.ffn_norm.data(), h.data(), E, c.eps);
        g.assign(ff, 0.0f); u.assign(ff, 0.0f);
        if (pf.on) pf.t_wo += pf.lap();
        {
            const MatVec* ms[2] = {&w.ffn_gate, &w.ffn_up};
            const size_t rs[2] = {ff, ff};
            float* ys[2] = {g.data(), u.data()};
            mv_apply_group(ms, rs, ys, 2, E, h.data());
        }
        if (pf.on) pf.t_gate += pf.lap();
        if (pf.on) pf.t_up += pf.lap();
        for (uint32_t i = 0; i < ff; ++i) g[i] = gelu(g[i]) * u[i];
        mv_apply(w.ffn_down, E, ff, g.data(), tmp.data());
        rms_norm(tmp.data(), w.post_ffn_norm.data(), tmp.data(), E, c.eps);
        for (uint32_t i = 0; i < E; ++i) x[i] += tmp[i];

        if (pf.on) pf.t_ffn += pf.lap();
        if (P > 0) {
            std::vector<float> gp(P), pe(E);
            mv_apply(w.ple_gate, P, E, x.data(), gp.data());
            for (uint32_t i = 0; i < P; ++i) gp[i] = gelu(gp[i]) * ple[(size_t) l * P + i];
            mv_apply(w.ple_proj, E, P, gp.data(), pe.data());
            rms_norm(pe.data(), w.ple_post_norm.data(), pe.data(), E, c.eps);
            for (uint32_t i = 0; i < E; ++i) x[i] += pe[i];
        }
        if (pf.on) pf.t_pl += pf.lap();
        if (w.out_scale != 1.0f) for (auto& e : x) e *= w.out_scale;
    }
    ++n_seen_;
    if (!want_logits) return;
    rms_norm(x.data(), out_norm_.data(), h.data(), E, c.eps);
    logits.assign(c.n_vocab, 0.0f);
    mv_apply(tied_ ? tok_embd_ : output_, c.n_vocab, E, h.data(), logits.data());
    if (c.softcap > 0.0f) for (auto& l : logits) l = c.softcap * tanhf(l / c.softcap);
    if (pf.on) {
        pf.t_logits = pf.lap();
        std::fprintf(stderr, "g4 wo %.1f gate %.1f up %.1f | ", pf.t_wo, pf.t_gate, pf.t_up);
        std::fprintf(stderr, "g4 ms: ple %.1f qkv+rope %.1f attn %.1f wo+ffn %.1f pl %.1f logits %.1f\n", pf.t_ple, pf.t_attn_mv, pf.t_attn, pf.t_ffn, pf.t_pl, pf.t_logits);
    }
}

bool Gemma4Forward::step(ModelReader&, const int32_t* tokens, size_t n_tokens,
                         std::vector<float>& last_logits, std::vector<float>* all_logits) {
    cancelled_ = false;
    if (all_logits) all_logits->clear();
    if (dev_ && n_seen_ + n_tokens > dev_cap_) { last_fail_ = "gemma4: context beyond the device cache (DESIREEIA_G4_CTX)"; return false; }
    for (size_t i = 0; i < n_tokens; ++i) {
        if (hooks_ && hooks_->cancel && hooks_->cancel->load()) { cancelled_ = true; return false; }
        const bool last = i + 1 == n_tokens;
        std::vector<float> lg;
        step_token(tokens[i], last || all_logits != nullptr, lg);
        if (all_logits) all_logits->insert(all_logits->end(), lg.begin(), lg.end());
        if (cancelled_) return false;
        if (last) last_logits = std::move(lg);
        if (hooks_ && hooks_->progress) hooks_->progress(i + 1, n_tokens, hooks_->user);
    }
    return true;
}

}
