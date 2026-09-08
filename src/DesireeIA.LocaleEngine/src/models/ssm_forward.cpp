#include "ssm_forward.h"
#include "core/thread_pool.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace desireeia {

namespace {

// y[j] = sum_i w[j*c+i]*x[i], accumulatore double (stessa scelta di
// dense_forward.cpp: precisione, non velocita', qui non e' il collo di
// bottiglia dominante rispetto alla scansione SSM stessa). Parallelizzato
// sulle righe per ssm_in/ssm_out (le uniche matrici davvero grandi qui).
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

float silu(float x) { return x / (1.0f + expf(-x)); }
float softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }

void rms_norm_vec(const float* x, const float* w, float* y, size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float scale = 1.0f / sqrtf((float) (ss / (double) n) + eps);
    for (size_t i = 0; i < n; ++i) y[i] = x[i] * scale * w[i];
}

}

bool SsmForward::load_embd(ModelReader& rd) {
    // Semplificazione voluta (prima passata di correttezza, nessun GGUF
    // Mamba2 reale disponibile per testare): embedding e lm_head tenuti
    // interamente in float, senza il dequant-per-riga a domanda che
    // DenseForward usa per i modelli grandi. I modelli Mamba2 pubblicati
    // sono tipicamente piccoli-medi (130M-7B): gap noto, documentato,
    // ottimizzabile in seguito se servisse su un modello con vocabolario
    // enorme.
    if (!rd.read_tensor("token_embd.weight", tok_embd_) ||
        tok_embd_.size() != (size_t) cfg_.n_vocab * cfg_.n_embd) return false;
    if (!rd.read_tensor("output.weight", output_) ||
        output_.size() != (size_t) cfg_.n_vocab * cfg_.n_embd) {
        output_ = tok_embd_; // tied embeddings
    }
    return true;
}

bool SsmForward::load_norms(ModelReader& rd) {
    if (!rd.read_tensor("output_norm.weight", out_norm_) || out_norm_.size() != cfg_.n_embd) return false;
    return true;
}

bool SsmForward::load_layer_data(ModelReader& rd, uint32_t il, SsmLayerWeights& w) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "blk.%u.", il);
    const std::string p = buf;

    if (!rd.read_tensor(p + "attn_norm.weight", w.attn_norm) || w.attn_norm.size() != cfg_.n_embd) return false;

    const uint32_t d_in_proj = cfg_.d_in_proj();
    if (!rd.read_tensor(p + "ssm_in.weight", w.ssm_in) || w.ssm_in.size() != (size_t) d_in_proj * cfg_.n_embd) return false;

    const uint32_t conv_dim = cfg_.conv_dim();
    if (!rd.read_tensor(p + "ssm_conv1d.weight", w.ssm_conv1d) || w.ssm_conv1d.size() != (size_t) conv_dim * cfg_.d_conv) return false;
    if (!rd.read_tensor(p + "ssm_conv1d.bias", w.ssm_conv1d_b) || w.ssm_conv1d_b.size() != conv_dim) return false;

    if (!rd.read_tensor(p + "ssm_dt.bias", w.ssm_dt_b) || w.ssm_dt_b.size() != cfg_.n_head) return false;
    if (!rd.read_tensor(p + "ssm_a", w.ssm_a) || w.ssm_a.size() != cfg_.n_head) return false;
    if (!rd.read_tensor(p + "ssm_d", w.ssm_d) || w.ssm_d.size() != cfg_.n_head) return false;

    w.ssm_norm.clear();
    rd.read_tensor(p + "ssm_norm.weight", w.ssm_norm);
    if (w.ssm_norm.size() != cfg_.d_inner) w.ssm_norm.clear(); // gruppo singolo/assente: gap noto, nessuna norm applicata

    if (!rd.read_tensor(p + "ssm_out.weight", w.ssm_out) || w.ssm_out.size() != (size_t) cfg_.n_embd * cfg_.d_inner) return false;

    return true;
}

const SsmLayerWeights* SsmForward::get_layer(ModelReader& rd, uint32_t il) {
    if (cache_enabled_) {
        SsmLayerWeights& w = layer_cache_[il];
        if (w.ssm_in.empty()) {
            if (!load_layer_data(rd, il, w)) return nullptr;
        }
        return &w;
    }
    if (!load_layer_data(rd, il, scratch_)) return nullptr;
    return &scratch_;
}

bool SsmForward::open(ModelReader& rd, const ModelMeta& meta, uint64_t ram_budget_mb) {
    const std::string arch_tag = meta.arch;
    const std::string kp = arch_tag + ".";

    cfg_.n_vocab = meta.n_vocab;
    cfg_.n_layers = meta.n_layers;
    uint32_t v32 = 0;
    float f32 = 0.0f;
    if (rd.meta_u32(kp + "embedding_length", v32)) cfg_.n_embd = v32;
    if (rd.meta_f32(kp + "attention.layer_norm_rms_epsilon", f32)) cfg_.rms_eps = f32;

    if (rd.meta_u32(kp + "ssm.conv_kernel", v32)) cfg_.d_conv = v32;
    if (rd.meta_u32(kp + "ssm.inner_size", v32)) cfg_.d_inner = v32;
    if (rd.meta_u32(kp + "ssm.state_size", v32)) cfg_.d_state = v32;
    // mamba2 reuses the "time_step_rank" metadata key as the head count.
    if (rd.meta_u32(kp + "ssm.time_step_rank", v32)) cfg_.n_head = v32;
    cfg_.n_group = 1;
    rd.meta_u32(kp + "ssm.group_count", cfg_.n_group);
    if (cfg_.n_group == 0) cfg_.n_group = 1;

    if (cfg_.n_vocab == 0 || cfg_.n_layers == 0 || cfg_.n_embd == 0 ||
        cfg_.d_conv < 2 || cfg_.d_inner == 0 || cfg_.d_state == 0 || cfg_.n_head == 0 ||
        cfg_.d_inner % cfg_.n_head != 0 || cfg_.d_inner % cfg_.n_group != 0) {
        return false;
    }

    if (!load_embd(rd)) return false;
    if (!load_norms(rd)) return false;

    // Stima RAM: pesi SSM per layer + i due grandi (ssm_in/ssm_out) contro
    // il budget pianificato, stessa soglia conservativa (60%) usata da
    // DenseForward per lasciare margine a embeddings/overhead — qui lo
    // stato ricorrente e' trascurabile (dimensione fissa, non cresce col
    // contesto) quindi non serve considerarlo nel budget.
    const size_t bytes_per_layer =
        (size_t) cfg_.d_in_proj() * cfg_.n_embd * sizeof(float) +
        (size_t) cfg_.n_embd * cfg_.d_inner * sizeof(float) +
        (size_t) cfg_.conv_dim() * cfg_.d_conv * sizeof(float);
    const uint64_t total_bytes = (uint64_t) bytes_per_layer * cfg_.n_layers;
    const uint64_t budget_bytes = ram_budget_mb * 1024ull * 1024ull;
    cache_enabled_ = ram_budget_mb > 0 && total_bytes <= (budget_bytes * 60 / 100);
    if (cache_enabled_) {
        layer_cache_.assign(cfg_.n_layers, SsmLayerWeights{});
    }

    conv_state_.assign((size_t) cfg_.n_layers * cfg_.conv_dim() * (cfg_.d_conv - 1), 0.0f);
    ssm_state_.assign((size_t) cfg_.n_layers * cfg_.n_head * cfg_.head_dim() * cfg_.d_state, 0.0f);
    n_tokens_seen_ = 0;

    return true;
}

void SsmForward::reset_cache() {
    std::fill(conv_state_.begin(), conv_state_.end(), 0.0f);
    std::fill(ssm_state_.begin(), ssm_state_.end(), 0.0f);
    n_tokens_seen_ = 0;
}

uint64_t SsmForward::kv_bytes() const {
    // Stato a dimensione FISSA (il punto degli SSM): non cresce col
    // contesto, a differenza della KV-cache del motore denso.
    return (uint64_t) (conv_state_.size() + ssm_state_.size()) * sizeof(float);
}

void SsmForward::step_token(ModelReader& rd, int32_t token, bool want_logits, std::vector<float>& logits_out) {
    const uint32_t n_embd = cfg_.n_embd;
    const uint32_t d_inner = cfg_.d_inner;
    const uint32_t d_state = cfg_.d_state;
    const uint32_t d_conv = cfg_.d_conv;
    const uint32_t n_head = cfg_.n_head;
    const uint32_t head_dim = cfg_.head_dim();
    const uint32_t n_group = cfg_.n_group;
    const uint32_t conv_dim = cfg_.conv_dim();
    const uint32_t d_in_proj = cfg_.d_in_proj();

    std::vector<float> x(n_embd);
    std::memcpy(x.data(), tok_embd_.data() + (size_t) token * n_embd, n_embd * sizeof(float));

    std::vector<float> xn(n_embd);
    std::vector<float> zxBCdt(d_in_proj);
    std::vector<float> conv_out(conv_dim);
    std::vector<float> y(d_inner);
    std::vector<float> y_normed(d_inner);
    std::vector<float> out(n_embd);

    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const SsmLayerWeights* lw = get_layer(rd, l);
        if (!lw) return;

        rms_norm_vec(x.data(), lw->attn_norm.data(), xn.data(), n_embd, cfg_.rms_eps);

        matvec_f32(lw->ssm_in.data(), d_in_proj, n_embd, xn.data(), zxBCdt.data());

        // Split: z [d_inner] | xBC [conv_dim] | dt [n_head], in that
        // order (dt sits at offset 2*d_inner+2*n_group*d_state).
        const float* z = zxBCdt.data();
        const float* xBC_raw = zxBCdt.data() + d_inner;
        const float* dt_raw = zxBCdt.data() + d_inner + conv_dim;

        // Causal depthwise convolution (kernel d_conv, one channel per
        // row of ssm_conv1d): window = (d_conv-1) historical values from
        // conv_state plus the current value. The oldest tap is index 0,
        // the most recent (the current token) is index d_conv-1.
        float* cstate = conv_state_.data() + (size_t) l * conv_dim * (d_conv - 1);
        for (uint32_t ch = 0; ch < conv_dim; ++ch) {
            const float* hist = cstate + (size_t) ch * (d_conv - 1);
            const float* cw = lw->ssm_conv1d.data() + (size_t) ch * d_conv;
            double acc = 0.0;
            for (uint32_t k = 0; k < d_conv - 1; ++k) acc += (double) hist[k] * cw[k];
            acc += (double) xBC_raw[ch] * cw[d_conv - 1];
            acc += lw->ssm_conv1d_b[ch];
            conv_out[ch] = silu((float) acc);
            // Slide the window: drop the oldest value, append the
            // current RAW (pre-convolution) value as the newest.
            float* hist_w = cstate + (size_t) ch * (d_conv - 1);
            for (uint32_t k = 0; k + 1 < d_conv - 1; ++k) hist_w[k] = hist_w[k + 1];
            if (d_conv >= 2) hist_w[d_conv - 2] = xBC_raw[ch];
        }

        const float* x_ssm = conv_out.data();
        const float* B = conv_out.data() + d_inner;
        const float* C = conv_out.data() + d_inner + n_group * d_state;

        float* sstate = ssm_state_.data() + (size_t) l * n_head * head_dim * d_state;
        const uint32_t heads_per_group = n_head / n_group;
        for (uint32_t h = 0; h < n_head; ++h) {
            const float dt_h = dt_raw[h] + lw->ssm_dt_b[h];
            const float dt_soft_plus = softplus(dt_h);
            const float dA = expf(dt_soft_plus * lw->ssm_a[h]);
            const uint32_t g = h / heads_per_group;
            const float* Bg = B + (size_t) g * d_state;
            const float* Cg = C + (size_t) g * d_state;

            for (uint32_t i1 = 0; i1 < head_dim; ++i1) {
                const uint32_t ii = h * head_dim + i1;
                const float x_dt = x_ssm[ii] * dt_soft_plus;
                float* srow = sstate + (size_t) ii * d_state;
                double sumf = 0.0;
                for (uint32_t i0 = 0; i0 < d_state; ++i0) {
                    const float state_val = srow[i0] * dA + Bg[i0] * x_dt;
                    sumf += (double) state_val * Cg[i0];
                    srow[i0] = state_val;
                }
                y[ii] = (float) sumf + x_ssm[ii] * lw->ssm_d[h];
            }
        }

        // y = silu(z) * y (gated SiLU, gate=z, value=y).
        for (uint32_t i = 0; i < d_inner; ++i) y[i] = silu(z[i]) * y[i];

        const float* y_final = y.data();
        if (!lw->ssm_norm.empty()) {
            // RMSNorm a gruppi: n_group blocchi contigui di d_inner/n_group
            // canali, ciascuno normalizzato per conto proprio col proprio
            // segmento di peso (vedi la nota sul layout in ssm_forward.h).
            const uint32_t grp_w = d_inner / n_group;
            for (uint32_t g = 0; g < n_group; ++g) {
                rms_norm_vec(y.data() + (size_t) g * grp_w, lw->ssm_norm.data() + (size_t) g * grp_w,
                             y_normed.data() + (size_t) g * grp_w, grp_w, cfg_.rms_eps);
            }
            y_final = y_normed.data();
        }

        matvec_f32(lw->ssm_out.data(), n_embd, d_inner, y_final, out.data());
        for (uint32_t i = 0; i < n_embd; ++i) x[i] += out[i];
    }

    ++n_tokens_seen_;

    if (want_logits) {
        rms_norm_vec(x.data(), out_norm_.data(), xn.data(), n_embd, cfg_.rms_eps);
        logits_out.assign(cfg_.n_vocab, 0.0f);
        matvec_f32(output_.data(), cfg_.n_vocab, n_embd, xn.data(), logits_out.data());
    }
}

bool SsmForward::step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
                       std::vector<float>& last_logits, std::vector<float>* all_logits) {
    if (n_tokens == 0 || tokens == nullptr || tok_embd_.empty()) return false;
    if (all_logits) all_logits->clear();

    for (size_t p = 0; p < n_tokens; ++p) {
        const int32_t t = tokens[p];
        if (t < 0 || (uint32_t) t >= cfg_.n_vocab) return false;
        const bool is_last = (p + 1 == n_tokens);
        const bool want = is_last || all_logits != nullptr;
        std::vector<float> logits;
        step_token(rd, t, want, logits);
        if (is_last) last_logits = logits;
        if (all_logits && want) {
            all_logits->insert(all_logits->end(), logits.begin(), logits.end());
        }
    }
    return true;
}

}
