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
    uint32_t n_head = 0;   // per mamba2: hparams.ssm_dt_rank riusato come conteggio teste
    uint32_t n_group = 1;  // gruppi B/C condivisi fra teste (mamba2; 1 per mamba1)

    uint32_t head_dim() const { return n_head > 0 ? d_inner / n_head : 0; }
    uint32_t conv_dim() const { return d_inner + 2 * n_group * d_state; }
    uint32_t d_in_proj() const { return 2 * d_inner + 2 * n_group * d_state + n_head; } // z + xBC + dt
};

struct SsmLayerWeights {
    std::vector<float> attn_norm;    // dim n_embd
    std::vector<float> ssm_in;       // [d_in_proj, n_embd] riga = riga d_in_proj, n_embd colonne
    std::vector<float> ssm_conv1d;   // [conv_dim, d_conv] riga = canale, d_conv colonne (tap piu' vecchio prima)
    std::vector<float> ssm_conv1d_b; // dim conv_dim
    std::vector<float> ssm_dt_b;     // dim n_head
    std::vector<float> ssm_a;        // dim n_head (scalare per testa, mamba2)
    std::vector<float> ssm_d;        // dim n_head (scalare per testa, mamba2)
    std::vector<float> ssm_norm;     // dim d_inner (opzionale: n_group blocchi da d_inner/n_group, concatenati)
    std::vector<float> ssm_out;      // [n_embd, d_inner] riga = riga n_embd, d_inner colonne
};

// Forward path per Mamba2. Copre SOLO la variante "pura" (norm -> mixer ->
// residuo ad ogni layer, nessuna attenzione): le varianti ibride
// (falcon-h1, jamba, granite-hybrid, nemotron-h — che alternano layer SSM e
// layer di attenzione classica nello stesso modello) NON sono coperte,
// richiederebbero anche il forward path denso nello stesso modello: gap
// noto, documentato. Mamba1 (A per-stato invece che scalare per testa) e le
// famiglie RWKV (ricorrenza diversa, non SSM) restano fuori: gap noti.
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

    // Esegue UN token attraverso tutti i layer, aggiornando lo stato
    // ricorrente in place. Se want_logits e' true scrive i logit in
    // logits_out (altrimenti li lascia vuoti, per i token intermedi di un
    // batch quando all_logits non e' richiesto: risparmia il matvec piu'
    // grande del modello, lm_head, quando il chiamante non lo vuole).
    void step_token(ModelReader& rd, int32_t token, bool want_logits, std::vector<float>& logits_out);

    SsmConfig cfg_;
    std::vector<float> tok_embd_; // [n_vocab, n_embd], sempre float (vedi nota in ssm_forward.cpp)
    std::vector<float> output_;   // [n_vocab, n_embd], puo' essere lo stesso buffer di tok_embd_ (tied)
    std::vector<float> out_norm_;

    bool cache_enabled_ = false;
    std::vector<SsmLayerWeights> layer_cache_;
    SsmLayerWeights scratch_;

    // Stato ricorrente: SOVRASCRITTO ad ogni token (non append-only come la
    // KV-cache del motore denso) — questa e' la proprieta' che rende gli
    // SSM a memoria costante indipendente dalla lunghezza del contesto.
    std::vector<float> conv_state_; // [n_layers, conv_dim, d_conv-1]
    std::vector<float> ssm_state_;  // [n_layers, n_head, head_dim, d_state]
    size_t n_tokens_seen_ = 0;
};

}

#endif
