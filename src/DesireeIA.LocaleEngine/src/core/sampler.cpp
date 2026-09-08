#include "core/sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace desireeia {

void Sampler::configure(const SamplerParams& p) {
    params_ = p;
    if (p.seed != 0) {
        rng_.seed(p.seed);
        seeded_ = true;
    } else if (!seeded_) {
        rng_.seed(std::random_device{}());
        seeded_ = true;
    }
}

int32_t Sampler::sample(std::vector<float>& logits, const std::vector<int32_t>& history) {
    const size_t n_vocab = logits.size();
    if (n_vocab == 0) return -1;

    // --- 1. Repetition / frequency / presence penalties -------------
    // NOT a plain division. Dividing a NEGATIVE logit would push it
    // toward zero, i.e. make an already-emitted token MORE likely — the
    // opposite of what's intended. Multiply when the logit is <= 0 and
    // divide when it's > 0, so the penalty always pushes the score down.
    if (params_.penalty_last_n > 0 &&
        (params_.penalty_repeat != 1.0f || params_.penalty_freq != 0.0f ||
         params_.penalty_present != 0.0f)) {
        const size_t look = std::min((size_t) params_.penalty_last_n, history.size());
        std::unordered_map<int32_t, int32_t> counts;
        counts.reserve(look * 2);
        for (size_t i = history.size() - look; i < history.size(); ++i) {
            ++counts[history[i]];
        }
        for (const auto& kv : counts) {
            const int32_t id = kv.first;
            if (id < 0 || (size_t) id >= n_vocab) continue;
            float& lg = logits[(size_t) id];
            if (lg <= 0.0f) lg *= params_.penalty_repeat;
            else            lg /= params_.penalty_repeat;
            lg -= (float) kv.second * params_.penalty_freq + params_.penalty_present;
        }
    }

    // --- 2. Greedy -----------------------------------------------------
    // Percorso separato e senza allocazioni: e' il default, e non deve
    // pagare nulla del macchinario di campionamento.
    if (params_.temperature <= 0.0f) {
        size_t best = 0;
        float  best_v = logits[0];
        for (size_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        }
        return (int32_t) best;
    }

    // --- 3. Top-k ------------------------------------------------------
    // Si riduce subito il numero di candidati: tutto quel che segue
    // (softmax, top-p, estrazione) lavora su k voci invece che su 262144.
    size_t k = (params_.top_k > 0)
        ? std::min((size_t) params_.top_k, n_vocab)
        : n_vocab;

    idx_.resize(n_vocab);
    for (size_t i = 0; i < n_vocab; ++i) idx_[i] = (int32_t) i;

    if (k < n_vocab) {
        std::partial_sort(idx_.begin(), idx_.begin() + (long) k, idx_.end(),
            [&](int32_t a, int32_t b) { return logits[(size_t) a] > logits[(size_t) b]; });
        idx_.resize(k);
    } else {
        std::sort(idx_.begin(), idx_.end(),
            [&](int32_t a, int32_t b) { return logits[(size_t) a] > logits[(size_t) b]; });
    }

    // --- 4. Temperatura + softmax --------------------------------------
    // Il massimo viene sottratto prima dell'esponenziale: senza, exp() di
    // logit grandi va in overflow e la distribuzione diventa NaN.
    const float inv_t = 1.0f / params_.temperature;
    const float max_l = logits[(size_t) idx_[0]] * inv_t;
    probs_.resize(k);
    float sum = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        const float p = std::exp(logits[(size_t) idx_[i]] * inv_t - max_l);
        probs_[i] = p;
        sum += p;
    }
    if (sum <= 0.0f) return idx_[0];
    for (size_t i = 0; i < k; ++i) probs_[i] /= sum;

    // --- 5. Top-p (nucleus) --------------------------------------------
    // idx_ e' gia' ordinato per logit decrescente, quindi anche per
    // probabilita' decrescente: basta troncare dove la massa cumulata
    // supera top_p. Si tiene sempre almeno un candidato.
    size_t keep = k;
    if (params_.top_p < 1.0f) {
        float cum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            cum += probs_[i];
            if (cum >= params_.top_p) { keep = i + 1; break; }
        }
        if (keep < 1) keep = 1;
    }

    // --- 6. Estrazione multinomiale ------------------------------------
    float cum = 0.0f;
    for (size_t i = 0; i < keep; ++i) cum += probs_[i];
    if (cum <= 0.0f) return idx_[0];

    std::uniform_real_distribution<float> dist(0.0f, cum);
    const float r = dist(rng_);
    float acc = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        acc += probs_[i];
        if (r <= acc) return idx_[i];
    }
    return idx_[keep - 1];
}

}
