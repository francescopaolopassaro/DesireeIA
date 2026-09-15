// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

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
    // Separate, allocation-free path: this is the default, and it must
    // not pay anything for the sampling machinery.
    if (params_.temperature <= 0.0f) {
        size_t best = 0;
        float  best_v = logits[0];
        for (size_t i = 1; i < n_vocab; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        }
        return (int32_t) best;
    }

    // --- 3. Top-k ------------------------------------------------------
    // The candidate count is reduced right away: everything that follows
    // (softmax, top-p, extraction) works on k entries instead of 262144.
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

    // --- 4. Temperature + softmax ---------------------------------------
    // The maximum is subtracted before the exponential: without it,
    // exp() of large logits overflows and the distribution becomes NaN.
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
    // idx_ is already sorted by decreasing logit, hence also by
    // decreasing probability: it's enough to truncate where the
    // cumulative mass exceeds top_p. At least one candidate is always kept.
    size_t keep = k;
    if (params_.top_p < 1.0f) {
        float cum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            cum += probs_[i];
            if (cum >= params_.top_p) { keep = i + 1; break; }
        }
        if (keep < 1) keep = 1;
    }

    // --- 6. Multinomial extraction ---------------------------------------
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
