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
        // Two passes the compiler vectorizes (eight running maxima, then the
        // first index holding the maximum) instead of one compare-and-branch
        // per logit: same result - the first index of the largest value -
        // in a fraction of the time on a 262k vocabulary.
        const float* lg = logits.data();
        float lane[8];
        for (int j = 0; j < 8; ++j) lane[j] = lg[0];
        size_t i = 0;
        for (; i + 8 <= n_vocab; i += 8) {
            for (int j = 0; j < 8; ++j) lane[j] = lg[i + j] > lane[j] ? lg[i + j] : lane[j];
        }
        float best_v = lane[0];
        for (int j = 1; j < 8; ++j) best_v = lane[j] > best_v ? lane[j] : best_v;
        for (; i < n_vocab; ++i) best_v = lg[i] > best_v ? lg[i] : best_v;
        for (size_t b = 0; b < n_vocab; ++b) {
            if (lg[b] == best_v) return (int32_t) b;
        }
        return 0;
    }

    // --- 3. Top-k ------------------------------------------------------
    // The candidate count is reduced right away: everything that follows
    // (softmax, top-p, extraction) works on k entries instead of 262144.
    size_t k = (params_.top_k > 0)
        ? std::min((size_t) params_.top_k, n_vocab)
        : n_vocab;

    if (k < n_vocab) {
        // One pass with a min-heap of the k best so far: nearly every logit
        // is rejected by a single compare against the heap's smallest,
        // instead of partial_sort's indirect compares over the whole
        // vocabulary. Same candidates (ties at the k-th value aside, which
        // partial_sort leaves unspecified too), sorted as before.
        auto worse = [&](int32_t a, int32_t b) {       // heap order: smallest logit on top
            return logits[(size_t) a] > logits[(size_t) b] ||
                   (logits[(size_t) a] == logits[(size_t) b] && a < b);
        };
        idx_.clear();
        idx_.reserve(k);
        for (size_t i = 0; i < k; ++i) idx_.push_back((int32_t) i);
        std::make_heap(idx_.begin(), idx_.end(), worse);
        for (size_t i = k; i < n_vocab; ++i) {
            const float v = logits[i];
            const int32_t top = idx_.front();
            if (v > logits[(size_t) top]) {
                std::pop_heap(idx_.begin(), idx_.end(), worse);
                idx_.back() = (int32_t) i;
                std::push_heap(idx_.begin(), idx_.end(), worse);
            }
        }
        std::sort(idx_.begin(), idx_.end(), [&](int32_t a, int32_t b) {
            return logits[(size_t) a] > logits[(size_t) b] ||
                   (logits[(size_t) a] == logits[(size_t) b] && a < b);
        });
    } else {
        idx_.resize(n_vocab);
        for (size_t i = 0; i < n_vocab; ++i) idx_[i] = (int32_t) i;
        std::sort(idx_.begin(), idx_.end(),
            [&](int32_t a, int32_t b) { return logits[(size_t) a] > logits[(size_t) b]; });
    }

    return finish_(logits, k);
}

int32_t Sampler::finish_(const std::vector<float>& logits, size_t k) {
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

uint32_t Sampler::candidates_needed(const std::vector<int32_t>& history) const {
    const bool pen = params_.penalty_last_n > 0 &&
        (params_.penalty_repeat != 1.0f || params_.penalty_freq != 0.0f || params_.penalty_present != 0.0f);
    if (pen && (params_.penalty_repeat < 1.0f || params_.penalty_freq < 0.0f || params_.penalty_present < 0.0f)) {
        return 0;                                   // a penalty could raise a logit
    }
    size_t need = params_.temperature <= 0.0f ? 1 : (params_.top_k > 0 ? (size_t) params_.top_k : 0);
    if (need == 0) return 0;
    if (pen) {
        const size_t look = std::min((size_t) params_.penalty_last_n, history.size());
        std::vector<int32_t> seen(history.end() - (long) look, history.end());
        std::sort(seen.begin(), seen.end());
        need += (size_t) (std::unique(seen.begin(), seen.end()) - seen.begin());
    }
    return need <= 256 ? (uint32_t) need : 0;
}

bool Sampler::device_params(DeviceParams& out) const {
    const bool pen = params_.penalty_last_n > 0 &&
        (params_.penalty_repeat != 1.0f || params_.penalty_freq != 0.0f || params_.penalty_present != 0.0f);
    if (pen && (params_.penalty_repeat < 1.0f || params_.penalty_freq < 0.0f || params_.penalty_present < 0.0f)) {
        return false;                               // a penalty could raise a logit: whole vocabulary needed
    }
    size_t need = params_.temperature <= 0.0f ? 1 : (params_.top_k > 0 ? (size_t) params_.top_k : 0);
    if (need == 0) return false;
    if (pen) {
        if (params_.penalty_last_n > 256) return false;          // the device history is 256 tokens
        need += (size_t) params_.penalty_last_n;                 // every distinct token of the window
    }
    if (need > 256) return false;
    out.temperature = params_.temperature;
    out.top_k = params_.top_k;
    out.top_p = params_.top_p;
    out.penalty_repeat = params_.penalty_repeat;
    out.penalty_freq = params_.penalty_freq;
    out.penalty_present = params_.penalty_present;
    out.penalty_last_n = pen ? params_.penalty_last_n : 0;
    out.kk = (uint32_t) need;
    return true;
}

int32_t Sampler::sample_candidates(std::vector<int32_t>& ids, std::vector<float>& vals,
                                   const std::vector<int32_t>& history) {
    const size_t n = ids.size();
    if (n == 0 || vals.size() != n) return -1;
    // 1. penalties, as sample() applies them, on the candidates that carry them
    if (params_.penalty_last_n > 0 &&
        (params_.penalty_repeat != 1.0f || params_.penalty_freq != 0.0f || params_.penalty_present != 0.0f)) {
        const size_t look = std::min((size_t) params_.penalty_last_n, history.size());
        std::unordered_map<int32_t, int32_t> counts;
        counts.reserve(look * 2);
        for (size_t i = history.size() - look; i < history.size(); ++i) ++counts[history[i]];
        for (size_t j = 0; j < n; ++j) {
            auto it = counts.find(ids[j]);
            if (it == counts.end()) continue;
            float& lg = vals[j];
            if (lg <= 0.0f) lg *= params_.penalty_repeat;
            else            lg /= params_.penalty_repeat;
            lg -= (float) it->second * params_.penalty_freq + params_.penalty_present;
        }
    }
    // Order of the whole-vocabulary path: larger logit first, lower token id on ties.
    idx_.resize(n);
    for (size_t j = 0; j < n; ++j) idx_[j] = (int32_t) j;
    std::sort(idx_.begin(), idx_.end(), [&](int32_t a, int32_t b) {
        return vals[(size_t) a] > vals[(size_t) b] ||
               (vals[(size_t) a] == vals[(size_t) b] && ids[(size_t) a] < ids[(size_t) b]);
    });
    // 2. greedy: the largest, the lowest id among equals
    if (params_.temperature <= 0.0f) return ids[(size_t) idx_[0]];
    // 3. top-k, then the shared tail
    const size_t k = std::min((size_t) params_.top_k, n);
    idx_.resize(k);
    return ids[(size_t) finish_(vals, k)];
}

}
