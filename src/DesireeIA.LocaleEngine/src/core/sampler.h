// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_SAMPLER_H
#define DESIREEIA_SAMPLER_H

#include "desireeia/abi.h"

#include <cstdint>
#include <random>
#include <vector>

namespace desireeia {

// Sampling parameters. The defaults match common practice for
// interactive generation.
//
// temperature <= 0 means GREEDY (argmax): deterministic, no sampling.
// This stays the default because it's the only behavior the existing
// end-to-end tests expect — whoever wants real generation raises the
// temperature explicitly.
struct SamplerParams {
    float    temperature     = 0.0f;   // 0 = greedy
    int32_t  top_k           = 40;     // <= 0 = disabled
    float    top_p           = 0.95f;  // >= 1 = disabled
    float    penalty_repeat  = 1.0f;   // 1.0 = disabled
    float    penalty_freq    = 0.0f;
    float    penalty_present = 0.0f;
    int32_t  penalty_last_n  = 64;     // how many tokens back to look
    uint32_t seed            = 0;      // 0 = seed from random_device
};

// Stateful sampler (the pseudo-random generator). Not thread-safe: one
// instance per inference context, protected by the context's mutex.
class Sampler {
public:
    void configure(const SamplerParams& p);
    const SamplerParams& params() const { return params_; }

    // Picks the next token from the logits. `history` are the tokens
    // already present in the sequence (prompt + generated), used for the
    // repetition penalties: the last penalty_last_n of them are looked at.
    //
    // `logits` is modified in place (penalties and temperature).
    int32_t sample(std::vector<float>& logits, const std::vector<int32_t>& history);

    // How many of the largest logits sample() can work from and still give
    // exactly the result it would give on the whole vocabulary: top_k (1 for
    // greedy) plus one per distinct token the penalties touch - penalties
    // only ever lower a logit, so the final top_k is inside that set. 0 when
    // that does not hold (top_k disabled or large, penalties that can raise
    // a logit): then the whole vocabulary is needed.
    uint32_t candidates_needed(const std::vector<int32_t>& history) const;
    // sample() over the candidates only (ids with their logits, any order).
    int32_t sample_candidates(std::vector<int32_t>& ids, std::vector<float>& vals,
                              const std::vector<int32_t>& history);

    // Pipelined decode (the draw made on the device, see CudaSampleArgs):
    // whether these parameters can be replicated there, and with how many
    // candidates - enough for any history of up to penalty_last_n tokens,
    // so the count does not change from token to token.
    struct DeviceParams {
        float temperature; int32_t top_k; float top_p;
        float penalty_repeat; float penalty_freq; float penalty_present; int32_t penalty_last_n;
        uint32_t kk;
    };
    bool device_params(DeviceParams& out) const;
    // The uniform number sample() would draw for its extraction: the same
    // generator call, so the sequence of draws is the host path's.
    float draw_uniform() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_); }
    // Generator state, saved before a speculative draw and restored when
    // that draw is discarded.
    const std::mt19937& rng_state() const { return rng_; }
    void set_rng_state(const std::mt19937& r) { rng_ = r; }

private:
    SamplerParams params_;
    std::mt19937  rng_{std::random_device{}()};
    bool          seeded_ = false;

    // Buffers reused across calls to avoid reallocating on every token:
    // gemma3's vocabulary is 262144 entries, allocating it at every step
    // would be a non-negligible fraction of decode time.
    std::vector<int32_t> idx_;
    // Steps 4-6 of sample() (temperature, softmax, top-p, draw) over the
    // first k entries of idx_, sorted; returns one of them.
    int32_t finish_(const std::vector<float>& logits, size_t k);
    std::vector<float>   probs_;
};

}

#endif
