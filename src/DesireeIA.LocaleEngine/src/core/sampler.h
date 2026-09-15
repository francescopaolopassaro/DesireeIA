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

private:
    SamplerParams params_;
    std::mt19937  rng_{std::random_device{}()};
    bool          seeded_ = false;

    // Buffers reused across calls to avoid reallocating on every token:
    // gemma3's vocabulary is 262144 entries, allocating it at every step
    // would be a non-negligible fraction of decode time.
    std::vector<int32_t> idx_;
    std::vector<float>   probs_;
};

}

#endif
