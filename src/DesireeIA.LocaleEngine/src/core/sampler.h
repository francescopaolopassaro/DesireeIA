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

// Extra samplers, off by default (desireeia_set_sampling_ext).
//  - min_p: drops every candidate whose probability is below min_p times the
//    probability of the best one (after top-k, before top-p). Adapts the
//    cut to how sure the model is: tight when it is confident, wide when not.
//  - DRY: penalises a token that would extend a sequence already written in
//    THIS generation (not the prompt): the longer the repeated run, the
//    stronger the penalty, multiplier * base^(run - allowed_length). Breaks
//    the "same sentence over and over" loops small models fall into without
//    touching words that merely occur twice.
struct SamplerExt {
    float   min_p            = 0.0f;   // 0 = off
    float   dry_multiplier   = 0.0f;   // 0 = off
    float   dry_base         = 1.75f;
    int32_t dry_allowed_len  = 2;
    int32_t dry_last_n       = 256;    // how many generated tokens back to look
};

// Confidence of the last token drawn (see Sampler::set_track_stats).
struct TokenStats {
    float logprob  = 0.0f;   // ln P(chosen) over the whole vocabulary
    float entropy  = 0.0f;   // entropy of the distribution, in nats
    float max_prob = 0.0f;   // probability of the most likely token
    bool  valid    = false;
};

// Stateful sampler (the pseudo-random generator). Not thread-safe: one
// instance per inference context, protected by the context's mutex.
class Sampler {
public:
    void configure(const SamplerParams& p);
    const SamplerParams& params() const { return params_; }
    void configure_ext(const SamplerExt& e) { ext_ = e; }
    const SamplerExt& ext() const { return ext_; }
    // Where the generation starts in the history (the prompt length): DRY
    // only looks at what was generated after it.
    void set_generation_start(size_t n) { gen_start_ = n; }
    // DRY is switched off while a constrained tool call is being written
    // (its JSON repeats keys and braces by design).
    void set_dry_suspended(bool v) { dry_suspended_ = v; }
    // Per-token confidence, exact over the whole vocabulary: costs one more
    // pass over the logits and keeps the sampling on the host.
    void set_track_stats(bool v) { track_ = v; last_stats_ = TokenStats{}; }
    bool track_stats() const { return track_; }
    const TokenStats& last_stats() const { return last_stats_; }
    // The extras that the device draw cannot reproduce.
    bool host_only() const { return ext_.min_p > 0.0f || ext_.dry_multiplier > 0.0f || track_; }

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
    SamplerExt    ext_;
    size_t        gen_start_ = 0;
    bool          dry_suspended_ = false;
    bool          track_ = false;
    TokenStats    last_stats_;
    int32_t sample_impl_(std::vector<float>& logits, const std::vector<int32_t>& history);
    void apply_dry_(std::vector<float>& logits, const std::vector<int32_t>& history) const;
    void compute_stats_(const std::vector<float>& logits, int32_t chosen);
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
