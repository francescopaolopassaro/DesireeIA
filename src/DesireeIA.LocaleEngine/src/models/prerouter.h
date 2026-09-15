// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_PREROUTER_H
#define DESIREEIA_PREROUTER_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace desireeia {

// A trained per-layer routing-prediction head. Predicts, from a feature
// vector built at layer L (the "owner"), which experts layer L+1 (the
// "consumer") is likely to route to — so their weights can be prefetched
// a full layer earlier than layer L+1's own router would resolve them.
// See DenseForward::maybe_predict_next_layer in dense_forward.cpp for the
// wiring and DenseForward::load_prerouter for the file format (this
// engine's own GGUF tensor-naming convention, not byte-compatible with
// any external reference implementation's format).
//
// out = linear_init(feat) + fc2(gelu_erf(fc1(feat)))
// fc1: [prerouter_hidden, feat_dim], fc2: [n_expert, prerouter_hidden],
// linear_init (lin): [n_expert, feat_dim] — row-major [out,in], the same
// convention as every weight matrix in this engine.
struct PrerouterHead {
    std::vector<float> fc1;
    std::vector<float> fc2;
    std::vector<float> lin;
    uint32_t hidden = 0;
    uint32_t prerouter_hidden = 0;
    uint32_t n_expert = 0;
};

// Computes the head's output logits and writes the top-`k` expert indices
// (by logit value — no softmax/normalization needed, only indices are
// used to trigger a prefetch, never routing weights) into out_idx.
// out_idx is cleared first; left empty if `head`/`feat_dim` don't match
// (caller's prefetch then simply does nothing, never a wrong guess).
void prerouter_predict(const PrerouterHead& head, const float* feat, size_t feat_dim,
                        uint32_t k, std::vector<uint32_t>& out_idx);

}

#endif
