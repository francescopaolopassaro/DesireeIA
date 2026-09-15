// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "../core/engine.h"

namespace desireeia {

KvCache::KvCache(uint32_t n_layers, uint32_t n_kv, bool compressed)
    : n_layers_(n_layers), n_kv_(n_kv), compressed_(compressed) {
}

void KvCache::push(const std::vector<float>& keys, const std::vector<float>& vals) {
    keys_.push_back(keys);
    vals_.push_back(vals);
}

uint64_t KvCache::bytes() const {
    uint64_t n = 0;
    for (const auto& k : keys_) n += k.size() * sizeof(float);
    for (const auto& v : vals_) n += v.size() * sizeof(float);
    return n;
}

}
