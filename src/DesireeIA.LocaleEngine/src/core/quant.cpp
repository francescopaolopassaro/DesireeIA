// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "engine.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace desireeia {

void quantize_q8_0(const float* src, size_t n, std::vector<int8_t>& q,
                   std::vector<float>& scales, std::vector<uint8_t>& signs) {
    constexpr size_t BLOCK = 32;
    size_t nblocks = (n + BLOCK - 1) / BLOCK;

    q.resize(nblocks * BLOCK);
    scales.resize(nblocks);
    signs.resize(0);
    signs.clear();

    const float qmax = 127.0f;

    for (size_t b = 0; b < nblocks; ++b) {
        size_t start = b * BLOCK;
        size_t len = std::min(BLOCK, n - start);

        float vmax = 0.0f;
        for (size_t i = 0; i < len; ++i) {
            vmax = std::max(vmax, std::fabs(src[start + i]));
        }

        float d = vmax > 0.0f ? vmax / qmax : 0.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        scales[b] = d;

        for (size_t i = 0; i < len; ++i) {
            int32_t v = static_cast<int32_t>(std::lrintf(src[start + i] * id));
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q[start + i] = static_cast<int8_t>(v);
        }
        for (size_t i = len; i < BLOCK; ++i) {
            q[start + i] = 0;
        }
    }
}

}
