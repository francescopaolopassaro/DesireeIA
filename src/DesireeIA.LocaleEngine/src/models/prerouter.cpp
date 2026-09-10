#include "prerouter.h"
#include <algorithm>
#include <cmath>

namespace desireeia {

namespace {
// Exact erf-based GELU (not the tanh approximation used elsewhere in this
// engine's dense FFN activation): 0.5*x*(1+erf(x/sqrt(2))).
inline float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}
}

void prerouter_predict(const PrerouterHead& head, const float* feat, size_t feat_dim,
                        uint32_t k, std::vector<uint32_t>& out_idx) {
    out_idx.clear();
    if (head.n_expert == 0 || head.prerouter_hidden == 0 || feat_dim == 0) return;
    if (head.fc1.size() != (size_t) head.prerouter_hidden * feat_dim) return;
    if (head.fc2.size() != (size_t) head.n_expert * head.prerouter_hidden) return;
    if (head.lin.size() != (size_t) head.n_expert * feat_dim) return;

    std::vector<float> h1(head.prerouter_hidden, 0.0f);
    for (uint32_t j = 0; j < head.prerouter_hidden; ++j) {
        const float* row = head.fc1.data() + (size_t) j * feat_dim;
        double acc = 0.0;
        for (size_t i = 0; i < feat_dim; ++i) acc += (double) row[i] * feat[i];
        h1[j] = gelu_erf((float) acc);
    }

    std::vector<float> logits(head.n_expert, 0.0f);
    for (uint32_t j = 0; j < head.n_expert; ++j) {
        const float* row2 = head.fc2.data() + (size_t) j * head.prerouter_hidden;
        double acc2 = 0.0;
        for (uint32_t i = 0; i < head.prerouter_hidden; ++i) acc2 += (double) row2[i] * h1[i];

        const float* rowl = head.lin.data() + (size_t) j * feat_dim;
        double accl = 0.0;
        for (size_t i = 0; i < feat_dim; ++i) accl += (double) rowl[i] * feat[i];

        logits[j] = (float) (acc2 + accl);
    }

    const uint32_t kk = std::min(k, head.n_expert);
    std::vector<uint32_t> idx(head.n_expert);
    for (uint32_t i = 0; i < head.n_expert; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                       [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
    out_idx.assign(idx.begin(), idx.begin() + kk);
}

}
