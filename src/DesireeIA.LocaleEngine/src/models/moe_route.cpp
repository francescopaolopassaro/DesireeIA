#include "moe_route.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace desireeia {

std::vector<std::pair<uint32_t, float>> moe_route_ex(
    const std::vector<float>& logits, uint32_t n_expert_used, bool norm_w, float w_scale,
    MoeGatingFunc gating, const std::vector<float>* sel_bias) {
    const size_t n = logits.size();
    std::vector<std::pair<uint32_t, float>> out;
    if (n == 0) return out;

    std::vector<float> probs(n);
    if (gating == MoeGatingFunc::Sigmoid) {
        for (size_t i = 0; i < n; ++i) probs[i] = 1.0f / (1.0f + expf(-logits[i]));
    } else {
        float m = logits[0];
        for (float v : logits) m = std::max(m, v);
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            probs[i] = expf(logits[i] - m);
            sum += probs[i];
        }
        for (size_t i = 0; i < n; ++i) probs[i] = (float) (probs[i] / sum);
    }

    // Bias di selezione (DeepSeek-V3, blk.N.exp_probs_b): sposta solo QUALI
    // esperti finiscono nel top-k, non il peso di combinazione (vedi la
    // nota in moe_route.h).
    std::vector<float> sel_probs = probs;
    if (sel_bias && sel_bias->size() == n) {
        for (size_t i = 0; i < n; ++i) sel_probs[i] += (*sel_bias)[i];
    }

    // A NaN here is not just a bad routing decision, it is undefined
    // behaviour: `a > b` is false in both directions for NaN, which breaks
    // the strict weak ordering std::partial_sort requires, and the standard
    // library is then free to walk off the end of the range — an access
    // violation, observed for real on a model whose forward pass produced
    // NaN upstream. Sorting must stay safe no matter what the network hands
    // it, so NaNs are pushed to the bottom instead of being compared.
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(sel_probs[i])) sel_probs[i] = -std::numeric_limits<float>::infinity();
        if (std::isnan(probs[i])) probs[i] = 0.0f;
    }

    std::vector<uint32_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = (uint32_t) i;
    const uint32_t k = std::min((uint32_t) n, n_expert_used);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                       [&](uint32_t a, uint32_t b) { return sel_probs[a] > sel_probs[b]; });

    out.resize(k);
    float wsum = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        out[i] = { idx[i], probs[idx[i]] };
        wsum += probs[idx[i]];
    }
    if (norm_w && wsum > 1e-9f) {
        for (auto& p : out) p.second /= wsum;
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        for (auto& p : out) p.second *= w_scale;
    }
    return out;
}

std::vector<std::pair<uint32_t, float>> moe_route(
    const std::vector<float>& logits, uint32_t n_expert_used, bool norm_w, float w_scale) {
    return moe_route_ex(logits, n_expert_used, norm_w, w_scale, MoeGatingFunc::Softmax, nullptr);
}

}
