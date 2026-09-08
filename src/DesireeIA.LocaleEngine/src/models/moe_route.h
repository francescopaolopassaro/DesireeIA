#ifndef DESIREEIA_MOE_ROUTE_H
#define DESIREEIA_MOE_ROUTE_H

#include "core/engine.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace desireeia {

// Standard softmax MoE router: softmax the logits over every expert, then
// select the top n_expert_used by probability, with optional
// renormalization of the selected weights to sum to 1 and a final scale
// factor. Covers standard SOFTMAX gating. Implemented on top of
// moe_route_ex below (softmax, no selection bias): identical behavior to
// before.
DESIREEIA_INTERNAL std::vector<std::pair<uint32_t, float>> moe_route(
    const std::vector<float>& logits, uint32_t n_expert_used, bool norm_w, float w_scale);

enum class MoeGatingFunc { Softmax, Sigmoid };

// General variant (DeepSeek2 and siblings): softmax or sigmoid gating,
// with an optional SELECTION bias that shifts which experts end up in the
// top-k but NOT the weight their output is combined with — the bias is
// added only for the top-k selection itself, the final combination
// weights are read from the unbiased probabilities. Doesn't implement the
// grouped top-k routing used only by the largest model in this family:
// known, documented gap — no currently testable checkpoint needs it.
DESIREEIA_INTERNAL std::vector<std::pair<uint32_t, float>> moe_route_ex(
    const std::vector<float>& logits, uint32_t n_expert_used, bool norm_w, float w_scale,
    MoeGatingFunc gating, const std::vector<float>* sel_bias);

}

#endif
