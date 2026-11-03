// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_DECISION_MODEL_H
#define DESIREEIA_DECISION_MODEL_H

#include "decision_ops.h"
#include "modernbert_forward.h"
#include "tokenizer/hf_bpe_tokenizer.h"
#include <map>
#include <string>
#include <vector>

namespace desireeia {

// Non-autoregressive decision model: a bidirectional encoder plus a small
// typed decision head. It never generates text. The caller supplies a state
// (free text or structured data) and a list of typed questions; every answer
// option is marked with its own mask token in the input, the head scores the
// hidden state at each marker, and a softmax over a question's options gives
// its answer distribution.
//
// Question types:
//   choice - one of N labelled options (criteria are label -> description)
//   score  - an ordinal level 0..N-1 (criteria describe each level)
//   noul   - a yes/no statement ("true"/"false" descriptions are optional)
//
// A model lives in a directory:
//   rl_agent_config.json   head/calibration/length settings
//   model.safetensors      encoder.* + head.* + scorer.* + act_head.* + type_emb.*
//   encoder/config.json    encoder architecture
//   tokenizer/tokenizer.json
// It is a completely separate backend from the generative and embedding
// paths of the engine: nothing here is reachable unless a model directory of
// this kind was opened.
class DESIREEIA_INTERNAL DecisionModel {
public:
    // True when `path` is a directory laid out as described above.
    static bool is_decision_dir(const std::string& path);

    bool open(const std::string& dir, std::string& err);
    // Runs the encoder, and when it fits the decision head too, on an NVIDIA GPU
    // (see ModernBertEncoder). False with the reason when the encoder cannot go to
    // the GPU; a head that cannot is not an error, it just stays on the CPU.
    bool enable_gpu(std::string& err);
    bool gpu_enabled() const { return enc_.gpu_enabled(); }
    bool gpu_head_ready() const { return enc_.gpu_head_ready(); }

    // Runs a decision request and returns the response, both as JSON text.
    //
    // Request:
    //   { "state": <string | any JSON value>,
    //     "questions": [
    //        { "type": "choice", "instruction": "...", "options": {"label": "description", ...} },
    //        { "type": "score",  "instruction": "...", "options": ["level 0 text", ...] },
    //        { "type": "noul",   "instruction": "...", "true": "...", "false": "..." } ],
    //     "calibrate": true,          // optional, default true
    //     "truncate_left": false }    // optional, keep the END of a long state
    // (The short keys "t", "ins" and "crit" are accepted as aliases.)
    //
    // Response: { "model": ..., "answers": [ { "type", "options", "probabilities",
    //   "logits", "index", "choice", "confidence", "act_probability",
    //   "act_probabilities", "n_tokens" } ] }
    bool decide(const std::string& request_json, std::string& response_json, std::string& err);

    const std::string& name() const { return name_; }
    uint32_t max_len() const { return max_len_; }

private:
    struct HeadLayer {
        std::vector<float> n1w, n1b, n2w, n2b;
        HalfMat in_proj, out_proj, lin1, lin2;
        std::vector<float> in_proj_b, out_proj_b, lin1_b, lin2_b;
    };
    struct Question {
        std::string type;
        std::string instruction;
        std::vector<std::string> labels;   // answer labels, in option order
        std::vector<std::string> options;  // rendered option texts, same order
    };
    struct Answer {
        std::vector<float> logits;
        std::vector<float> act;
        size_t n_tokens = 0;
    };

    bool build_input(const std::string& state_text, const Question& q, bool truncate_left,
                     std::vector<int32_t>& ids, std::vector<size_t>& markers) const;
    bool run(const std::vector<int32_t>& ids, const std::vector<size_t>& markers, int qtype,
             Answer& out) const;
    float temperature_for(int qtype, size_t k) const;

    std::string name_;
    StFile st_;
    ModernBertEncoder enc_;
    HfBpeTokenizer tok_;
    std::vector<HeadLayer> head_;
    uint32_t d_ = 0;
    uint32_t max_len_ = 1024;
    uint32_t head_max_len_ = 256;
    std::vector<float> type_emb_;                // [3, d]
    std::vector<float> sc_nw_, sc_nb_;           // scorer LayerNorm
    HalfMat sc_l1_;
    std::vector<float> sc_l1_b_;
    std::vector<float> sc_l2_w_;                 // [d]
    float sc_l2_b_ = 0.0f;
    HalfMat act_l0_, act_l2_;
    std::vector<float> act_l0_b_, act_l2_b_;
    std::vector<float> temp_by_type_;            // [3]
    std::map<std::string, float> temp_by_bucket_;
};

}

#endif
