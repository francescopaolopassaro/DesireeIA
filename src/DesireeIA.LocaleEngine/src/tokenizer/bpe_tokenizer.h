// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_BPE_TOKENIZER_H
#define DESIREEIA_BPE_TOKENIZER_H

#include "core/engine.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace desireeia {

// Byte-level GPT2-style BPE tokenizer (used by qwen2 and many
// non-SentencePiece families in GGUF files, tag "tokenizer.ggml.model" =
// "gpt2"). No external dependency.
//
// Pipeline: pre-tokenization into chunks (letters/digits/punctuation/spaces,
// an approximation of the GPT2 regex pattern — does not handle full
// Unicode categories, known gap) -> byte->"visible" unicode mapping
// (standard GPT2 table) -> greedy BPE merge by rank on each chunk ->
// vocabulary lookup.
class DESIREEIA_INTERNAL BpeTokenizer {
public:
    bool load(const VocabData& vocab);
    std::vector<int32_t> encode(const std::string& text, bool add_bos) const;
    bool piece(int32_t id, std::string& out) const;
    // Exact vocab lookup: checks the special-token cache first (raw token
    // bytes as stored in vocab, e.g. "<image>"/"<|im_start|>"), then the
    // regular piece map. -1 if absent.
    int32_t token_to_id(const std::string& piece) const;
    int32_t bos_id() const { return bos_id_; }
    bool ready() const { return !id_to_piece_.empty(); }

private:
    std::vector<std::string> bpe_merge(const std::string& chunk_mapped) const;

    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t unk_id_ = -1;
    // Mirrors tokenizer.ggml.add_bos_token. True when the model says
    // nothing, matching the format's own default and keeping every model
    // that predates this check behaving exactly as before; load() then
    // overwrites it with whatever the file actually declares.
    bool add_bos_default_ = true;

    // Special-token cache (CONTROL/USER_DEFINED/UNKNOWN), sorted by
    // decreasing text length so the longest match wins.
    // Needed because, without it, markers like "<|im_start|>" (used by the
    // ChatML/Llama3/Phi/... chat templates) did NOT tokenize at all as an
    // atomic vocabulary entry: they ended up split apart by ordinary
    // pre-tokenization/BPE merging into "<", "|", "im", "_", "start", "|",
    // ">" — the model never saw the real control token, and generation no
    // longer stopped at the turn boundary.
    std::vector<std::pair<std::string, int32_t>> special_tokens_;
};

}

#endif
