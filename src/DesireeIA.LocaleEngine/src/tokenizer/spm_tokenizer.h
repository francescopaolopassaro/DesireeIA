// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_SPM_TOKENIZER_H
#define DESIREEIA_SPM_TOKENIZER_H

#include "core/engine.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace desireeia {

// SentencePiece Unigram tokenizer with byte-fallback, driven entirely by
// the format's metadata (tokens/scores/token_type), with no external
// dependencies. Segmentation via Viterbi (dynamic programming) over UTF-8
// codepoint boundaries, with a byte-by-byte fallback when no vocabulary
// piece covers a codepoint (requires the <0xXX> tokens in the vocabulary).
// Normalization is limited to: space -> U+2581, leading U+2581 prefix.
// NFKC is not applied (known gap, see docs/engine_gap_analysis.md).
class DESIREEIA_INTERNAL SpmTokenizer {
public:
    bool load(const VocabData& vocab);
    std::vector<int32_t> encode(const std::string& text, bool add_bos) const;
    bool piece(int32_t id, std::string& out) const;
    // Exact vocab lookup (raw token bytes, e.g. "<image>"). -1 if absent.
    int32_t token_to_id(const std::string& piece) const;
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    bool ready() const { return !id_to_piece_.empty(); }

private:
    struct Entry { int32_t id; float score; };

    std::unordered_map<std::string, Entry> piece_to_id_;
    std::vector<std::string> id_to_piece_;
    std::unordered_map<int32_t, int32_t> byte_token_id_; // byte value (0-255) -> vocabulary id
    size_t max_piece_cp_ = 1;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t unk_id_ = -1;
};

}

#endif
