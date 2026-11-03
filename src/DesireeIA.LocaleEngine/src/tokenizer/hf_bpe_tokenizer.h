// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_HF_BPE_TOKENIZER_H
#define DESIREEIA_HF_BPE_TOKENIZER_H

#include "core/engine.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace desireeia {

// BPE tokenizer driven by a tokenizer.json file (the format used next to
// encoder checkpoints distributed as plain directories). It is a separate
// class from BpeTokenizer on purpose: that one is fed from GGUF metadata and
// uses an approximate pre-tokenizer, whereas the decision models need token
// ids that match the reference implementation exactly.
//
// Two flavours are supported, picked from the file's pre-tokenizer:
//   - byte-level (English checkpoints): the GPT2 pre-tokenization pattern,
//     bytes mapped to the visible unicode alphabet, BPE merges by rank;
//   - metaspace (multilingual checkpoints): spaces become U+2581, the text is
//     cut before every U+2581, BPE runs over characters, and a character that
//     is not in the vocabulary falls back to its UTF-8 bytes (<0xHH> pieces).
// In both, added tokens are matched atomically (longest first, honouring the
// lstrip/rstrip flags) before anything else.
//
// Known gaps: Unicode normalization (NFC) is not applied, so input that is
// not already NFC-composed can tokenize differently; for the byte-level
// flavour, letter/number classification of non-ASCII code points is
// table-based, not the full Unicode database.
class DESIREEIA_INTERNAL HfBpeTokenizer {
public:
    // Loads vocab, merges and added tokens from a tokenizer.json file. The
    // names of the special tokens come from a tokenizer_config.json in the
    // same folder when there is one.
    bool load_file(const std::string& path);
    // Same, from the file contents already in memory (config_json may be empty).
    bool load_json(const std::string& json_text, const std::string& config_json = std::string());

    // Encodes text without adding any special token of its own.
    std::vector<int32_t> encode(const std::string& text) const;

    // Id of an exact token string (added tokens first, then the vocabulary),
    // or -1 when absent.
    int32_t token_id(const std::string& piece) const;
    size_t vocab_size() const { return n_vocab_; }
    bool ready() const { return n_vocab_ > 0; }

    int32_t cls_id() const { return cls_id_; }
    int32_t sep_id() const { return sep_id_; }
    int32_t mask_id() const { return mask_id_; }
    int32_t pad_id() const { return pad_id_; }
    const std::string& mask_text() const { return mask_text_; }

private:
    struct Added {
        std::string content;
        int32_t id = -1;
        bool lstrip = false;
        bool rstrip = false;
    };

    void encode_plain(const std::string& text, bool first_segment, std::vector<int32_t>& out) const;
    void bpe_chunk(const std::string& chunk, std::vector<int32_t>& out) const;
    void metaspace_piece(const std::string& piece, std::vector<int32_t>& out) const;

    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<Added> added_;                                  // longest content first
    std::unordered_map<unsigned char, std::vector<size_t>> added_by_first_;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
    size_t n_vocab_ = 0;
    int32_t cls_id_ = -1;
    int32_t sep_id_ = -1;
    int32_t mask_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = -1;
    std::string mask_text_ = "[MASK]";

    // metaspace flavour
    bool metaspace_ = false;
    bool replace_space_ = false;      // the normalizer turns ' ' into the replacement
    bool byte_fallback_ = false;
    bool split_ = true;
    std::string replacement_ = "\xE2\x96\x81";   // U+2581
    std::string prepend_scheme_ = "always";
};

}

#endif
