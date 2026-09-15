// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "tokenizer.h"

namespace desireeia {

bool Tokenizer::load(const VocabData& vocab) {
    kind_ = detect_tokenizer(vocab.tokenizer_tag);
    switch (kind_) {
        case TokenizerKind::SpmUnigram:
            if (!spm_.load(vocab)) { kind_ = TokenizerKind::Unknown; return false; }
            return true;
        case TokenizerKind::Bpe:
            if (!bpe_.load(vocab)) { kind_ = TokenizerKind::Unknown; return false; }
            return true;
        default:
            return false;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    switch (kind_) {
        case TokenizerKind::SpmUnigram: return spm_.encode(text, add_bos);
        case TokenizerKind::Bpe:        return bpe_.encode(text, add_bos);
        default:                        return {};
    }
}

bool Tokenizer::piece(int32_t id, std::string& out) const {
    switch (kind_) {
        case TokenizerKind::SpmUnigram: return spm_.piece(id, out);
        case TokenizerKind::Bpe:        return bpe_.piece(id, out);
        default:                        return false;
    }
}

int32_t Tokenizer::token_to_id(const std::string& piece) const {
    switch (kind_) {
        case TokenizerKind::SpmUnigram: return spm_.token_to_id(piece);
        case TokenizerKind::Bpe:        return bpe_.token_to_id(piece);
        default:                        return -1;
    }
}

}
