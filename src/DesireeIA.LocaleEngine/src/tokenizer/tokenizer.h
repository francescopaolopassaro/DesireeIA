#ifndef DESIREEIA_TOKENIZER_H
#define DESIREEIA_TOKENIZER_H

#include "core/engine.h"
#include "core/arch_tags.h"
#include "spm_tokenizer.h"
#include "bpe_tokenizer.h"
#include <string>
#include <vector>

namespace desireeia {

// Wrapper uniforme sui due tokenizer implementati (SentencePiece Unigram e
// BPE byte-level), selezionato in base a detect_tokenizer(vocab.tokenizer_tag).
class DESIREEIA_INTERNAL Tokenizer {
public:
    bool load(const VocabData& vocab);
    std::vector<int32_t> encode(const std::string& text, bool add_bos) const;
    bool piece(int32_t id, std::string& out) const;
    // Exact token lookup (raw piece bytes, e.g. "<image>"), -1 if absent.
    int32_t token_to_id(const std::string& piece) const;
    bool ready() const { return kind_ != TokenizerKind::Unknown; }

private:
    TokenizerKind kind_ = TokenizerKind::Unknown;
    SpmTokenizer spm_;
    BpeTokenizer bpe_;
};

}

#endif
