#ifndef DESIREEIA_SPM_TOKENIZER_H
#define DESIREEIA_SPM_TOKENIZER_H

#include "core/engine.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace desireeia {

// Tokenizer SentencePiece Unigram con byte-fallback, guidato interamente dai
// metadati del formato (tokens/scores/token_type), senza dipendenze esterne.
// Segmentazione via Viterbi (programmazione dinamica) sui confini di
// codepoint UTF-8, con fallback byte-per-byte quando nessun pezzo del
// vocabolario copre un codepoint (richiede i token <0xXX> nel vocabolario).
// Normalizzazione limitata a: spazio -> U+2581, prefisso U+2581 iniziale.
// NFKC non e' applicata (gap noto, vedi docs/engine_gap_analysis.md).
class DESIREEIA_INTERNAL SpmTokenizer {
public:
    bool load(const VocabData& vocab);
    std::vector<int32_t> encode(const std::string& text, bool add_bos) const;
    bool piece(int32_t id, std::string& out) const;
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    bool ready() const { return !id_to_piece_.empty(); }

private:
    struct Entry { int32_t id; float score; };

    std::unordered_map<std::string, Entry> piece_to_id_;
    std::vector<std::string> id_to_piece_;
    std::unordered_map<int32_t, int32_t> byte_token_id_; // valore byte (0-255) -> id vocabolario
    size_t max_piece_cp_ = 1;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t unk_id_ = -1;
};

}

#endif
