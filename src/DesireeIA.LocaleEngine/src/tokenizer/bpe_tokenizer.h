#ifndef DESIREEIA_BPE_TOKENIZER_H
#define DESIREEIA_BPE_TOKENIZER_H

#include "core/engine.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace desireeia {

// Tokenizer BPE byte-level in stile GPT2 (usato da qwen2 e da molte
// famiglie non-SentencePiece nei file GGUF, tag "tokenizer.ggml.model" =
// "gpt2"). Nessuna dipendenza esterna.
//
// Pipeline: pre-tokenizzazione in chunk (lettere/cifre/punteggiatura/spazi,
// approssimazione del pattern regex GPT2 — non gestisce categorie Unicode
// complete, gap noto) -> mappatura byte->unicode "visibile" (tabella GPT2
// standard) -> merge BPE greedy per rank su ogni chunk -> lookup vocabolario.
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
    bool add_bos_default_ = false;

    // Special-token cache (CONTROL/USER_DEFINED/UNKNOWN), sorted by
    // decreasing text length so the longest match wins.
    // Serve perche' senza, marcatori come "<|im_start|>" (usati dai
    // template di chat ChatML/Llama3/Phi/...) NON tokenizzavano affatto
    // come voce atomica del vocabolario: finivano spezzettati dalla
    // pre-tokenizzazione/merge BPE ordinaria in "<", "|", "im", "_",
    // "start", "|", ">" — il modello non vedeva mai il vero token di
    // controllo, e la generazione non si fermava piu' sul turno.
    std::vector<std::pair<std::string, int32_t>> special_tokens_;
};

}

#endif
