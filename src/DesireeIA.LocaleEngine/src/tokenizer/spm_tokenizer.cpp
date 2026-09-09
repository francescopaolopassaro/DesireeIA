#include "spm_tokenizer.h"
#include <limits>

namespace desireeia {

namespace {

size_t utf8_len(unsigned char c0) {
    if ((c0 & 0x80) == 0x00) return 1;
    if ((c0 & 0xE0) == 0xC0) return 2;
    if ((c0 & 0xF0) == 0xE0) return 3;
    if ((c0 & 0xF8) == 0xF0) return 4;
    return 1; // byte non valido: trattato come codepoint di 1 byte
}

int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

constexpr float kByteFallbackPenalty = -10.0f;
constexpr float kUnkPenalty = -1000.0f;

}

bool SpmTokenizer::load(const VocabData& vocab) {
    if (vocab.tokens.empty()) return false;

    id_to_piece_ = vocab.tokens;
    piece_to_id_.clear();
    byte_token_id_.clear();
    max_piece_cp_ = 1;

    for (size_t i = 0; i < vocab.tokens.size(); ++i) {
        const std::string& piece = vocab.tokens[i];
        const float score = i < vocab.scores.size() ? vocab.scores[i] : 0.0f;
        piece_to_id_[piece] = Entry{ (int32_t) i, score };

        size_t cp_count = 0;
        for (size_t p = 0; p < piece.size();) {
            const size_t len = utf8_len((unsigned char) piece[p]);
            p += len;
            cp_count++;
        }
        if (cp_count > max_piece_cp_) max_piece_cp_ = cp_count;

        if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
            const int hi = hex_val(piece[3]);
            const int lo = hex_val(piece[4]);
            if (hi >= 0 && lo >= 0) {
                byte_token_id_[hi * 16 + lo] = (int32_t) i;
            }
        }
    }

    bos_id_ = vocab.bos_id;
    eos_id_ = vocab.eos_id;
    unk_id_ = vocab.unk_id;
    return true;
}

int32_t SpmTokenizer::token_to_id(const std::string& piece) const {
    auto it = piece_to_id_.find(piece);
    return it == piece_to_id_.end() ? -1 : it->second.id;
}

bool SpmTokenizer::piece(int32_t id, std::string& out) const {
    if (id < 0 || (size_t) id >= id_to_piece_.size()) return false;
    const std::string& raw = id_to_piece_[(size_t) id];

    // Detokenizzazione: inversa della normalizzazione fatta in encode().
    // 1) U+2581 (\xe2\x96\x81, il marcatore di spazio SentencePiece) torna
    //    a essere uno spazio normale — prima veniva restituito grezzo e
    //    finiva letteralmente nell'output ("The▁capital▁of...").
    // 2) I token di byte-fallback "<0xNN>" tornano al byte che
    //    rappresentano (altrimenti resterebbero visibili come testo).
    if (raw.size() == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>') {
        const int hi = hex_val(raw[3]);
        const int lo = hex_val(raw[4]);
        if (hi >= 0 && lo >= 0) {
            out.assign(1, (char) (unsigned char) ((hi << 4) | lo));
            return true;
        }
    }

    out.clear();
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size();) {
        if (i + 3 <= raw.size() &&
            (unsigned char) raw[i] == 0xE2 &&
            (unsigned char) raw[i + 1] == 0x96 &&
            (unsigned char) raw[i + 2] == 0x81) {
            out += ' ';
            i += 3;
        } else {
            out += raw[i];
            ++i;
        }
    }
    return true;
}

std::vector<int32_t> SpmTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> result;
    if (!ready()) return result;

    // Normalizzazione minima SentencePiece: spazio -> U+2581, prefisso
    // U+2581 iniziale (add_dummy_prefix). NFKC non applicata (gap noto).
    std::string norm;
    norm.reserve(text.size() + 4);
    norm += "\xe2\x96\x81";
    for (char c : text) {
        if (c == ' ') norm += "\xe2\x96\x81";
        else norm += c;
    }

    struct Span { size_t start; size_t len; };
    std::vector<Span> cps;
    cps.reserve(norm.size());
    for (size_t i = 0; i < norm.size();) {
        size_t len = utf8_len((unsigned char) norm[i]);
        if (i + len > norm.size()) len = 1;
        cps.push_back({ i, len });
        i += len;
    }

    const size_t n = cps.size();
    const float neg_inf = -std::numeric_limits<float>::max() / 4.0f;
    std::vector<float> best(n + 1, neg_inf);
    std::vector<int32_t> prev(n + 1, -1);
    std::vector<std::vector<int32_t>> emit(n + 1);
    best[0] = 0.0f;

    for (size_t idx = 1; idx <= n; ++idx) {
        const size_t jmin = idx > max_piece_cp_ ? idx - max_piece_cp_ : 0;
        for (size_t j = jmin; j < idx; ++j) {
            if (best[j] <= neg_inf) continue;
            const size_t byte_start = cps[j].start;
            const size_t byte_end = cps[idx - 1].start + cps[idx - 1].len;
            const std::string sub = norm.substr(byte_start, byte_end - byte_start);
            auto it = piece_to_id_.find(sub);
            if (it == piece_to_id_.end()) continue;
            const float cand = best[j] + it->second.score;
            if (cand > best[idx]) {
                best[idx] = cand;
                prev[idx] = (int32_t) j;
                emit[idx] = { it->second.id };
            }
        }

        if (best[idx] <= neg_inf) {
            const size_t j = idx - 1;
            const std::string sub = norm.substr(cps[j].start, cps[j].len);
            std::vector<int32_t> ids;
            bool ok = true;
            for (unsigned char b : sub) {
                auto bit = byte_token_id_.find((int32_t) b);
                if (bit == byte_token_id_.end()) { ok = false; break; }
                ids.push_back(bit->second);
            }
            if (ok && best[j] > neg_inf) {
                const float cand = best[j] + kByteFallbackPenalty * (float) sub.size();
                if (cand > best[idx]) {
                    best[idx] = cand;
                    prev[idx] = (int32_t) j;
                    emit[idx] = ids;
                }
            } else if (unk_id_ >= 0 && best[j] > neg_inf) {
                const float cand = best[j] + kUnkPenalty;
                if (cand > best[idx]) {
                    best[idx] = cand;
                    prev[idx] = (int32_t) j;
                    emit[idx] = { unk_id_ };
                }
            }
        }
    }

    std::vector<int32_t> rev;
    size_t cur = n;
    while (cur > 0 && prev[cur] != -1) {
        const auto& ids = emit[cur];
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) rev.push_back(*it);
        cur = (size_t) prev[cur];
    }

    if (add_bos && bos_id_ >= 0) result.push_back(bos_id_);
    for (auto it = rev.rbegin(); it != rev.rend(); ++it) result.push_back(*it);
    return result;
}

}
