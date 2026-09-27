// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "spm_tokenizer.h"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <queue>

namespace desireeia {

namespace {

size_t utf8_len(unsigned char c0) {
    if ((c0 & 0x80) == 0x00) return 1;
    if ((c0 & 0xE0) == 0xC0) return 2;
    if ((c0 & 0xF0) == 0xE0) return 3;
    if ((c0 & 0xF8) == 0xF0) return 4;
    return 1; // invalid byte: treated as a 1-byte codepoint
}

int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

constexpr float kByteFallbackPenalty = -10.0f;
constexpr float kUnkPenalty = -1000.0f;
constexpr const char* SPACE_MARK = "\xe2\x96\x81";   // U+2581

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

    for (auto& bucket : specials_) bucket.clear();
    for (size_t i = 0; i < vocab.tokens.size() && i < vocab.token_type.size(); ++i) {
        const int32_t type = vocab.token_type[i];
        const std::string& t = vocab.tokens[i];
        // pieces written with the space marker are reached by the merges
        if ((type == 3 || type == 4) && !t.empty() && t.find(SPACE_MARK) == std::string::npos)
            specials_[(unsigned char) t[0]].emplace_back(t, (int32_t) i);
    }
    for (auto& bucket : specials_)
        std::sort(bucket.begin(), bucket.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    bos_id_ = vocab.bos_id;
    add_space_prefix_ = vocab.add_space_prefix;
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

    // Detokenization: inverse of the normalization done in encode().
    // 1) U+2581 (\xe2\x96\x81, the SentencePiece space marker) is turned
    //    back into a normal space — previously it was returned raw and
    //    ended up literally in the output ("The▁capital▁of...").
    // 2) Byte-fallback tokens "<0xNN>" are turned back into the byte they
    //    represent (otherwise they would stay visible as text).
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
    if (add_bos && bos_id_ >= 0) result.push_back(bos_id_);
    static const bool viterbi = std::getenv("DESIREEIA_SPM_VITERBI") != nullptr;

    // Text fragments between special tokens. Minimal SentencePiece
    // normalization of each: space -> U+2581, and the leading U+2581 only
    // at the very start of the text when the model asks for it
    // (add_space_prefix: Gemma does not - with it every prompt began with a
    // stray space token before <start_of_turn>). NFKC is not applied
    // (known gap).
    auto segment = [&](size_t from, size_t to) {
        if (from >= to) return;
        std::string norm;
        norm.reserve(to - from + 4);
        if (add_space_prefix_ && from == 0) norm += "\xe2\x96\x81";
        for (size_t i = from; i < to; ++i) {
            if (text[i] == ' ') norm += "\xe2\x96\x81";
            else norm += text[i];
        }
        if (viterbi) encode_viterbi(norm, result);
        else encode_merge(norm, result);
    };
    size_t start = 0;
    for (size_t i = 0; i < text.size();) {
        int32_t hit = -1;
        size_t hit_len = 0;
        {
            for (const auto& sp : specials_[(unsigned char) text[i]]) {
                if (sp.first.size() <= text.size() - i && text.compare(i, sp.first.size(), sp.first) == 0) {
                    hit = sp.second;
                    hit_len = sp.first.size();
                    break;
                }
            }
        }
        if (hit < 0) { ++i; continue; }
        segment(start, i);
        result.push_back(hit);
        i += hit_len;
        start = i;
    }
    segment(start, text.size());
    return result;
}

void SpmTokenizer::emit_piece(const std::string& piece, std::vector<int32_t>& out) const {
    auto it = piece_to_id_.find(piece);
    if (it != piece_to_id_.end()) { out.push_back(it->second.id); return; }
    // not in the vocabulary: its bytes, or the unknown token
    for (unsigned char c : piece) {
        auto bit = byte_token_id_.find((int32_t) c);
        if (bit != byte_token_id_.end()) out.push_back(bit->second);
        else if (unk_id_ >= 0) { out.push_back(unk_id_); return; }
    }
}

void SpmTokenizer::encode_merge(const std::string& norm, std::vector<int32_t>& out) const {
    struct Sym { size_t start; size_t len; int prev; int next; };
    std::vector<Sym> sym;
    for (size_t i = 0; i < norm.size();) {
        size_t len = utf8_len((unsigned char) norm[i]);
        if (i + len > norm.size()) len = norm.size() - i;
        sym.push_back({ i, len, (int) sym.size() - 1, (int) sym.size() + 1 });
        i += len;
    }
    if (sym.empty()) return;
    sym.back().next = -1;

    struct Pair { int left; int right; float score; size_t size; };
    auto worse = [](const Pair& x, const Pair& y) {       // queue top: highest score, then leftmost
        return x.score < y.score || (x.score == y.score && x.left > y.left);
    };
    std::priority_queue<Pair, std::vector<Pair>, decltype(worse)> queue(worse);
    auto try_pair = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        const size_t size = sym[(size_t) l].len + sym[(size_t) r].len;
        auto it = piece_to_id_.find(norm.substr(sym[(size_t) l].start, size));
        if (it == piece_to_id_.end()) return;
        queue.push({ l, r, it->second.score, size });
    };
    for (size_t i = 1; i < sym.size(); ++i) try_pair((int) i - 1, (int) i);

    while (!queue.empty()) {
        const Pair p = queue.top();
        queue.pop();
        Sym& l = sym[(size_t) p.left];
        Sym& r = sym[(size_t) p.right];
        if (l.len == 0 || r.len == 0 || l.len + r.len != p.size) continue;   // superseded by another merge
        l.len += r.len;
        r.len = 0;
        l.next = r.next;
        if (r.next >= 0) sym[(size_t) r.next].prev = p.left;
        try_pair(l.prev, p.left);
        try_pair(p.left, l.next);
    }
    for (int i = 0; i >= 0; i = sym[(size_t) i].next) {
        if (sym[(size_t) i].len > 0) emit_piece(norm.substr(sym[(size_t) i].start, sym[(size_t) i].len), out);
    }
}

void SpmTokenizer::encode_viterbi(const std::string& norm, std::vector<int32_t>& result) const {
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

    for (auto it = rev.rbegin(); it != rev.rend(); ++it) result.push_back(*it);
}

}
