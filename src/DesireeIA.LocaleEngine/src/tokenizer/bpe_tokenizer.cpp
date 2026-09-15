// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "bpe_tokenizer.h"
#include <algorithm>
#include <array>
#include <functional>
#include <unordered_map>

namespace desireeia {

namespace {

std::string utf8_encode_cp(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
    return out;
}

// Standard GPT2 byte<->"visible" unicode table: every byte 0..255 is
// mapped to a codepoint that, once re-encoded as UTF-8, is the character
// used in the BPE vocabulary pieces in GGUF files.
const std::array<std::string, 256>& byte_to_unicode() {
    static const std::array<std::string, 256> table = [] {
        std::array<std::string, 256> t;
        std::array<bool, 256> assigned{};
        int n = 0;
        auto put = [&](int b) { t[b] = utf8_encode_cp((uint32_t) b); assigned[b] = true; };
        for (int b = '!'; b <= '~'; ++b) put(b);
        for (int b = 0xA1; b <= 0xAC; ++b) put(b);
        for (int b = 0xAE; b <= 0xFF; ++b) put(b);
        for (int b = 0; b < 256; ++b) {
            if (!assigned[b]) {
                t[b] = utf8_encode_cp((uint32_t) (256 + n));
                n++;
            }
        }
        return t;
    }();
    return table;
}

std::string map_bytes(const std::string& raw) {
    const auto& tbl = byte_to_unicode();
    std::string out;
    out.reserve(raw.size() * 2);
    for (unsigned char b : raw) out += tbl[b];
    return out;
}


size_t utf8_len(unsigned char c0) {
    if ((c0 & 0x80) == 0x00) return 1;
    if ((c0 & 0xE0) == 0xC0) return 2;
    if ((c0 & 0xF0) == 0xE0) return 3;
    if ((c0 & 0xF8) == 0xF0) return 4;
    return 1;
}

// Inverse of map_bytes: from the "visible" characters of the BPE
// vocabulary back to the real bytes. Needed for detokenization — without
// it, pieces came back raw and the output literally showed the GPT2
// markers (Ġ for space, Ċ for newline) instead of the actual text.
const std::unordered_map<std::string, unsigned char>& unicode_to_byte() {
    static const std::unordered_map<std::string, unsigned char> table = [] {
        std::unordered_map<std::string, unsigned char> m;
        const auto& tbl = byte_to_unicode();
        for (int b = 0; b < 256; ++b) m[tbl[b]] = (unsigned char) b;
        return m;
    }();
    return table;
}

std::string unmap_bytes(const std::string& mapped) {
    const auto& tbl = unicode_to_byte();
    std::string out;
    out.reserve(mapped.size());
    for (size_t i = 0; i < mapped.size();) {
        size_t len = utf8_len((unsigned char) mapped[i]);
        if (i + len > mapped.size()) len = 1;
        const std::string ch = mapped.substr(i, len);
        auto it = tbl.find(ch);
        if (it != tbl.end()) {
            out += (char) it->second;
        } else {
            // Not a character from the GPT2 map (e.g. a special token like
            // <|im_end|>): leave it as is.
            out += ch;
        }
        i += len;
    }
    return out;
}

bool is_ascii_alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// Approximate pre-tokenization of the GPT2 pattern: handles common English
// contractions, then groups letters / digits / other / spaces into
// separate chunks, with the optional leading space attached to the
// following chunk (as in the original pattern). Full Unicode categories
// (\p{L}, \p{N}) are not implemented: every byte >=0x80 (part of a
// multi-byte UTF-8 sequence) is treated as a "letter" as an approximation
// (known gap).
std::vector<std::string> pre_tokenize(const std::string& text) {
    std::vector<std::string> chunks;
    static const char* contractions[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        bool matched = false;
        for (const char* c : contractions) {
            const size_t len = std::char_traits<char>::length(c);
            if (i + len <= n && text.compare(i, len, c) == 0) {
                chunks.push_back(text.substr(i, len));
                i += len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        size_t start = i;
        bool has_leading_space = false;
        if (text[i] == ' ' && i + 1 < n && !is_space((unsigned char) text[i + 1])) {
            has_leading_space = true;
            i++;
        }
        if (i >= n) { chunks.push_back(text.substr(start)); break; }

        unsigned char c0 = (unsigned char) text[i];
        if (is_ascii_alpha(c0) || c0 >= 0x80) {
            size_t j = i;
            while (j < n) {
                unsigned char cc = (unsigned char) text[j];
                if (is_ascii_alpha(cc) || cc >= 0x80) j += utf8_len(cc);
                else break;
            }
            chunks.push_back(text.substr(start, j - start));
            i = j;
        } else if (is_ascii_digit(c0)) {
            size_t j = i;
            while (j < n && is_ascii_digit((unsigned char) text[j])) j++;
            chunks.push_back(text.substr(start, j - start));
            i = j;
        } else if (is_space(c0)) {
            size_t j = i;
            while (j < n && is_space((unsigned char) text[j])) j++;
            chunks.push_back(text.substr(start, j - start));
            i = j;
        } else {
            size_t j = i;
            while (j < n) {
                unsigned char cc = (unsigned char) text[j];
                if (!is_ascii_alpha(cc) && !is_ascii_digit(cc) && !is_space(cc) && cc < 0x80) j++;
                else break;
            }
            if (j == i) j++; // ensures progress on unexpected bytes
            chunks.push_back(text.substr(start, j - start));
            i = j;
        }
        (void) has_leading_space;
    }
    return chunks;
}

}

bool BpeTokenizer::load(const VocabData& vocab) {
    if (vocab.tokens.empty()) return false;
    id_to_piece_ = vocab.tokens;
    piece_to_id_.clear();
    for (size_t i = 0; i < vocab.tokens.size(); ++i) {
        piece_to_id_[vocab.tokens[i]] = (int32_t) i;
    }
    merge_rank_.clear();
    for (size_t i = 0; i < vocab.merges.size(); ++i) {
        merge_rank_[vocab.merges[i]] = (int32_t) i;
    }
    bos_id_ = vocab.bos_id;
    eos_id_ = vocab.eos_id;
    unk_id_ = vocab.unk_id;
    add_bos_default_ = vocab.add_bos;

    // GGUF token types: 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED. These three
    // types are recognized as atomic BEFORE ordinary pre-tokenization/BPE
    // merging, instead of being split apart like normal text.
    special_tokens_.clear();
    if (vocab.token_type.size() == vocab.tokens.size()) {
        for (size_t i = 0; i < vocab.tokens.size(); ++i) {
            const int32_t tt = vocab.token_type[i];
            if (tt == 2 || tt == 3 || tt == 4) {
                special_tokens_.emplace_back(vocab.tokens[i], (int32_t) i);
            }
        }
        // Longest first: if a special token were ever a prefix of another
        // (doesn't happen in known formats, but it costs nothing to guard
        // against), the longest match must win.
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }
    return true;
}

int32_t BpeTokenizer::token_to_id(const std::string& piece) const {
    for (const auto& st : special_tokens_) {
        if (st.first == piece) return st.second;
    }
    auto it = piece_to_id_.find(piece);
    return it == piece_to_id_.end() ? -1 : it->second;
}

bool BpeTokenizer::piece(int32_t id, std::string& out) const {
    if (id < 0 || (size_t) id >= id_to_piece_.size()) return false;
    // Detokenization: inverse of the byte->"visible" unicode GPT2 map
    // applied in encode(). Without it, the output would literally contain
    // the markers (Ġ instead of space, Ċ instead of newline).
    out = unmap_bytes(id_to_piece_[(size_t) id]);
    return true;
}

std::vector<std::string> BpeTokenizer::bpe_merge(const std::string& chunk_mapped) const {
    std::vector<std::string> symbols;
    for (size_t i = 0; i < chunk_mapped.size();) {
        size_t len = utf8_len((unsigned char) chunk_mapped[i]);
        if (i + len > chunk_mapped.size()) len = 1;
        symbols.push_back(chunk_mapped.substr(i, len));
        i += len;
    }
    if (symbols.empty()) return symbols;

    for (;;) {
        int best_rank = -1;
        size_t best_idx = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const std::string key = symbols[i] + " " + symbols[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_rank < 0) break;
        symbols[best_idx] += symbols[best_idx + 1];
        symbols.erase(symbols.begin() + (long) best_idx + 1);
    }
    return symbols;
}

namespace {
// Applies the ordinary BPE pipeline (pre-tokenization -> byte map -> merge
// -> lookup) to a stretch of text that does NOT contain special tokens.
void encode_plain(const std::string& text,
                  const std::unordered_map<std::string, int32_t>& piece_to_id,
                  int32_t unk_id,
                  const std::function<std::vector<std::string>(const std::string&)>& bpe_merge_fn,
                  std::vector<int32_t>& out) {
    for (const std::string& chunk : pre_tokenize(text)) {
        const std::string mapped = map_bytes(chunk);
        for (const std::string& sym : bpe_merge_fn(mapped)) {
            auto it = piece_to_id.find(sym);
            if (it != piece_to_id.end()) {
                out.push_back(it->second);
            } else if (unk_id >= 0) {
                out.push_back(unk_id);
            }
        }
    }
}
}

std::vector<int32_t> BpeTokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> out;
    if (!ready()) return out;

    if (add_bos && bos_id_ >= 0) out.push_back(bos_id_);

    auto merge_fn = [this](const std::string& s) { return bpe_merge(s); };

    if (special_tokens_.empty()) {
        encode_plain(text, piece_to_id_, unk_id_, merge_fn, out);
        return out;
    }

    // Linear scan: at every position, try the longest special token that
    // matches (special_tokens_ is sorted by decreasing length). Plain text
    // between two special tokens (or before the first / after the last) is
    // accumulated and only run through the ordinary BPE pipeline once the
    // stretch closes — not character by character, otherwise merges
    // between adjacent bytes would be lost.
    size_t pos = 0;
    size_t plain_start = 0;
    while (pos < text.size()) {
        int32_t matched_id = -1;
        size_t matched_len = 0;
        for (const auto& st : special_tokens_) {
            const std::string& s = st.first;
            if (!s.empty() && text.compare(pos, s.size(), s) == 0) {
                matched_id = st.second;
                matched_len = s.size();
                break; // sorted by decreasing length: the first one is the longest
            }
        }
        if (matched_id < 0) {
            ++pos;
            continue;
        }
        if (pos > plain_start) {
            encode_plain(text.substr(plain_start, pos - plain_start), piece_to_id_, unk_id_, merge_fn, out);
        }
        out.push_back(matched_id);
        pos += matched_len;
        plain_start = pos;
    }
    if (plain_start < text.size()) {
        encode_plain(text.substr(plain_start), piece_to_id_, unk_id_, merge_fn, out);
    }
    return out;
}

}
