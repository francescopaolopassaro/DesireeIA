// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "hf_bpe_tokenizer.h"
#include "models/decision_json.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iterator>

namespace desireeia {

namespace {

// ---- UTF-8 helpers ---------------------------------------------------------

// Decodes the code point starting at s[i]. Invalid sequences decode as the
// single byte (so the scanner always makes progress); len receives the
// number of bytes consumed.
uint32_t decode_cp(const std::string& s, size_t i, size_t& len) {
    const unsigned char c0 = (unsigned char) s[i];
    size_t n = 1;
    uint32_t cp = c0;
    if (c0 >= 0xF0 && c0 <= 0xF4) { n = 4; cp = c0 & 0x07; }
    else if (c0 >= 0xE0) { n = 3; cp = c0 & 0x0F; }
    else if (c0 >= 0xC2 && c0 < 0xE0) { n = 2; cp = c0 & 0x1F; }
    else { len = 1; return c0; }
    if (c0 >= 0xF5 || i + n > s.size()) { len = 1; return c0; }
    for (size_t k = 1; k < n; ++k) {
        const unsigned char c = (unsigned char) s[i + k];
        if ((c & 0xC0) != 0x80) { len = 1; return c0; }
        cp = (cp << 6) | (c & 0x3F);
    }
    len = n;
    return cp;
}

std::string encode_cp(uint32_t cp) {
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

// ---- character classes -----------------------------------------------------

bool is_space_cp(uint32_t c) {
    if (c == ' ' || (c >= 0x09 && c <= 0x0D)) return true;
    if (c < 0x80) return false;
    return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_number_cp(uint32_t c) {
    if (c < 0x80) return c >= '0' && c <= '9';
    return c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) ||
           (c >= 0x966 && c <= 0x96F) || (c >= 0x2070 && c <= 0x2079) ||
           (c >= 0x2080 && c <= 0x2089) || (c >= 0x2150 && c <= 0x218B) ||
           (c >= 0x2460 && c <= 0x249B) || (c >= 0xFF10 && c <= 0xFF19);
}

// Non-letter, non-space, non-number code points above ASCII: punctuation,
// symbols, format characters, emoji. Everything else >= 0x80 counts as a
// letter, which is right for the scripts that dominate real text.
bool is_symbol_cp(uint32_t c) {
    if (c < 0x80) return false;
    if (c <= 0xBF) return c != 0xAA && c != 0xB5 && c != 0xBA && c != 0xB2 && c != 0xB3 &&
                          c != 0xB9 && !(c >= 0xBC && c <= 0xBE) && c != 0x85 && c != 0xA0;
    if (c == 0xD7 || c == 0xF7) return true;
    return (c >= 0x2010 && c <= 0x2027) || (c >= 0x2030 && c <= 0x205E) ||
           (c >= 0x2060 && c <= 0x206F) || (c >= 0x20A0 && c <= 0x20CF) ||
           (c >= 0x2100 && c <= 0x214F && c != 0x2126) || (c >= 0x2190 && c <= 0x2BFF) ||
           (c >= 0x2E00 && c <= 0x2E7F) || (c >= 0x3001 && c <= 0x3004) ||
           (c >= 0x3008 && c <= 0x3020) || (c >= 0xFE00 && c <= 0xFE0F) ||
           (c >= 0xFE30 && c <= 0xFE6F) || (c >= 0xFF01 && c <= 0xFF0F) ||
           (c >= 0xFF1A && c <= 0xFF20) || c == 0xFEFF || c == 0x200B || c == 0x200C ||
           c == 0x200D || (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0xE000 && c <= 0xF8FF);
}

enum Cls { Letter, Number, Other, Space };

Cls classify(uint32_t c) {
    if (c < 0x80) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return Letter;
        if (c >= '0' && c <= '9') return Number;
        if (is_space_cp(c)) return Space;
        return Other;
    }
    if (is_space_cp(c)) return Space;
    if (is_number_cp(c)) return Number;
    if (is_symbol_cp(c)) return Other;
    return Letter;
}

// Standard byte -> visible unicode table of byte-level BPE vocabularies.
const std::array<std::string, 256>& byte_table() {
    static const std::array<std::string, 256> table = [] {
        std::array<std::string, 256> t;
        std::array<bool, 256> assigned{};
        auto put = [&](int b) { t[(size_t) b] = encode_cp((uint32_t) b); assigned[(size_t) b] = true; };
        for (int b = '!'; b <= '~'; ++b) put(b);
        for (int b = 0xA1; b <= 0xAC; ++b) put(b);
        for (int b = 0xAE; b <= 0xFF; ++b) put(b);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!assigned[(size_t) b]) t[(size_t) b] = encode_cp((uint32_t) (256 + n++));
        }
        return t;
    }();
    return table;
}

// Splits text following the GPT2 pattern:
//   's|'t|'re|'ve|'m|'ll|'d| ?L+| ?N+| ?[^\sLN]+|\s+(?!\S)|\s+
void pre_tokenize(const std::string& text, std::vector<std::string>& chunks) {
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        size_t len = 0;
        const uint32_t c = decode_cp(text, i, len);

        // contractions
        if (c == '\'') {
            static const char* const contr[] = { "s", "t", "re", "ve", "m", "ll", "d" };
            bool hit = false;
            for (const char* k : contr) {
                const size_t kl = std::char_traits<char>::length(k);
                if (text.compare(i + 1, kl, k) == 0) {
                    chunks.push_back(text.substr(i, 1 + kl));
                    i += 1 + kl;
                    hit = true;
                    break;
                }
            }
            if (hit) continue;
        }

        // optional single literal space, then a run of one class
        size_t j = i;
        size_t first_len = len;
        uint32_t first = c;
        if (c == ' ' && i + 1 < n) {
            first = decode_cp(text, i + 1, first_len);
            j = i + 1;
        }
        const Cls k = classify(first);
        if (k != Space) {
            // a leading space only applies when followed by a non-space
            size_t p = j;
            while (p < n) {
                size_t l2 = 0;
                const uint32_t cc = decode_cp(text, p, l2);
                if (classify(cc) != k) break;
                p += l2;
            }
            chunks.push_back(text.substr(i, p - i));
            i = p;
            continue;
        }

        // whitespace run: \s+(?!\S) then \s+
        size_t p = i;
        std::vector<size_t> ends; // end offset after each whitespace code point
        while (p < n) {
            size_t l2 = 0;
            const uint32_t cc = decode_cp(text, p, l2);
            if (classify(cc) != Space) break;
            p += l2;
            ends.push_back(p);
        }
        if (p >= n) {                       // run reaches the end of text
            chunks.push_back(text.substr(i, p - i));
            i = p;
        } else if (ends.size() > 1) {       // leave the last space for the next token
            const size_t e = ends[ends.size() - 2];
            chunks.push_back(text.substr(i, e - i));
            i = e;
        } else {                            // single space before a non-space
            chunks.push_back(text.substr(i, p - i));
            i = p;
        }
    }
}

bool is_ws_byte(unsigned char c) { return c == ' ' || (c >= 0x09 && c <= 0x0D); }

}

bool HfBpeTokenizer::load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return load_json(content);
}

bool HfBpeTokenizer::load_json(const std::string& json_text) {
    djson::Value root;
    djson::Parser parser(json_text);
    if (!parser.parse(root) || !root.is_obj()) return false;

    const djson::Value* model = root.find("model");
    if (!model || !model->is_obj()) return false;
    const djson::Value* type = model->find("type");
    if (type && type->is_str() && type->str != "BPE") return false;
    const djson::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->is_obj()) return false;

    vocab_.clear();
    merge_rank_.clear();
    added_.clear();
    added_by_first_.clear();
    cache_.clear();
    n_vocab_ = 0;

    vocab_.reserve(vocab->obj.size());
    int32_t max_id = -1;
    for (const auto& kv : vocab->obj) {
        const int32_t id = (int32_t) kv.second.num;
        vocab_[kv.first] = id;
        max_id = std::max(max_id, id);
    }

    const djson::Value* merges = model->find("merges");
    if (merges && merges->is_arr()) {
        merge_rank_.reserve(merges->arr.size());
        int32_t rank = 0;
        for (const auto& m : merges->arr) {
            if (m.is_str()) {
                merge_rank_.emplace(m.str, rank);
            } else if (m.is_arr() && m.arr.size() == 2) {
                merge_rank_.emplace(m.arr[0].str + " " + m.arr[1].str, rank);
            }
            ++rank;
        }
    }

    const djson::Value* added = root.find("added_tokens");
    if (added && added->is_arr()) {
        for (const auto& a : added->arr) {
            if (!a.is_obj()) continue;
            Added t;
            const djson::Value* v = a.find("content");
            if (!v || !v->is_str() || v->str.empty()) continue;
            t.content = v->str;
            v = a.find("id");
            if (!v) continue;
            t.id = (int32_t) v->num;
            v = a.find("lstrip");
            t.lstrip = v && v->type == djson::Value::Bool && v->b;
            v = a.find("rstrip");
            t.rstrip = v && v->type == djson::Value::Bool && v->b;
            max_id = std::max(max_id, t.id);
            added_.push_back(std::move(t));
        }
        std::stable_sort(added_.begin(), added_.end(),
                         [](const Added& a, const Added& b) { return a.content.size() > b.content.size(); });
        for (size_t i = 0; i < added_.size(); ++i) {
            added_by_first_[(unsigned char) added_[i].content[0]].push_back(i);
        }
    }

    n_vocab_ = (size_t) (max_id + 1);
    cls_id_ = token_id("[CLS]");
    sep_id_ = token_id("[SEP]");
    mask_id_ = token_id("[MASK]");
    pad_id_ = token_id("[PAD]");
    return !vocab_.empty() && cls_id_ >= 0 && sep_id_ >= 0 && mask_id_ >= 0;
}

int32_t HfBpeTokenizer::token_id(const std::string& piece) const {
    for (const auto& a : added_) {
        if (a.content == piece) return a.id;
    }
    auto it = vocab_.find(piece);
    return it == vocab_.end() ? -1 : it->second;
}

void HfBpeTokenizer::bpe_chunk(const std::string& chunk, std::vector<int32_t>& out) const {
    auto hit = cache_.find(chunk);
    if (hit != cache_.end()) {
        out.insert(out.end(), hit->second.begin(), hit->second.end());
        return;
    }

    const auto& tbl = byte_table();
    std::vector<std::string> sym;
    sym.reserve(chunk.size());
    for (unsigned char b : chunk) sym.push_back(tbl[b]);

    for (;;) {
        int32_t best = -1;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_rank_.end() && (best < 0 || it->second < best)) {
                best = it->second;
                at = i;
            }
        }
        if (best < 0) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + (long) at + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(sym.size());
    for (const auto& s : sym) {
        auto it = vocab_.find(s);
        if (it != vocab_.end()) ids.push_back(it->second);
    }
    if (cache_.size() < 100000) cache_.emplace(chunk, ids);
    out.insert(out.end(), ids.begin(), ids.end());
}

void HfBpeTokenizer::encode_plain(const std::string& text, std::vector<int32_t>& out) const {
    if (text.empty()) return;
    std::vector<std::string> chunks;
    pre_tokenize(text, chunks);
    for (const auto& c : chunks) bpe_chunk(c, out);
}

std::vector<int32_t> HfBpeTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    if (!ready()) return out;
    if (added_.empty()) {
        encode_plain(text, out);
        return out;
    }

    size_t pos = 0;
    size_t plain_start = 0;
    while (pos < text.size()) {
        const Added* hit = nullptr;
        auto cand = added_by_first_.find((unsigned char) text[pos]);
        if (cand != added_by_first_.end()) {
            for (size_t idx : cand->second) {
                const Added& a = added_[idx];
                if (text.compare(pos, a.content.size(), a.content) == 0) { hit = &a; break; }
            }
        }
        if (!hit) { ++pos; continue; }

        size_t plain_end = pos;
        if (hit->lstrip) {
            while (plain_end > plain_start && is_ws_byte((unsigned char) text[plain_end - 1])) --plain_end;
        }
        if (plain_end > plain_start) {
            encode_plain(text.substr(plain_start, plain_end - plain_start), out);
        }
        out.push_back(hit->id);
        pos += hit->content.size();
        if (hit->rstrip) {
            while (pos < text.size() && is_ws_byte((unsigned char) text[pos])) ++pos;
        }
        plain_start = pos;
    }
    if (plain_start < text.size()) encode_plain(text.substr(plain_start), out);
    return out;
}

}
