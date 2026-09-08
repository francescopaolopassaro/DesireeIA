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

// Tabella standard GPT2 byte<->unicode "visibile": ogni byte 0..255 e'
// mappato su un codepoint che, quando ricodificato UTF-8, e' il carattere
// usato nei pezzi del vocabolario BPE nei file GGUF.
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

// Inversa di map_bytes: dai caratteri "visibili" del vocabolario BPE ai
// byte reali. Serve in detokenizzazione — senza, i pezzi tornavano grezzi
// e nell'output comparivano letteralmente i marcatori GPT2 (Ġ per lo
// spazio, Ċ per il newline) invece del testo vero.
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
            // Non e' un carattere della mappa GPT2 (es. un token speciale
            // come <|im_end|>): lo si lascia com'e'.
            out += ch;
        }
        i += len;
    }
    return out;
}

bool is_ascii_alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
bool is_space(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// Pre-tokenizzazione approssimata del pattern GPT2: gestisce contrazioni
// inglesi comuni, poi raggruppa lettere / cifre / altro / spazi in chunk
// separati, con lo spazio iniziale opzionale attaccato al chunk successivo
// (come nel pattern originale). Categorie Unicode complete (\p{L}, \p{N})
// non sono implementate: ogni byte >=0x80 (parte di sequenza UTF-8
// multi-byte) e' trattato come "lettera" per approssimazione (gap noto).
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
            if (j == i) j++; // garantisce progresso su byte imprevisti
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

bool BpeTokenizer::piece(int32_t id, std::string& out) const {
    if (id < 0 || (size_t) id >= id_to_piece_.size()) return false;
    // Detokenizzazione: inversa della mappa byte->unicode "visibile" di
    // GPT2 applicata in encode(). Senza, l'output conteneva letteralmente
    // i marcatori (Ġ al posto dello spazio, Ċ al posto del newline).
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
// Applica la pipeline BPE ordinaria (pre-tokenizzazione -> mappa byte ->
// merge -> lookup) a un tratto di testo che NON contiene token speciali.
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

    // Scansione lineare: a ogni posizione si prova il token speciale piu'
    // lungo che combacia (special_tokens_ e' ordinata per lunghezza
    // decrescente). Il testo semplice fra due token speciali (o prima del
    // primo/dopo l'ultimo) si accumula e passa dalla pipeline BPE
    // ordinaria solo quando si chiude il tratto — non carattere per
    // carattere, altrimenti si perderebbero i merge fra byte adiacenti.
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
                break; // ordinata per lunghezza decrescente: il primo e' il piu' lungo
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
