// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "memory/context_memory.h"

#include <brotli/decode.h>
#include <brotli/encode.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

namespace desireeia {

namespace {

constexpr uint32_t CHUNK_LINES = 40;
constexpr uint32_t PREVIEW_LINES = 20;
constexpr size_t MAX_OUTLINE_ENTRIES = 60;
constexpr size_t MAX_LINE_BYTES = 2000;
constexpr uint32_t MAX_READ_LINES = 400;
constexpr size_t MAX_READ_CHARS = 12000;
constexpr uint32_t MAX_SEARCH_RESULTS = 8;
constexpr int BROTLI_QUALITY = 5;  // good ratio on text at stream-ingest speed; 11 is far slower
constexpr double VECTOR_WEIGHT = 0.5;
constexpr uint32_t META_MAGIC = 0x4D454D44;  // "DMEM"
constexpr uint32_t META_FORMAT = 2;
constexpr uint32_t FLAG_CRLF = 1;       // source file uses \r\n line endings
constexpr uint32_t FLAG_FINAL_EOL = 2;  // source file ends with a line ending

#pragma pack(push, 1)
struct ChunkRec {
    uint32_t start_line;
    uint32_t end_line;
    uint32_t n_tokens;
    uint32_t n_terms;
    uint64_t data_off;
    uint32_t data_len;
    uint32_t raw_len;
    uint64_t terms_off;
};
struct TermRec {
    uint32_t hash;
    uint32_t tf;
};
#pragma pack(pop)
static_assert(sizeof(ChunkRec) == 40, "ChunkRec is an on-disk record");
static_assert(sizeof(TermRec) == 8, "TermRec is an on-disk record");

struct Meta {
    uint32_t version = 0;
    uint32_t total_lines = 0;
    uint64_t size_bytes = 0;
    uint64_t digest_a = 0;
    uint64_t digest_b = 0;
    uint32_t vector_dim = 0;
    std::string name;
    // File-backed items: the memory is only a working copy, the file stays
    // the source of truth (re-synced when it changes, edits written back).
    std::string source_path;  // UTF-8, empty for text/stream items
    int64_t source_mtime = 0;
    uint64_t source_size = 0;
    uint32_t flags = 0;
};

// ---- small helpers ------------------------------------------------------

struct Digest {
    // Two independent FNV-1a 64 streams: change detection only, not crypto.
    uint64_t a = 0xcbf29ce484222325ULL;
    uint64_t b = 0x84222325cbf29ce4ULL;
    void update(const char* p, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            const auto c = static_cast<uint8_t>(p[i]);
            a = (a ^ c) * 0x100000001b3ULL;
            b = (b ^ c) * 0x100000001b3ULL ^ (b >> 29);
        }
    }
};

uint32_t hash32(const std::string& s) {
    uint32_t h = 2166136261u;
    for (unsigned char c : s) h = (h ^ c) * 16777619u;
    return h;
}

bool valid_session(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (unsigned char c : id) {
        if (!(std::isalnum(c) || c == '_' || c == '-')) return false;
    }
    return true;
}

bool valid_handle(const std::string& h) {
    if (h.size() != 19 || h.compare(0, 3, "sp_") != 0) return false;
    for (size_t i = 3; i < h.size(); ++i) {
        if (!std::isxdigit(static_cast<unsigned char>(h[i])) || std::isupper(static_cast<unsigned char>(h[i]))) return false;
    }
    return true;
}

// Cut at MAX_LINE_BYTES without splitting a UTF-8 sequence.
std::string clip_line(std::string line) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
    if (line.size() <= MAX_LINE_BYTES) return line;
    size_t cut = MAX_LINE_BYTES;
    while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) --cut;
    line.resize(cut);
    line += " [...line truncated]";
    return line;
}

// Words of >= 2 chars: ASCII alnum/_ lowercased, any non-ASCII byte kept as
// part of a word so accented and non-Latin text still tokenizes.
void tokenize_words(const std::string& text, std::vector<std::string>& out) {
    std::string cur;
    auto flush = [&] {
        if (cur.size() >= 2) out.push_back(cur);
        cur.clear();
    };
    for (unsigned char c : text) {
        if (c >= 0x80 || std::isalnum(c) || c == '_') {
            cur.push_back(static_cast<char>(c < 0x80 ? std::tolower(c) : c));
        } else {
            flush();
        }
    }
    flush();
}

bool is_outline_line(const std::string& line) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) ++i;
    if (i < line.size() && line[i] == '#') {
        size_t j = i;
        while (j < line.size() && line[j] == '#') ++j;
        return j - i <= 6 && j < line.size() && line[j] == ' ' && j + 1 < line.size();
    }
    static const std::unordered_set<std::string> modifiers = {
        "export", "public", "private", "protected", "internal", "static", "async", "abstract", "sealed", "partial"};
    static const std::unordered_set<std::string> keywords = {
        "def", "class", "function", "interface", "struct", "enum", "namespace", "record", "fn", "func", "impl", "trait"};
    for (int guard = 0; guard < 8; ++guard) {
        size_t j = i;
        while (j < line.size() && (std::isalnum(static_cast<unsigned char>(line[j])) || line[j] == '_')) ++j;
        if (j == i || j >= line.size() || line[j] != ' ') return false;
        const std::string word = line.substr(i, j - i);
        if (keywords.count(word)) {
            const unsigned char next = j + 1 < line.size() ? static_cast<unsigned char>(line[j + 1]) : 0;
            return std::isalpha(next) || next == '_';
        }
        if (!modifiers.count(word)) return false;
        i = j + 1;
    }
    return false;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::string q(const std::string& s) { return "\"" + json_escape(s) + "\""; }

std::string read_all(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_all_atomic(const fs::path& p, const std::string& data) {
    fs::path tmp = p;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("context memory: cannot write " + tmp.string());
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    fs::rename(tmp, p);  // a reader never sees a half-written file
}

template <class T>
void put_pod(std::string& s, const T& v) { s.append(reinterpret_cast<const char*>(&v), sizeof(T)); }

template <class T>
bool get_pod(const std::string& s, size_t& pos, T& v) {
    if (pos + sizeof(T) > s.size()) return false;
    std::memcpy(&v, s.data() + pos, sizeof(T));
    pos += sizeof(T);
    return true;
}

// ---- one stored item ----------------------------------------------------

class Item {
public:
    explicit Item(fs::path dir)
        : dir_(std::move(dir)), data_(dir_ / "data.br"), chunks_(dir_ / "chunks.bin"),
          terms_(dir_ / "terms.bin"), vectors_(dir_ / "vectors.f32"),
          outline_(dir_ / "outline.txt"), meta_(dir_ / "meta.bin") {}

    bool exists() const { return fs::is_regular_file(meta_); }

    bool load_meta(Meta& m) const {
        const std::string s = read_all(meta_);
        size_t pos = 0;
        uint32_t magic = 0, fmt = 0, name_len = 0;
        if (!get_pod(s, pos, magic) || magic != META_MAGIC || !get_pod(s, pos, fmt) || fmt != META_FORMAT) return false;
        if (!get_pod(s, pos, m.version) || !get_pod(s, pos, m.total_lines) || !get_pod(s, pos, m.size_bytes) ||
            !get_pod(s, pos, m.digest_a) || !get_pod(s, pos, m.digest_b) || !get_pod(s, pos, m.vector_dim) ||
            !get_pod(s, pos, name_len) || pos + name_len > s.size()) return false;
        m.name.assign(s.data() + pos, name_len);
        pos += name_len;
        uint32_t src_len = 0;
        if (!get_pod(s, pos, m.source_mtime) || !get_pod(s, pos, m.source_size) || !get_pod(s, pos, m.flags) ||
            !get_pod(s, pos, src_len) || pos + src_len > s.size()) return false;
        m.source_path.assign(s.data() + pos, src_len);
        return true;
    }

    void save_meta(const Meta& m) const {
        std::string s;
        put_pod(s, META_MAGIC);
        put_pod(s, META_FORMAT);
        put_pod(s, m.version);
        put_pod(s, m.total_lines);
        put_pod(s, m.size_bytes);
        put_pod(s, m.digest_a);
        put_pod(s, m.digest_b);
        put_pod(s, m.vector_dim);
        put_pod(s, static_cast<uint32_t>(m.name.size()));
        s += m.name;
        put_pod(s, m.source_mtime);
        put_pod(s, m.source_size);
        put_pod(s, m.flags);
        put_pod(s, static_cast<uint32_t>(m.source_path.size()));
        s += m.source_path;
        write_all_atomic(meta_, s);
    }

    void reset() {
        std::error_code ec;
        fs::remove_all(dir_, ec);
        fs::create_directories(dir_);
        for (const auto& p : {data_, chunks_, terms_, vectors_, outline_}) {
            std::ofstream(p, std::ios::binary | std::ios::trunc);
        }
    }

    std::vector<ChunkRec> chunks() const {
        const std::string s = read_all(chunks_);
        std::vector<ChunkRec> out(s.size() / sizeof(ChunkRec));
        if (!out.empty()) std::memcpy(out.data(), s.data(), out.size() * sizeof(ChunkRec));
        return out;
    }

    std::vector<std::string> block(const ChunkRec& c) const {
        std::ifstream in(data_, std::ios::binary);
        std::string packed(c.data_len, '\0');
        in.seekg(static_cast<std::streamoff>(c.data_off));
        in.read(packed.data(), c.data_len);
        std::string raw(c.raw_len, '\0');
        size_t decoded = raw.size();
        if (BrotliDecoderDecompress(packed.size(), reinterpret_cast<const uint8_t*>(packed.data()), &decoded,
                                    reinterpret_cast<uint8_t*>(raw.data())) != BROTLI_DECODER_RESULT_SUCCESS) {
            throw std::runtime_error("context memory: corrupt block in " + data_.string());
        }
        raw.resize(decoded);
        std::vector<std::string> lines;
        size_t start = 0;
        for (size_t i = 0; i <= raw.size(); ++i) {
            if (i == raw.size() || raw[i] == '\n') {
                lines.emplace_back(raw, start, i - start);
                start = i + 1;
            }
        }
        return lines;
    }

    // Append lines into compressed blocks. A trailing partial block from an
    // earlier append is reopened and refilled, so a stream of small appends
    // still ends up as full-size blocks.
    template <class LineSource>
    void append(LineSource&& next_line, Meta& meta, const std::function<std::vector<float>(const std::string&)>& embed) {
        std::vector<ChunkRec> chunks = this->chunks();
        std::vector<std::string> pending;
        if (!chunks.empty() && chunks.back().end_line - chunks.back().start_line + 1 < CHUNK_LINES) {
            const ChunkRec last = chunks.back();
            pending = block(last);
            chunks.pop_back();
            fs::resize_file(data_, last.data_off);
            fs::resize_file(terms_, last.terms_off);
            if (meta.vector_dim && fs::file_size(vectors_) > chunks.size() * meta.vector_dim * sizeof(float)) {
                fs::resize_file(vectors_, chunks.size() * meta.vector_dim * sizeof(float));
            }
            drop_outline_from(last.start_line);
        }
        uint32_t next = chunks.empty() ? 1 : chunks.back().end_line + 1;
        size_t outline_count = count_outline();

        std::ofstream data(data_, std::ios::binary | std::ios::app);
        std::ofstream terms(terms_, std::ios::binary | std::ios::app);
        std::ofstream vecs(vectors_, std::ios::binary | std::ios::app);
        std::ofstream outline(outline_, std::ios::binary | std::ios::app);
        uint64_t data_off = fs::file_size(data_);
        uint64_t terms_off = fs::file_size(terms_);

        auto flush = [&](const std::vector<std::string>& lines) {
            std::string text;
            for (size_t i = 0; i < lines.size(); ++i) {
                if (i) text.push_back('\n');
                text += lines[i];
                if (outline_count < MAX_OUTLINE_ENTRIES && is_outline_line(lines[i])) {
                    outline << (next + i) << '\t' << lines[i].substr(0, 160) << '\n';
                    ++outline_count;
                }
            }
            std::vector<uint8_t> packed(BrotliEncoderMaxCompressedSize(text.size()) + 16);
            size_t packed_size = packed.size();
            if (!BrotliEncoderCompress(BROTLI_QUALITY, BROTLI_DEFAULT_WINDOW, BROTLI_MODE_TEXT, text.size(),
                                       reinterpret_cast<const uint8_t*>(text.data()), &packed_size, packed.data())) {
                throw std::runtime_error("context memory: brotli compression failed");
            }
            data.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed_size));

            std::vector<std::string> words;
            tokenize_words(text, words);
            std::unordered_map<uint32_t, uint32_t> tf;
            for (const auto& w : words) ++tf[hash32(w)];
            for (const auto& [h, n] : tf) {
                const TermRec rec{h, n};
                terms.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
            }

            ChunkRec c{};
            c.start_line = next;
            c.end_line = next + static_cast<uint32_t>(lines.size()) - 1;
            c.n_tokens = static_cast<uint32_t>(words.size());
            c.n_terms = static_cast<uint32_t>(tf.size());
            c.data_off = data_off;
            c.data_len = static_cast<uint32_t>(packed_size);
            c.raw_len = static_cast<uint32_t>(text.size());
            c.terms_off = terms_off;
            chunks.push_back(c);
            data_off += packed_size;
            terms_off += tf.size() * sizeof(TermRec);
            next = c.end_line + 1;

            if (embed) {
                const std::vector<float> v = embed(text);
                // Vectors only stay usable while every block has one of the
                // same size; a gap would shift every later row.
                if (!v.empty() && meta.vector_dim == 0 && chunks.size() == 1) {
                    meta.vector_dim = static_cast<uint32_t>(v.size());
                }
                if (!v.empty() && v.size() == meta.vector_dim) {
                    vecs.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
                }
            }
        };

        std::string line;
        while (next_line(line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            pending.push_back(line);
            if (pending.size() >= CHUNK_LINES) {
                flush(pending);
                pending.clear();
            }
        }
        if (!pending.empty()) flush(pending);
        data.close();
        terms.close();
        vecs.close();
        outline.close();

        std::string raw(chunks.size() * sizeof(ChunkRec), '\0');
        if (!chunks.empty()) std::memcpy(raw.data(), chunks.data(), raw.size());
        write_all_atomic(chunks_, raw);
        meta.total_lines = chunks.empty() ? 0 : chunks.back().end_line;
        // Vectors are only trusted when there is exactly one row per block.
        if (meta.vector_dim && fs::file_size(vectors_) != chunks.size() * meta.vector_dim * sizeof(float)) {
            meta.vector_dim = 0;
            fs::resize_file(vectors_, 0);
        }
    }

    std::vector<std::pair<uint32_t, std::string>> window(uint32_t start, uint32_t count) const {
        std::vector<std::pair<uint32_t, std::string>> out;
        const uint32_t end = start + count - 1;
        for (const ChunkRec& c : chunks()) {
            if (c.end_line < start) continue;
            if (c.start_line > end) break;
            const auto lines = block(c);
            for (uint32_t n = std::max(start, c.start_line); n <= std::min(end, c.end_line); ++n) {
                out.emplace_back(n, lines[n - c.start_line]);
            }
        }
        return out;
    }

    std::vector<std::string> outline() const {
        std::vector<std::string> out;
        std::ifstream in(outline_, std::ios::binary);
        std::string line;
        while (std::getline(in, line)) out.push_back(line);
        return out;
    }

    // Hybrid ranking: BM25 normalized to [0,1] blended with cosine
    // similarity when vectors exist. -> (score, chunk)
    std::vector<std::pair<double, ChunkRec>> search(const std::string& query, const std::vector<float>& qvec,
                                                    const Meta& meta, uint32_t k) const {
        const std::vector<ChunkRec> chunks = this->chunks();
        std::vector<std::pair<double, ChunkRec>> out;
        if (chunks.empty()) return out;

        std::vector<std::string> words;
        tokenize_words(query, words);
        std::unordered_set<uint32_t> qterms;
        for (const auto& w : words) qterms.insert(hash32(w));

        std::vector<double> bm25(chunks.size(), 0.0);
        if (!qterms.empty()) {
            // Streamed from SSD in block order: RAM holds one buffered read
            // and the query's matches, never the whole index.
            std::ifstream terms(terms_, std::ios::binary);
            std::unordered_map<uint32_t, uint32_t> df;
            std::vector<std::vector<std::pair<uint32_t, uint32_t>>> hits(chunks.size());
            double total_tokens = 0;
            std::vector<TermRec> recs;
            for (size_t ci = 0; ci < chunks.size(); ++ci) {
                const ChunkRec& c = chunks[ci];
                total_tokens += c.n_tokens;
                recs.resize(c.n_terms);
                terms.seekg(static_cast<std::streamoff>(c.terms_off));
                if (!terms.read(reinterpret_cast<char*>(recs.data()), static_cast<std::streamsize>(c.n_terms * sizeof(TermRec)))) {
                    terms.clear();
                    continue;
                }
                for (const TermRec& r : recs) {
                    if (qterms.count(r.hash)) {
                        ++df[r.hash];
                        hits[ci].emplace_back(r.hash, r.tf);
                    }
                }
            }
            const double n = static_cast<double>(chunks.size());
            const double avg = std::max(1.0, total_tokens / n);
            for (size_t ci = 0; ci < chunks.size(); ++ci) {
                for (const auto& [h, tf] : hits[ci]) {
                    const double d = df[h];
                    const double idf = std::log(1.0 + (n - d + 0.5) / (d + 0.5));
                    bm25[ci] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * chunks[ci].n_tokens / avg));
                }
            }
        }
        const double top = std::max(1e-9, *std::max_element(bm25.begin(), bm25.end()));

        // One embedding row in RAM at a time, read sequentially from SSD.
        std::vector<float> sims;
        if (!qvec.empty() && meta.vector_dim == qvec.size()) {
            std::error_code ec;
            if (fs::file_size(vectors_, ec) == chunks.size() * meta.vector_dim * sizeof(float)) {
                std::ifstream vin(vectors_, std::ios::binary);
                std::vector<float> row(meta.vector_dim);
                sims.resize(chunks.size());
                for (size_t ci = 0; ci < chunks.size(); ++ci) {
                    vin.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
                    double dot = 0.0;
                    for (uint32_t d = 0; d < meta.vector_dim; ++d) dot += static_cast<double>(qvec[d]) * row[d];
                    sims[ci] = static_cast<float>(dot);
                }
            }
        }
        const double weight = sims.empty() ? 0.0 : VECTOR_WEIGHT;
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const double sim = sims.empty() ? 0.0 : sims[ci];
            const double score = (1.0 - weight) * bm25[ci] / top + weight * std::max(0.0, sim);
            if (score > 0) out.emplace_back(score, chunks[ci]);
        }
        const size_t keep = std::min<size_t>(k, out.size());
        std::partial_sort(out.begin(), out.begin() + keep, out.end(),
                          [](const auto& x, const auto& y) { return x.first > y.first; });
        out.resize(keep);
        return out;
    }

private:
    size_t count_outline() const { return outline().size(); }

    void drop_outline_from(uint32_t first_line) const {
        std::string kept;
        for (const auto& l : outline()) {
            if (std::strtoul(l.c_str(), nullptr, 10) < first_line) kept += l + "\n";
        }
        write_all_atomic(outline_, kept);
    }

    fs::path dir_, data_, chunks_, terms_, vectors_, outline_, meta_;
};

std::string stub_json(const Item& item, const std::string& handle) {
    Meta m;
    item.load_meta(m);
    std::string preview;
    for (const auto& [n, text] : item.window(1, PREVIEW_LINES)) {
        if (!preview.empty()) preview.push_back('\n');
        preview += std::to_string(n) + "\t" + clip_line(text);
    }
    std::string outline = "[";
    bool first = true;
    for (const auto& l : item.outline()) {
        if (!first) outline += ",";
        outline += q(l);
        first = false;
    }
    outline += "]";
    return "{\"spilled\":true,\"handle\":" + q(handle) + ",\"name\":" + q(m.name) +
           ",\"version\":" + std::to_string(m.version) + ",\"total_lines\":" + std::to_string(m.total_lines) +
           ",\"size_bytes\":" + std::to_string(m.size_bytes) + ",\"preview\":" + q(preview) +
           ",\"outline\":" + outline +
           ",\"note\":\"Full content is stored outside the context. Use memory_search(handle, query) to find the "
           "relevant part and memory_read(handle, offset, limit) to read a line window. Do not try to read it all "
           "at once.\"}";
}

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

std::string memory_handle_for(const std::string& name) {
    Digest d;
    d.update(name.data(), name.size());
    char buf[20];
    std::snprintf(buf, sizeof(buf), "sp_%016llx", static_cast<unsigned long long>(d.a));
    return buf;
}

struct ContextMemory::Session {
    std::mutex mu;
    std::atomic<double> last_seen{0.0};
    fs::path dir;
};

namespace {

// Holds a session for one operation; activity is stamped again on release
// so an operation longer than the TTL can't make its own session look idle.
class SessionLock {
public:
    template <class S>
    explicit SessionLock(S& s) : lk_(s.mu), stamp_([&s] { s.last_seen = now_seconds(); }) {}
    ~SessionLock() { stamp_(); }
private:
    std::lock_guard<std::mutex> lk_;
    std::function<void()> stamp_;
};

} // namespace

ContextMemory::ContextMemory(fs::path root, double ttl_seconds)
    : root_(std::move(root)), ttl_seconds_(ttl_seconds) {
    fs::create_directories(root_);
    if (ttl_seconds_ > 0) sweeper_ = std::thread([this] { sweeper_loop(); });
}

ContextMemory::~ContextMemory() {
    {
        std::lock_guard<std::mutex> lk(guard_);
        stop_ = true;
    }
    stop_cv_.notify_all();
    if (sweeper_.joinable()) sweeper_.join();
}

void ContextMemory::set_embedder(MemoryEmbedder embedder) {
    std::lock_guard<std::mutex> lk(embed_mu_);
    embedder_ = std::move(embedder);
}

std::vector<float> ContextMemory::embed(const std::string& text) {
    std::lock_guard<std::mutex> lk(embed_mu_);
    if (!embedder_) return {};
    try {
        std::vector<float> v = embedder_(text);
        if (v.empty()) embedder_ = nullptr;  // model can't embed: stop asking for every block
        return v;
    } catch (...) {
        embedder_ = nullptr;
        return {};
    }
}

std::shared_ptr<ContextMemory::Session> ContextMemory::session(const std::string& id) {
    if (!valid_session(id)) throw std::invalid_argument("invalid session id");
    std::lock_guard<std::mutex> lk(guard_);
    auto& s = sessions_[id];
    if (!s) {
        s = std::make_shared<Session>();
        s->dir = root_ / id;
    }
    s->last_seen = now_seconds();
    return s;
}

void ContextMemory::touch(const std::string& id) {
    if (valid_session(id)) session(id);
}

namespace {

// Line source over an in-memory string; "a\nb\n" yields a, b.
std::function<bool(std::string&)> text_lines(const std::string& text) {
    auto pos = std::make_shared<size_t>(0);
    return [&text, pos](std::string& line) {
        if (*pos >= text.size()) return false;
        const size_t nl = text.find('\n', *pos);
        line.assign(text, *pos, nl == std::string::npos ? std::string::npos : nl - *pos);
        *pos = nl == std::string::npos ? text.size() : nl + 1;
        return true;
    };
}

template <class LineSource>
std::string put_impl(const fs::path& dir, const std::string& name, const Digest& digest, uint64_t size_bytes,
                     LineSource&& lines, const std::function<std::vector<float>(const std::string&)>& embed) {
    const std::string handle = memory_handle_for(name);
    Item item(dir / handle);
    Meta meta;
    const bool had = item.exists() && item.load_meta(meta);
    if (!had || meta.digest_a != digest.a || meta.digest_b != digest.b || meta.name != name) {
        const uint32_t version = had ? meta.version + 1 : 1;
        item.reset();
        meta = Meta{};
        meta.name = name;
        meta.version = version;
        item.append(lines, meta, embed);
        meta.size_bytes = size_bytes;
        meta.digest_a = digest.a;
        meta.digest_b = digest.b;
        item.save_meta(meta);
    }
    return stub_json(item, handle);
}

} // namespace

namespace {

bool stat_source(const fs::path& path, int64_t& mtime, uint64_t& size) {
    std::error_code ec;
    size = fs::file_size(path, ec);
    if (ec) return false;
    const auto t = fs::last_write_time(path, ec);
    if (ec) return false;
    mtime = static_cast<int64_t>(t.time_since_epoch().count());
    return true;
}

// Stream an item to `path` atomically, restoring the source line endings.
uint64_t write_item(const Item& item, const Meta& meta, const fs::path& path) {
    fs::path tmp = path;
    tmp += ".desireeia.tmp";
    const char* eol = (meta.flags & FLAG_CRLF) ? "\r\n" : "\n";
    const size_t eol_len = (meta.flags & FLAG_CRLF) ? 2 : 1;
    const bool final_eol = meta.source_path.empty() || (meta.flags & FLAG_FINAL_EOL);
    uint64_t written = 0;
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write " + tmp.u8string());
        uint32_t n = 0;
        for (const ChunkRec& c : item.chunks()) {
            for (const auto& line : item.block(c)) {
                ++n;
                out.write(line.data(), static_cast<std::streamsize>(line.size()));
                written += line.size();
                if (n < meta.total_lines || final_eol) {
                    out.write(eol, static_cast<std::streamsize>(eol_len));
                    written += eol_len;
                }
            }
        }
    }
    fs::rename(tmp, path);
    return written;
}

} // namespace

std::string ContextMemory::put_file_locked(const fs::path& dir, const std::string& name, const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.u8string());
    Digest digest;
    std::vector<char> buf(1 << 20);
    uint64_t size = 0;
    uint32_t flags = 0;
    char last = 0;
    bool first_block = true;
    while (in.read(buf.data(), static_cast<std::streamsize>(buf.size())) || in.gcount() > 0) {
        const size_t got = static_cast<size_t>(in.gcount());
        if (first_block) {
            const std::string head(buf.data(), got);
            const size_t nl = head.find('\n');
            if (nl != std::string::npos && nl > 0 && head[nl - 1] == '\r') flags |= FLAG_CRLF;
            first_block = false;
        }
        digest.update(buf.data(), got);
        size += got;
        last = buf[got - 1];
    }
    if (last == '\n') flags |= FLAG_FINAL_EOL;
    in.clear();
    in.seekg(0);
    auto next = [&in](std::string& line) { return static_cast<bool>(std::getline(in, line)); };
    auto embed = [this](const std::string& t) { return this->embed(t); };
    put_impl(dir, name, digest, size, next, embed);

    const std::string handle = memory_handle_for(name);
    Item item(dir / handle);
    Meta meta;
    item.load_meta(meta);
    meta.source_path = fs::absolute(path).u8string();
    meta.flags = flags;
    stat_source(path, meta.source_mtime, meta.source_size);
    item.save_meta(meta);
    return stub_json(item, handle);
}

// Re-sync a file-backed item whose source changed on disk. Returns true
// when the stored copy was rebuilt (version bumped).
bool ContextMemory::refresh_locked(const fs::path& dir, const std::string& handle) {
    Item item(dir / handle);
    Meta meta;
    if (!(item.exists() && item.load_meta(meta)) || meta.source_path.empty()) return false;
    int64_t mtime = 0;
    uint64_t size = 0;
    const fs::path src = fs::u8path(meta.source_path);
    if (!stat_source(src, mtime, size)) return false;  // source gone: keep the last copy
    if (mtime == meta.source_mtime && size == meta.source_size) return false;
    const uint32_t before = meta.version;
    put_file_locked(dir, meta.name, src);
    Meta after;
    item.load_meta(after);
    return after.version != before;
}

std::string ContextMemory::put_file(const std::string& sid, const std::string& name, const fs::path& path) {
    auto s = session(sid);
    SessionLock lk(*s);
    return put_file_locked(s->dir, name, path);
}

std::string ContextMemory::put_text(const std::string& sid, const std::string& name, const std::string& text) {
    auto s = session(sid);
    SessionLock lk(*s);
    Digest digest;
    digest.update(text.data(), text.size());
    return put_impl(s->dir, name, digest, text.size(), text_lines(text),
                    [this](const std::string& t) { return this->embed(t); });
}

std::string ContextMemory::replace_lines(const std::string& sid, const std::string& handle, uint32_t first,
                                         uint32_t last, const std::string& text) {
    if (!valid_handle(handle)) throw std::invalid_argument("invalid handle");
    auto s = session(sid);
    SessionLock lk(*s);
    if (refresh_locked(s->dir, handle)) {
        // The real file changed since the model looked at it: its line
        // numbers may point at different code now, so don't edit blindly.
        return "{\"error\":\"source file changed on disk; re-read it before editing\",\"stub\":" +
               stub_json(Item(s->dir / handle), handle) + "}";
    }
    Item item(s->dir / handle);
    Meta meta;
    if (!(item.exists() && item.load_meta(meta))) {
        return "{\"error\":" + q("unknown handle " + handle) + "}";
    }
    if (first < 1 || first > meta.total_lines + 1 || last + 1 < first || last > meta.total_lines) {
        return "{\"error\":\"line range out of bounds\"}";
    }
    // Rebuild into a sibling directory streaming block by block, so the
    // edit never needs the whole file in RAM, then swap it in.
    fs::path tmp_dir = s->dir / (handle + ".edit");
    std::error_code ec;
    fs::remove_all(tmp_dir, ec);
    Item fresh(tmp_dir);
    fresh.reset();
    const std::vector<ChunkRec> chunks = item.chunks();
    size_t ci = 0;
    std::vector<std::string> cur;
    size_t cur_pos = 0;
    uint32_t cur_line = 1;
    bool inserted = false;
    auto replacement = text_lines(text);
    auto next = [&](std::string& line) -> bool {
        for (;;) {
            if (!inserted && cur_line == first) {
                if (replacement(line)) return true;
                inserted = true;
                // skip the replaced range of the original
                while (cur_line <= last) {
                    if (cur_pos >= cur.size()) {
                        if (ci >= chunks.size()) break;
                        cur = item.block(chunks[ci++]);
                        cur_pos = 0;
                    }
                    ++cur_pos;
                    ++cur_line;
                }
                continue;
            }
            if (cur_pos >= cur.size()) {
                if (ci >= chunks.size()) {
                    if (!inserted && cur_line == first) continue;
                    return false;
                }
                cur = item.block(chunks[ci++]);
                cur_pos = 0;
                continue;
            }
            line = cur[cur_pos++];
            ++cur_line;
            return true;
        }
    };
    Meta m2;
    m2.name = meta.name;
    m2.version = meta.version + 1;
    fresh.append(next, m2, [this](const std::string& t) { return this->embed(t); });
    Digest d;
    d.a = meta.digest_a ^ 0x5bd1e9955bd1e995ULL;
    d.b = meta.digest_b;
    d.update(text.data(), text.size());
    m2.digest_a = d.a;
    m2.digest_b = d.b;
    m2.source_path = meta.source_path;
    m2.flags = meta.flags;
    m2.size_bytes = meta.size_bytes + text.size();  // approximate until written back / exported
    fresh.save_meta(m2);
    fs::remove_all(s->dir / handle, ec);
    fs::rename(tmp_dir, s->dir / handle);
    Item updated(s->dir / handle);
    if (!m2.source_path.empty()) {
        // Write-through: the edit lands in the real file, the memory copy
        // only mirrors it.
        const fs::path src = fs::u8path(m2.source_path);
        m2.size_bytes = write_item(updated, m2, src);
        stat_source(src, m2.source_mtime, m2.source_size);
        updated.save_meta(m2);
    }
    // A compact confirmation with just the edited region (plus 2 lines of
    // context), not the whole stub again: measured with a 4B model, getting
    // the full stub back after an edit read as "nothing happened" and the
    // model re-applied the same edit every hop, growing the context ~1.7k
    // tokens per turn.
    uint32_t new_lines = 0;
    {
        auto count = text_lines(text);
        std::string tmp;
        while (count(tmp)) ++new_lines;
    }
    const uint32_t from = first > 2 ? first - 2 : 1;
    std::string window;
    for (const auto& [n, line] : updated.window(from, new_lines + 4)) {
        if (!window.empty()) window.push_back('\n');
        window += std::to_string(n) + "\t" + clip_line(line);
    }
    return "{\"ok\":true,\"handle\":" + q(handle) + ",\"version\":" + std::to_string(m2.version) +
           ",\"total_lines\":" + std::to_string(m2.total_lines) + ",\"replaced\":[" + std::to_string(first) + "," +
           std::to_string(last) + "],\"inserted_lines\":" + std::to_string(new_lines) +
           ",\"written_to_source\":" + (m2.source_path.empty() ? "false" : "true") + ",\"window\":" + q(window) +
           ",\"note\":\"Edit applied. Do not repeat it; verify with memory_read only if needed.\"}";
}

std::string ContextMemory::export_to(const std::string& sid, const std::string& handle, const fs::path& path) {
    if (!valid_handle(handle)) throw std::invalid_argument("invalid handle");
    auto s = session(sid);
    SessionLock lk(*s);
    Item item(s->dir / handle);
    Meta meta;
    if (!(item.exists() && item.load_meta(meta))) {
        return "{\"error\":" + q("unknown handle " + handle) + "}";
    }
    const uint64_t written = write_item(item, meta, path);
    return "{\"handle\":" + q(handle) + ",\"path\":" + q(path.string()) + ",\"bytes_written\":" +
           std::to_string(written) + ",\"version\":" + std::to_string(meta.version) + "}";
}

std::string ContextMemory::append(const std::string& sid, const std::string& name, const std::string& text) {
    auto s = session(sid);
    SessionLock lk(*s);
    const std::string handle = memory_handle_for(name);
    Item item(s->dir / handle);
    Meta meta;
    if (!(item.exists() && item.load_meta(meta))) {
        item.reset();
        meta = Meta{};
        meta.name = name;
    }
    item.append(text_lines(text), meta, [this](const std::string& t) { return this->embed(t); });
    Digest d;
    d.a = meta.digest_a ^ 0x9e3779b97f4a7c15ULL;
    d.b = meta.digest_b;
    d.update(text.data(), text.size());
    meta.digest_a = d.a;
    meta.digest_b = d.b;
    meta.size_bytes += text.size();
    meta.version += 1;
    item.save_meta(meta);
    return stub_json(item, handle);
}

std::string ContextMemory::read(const std::string& sid, const std::string& handle, uint32_t offset, uint32_t limit) {
    if (!valid_handle(handle)) throw std::invalid_argument("invalid handle");
    auto s = session(sid);
    SessionLock lk(*s);
    refresh_locked(s->dir, handle);
    Item item(s->dir / handle);
    Meta meta;
    if (!(item.exists() && item.load_meta(meta))) {
        return "{\"error\":" + q("unknown handle " + handle + " (session expired or wrong id)") + "}";
    }
    offset = std::max<uint32_t>(1, offset);
    limit = std::clamp<uint32_t>(limit ? limit : 200, 1, MAX_READ_LINES);
    std::string content;
    uint32_t last = offset - 1;
    for (const auto& [n, text] : item.window(offset, limit)) {
        std::string line = std::to_string(n) + "\t" + clip_line(text);
        if (!content.empty() && content.size() + line.size() > MAX_READ_CHARS) break;
        if (!content.empty()) content.push_back('\n');
        content += line;
        last = n;
    }
    const std::string next = last < meta.total_lines ? std::to_string(last + 1) : "null";
    return "{\"handle\":" + q(handle) + ",\"version\":" + std::to_string(meta.version) + ",\"content\":" + q(content) +
           ",\"start_line\":" + std::to_string(offset) + ",\"end_line\":" + std::to_string(last) +
           ",\"total_lines\":" + std::to_string(meta.total_lines) + ",\"next_offset\":" + next + "}";
}

std::string ContextMemory::search(const std::string& sid, const std::string& query, const std::string& handle, uint32_t k) {
    if (!handle.empty() && !valid_handle(handle)) throw std::invalid_argument("invalid handle");
    auto s = session(sid);
    k = std::clamp<uint32_t>(k ? k : 5, 1, MAX_SEARCH_RESULTS);
    const std::vector<float> qvec = embed(query);
    SessionLock lk(*s);

    std::vector<std::string> handles;
    if (!handle.empty()) {
        handles.push_back(handle);
    } else if (fs::is_directory(s->dir)) {
        for (const auto& e : fs::directory_iterator(s->dir)) {
            const std::string h = e.path().filename().string();
            if (valid_handle(h)) handles.push_back(h);
        }
    }
    struct Hit { double score; std::string handle; ChunkRec chunk; };
    std::vector<Hit> hits;
    for (const auto& h : handles) {
        refresh_locked(s->dir, h);
        Item item(s->dir / h);
        Meta meta;
        if (!(item.exists() && item.load_meta(meta))) continue;
        for (const auto& [score, c] : item.search(query, qvec, meta, k)) hits.push_back({score, h, c});
    }
    std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.score > b.score; });
    if (hits.size() > k) hits.resize(k);

    std::string out = "{\"results\":[";
    for (size_t i = 0; i < hits.size(); ++i) {
        Item item(s->dir / hits[i].handle);
        std::string snippet;
        const ChunkRec& c = hits[i].chunk;
        for (const auto& [n, text] : item.window(c.start_line, c.end_line - c.start_line + 1)) {
            if (!snippet.empty()) snippet.push_back('\n');
            snippet += std::to_string(n) + "\t" + clip_line(text);
            if (snippet.size() > MAX_READ_CHARS / k) break;
        }
        char score[32];
        std::snprintf(score, sizeof(score), "%.3f", hits[i].score);
        if (i) out += ",";
        out += "{\"handle\":" + q(hits[i].handle) + ",\"start_line\":" + std::to_string(c.start_line) +
               ",\"end_line\":" + std::to_string(c.end_line) + ",\"score\":" + score + ",\"snippet\":" + q(snippet) + "}";
    }
    return out + "]}";
}

std::string ContextMemory::list(const std::string& sid) {
    auto s = session(sid);
    SessionLock lk(*s);
    std::string out = "{\"items\":[";
    bool first = true;
    if (fs::is_directory(s->dir)) {
        for (const auto& e : fs::directory_iterator(s->dir)) {
            const std::string h = e.path().filename().string();
            Item item(e.path());
            Meta m;
            if (!valid_handle(h) || !item.exists() || !item.load_meta(m)) continue;
            std::error_code ec;
            const auto stored = fs::file_size(e.path() / "data.br", ec);
            if (!first) out += ",";
            first = false;
            out += "{\"handle\":" + q(h) + ",\"name\":" + q(m.name) + ",\"version\":" + std::to_string(m.version) +
                   ",\"total_lines\":" + std::to_string(m.total_lines) + ",\"size_bytes\":" + std::to_string(m.size_bytes) +
                   ",\"stored_bytes\":" + std::to_string(ec ? 0 : stored) + "}";
        }
    }
    return out + "]}";
}

void ContextMemory::drop_session(const std::string& id) {
    auto s = session(id);
    {
        SessionLock lk(*s);
        std::error_code ec;
        fs::remove_all(s->dir, ec);
    }
    std::lock_guard<std::mutex> lk(guard_);
    sessions_.erase(id);
}

int ContextMemory::sweep() {
    if (ttl_seconds_ <= 0 || !fs::is_directory(root_)) return 0;
    const double now = now_seconds();
    std::vector<std::string> expired;
    for (const auto& e : fs::directory_iterator(root_)) {
        if (!e.is_directory()) continue;
        const std::string id = e.path().filename().string();
        double idle;
        {
            std::lock_guard<std::mutex> lk(guard_);
            auto it = sessions_.find(id);
            if (it != sessions_.end()) {
                idle = now - it->second->last_seen.load();
            } else {
                // Left from a previous run: judge it by its last write.
                std::error_code ec;
                const auto mtime = fs::last_write_time(e.path(), ec);
                idle = ec ? 1e18
                          : std::chrono::duration<double>(fs::file_time_type::clock::now() - mtime).count();
            }
        }
        if (idle > ttl_seconds_) expired.push_back(id);
    }
    int removed = 0;
    for (const auto& id : expired) {
        std::shared_ptr<Session> s;
        {
            std::lock_guard<std::mutex> lk(guard_);
            auto it = sessions_.find(id);
            if (it != sessions_.end()) s = it->second;
        }
        if (s) {
            // Re-check under the session lock: it may have been used since.
            std::lock_guard<std::mutex> lk(s->mu);
            if (now_seconds() - s->last_seen.load() <= ttl_seconds_) continue;
            std::error_code ec;
            fs::remove_all(s->dir, ec);
            std::lock_guard<std::mutex> g(guard_);
            sessions_.erase(id);
        } else {
            std::error_code ec;
            fs::remove_all(root_ / id, ec);
        }
        ++removed;
    }
    return removed;
}

void ContextMemory::sweeper_loop() {
    const auto interval = std::chrono::duration<double>(std::clamp(ttl_seconds_ / 4.0, 5.0, 60.0));
    std::unique_lock<std::mutex> lk(guard_);
    while (!stop_) {
        stop_cv_.wait_for(lk, interval, [this] { return stop_; });
        if (stop_) break;
        lk.unlock();
        try {
            sweep();
        } catch (...) {
            // a file locked by another process: retry on the next round
        }
        lk.lock();
    }
}

} // namespace desireeia
