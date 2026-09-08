#include "../core/engine.h"
#include "../quant/quant.h"
#include <fstream>
#include <cstring>
#include <unordered_map>

namespace desireeia {

namespace {
constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t GGUF_MAGIC_V3 = 0x47554746;

template <typename T>
static bool read_at(std::istream& f, T& out) {
    return static_cast<bool>(f.read(reinterpret_cast<char*>(&out), sizeof(T)));
}

bool read_string(std::istream& f, std::string& out) {
    uint64_t len = 0;
    if (!read_at(f, len)) return false;
    out.resize(len);
    if (len > 0 && !f.read(out.data(), static_cast<std::streamsize>(len))) {
        return false;
    }
    return true;
}

constexpr size_t gguf_type_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7:             return 1;          // U8, I8, BOOL
        case 2: case 3:                     return 2;          // U16, I16
        case 4: case 5: case 6:             return 4;          // U32, I32, F32
        case 10: case 11: case 12:          return 8;          // U64, I64, F64
        default:                            return 0;
    }
}

bool skip_gguf_value(std::istream& f, uint32_t type) {
    size_t sz = gguf_type_size(type);
    if (sz > 0) {
        f.seekg(static_cast<std::streamoff>(sz), std::ios::cur);
        return true;
    }
    if (type == 8) {                                    // STRING
        uint64_t len = 0;
        if (!read_at(f, len)) return false;
        f.seekg(static_cast<std::streamoff>(len), std::ios::cur);
        return true;
    }
    if (type == 9) {                                    // ARRAY
        uint32_t elem = 0;
        if (!read_at(f, elem)) return false;
        uint64_t n = 0;
        if (!read_at(f, n)) return false;
        for (uint64_t i = 0; i < n; ++i) {
            if (!skip_gguf_value(f, elem)) return false;
        }
        return true;
    }
    return false;
}

struct TensorInfo {
    std::string name;
    uint32_t type = 0;
    std::vector<uint64_t> dims;
    uint64_t offset = 0;
};

bool read_tensor_info(std::istream& f, TensorInfo& t) {
    if (!read_string(f, t.name)) return false;
    uint32_t n_dims = 0;
    if (!read_at(f, n_dims)) return false;
    t.dims.resize(n_dims);
    for (uint32_t i = 0; i < n_dims; ++i) {
        if (!read_at(f, t.dims[i])) return false;
    }
    if (!read_at(f, t.type)) return false;
    if (!read_at(f, t.offset)) return false;
    return true;
}

uint64_t tensor_count(const std::vector<uint64_t>& dims) {
    uint64_t n = 1;
    for (uint64_t d : dims) n *= d;
    return n;
}
}

class GgufReader final : public ModelReader {
public:
    bool open(const std::string& path, ModelMeta& meta) override {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        uint32_t magic = 0;
        if (!read_at(f, magic)) return false;
        if (magic != GGUF_MAGIC) return false;

        meta.format = DESIREEIA_FORMAT_GGUF;
        meta.path = path;
        path_ = path;

        uint32_t version = 0;
        if (!read_at(f, version)) return false;
        version_ = version;

        uint64_t n_tensors = 0;
        if (!read_at(f, n_tensors)) return false;

        uint64_t n_kv = 0;
        if (!read_at(f, n_kv)) return false;

        std::string arch;
        uint64_t n_tokens = 0;
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key;
            if (!read_string(f, key)) return false;
            uint32_t type = 0;
            if (!read_at(f, type)) return false;

            if (type == 8) {
                std::string val;
                if (!read_string(f, val)) return false;
                if (key == "general.architecture") arch = val;
                if (key == "tokenizer.ggml.model") vocab_.tokenizer_tag = val;
                continue;
            }
            if (type == 9) {
                uint32_t elem = 0;
                uint64_t n = 0;
                if (!read_at(f, elem)) return false;
                if (!read_at(f, n)) return false;

                if (key == "tokenizer.ggml.tokens" && elem == 8) {
                    n_tokens = n;
                    vocab_.tokens.resize((size_t) n);
                    for (uint64_t a = 0; a < n; ++a) {
                        if (!read_string(f, vocab_.tokens[(size_t) a])) return false;
                    }
                    continue;
                }
                if (key == "tokenizer.ggml.merges" && elem == 8) {
                    vocab_.merges.resize((size_t) n);
                    for (uint64_t a = 0; a < n; ++a) {
                        if (!read_string(f, vocab_.merges[(size_t) a])) return false;
                    }
                    continue;
                }
                if (key == "tokenizer.ggml.scores" && elem == 6) {
                    vocab_.scores.resize((size_t) n);
                    for (uint64_t a = 0; a < n; ++a) {
                        if (!read_at(f, vocab_.scores[(size_t) a])) return false;
                    }
                    continue;
                }
                if (key == "tokenizer.ggml.token_type" && (elem == 5 || elem == 4)) {
                    vocab_.token_type.resize((size_t) n);
                    for (uint64_t a = 0; a < n; ++a) {
                        int32_t v = 0;
                        if (!read_at(f, v)) return false;
                        vocab_.token_type[(size_t) a] = v;
                    }
                    continue;
                }
                if (key == "tokenizer.ggml.tokens") n_tokens = n;
                for (uint64_t a = 0; a < n; ++a) {
                    if (!skip_gguf_value(f, elem)) return false;
                }
                continue;
            }
            if (type == 10) {
                uint64_t v = 0;
                if (!read_at(f, v)) return false;
                kv_int_[key] = v;
                continue;
            }
            if (type == 11) {
                int64_t v = 0;
                if (!read_at(f, v)) return false;
                kv_int_[key] = static_cast<uint64_t>(v);
                continue;
            }
            if (type == 4 || type == 5) {
                uint32_t v = 0;
                if (!read_at(f, v)) return false;
                kv_int_[key] = v;
                continue;
            }
            if (type == 6) {
                float v = 0.0f;
                if (!read_at(f, v)) return false;
                kv_float_[key] = v;
                continue;
            }
            if (type == 7) {
                uint8_t v = 0;
                if (!read_at(f, v)) return false;
                kv_int_[key] = v;
                continue;
            }
            // type 8 = GGUF_TYPE_STRING. Prima veniva sempre scartata
            // (skip_gguf_value): il motore non leggeva NESSUNA stringa a
            // livello di chiave/valore. Serve per "tokenizer.chat_template"
            // (Fase template di chat, 2026-09-08): senza, la CLI non ha modo
            // di sapere quale formato di chat usa il modello e deve
            // indovinare dall'architettura, che per famiglie con piu'
            // varianti (llama3 vs llama2, mistral v1/v3/v7...) non basta.
            if (type == 8) {
                std::string v;
                if (!read_string(f, v)) return false;
                kv_string_[key] = std::move(v);
                continue;
            }
            if (!skip_gguf_value(f, type)) return false;
        }

        auto get_i32 = [this](const std::string& k, int32_t def) -> int32_t {
            auto it = kv_int_.find(k);
            return it != kv_int_.end() ? static_cast<int32_t>(it->second) : def;
        };
        vocab_.bos_id = get_i32("tokenizer.ggml.bos_token_id", -1);
        vocab_.eos_id = get_i32("tokenizer.ggml.eos_token_id", -1);
        vocab_.unk_id = get_i32("tokenizer.ggml.unknown_token_id", -1);
        vocab_.pad_id = get_i32("tokenizer.ggml.padding_token_id", -1);
        vocab_.add_bos = get_i32("tokenizer.ggml.add_bos_token", 1) != 0;

        if (!arch.empty()) {
            auto it = kv_int_.find(arch + ".block_count");
            if (it != kv_int_.end()) {
                meta.n_layers = static_cast<uint32_t>(it->second);
            }
            auto exp_it = kv_int_.find(arch + ".expert_count");
            if (exp_it != kv_int_.end() && exp_it->second > 0) {
                meta.n_experts = static_cast<uint32_t>(exp_it->second);
            }
        }

        tensors_.clear();
        tensors_.reserve(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            if (!read_tensor_info(f, t)) return false;
            name_to_idx_[t.name] = tensors_.size();
            tensors_.push_back(t);
        }

        uint64_t alignment = 32;
        auto al_it = kv_int_.find("general.alignment");
        if (al_it != kv_int_.end() && al_it->second > 0 && (al_it->second & (al_it->second - 1)) == 0) {
            alignment = al_it->second;
        }
        const uint64_t pos = static_cast<uint64_t>(f.tellg());
        // GGUF v2+ stores tensor offsets relative to the data section start,
        // which is the position after the tensor infos padded to the alignment
        data_start_ = version_ >= 2 ? ((pos + alignment - 1) / alignment) * alignment : 0;

        meta.expert_model = meta.n_experts > 0;
        if (!arch.empty()) {
            meta.arch = arch;
            meta.n_vocab = static_cast<uint32_t>(n_tokens);
        }
        return true;
    }

    bool read_tensor_raw(const std::string& name, std::vector<uint8_t>& raw,
                          int& quant_type, uint64_t& ne0, uint64_t& rows) override {
        auto it = name_to_idx_.find(name);
        if (it == name_to_idx_.end()) return false;
        const TensorInfo& t = tensors_[it->second];
        if (t.dims.empty()) return false;

        const size_t row_bytes = desireeia_row_size((int) t.type, (int64_t) t.dims[0]);
        if (row_bytes == 0) return false;

        rows = t.dims.size() > 1 ? tensor_count(std::vector<uint64_t>(t.dims.begin() + 1, t.dims.end())) : 1;
        ne0 = t.dims[0];
        quant_type = (int) t.type;

        std::ifstream f(path_, std::ios::binary);
        if (!f) return false;
        f.seekg(static_cast<std::streamoff>(data_start_ + t.offset), std::ios::beg);

        raw.resize(row_bytes * rows);
        return (bool) f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize) raw.size());
    }

    bool read_tensor(const std::string& name, std::vector<float>& out) override {
        auto it = name_to_idx_.find(name);
        if (it == name_to_idx_.end()) return false;
        const TensorInfo& t = tensors_[it->second];
        uint64_t n = tensor_count(t.dims);
        if (n == 0) { out.clear(); return true; }

        size_t row = desireeia_row_size((int)t.type, (int64_t)t.dims[0]);
        if (row == 0) return false;

        std::ifstream f(path_, std::ios::binary);
        if (!f) return false;
        f.seekg(static_cast<std::streamoff>(data_start_ + t.offset), std::ios::beg);

        uint64_t rows = t.dims.size() > 1 ? tensor_count(std::vector<uint64_t>(t.dims.begin() + 1, t.dims.end())) : 1;
        std::vector<unsigned char> raw(row * rows);
        if (!f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()))) {
            return false;
        }

        int64_t ne0 = (int64_t)t.dims[0];
        int64_t per_row = (int64_t)desireeia_row_size((int)t.type, ne0);
        out.resize(n);
        const unsigned char* p = raw.data();
        int64_t off = 0;
        for (uint64_t r = 0; r < rows; ++r) {
            int rc = desireeia_dequantize_row((int)t.type, p, out.data() + off, ne0);
            if (rc != 0) return false;
            p += per_row;
            off += ne0;
        }
        return true;
    }

    // MoE tensor convention: blk.N.ffn_{gate,up,down}_exps.weight, shape
    // [n_embd, n_ff, n_expert] per gate/up e [n_ff, n_embd, n_expert] per
    // down (dims[0] = larghezza riga, dims[1] = righe per esperto,
    // dims[2] = n_expert, esperti contigui in righe [idx*dims[1],
    // (idx+1)*dims[1])). Si legge solo il sotto-blocco dell'esperto
    // richiesto, non l'intero tensore.
    bool read_expert(uint32_t layer, uint32_t idx, ExpertPart part, std::vector<float>& out) override {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "blk.%u.", layer);
        std::string name = buf;
        switch (part) {
            case ExpertPart::Gate: name += "ffn_gate_exps.weight"; break;
            case ExpertPart::Up:   name += "ffn_up_exps.weight";   break;
            case ExpertPart::Down: name += "ffn_down_exps.weight"; break;
        }

        auto it = name_to_idx_.find(name);
        if (it == name_to_idx_.end()) return false;
        const TensorInfo& t = tensors_[it->second];
        if (t.dims.size() != 3) return false;

        const uint64_t rows_per_expert = t.dims[1];
        const uint64_t n_expert = t.dims[2];
        if (idx >= n_expert) return false;

        const size_t row = desireeia_row_size((int) t.type, (int64_t) t.dims[0]);
        if (row == 0) return false;

        std::ifstream f(path_, std::ios::binary);
        if (!f) return false;
        const uint64_t row_start = idx * rows_per_expert;
        f.seekg(static_cast<std::streamoff>(data_start_ + t.offset + row_start * row), std::ios::beg);

        std::vector<unsigned char> raw(row * rows_per_expert);
        if (!f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()))) {
            return false;
        }

        const int64_t ne0 = (int64_t) t.dims[0];
        out.resize((size_t)(ne0 * (int64_t) rows_per_expert));
        const unsigned char* p = raw.data();
        int64_t off = 0;
        for (uint64_t r = 0; r < rows_per_expert; ++r) {
            int rc = desireeia_dequantize_row((int) t.type, p, out.data() + off, ne0);
            if (rc != 0) return false;
            p += row;
            off += ne0;
        }
        return true;
    }

    bool meta_u32(const std::string& key, uint32_t& out) override {
        auto it = kv_int_.find(key);
        if (it == kv_int_.end()) return false;
        out = static_cast<uint32_t>(it->second);
        return true;
    }

    bool meta_f32(const std::string& key, float& out) override {
        auto it = kv_float_.find(key);
        if (it == kv_float_.end()) return false;
        out = it->second;
        return true;
    }

    bool meta_str(const std::string& key, std::string& out) override {
        auto it = kv_string_.find(key);
        if (it == kv_string_.end()) return false;
        out = it->second;
        return true;
    }

    bool read_vocab(VocabData& out) override {
        if (vocab_.tokens.empty()) return false;
        out = vocab_;
        return true;
    }

private:
    std::string path_;
    uint32_t version_ = 1;
    uint64_t data_start_ = 0;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> name_to_idx_;
    std::unordered_map<std::string, uint64_t> kv_int_;
    std::unordered_map<std::string, float> kv_float_;
    std::unordered_map<std::string, std::string> kv_string_;
    VocabData vocab_;
};

ModelReader* make_gguf_reader() {
    return new GgufReader;
}

}
