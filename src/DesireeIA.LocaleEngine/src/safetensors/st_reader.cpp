// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "../core/engine.h"
#include "../quant/quant.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <unordered_map>

namespace desireeia {

namespace {

// Minimal JSON parser, sufficient for the safetensors header and for
// safetensors.index.json (objects/arrays/strings/numbers, no external
// dependency). Not a general-purpose JSON parser: not needed, the header
// format is always a flat object of objects.
struct JsonValue {
    enum Type { Null, Str, Num, Arr, Obj } type = Null;
    std::string str;
    double num = 0.0;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* find(const std::string& key) const {
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s), i_(0) {}

    bool parse(JsonValue& out) {
        skip_ws();
        return parse_value(out);
    }

private:
    const std::string& s_;
    size_t i_;

    void skip_ws() {
        while (i_ < s_.size() && (unsigned char) s_[i_] <= ' ') i_++;
    }

    bool parse_value(JsonValue& v) {
        skip_ws();
        if (i_ >= s_.size()) return false;
        char c = s_[i_];
        if (c == '{') return parse_obj(v);
        if (c == '[') return parse_arr(v);
        if (c == '"') return parse_str_val(v);
        if (s_.compare(i_, 4, "null") == 0) { v.type = JsonValue::Null; i_ += 4; return true; }
        if (s_.compare(i_, 4, "true") == 0) { v.type = JsonValue::Num; v.num = 1; i_ += 4; return true; }
        if (s_.compare(i_, 5, "false") == 0) { v.type = JsonValue::Num; v.num = 0; i_ += 5; return true; }
        return parse_num(v);
    }

    bool parse_str_raw(std::string& out) {
        if (i_ >= s_.size() || s_[i_] != '"') return false;
        i_++;
        out.clear();
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_];
            if (c == '\\' && i_ + 1 < s_.size()) {
                i_++;
                char e = s_[i_];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    default: out += e; break;
                }
            } else {
                out += c;
            }
            i_++;
        }
        if (i_ >= s_.size()) return false;
        i_++;
        return true;
    }

    bool parse_str_val(JsonValue& v) {
        v.type = JsonValue::Str;
        return parse_str_raw(v.str);
    }

    bool parse_num(JsonValue& v) {
        size_t start = i_;
        while (i_ < s_.size() && (std::isdigit((unsigned char) s_[i_]) || s_[i_] == '-' ||
               s_[i_] == '+' || s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E')) {
            i_++;
        }
        if (i_ == start) return false;
        v.type = JsonValue::Num;
        v.num = std::strtod(s_.c_str() + start, nullptr);
        return true;
    }

    bool parse_arr(JsonValue& v) {
        v.type = JsonValue::Arr;
        i_++;
        skip_ws();
        if (i_ < s_.size() && s_[i_] == ']') { i_++; return true; }
        for (;;) {
            JsonValue item;
            if (!parse_value(item)) return false;
            v.arr.push_back(std::move(item));
            skip_ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { i_++; skip_ws(); continue; }
            if (s_[i_] == ']') { i_++; break; }
            return false;
        }
        return true;
    }

    bool parse_obj(JsonValue& v) {
        v.type = JsonValue::Obj;
        i_++;
        skip_ws();
        if (i_ < s_.size() && s_[i_] == '}') { i_++; return true; }
        for (;;) {
            skip_ws();
            std::string key;
            if (!parse_str_raw(key)) return false;
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != ':') return false;
            i_++;
            JsonValue val;
            if (!parse_value(val)) return false;
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { i_++; skip_ws(); continue; }
            if (s_[i_] == '}') { i_++; break; }
            return false;
        }
        return true;
    }
};

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string dirname_of(const std::string& path) {
    size_t p1 = path.find_last_of('/');
    size_t p2 = path.find_last_of('\\');
    size_t p = (p1 == std::string::npos) ? p2 : (p2 == std::string::npos ? p1 : std::max(p1, p2));
    if (p == std::string::npos) return ".";
    return path.substr(0, p);
}

}

struct StTensorEntry {
    std::string shard_path;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t off_start = 0;
    uint64_t off_end = 0;
};

class StReader final : public ModelReader {
public:
    bool open(const std::string& path, ModelMeta& meta) override {
        tensors_.clear();
        shard_data_base_.clear();

        bool ok;
        if (ends_with(path, ".safetensors.index.json")) {
            ok = open_index(path);
        } else {
            ok = parse_shard(path);
        }
        if (!ok || tensors_.empty()) return false;

        meta.format = DESIREEIA_FORMAT_SAFETENSORS;
        meta.path = path;
        meta.expert_model = true;
        return true;
    }

    bool read_tensor(const std::string& name, std::vector<float>& out) override {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) return false;
        const StTensorEntry& e = it->second;

        uint64_t n = 1;
        for (uint64_t d : e.shape) n *= d;
        if (n == 0) { out.clear(); return true; }

        auto base_it = shard_data_base_.find(e.shard_path);
        if (base_it == shard_data_base_.end()) return false;

        std::ifstream f(e.shard_path, std::ios::binary);
        if (!f) return false;
        f.seekg(static_cast<std::streamoff>(base_it->second + e.off_start), std::ios::beg);

        return decode_dtype(f, e.dtype, n, out);
    }

    // The expert tensor convention in safetensors files is not standardized
    // across HF checkpoints (it varies by family: mixtral, deepseek,
    // qwen2moe each use different naming schemes), unlike GGUF where
    // blk.N.ffn_{gate,up,down}_exps.weight is fixed. Only the Mixtral
    // convention is implemented here (the original HF checkpoint, not the
    // "merged" layout produced by a conversion tool), verified by reading
    // the reference converter in the external desireeialmn project
    // (conversion module, MixtralModel.modify_tensors class): each expert
    // is a separate tensor
    // "model.layers.{layer}.block_sparse_moe.experts.{idx}.{wid}.weight"
    // with wid in {w1=gate_proj, w2=down_proj, w3=up_proj} (mapping
    // confirmed via gguf-py/gguf/tensor_mapping.py: FFN_GATE_EXP -> w1,
    // FFN_DOWN_EXP -> w2, FFN_UP_EXP -> w3). Other families (deepseek,
    // qwen2moe) remain an open gap, see docs/engine_gap_analysis.md: there
    // is no verified reference for their naming as of this session.
    bool read_expert(uint32_t layer, uint32_t idx, ExpertPart part, std::vector<float>& out) override {
        const char* wid = nullptr;
        switch (part) {
            case ExpertPart::Gate: wid = "w1"; break;
            case ExpertPart::Down: wid = "w2"; break;
            case ExpertPart::Up:   wid = "w3"; break;
        }
        if (!wid) return false;

        std::string name = "model.layers." + std::to_string(layer) +
                            ".block_sparse_moe.experts." + std::to_string(idx) +
                            "." + wid + ".weight";
        return read_tensor(name, out);
    }

private:
    bool open_index(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

        JsonValue root;
        JsonParser parser(content);
        if (!parser.parse(root) || root.type != JsonValue::Obj) return false;

        const JsonValue* weight_map = root.find("weight_map");
        if (!weight_map || weight_map->type != JsonValue::Obj) return false;

        const std::string dir = dirname_of(path);
        std::unordered_map<std::string, bool> shards_seen;
        for (const auto& kv : weight_map->obj) {
            if (kv.second.type != JsonValue::Str) continue;
            const std::string shard_path = dir + "/" + kv.second.str;
            if (shards_seen.count(shard_path)) continue;
            shards_seen[shard_path] = true;
            if (!parse_shard(shard_path)) return false;
        }
        return true;
    }

    bool parse_shard(const std::string& shard_path) {
        std::ifstream f(shard_path, std::ios::binary);
        if (!f) return false;

        uint64_t header_len = 0;
        if (!f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len))) return false;

        std::string header;
        header.resize(static_cast<size_t>(header_len));
        if (header_len > 0 && !f.read(header.data(), static_cast<std::streamsize>(header_len))) {
            return false;
        }

        JsonValue root;
        JsonParser parser(header);
        if (!parser.parse(root) || root.type != JsonValue::Obj) return false;

        for (const auto& kv : root.obj) {
            if (kv.first == "__metadata__") continue;
            const JsonValue& tv = kv.second;
            if (tv.type != JsonValue::Obj) continue;

            StTensorEntry e;
            e.shard_path = shard_path;
            const JsonValue* dtype = tv.find("dtype");
            if (dtype && dtype->type == JsonValue::Str) e.dtype = dtype->str;
            const JsonValue* shape = tv.find("shape");
            if (shape && shape->type == JsonValue::Arr) {
                for (const auto& d : shape->arr) e.shape.push_back((uint64_t) d.num);
            }
            const JsonValue* offsets = tv.find("data_offsets");
            if (offsets && offsets->type == JsonValue::Arr && offsets->arr.size() == 2) {
                e.off_start = (uint64_t) offsets->arr[0].num;
                e.off_end = (uint64_t) offsets->arr[1].num;
            } else {
                continue; // tensor without valid data_offsets: unusable
            }
            tensors_[kv.first] = std::move(e);
        }
        shard_data_base_[shard_path] = sizeof(header_len) + header_len;
        return true;
    }

    static bool decode_dtype(std::ifstream& f, const std::string& dtype, uint64_t n, std::vector<float>& out) {
        out.resize((size_t) n);
        if (dtype == "F32") {
            return (bool) f.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n * 4));
        }
        if (dtype == "F16") {
            std::vector<uint16_t> raw((size_t) n);
            if (!f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(n * 2))) return false;
            for (uint64_t i = 0; i < n; ++i) out[(size_t) i] = desireeia_fp16_to_fp32(raw[(size_t) i]);
            return true;
        }
        if (dtype == "BF16") {
            std::vector<uint16_t> raw((size_t) n);
            if (!f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)(n * 2))) return false;
            for (uint64_t i = 0; i < n; ++i) out[(size_t) i] = desireeia_bf16_to_fp32((desireeia_bf16_t) raw[(size_t) i]);
            return true;
        }
        // I8/U8/I64/F8_E4M3/F8_E5M2 not yet supported: known gap.
        return false;
    }

    std::unordered_map<std::string, StTensorEntry> tensors_;
    std::unordered_map<std::string, uint64_t> shard_data_base_;
};

ModelReader* make_st_reader() {
    return new StReader;
}

}
