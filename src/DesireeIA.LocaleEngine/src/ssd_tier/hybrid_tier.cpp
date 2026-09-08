#include "hybrid_tier.h"
#include "ssd_io.h"
#include "core/engine.h"
#include "quant/quant.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <list>
#include <mutex>
#include <sstream>

namespace desireeia {

static constexpr uint32_t GGUF_MAGIC = 0x46554747;
static constexpr uint32_t GGUF_MAGIC_V3 = 0x47554746;

static uint64_t now_ns() {
    auto tp = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count());
}

static uint64_t make_key(uint32_t layer, uint32_t idx, uint32_t part) {
    return (static_cast<uint64_t>(layer) << 48) |
           (static_cast<uint64_t>(idx) << 8) |
           static_cast<uint64_t>(part);
}

static void unpack_key(uint64_t key, uint32_t& layer, uint32_t& idx, uint32_t& part) {
    layer = static_cast<uint32_t>(key >> 48);
    idx = static_cast<uint32_t>((key >> 8) & 0xFFFFFFFFu);
    part = static_cast<uint32_t>(key & 0xFFu);
}

// Minimal GGUF header parser: extracts data_start and tensor offsets.
// Uses ifstream for the one-time header parse (small sequential read).
struct GgufHeader {
    uint64_t data_start = 0;
    uint32_t alignment = 32;

    struct TensorInfo {
        std::string name;
        uint32_t type = 0;
        std::vector<uint64_t> dims;
        uint64_t offset = 0;
        uint64_t data_offset = 0;
    };
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;

    bool parse(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        uint32_t magic = 0;
        f.read(reinterpret_cast<char*>(&magic), 4);
        if (magic != GGUF_MAGIC && magic != GGUF_MAGIC_V3) return false;

        uint32_t version = 0;
        f.read(reinterpret_cast<char*>(&version), 4);

        uint64_t n_tensors = 0;
        f.read(reinterpret_cast<char*>(&n_tensors), 8);

        uint64_t n_kv = 0;
        f.read(reinterpret_cast<char*>(&n_kv), 8);

        for (uint64_t i = 0; i < n_kv; ++i) {
            uint64_t klen = 0;
            f.read(reinterpret_cast<char*>(&klen), 8);
            f.seekg(static_cast<std::streamoff>(klen), std::ios::cur);
            uint32_t vtype = 0;
            f.read(reinterpret_cast<char*>(&vtype), 4);
            if (!skip_value(f, vtype)) return false;
        }

        tensors.clear();
        tensors.reserve(n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            uint64_t nlen = 0;
            f.read(reinterpret_cast<char*>(&nlen), 8);
            t.name.resize(static_cast<size_t>(nlen));
            if (nlen > 0) f.read(t.name.data(), static_cast<std::streamoff>(nlen));
            uint32_t n_dims = 0;
            f.read(reinterpret_cast<char*>(&n_dims), 4);
            t.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                f.read(reinterpret_cast<char*>(&t.dims[d]), 8);
            }
            f.read(reinterpret_cast<char*>(&t.type), 4);
            f.read(reinterpret_cast<char*>(&t.offset), 8);

            name_to_idx[t.name] = tensors.size();
            tensors.push_back(std::move(t));
        }

        const uint64_t pos = static_cast<uint64_t>(f.tellg());
        data_start = ((pos + alignment - 1) / alignment) * alignment;

        for (auto& t : tensors) {
            t.data_offset = data_start + t.offset;
        }

        return true;
    }

    const TensorInfo* find_tensor(const std::string& name) const {
        auto it = name_to_idx.find(name);
        if (it == name_to_idx.end()) return nullptr;
        return &tensors[it->second];
    }

    static bool skip_value(std::istream& f, uint32_t type) {
        switch (type) {
            case 0: case 1: case 7: f.seekg(1, std::ios::cur); return true;
            case 2: case 3: f.seekg(2, std::ios::cur); return true;
            case 4: case 5: case 6: f.seekg(4, std::ios::cur); return true;
            case 10: case 11: case 12: f.seekg(8, std::ios::cur); return true;
            case 8: {
                uint64_t len = 0;
                f.read(reinterpret_cast<char*>(&len), 8);
                f.seekg(static_cast<std::streamoff>(len), std::ios::cur);
                return true;
            }
            case 9: {
                uint32_t elem = 0;
                uint64_t n = 0;
                f.read(reinterpret_cast<char*>(&elem), 4);
                f.read(reinterpret_cast<char*>(&n), 8);
                for (uint64_t i = 0; i < n; ++i) {
                    if (!skip_value(f, elem)) return false;
                }
                return true;
            }
            default: return false;
        }
    }

    static size_t row_size_for_type(uint32_t type, uint64_t ne0) {
        switch (type) {
            case 0: return ne0 * 4;
            case 1: return ne0 * 2;
            case 2: return (ne0 + 1) / 2 + 2;
            case 3: return ne0 / 2 + 4;
            case 7: return ne0 + 4;
            case 8: return ne0 + 8;
            case 10: return (ne0 + 255) / 256 * 144;
            case 11: return (ne0 + 255) / 256 * 176;
            case 12: return (ne0 + 255) / 256 * 210;
            default: return 0;
        }
    }
};

struct HybridTier::Impl {
    HybridTierConfig config;
    LogFn log;
    mutable std::mutex mtx;

    GgufHeader gguf_header;
    bool gguf_opened = false;

    // Where anything not in cache comes from. Not owned. When this is set,
    // every miss is served through it rather than through gguf_header, which
    // is what keeps the tier's view of the file identical to the engine's.
    ModelReader* source = nullptr;

    // Direct-I/O handle for tensor data reads.
    SsdFile ssd_file;
    AlignedSlab slab;

    struct TensorEntry {
        std::string name;
        uint64_t key = 0;
        std::vector<float> data;
        uint32_t hits;
        bool pinned;
    };
    std::list<TensorEntry> tensor_lru;
    std::unordered_map<std::string, std::list<TensorEntry>::iterator> tensor_map;

    std::list<TensorEntry> expert_lru;
    std::unordered_map<uint64_t, std::list<TensorEntry>::iterator> expert_map;

    // Cache of RAW, still-quantized tensor bytes.
    //
    // This is the cache that matters. The matmul kernels read weights through
    // read_tensor_raw and consume the quantized bytes directly — the float
    // path above is a fallback for formats without a direct kernel, and on a
    // K-quant model it is never taken. So the tier used to cache only the path
    // nothing used, and send every read on the path everything uses straight
    // to disk.
    //
    // Bytes, not entries, decide the budget here: tensors in one model range
    // from a few KiB of norm weights to hundreds of MiB for the output
    // projection, so a count of entries says nothing about memory used. And
    // quantized bytes are what gets stored: dequantizing to float first, as
    // the entries above do, costs 8x the RAM for the same coverage.
    struct RawEntry {
        std::string name;
        std::vector<uint8_t> raw;
        int quant_type = 0;
        uint64_t ne0 = 0;
        uint64_t rows = 0;
        uint32_t hits = 0;
        bool pinned = false;
    };
    std::list<RawEntry> raw_lru;
    std::unordered_map<std::string, std::list<RawEntry>::iterator> raw_map;
    uint64_t raw_bytes = 0;

    // Returns false when there was nothing evictable left, so the caller can
    // stop instead of looping forever on an all-pinned cache.
    bool evict_raw() {
        for (auto it = raw_lru.end(); it != raw_lru.begin();) {
            --it;
            if (!it->pinned) {
                raw_bytes -= it->raw.size();
                raw_map.erase(it->name);
                raw_lru.erase(it);
                return true;
            }
        }
        return false;
    }

    std::unordered_map<uint64_t, uint32_t> usage;
    std::unordered_map<std::string, uint32_t> tensor_usage;
    std::string usage_file;

    uint64_t ram_hits = 0;
    uint64_t ram_misses = 0;
    uint64_t ssd_reads = 0;
    uint64_t ssd_errors = 0;
    uint64_t total_bytes_read = 0;
    uint64_t total_read_ns = 0;
    uint32_t read_count = 0;

    void touch_tensor(std::list<TensorEntry>::iterator it) {
        tensor_lru.splice(tensor_lru.begin(), tensor_lru, it);
        it->hits++;
    }

    void touch_expert(std::list<TensorEntry>::iterator it) {
        expert_lru.splice(expert_lru.begin(), expert_lru, it);
        it->hits++;
    }

    void evict_tensor() {
        for (auto it = tensor_lru.end(); it != tensor_lru.begin();) {
            --it;
            if (!it->pinned) {
                tensor_map.erase(it->name);
                tensor_lru.erase(it);
                return;
            }
        }
    }

    void evict_expert() {
        for (auto it = expert_lru.end(); it != expert_lru.begin();) {
            --it;
            if (!it->pinned) {
                expert_map.erase(it->key);
                expert_lru.erase(it);
                return;
            }
        }
    }

    static uint64_t name_hash(const std::string& name) {
        uint64_t h = 14695981039346656037ull;
        for (char c : name) {
            h ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
            h *= 1099511628211ull;
        }
        return h;
    }

    // Read raw tensor bytes from GGUF via SsdFile pread.
    bool pread_tensor_raw(const GgufHeader::TensorInfo* t,
                          size_t row_start, size_t row_count,
                          std::vector<uint8_t>& out) {
        if (!t || !ssd_file.is_open()) return false;
        if (t->dims.empty()) return false;

        const size_t row_bytes = GgufHeader::row_size_for_type(t->type, t->dims[0]);
        if (row_bytes == 0) return false;

        const size_t byte_offset = t->data_offset + row_start * row_bytes;
        const size_t byte_count = row_bytes * row_count;

        if (!slab.alloc(byte_count)) return false;

        const size_t got = ssd_file.pread(slab.data, byte_count, byte_offset);
        if (got != byte_count) return false;

        ssd_file.fadvise_dontneed(byte_offset, byte_count);

        out.resize(byte_count);
        std::memcpy(out.data(), slab.data, byte_count);
        return true;
    }

    // Read a full tensor from GGUF via SsdFile pread, dequantize to float.
    bool read_tensor_from_ssd(const std::string& name, std::vector<float>& out) {
        if (!source) return false;

        const uint64_t t0 = now_ns();
        if (!source->read_tensor(name, out)) {
            ssd_errors++;
            return false;
        }
        const uint64_t t1 = now_ns();

        ssd_reads++;
        total_bytes_read += out.size() * sizeof(float);
        total_read_ns += (t1 - t0);
        read_count++;
        return true;
    }

    // Read raw (still quantized) tensor bytes through the source reader.
    bool read_tensor_raw_from_ssd(const std::string& name, std::vector<uint8_t>& raw,
                                   int& quant_type, uint64_t& ne0, uint64_t& rows) {
        if (!source) return false;

        const uint64_t t0 = now_ns();
        if (!source->read_tensor_raw(name, raw, quant_type, ne0, rows)) {
            ssd_errors++;
            return false;
        }
        const uint64_t t1 = now_ns();

        ssd_reads++;
        total_bytes_read += raw.size();
        total_read_ns += (t1 - t0);
        read_count++;
        return true;
    }

    // Read one MoE expert through the source reader, which already knows how
    // to slice a single expert out of the 3D stacked-expert tensor.
    bool read_expert_from_ssd(uint64_t key, std::vector<float>& out) {
        if (!source) return false;

        uint32_t layer, idx, part;
        unpack_key(key, layer, idx, part);
        if (part > 2) return false;

        const uint64_t t0 = now_ns();
        if (!source->read_expert(layer, idx, static_cast<ExpertPart>(part), out)) {
            ssd_errors++;
            return false;
        }
        const uint64_t t1 = now_ns();

        ssd_reads++;
        total_bytes_read += out.size() * sizeof(float);
        total_read_ns += (t1 - t0);
        read_count++;
        return true;
    }

    void log_msg(int32_t level, const char* msg) {
        if (log) log(level, msg);
    }
};

HybridTier::HybridTier(const HybridTierConfig& config, LogFn log)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
    impl_->log = log;
    impl_->usage_file = config.usage_file;

    // The file is opened only to hint the OS about the access pattern and to
    // confirm it is readable. Tensor data itself is never read from here:
    // that goes through the source reader (see set_source).
    if (!config.gguf_path.empty()) {
        impl_->ssd_file.open(config.gguf_path);
        if (impl_->ssd_file.is_open()) {
            impl_->ssd_file.fadvise_sequential();
        } else {
            impl_->log_msg(3, "ssd tier: model file could not be opened for read-ahead hints");
        }
    }
}

HybridTier::~HybridTier() = default;
HybridTier::HybridTier(HybridTier&&) noexcept = default;
HybridTier& HybridTier::operator=(HybridTier&&) noexcept = default;

void HybridTier::set_source(ModelReader* reader) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->source = reader;
    // "Available" now means "we have something to read through", not "we
    // managed to parse the file ourselves".
    impl_->gguf_opened = (reader != nullptr);
}

bool HybridTier::read_tensor(const std::string& name, std::vector<float>& out) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    auto it = impl_->tensor_map.find(name);
    if (it != impl_->tensor_map.end()) {
        impl_->touch_tensor(it->second);
        impl_->ram_hits++;
        out = it->second->data;
        return true;
    }

    impl_->ram_misses++;
    impl_->tensor_usage[name]++;

    if (!impl_->gguf_opened) return false;

    std::vector<float> data;
    if (!impl_->read_tensor_from_ssd(name, data)) {
        return false;
    }

    while (static_cast<int32_t>(impl_->tensor_map.size()) >= impl_->config.ram_capacity) {
        impl_->evict_tensor();
    }

    Impl::TensorEntry entry;
    entry.name = name;
    entry.data = std::move(data);
    entry.hits = 1;
    entry.pinned = (impl_->tensor_usage[name] >= impl_->config.ram_pin_threshold);

    impl_->tensor_lru.push_front(std::move(entry));
    impl_->tensor_map[name] = impl_->tensor_lru.begin();
    out = impl_->tensor_lru.front().data;
    return true;
}

bool HybridTier::read_tensor_raw(const std::string& name, std::vector<uint8_t>& raw,
                                  int& quant_type, uint64_t& ne0, uint64_t& rows) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    if (!impl_->gguf_opened) return false;
    return impl_->read_tensor_raw_from_ssd(name, raw, quant_type, ne0, rows);
}

bool HybridTier::write_tensor(const std::string& name, const std::vector<float>& data) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    auto it = impl_->tensor_map.find(name);
    if (it != impl_->tensor_map.end()) {
        it->second->data = data;
        impl_->touch_tensor(it->second);
    } else {
        while (static_cast<int32_t>(impl_->tensor_map.size()) >= impl_->config.ram_capacity) {
            impl_->evict_tensor();
        }

        Impl::TensorEntry entry;
        entry.name = name;
        entry.data = data;
        entry.hits = 1;
        entry.pinned = false;

        impl_->tensor_lru.push_front(std::move(entry));
        impl_->tensor_map[name] = impl_->tensor_lru.begin();
    }
    return true;
}

void HybridTier::prefetch_tensors(const std::vector<std::string>& names) {
    if (!impl_->config.prefetch_enabled || !impl_->gguf_opened) return;

    for (const auto& name : names) {
        if (impl_->tensor_map.find(name) != impl_->tensor_map.end()) continue;

        std::vector<float> data;
        if (impl_->read_tensor_from_ssd(name, data)) {
            while (static_cast<int32_t>(impl_->tensor_map.size()) >= impl_->config.ram_capacity) {
                impl_->evict_tensor();
            }

            Impl::TensorEntry entry;
            entry.name = name;
            entry.data = std::move(data);
            entry.hits = 0;
            entry.pinned = false;

            impl_->tensor_lru.push_front(std::move(entry));
            impl_->tensor_map[name] = impl_->tensor_lru.begin();
        }
    }
}

bool HybridTier::pin_tensor(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->tensor_map.find(name);
    if (it == impl_->tensor_map.end()) return false;
    it->second->pinned = true;
    return true;
}

bool HybridTier::unpin_tensor(const std::string& name) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->tensor_map.find(name);
    if (it == impl_->tensor_map.end()) return false;
    it->second->pinned = false;
    return true;
}

bool HybridTier::read_expert(uint32_t layer, uint32_t idx, uint32_t part,
                              std::vector<float>& out) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    uint64_t key = make_key(layer, idx, part);

    auto it = impl_->expert_map.find(key);
    if (it != impl_->expert_map.end()) {
        impl_->touch_expert(it->second);
        impl_->ram_hits++;
        out = it->second->data;
        return true;
    }

    impl_->ram_misses++;
    impl_->usage[key]++;

    if (!impl_->gguf_opened) return false;

    std::vector<float> data;
    if (!impl_->read_expert_from_ssd(key, data)) {
        return false;
    }

    while (static_cast<int32_t>(impl_->expert_map.size()) >= impl_->config.ram_capacity) {
        impl_->evict_expert();
    }

    Impl::TensorEntry entry;
    entry.name = std::to_string(key);
    entry.key = key;
    entry.data = std::move(data);
    entry.hits = 1;
    entry.pinned = (impl_->usage[key] >= impl_->config.ram_pin_threshold);

    impl_->expert_lru.push_front(std::move(entry));
    impl_->expert_map[key] = impl_->expert_lru.begin();
    out = impl_->expert_lru.front().data;
    return true;
}

bool HybridTier::write_expert(uint32_t layer, uint32_t idx, uint32_t part,
                               const std::vector<float>& data) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    uint64_t key = make_key(layer, idx, part);

    auto it = impl_->expert_map.find(key);
    if (it != impl_->expert_map.end()) {
        it->second->data = data;
        impl_->touch_expert(it->second);
    } else {
        while (static_cast<int32_t>(impl_->expert_map.size()) >= impl_->config.ram_capacity) {
            impl_->evict_expert();
        }

        Impl::TensorEntry entry;
        entry.name = std::to_string(key);
        entry.key = key;
        entry.data = data;
        entry.hits = 1;
        entry.pinned = false;

        impl_->expert_lru.push_front(std::move(entry));
        impl_->expert_map[key] = impl_->expert_lru.begin();
    }
    impl_->usage[key]++;
    return true;
}

void HybridTier::prefetch(const std::vector<uint32_t>& layers,
                           const std::vector<uint32_t>& idxs) {
    if (!impl_->config.prefetch_enabled || !impl_->gguf_opened) return;

    for (uint32_t lyr : layers) {
        for (uint32_t idx : idxs) {
            for (uint32_t part = 0; part < 3; ++part) {
                uint64_t key = make_key(lyr, idx, part);
                if (impl_->expert_map.find(key) != impl_->expert_map.end()) continue;

                std::vector<float> data;
                if (impl_->read_expert_from_ssd(key, data)) {
                    while (static_cast<int32_t>(impl_->expert_map.size()) >= impl_->config.ram_capacity) {
                        impl_->evict_expert();
                    }

                    Impl::TensorEntry entry;
                    entry.name = std::to_string(key);
                    entry.key = key;
                    entry.data = std::move(data);
                    entry.hits = 0;
                    entry.pinned = false;

                    impl_->expert_lru.push_front(std::move(entry));
                    impl_->expert_map[key] = impl_->expert_lru.begin();
                }
            }
        }
    }
}

bool HybridTier::promote(uint32_t layer, uint32_t idx, uint32_t part) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    uint64_t key = make_key(layer, idx, part);

    if (impl_->expert_map.find(key) != impl_->expert_map.end()) {
        impl_->expert_map[key]->pinned = true;
        return true;
    }

    if (!impl_->gguf_opened) return false;

    std::vector<float> data;
    if (!impl_->read_expert_from_ssd(key, data)) return false;

    while (static_cast<int32_t>(impl_->expert_map.size()) >= impl_->config.ram_capacity) {
        impl_->evict_expert();
    }

    Impl::TensorEntry entry;
    entry.name = std::to_string(key);
    entry.key = key;
    entry.data = std::move(data);
    entry.hits = 1;
    entry.pinned = true;

    impl_->expert_lru.push_front(std::move(entry));
    impl_->expert_map[key] = impl_->expert_lru.begin();
    return true;
}

bool HybridTier::demote(uint32_t layer, uint32_t idx, uint32_t part) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    uint64_t key = make_key(layer, idx, part);
    auto it = impl_->expert_map.find(key);
    if (it == impl_->expert_map.end()) return false;
    impl_->expert_map.erase(key);
    impl_->expert_lru.erase(it->second);
    return true;
}

bool HybridTier::pin(uint32_t layer, uint32_t idx, uint32_t part) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    uint64_t key = make_key(layer, idx, part);
    auto it = impl_->expert_map.find(key);
    if (it == impl_->expert_map.end()) return false;
    it->second->pinned = true;
    return true;
}

bool HybridTier::unpin(uint32_t layer, uint32_t idx, uint32_t part) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    uint64_t key = make_key(layer, idx, part);
    auto it = impl_->expert_map.find(key);
    if (it == impl_->expert_map.end()) return false;
    it->second->pinned = false;
    return true;
}

void HybridTier::load_usage() {
    if (impl_->usage_file.empty()) return;
    std::ifstream f(impl_->usage_file);
    if (!f) return;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        uint32_t layer, idx, part, count;
        char sep;
        if (ss >> layer >> sep >> idx >> sep >> part >> sep >> count) {
            impl_->usage[make_key(layer, idx, part)] = count;
        }
    }
}

void HybridTier::save_usage() {
    if (impl_->usage_file.empty()) return;
    std::ofstream f(impl_->usage_file);
    if (!f) return;

    for (const auto& kv : impl_->usage) {
        uint32_t layer, idx, part;
        unpack_key(kv.first, layer, idx, part);
        f << layer << ':' << idx << ':' << part << ':' << kv.second << '\n';
    }
}

HybridTier::Stats HybridTier::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    Stats s;
    s.ram_hits = impl_->ram_hits;
    s.ram_misses = impl_->ram_misses;
    s.ssd_reads = impl_->ssd_reads;
    s.ssd_errors = impl_->ssd_errors;
    s.ram_size = static_cast<int32_t>(impl_->tensor_map.size() + impl_->expert_map.size());
    s.pinned_count = 0;
    s.total_bytes_read = impl_->total_bytes_read;
    s.avg_read_ns = impl_->read_count > 0
        ? static_cast<double>(impl_->total_read_ns) / impl_->read_count
        : 0.0;
    for (const auto& entry : impl_->tensor_lru) {
        if (entry.pinned) s.pinned_count++;
    }
    for (const auto& entry : impl_->expert_lru) {
        if (entry.pinned) s.pinned_count++;
    }
    return s;
}

void HybridTier::reset_stats() {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->ram_hits = 0;
    impl_->ram_misses = 0;
    impl_->ssd_reads = 0;
    impl_->ssd_errors = 0;
    impl_->total_bytes_read = 0;
    impl_->total_read_ns = 0;
    impl_->read_count = 0;
}

bool HybridTier::gguf_available() const {
    return impl_->gguf_opened;
}

const std::string& HybridTier::model_path() const {
    return impl_->config.gguf_path;
}

} // namespace desireeia
