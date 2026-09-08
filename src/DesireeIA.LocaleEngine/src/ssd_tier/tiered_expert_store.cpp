#include "tiered_expert_store.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <list>
#include <sstream>
#include <unordered_map>

namespace desireeia {

// Convert expert key components to a single 64-bit key.
static uint64_t make_expert_key(uint32_t layer, uint32_t idx, uint32_t part) {
    return (static_cast<uint64_t>(layer) << 48) |
           (static_cast<uint64_t>(idx) << 8) |
           static_cast<uint64_t>(part);
}

// Extract components from a 64-bit expert key.
static void unpack_expert_key(uint64_t key, uint32_t& layer, uint32_t& idx, uint32_t& part) {
    layer = static_cast<uint32_t>(key >> 48);
    idx = static_cast<uint32_t>((key >> 8) & 0xFFFFFFFFu);
    part = static_cast<uint32_t>(key & 0xFFu);
}

struct TieredExpertStore::Impl {
    TieredExpertConfig config;
    LogFn log;

    // RAM cache: LRU list of expert data.
    struct RamEntry {
        uint64_t key;
        std::vector<float> data;
        uint32_t hits;
        bool pinned;
    };
    std::list<RamEntry> ram_lru;
    std::unordered_map<uint64_t, std::list<RamEntry>::iterator> ram_map;

    // SSD tier for cold data.
    std::unique_ptr<SsdTier> ssd_tier;

    // Mirror manager for fault tolerance.
    std::unique_ptr<MirrorManager> mirror_mgr;

    // Usage tracking for persistence.
    std::unordered_map<uint64_t, uint32_t> usage;
    std::string usage_file;

    // Statistics.
    uint64_t ram_hits = 0;
    uint64_t ram_misses = 0;
    uint64_t ssd_hits = 0;
    uint64_t ssd_misses = 0;

    // Move a RAM entry to the front of the LRU list.
    void touch_ram(std::list<RamEntry>::iterator it) {
        ram_lru.splice(ram_lru.begin(), ram_lru, it);
        it->hits++;
    }

    // Evict the least recently used unpinned entry from RAM.
    void evict_ram() {
        for (auto it = ram_lru.end(); it != ram_lru.begin();) {
            --it;
            if (!it->pinned) {
                // Demote to SSD before evicting from RAM.
                demote_to_ssd(it->key, it->data);
                ram_map.erase(it->key);
                ram_lru.erase(it);
                return;
            }
        }
    }

    // Demote expert data from RAM to SSD.
    void demote_to_ssd(uint64_t key, const std::vector<float>& data) {
        if (!ssd_tier) return;

        // Convert float data to bytes for SSD storage.
        std::vector<uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());

        ssd_tier->write(key, bytes.data(), bytes.size());
    }

    // Load expert data from SSD into a float vector.
    bool load_from_ssd(uint64_t key, std::vector<float>& out) {
        if (!ssd_tier) return false;

        std::vector<uint8_t> bytes;
        if (!ssd_tier->read(key, bytes)) {
            return false;
        }

        // Convert bytes back to floats.
        if (bytes.size() % sizeof(float) != 0) {
            return false;
        }
        size_t count = bytes.size() / sizeof(float);
        out.resize(count);
        std::memcpy(out.data(), bytes.data(), bytes.size());
        return true;
    }

    void log_msg(int32_t level, const char* msg) {
        if (log) {
            log(level, msg);
        }
    }
};

TieredExpertStore::TieredExpertStore(const TieredExpertConfig& config, LogFn log)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
    impl_->log = log;
    impl_->usage_file = config.usage_file;

    // Initialize SSD tier.
    SsdTierConfig ssd_config;
    ssd_config.cache_capacity = config.ssd_cache_capacity;
    ssd_config.model_path = config.model_path;
    ssd_config.mirror_path = config.mirror_path;
    impl_->ssd_tier = std::make_unique<SsdTier>(ssd_config, log);

    // Initialize mirror manager.
    if (!config.mirror_path.empty()) {
        MirrorConfig mirror_config;
        mirror_config.primary_path = config.model_path;
        mirror_config.mirror_path = config.mirror_path;
        impl_->mirror_mgr = std::make_unique<MirrorManager>(mirror_config, log);
    }
}

TieredExpertStore::~TieredExpertStore() = default;

TieredExpertStore::TieredExpertStore(TieredExpertStore&&) noexcept = default;
TieredExpertStore& TieredExpertStore::operator=(TieredExpertStore&&) noexcept = default;

bool TieredExpertStore::read(uint64_t key, std::vector<float>& out) {
    // Try RAM cache first.
    auto it = impl_->ram_map.find(key);
    if (it != impl_->ram_map.end()) {
        impl_->touch_ram(it->second);
        impl_->ram_hits++;
        out = it->second->data;
        return true;
    }

    // RAM cache miss.
    impl_->ram_misses++;
    impl_->usage[key]++;

    // Try SSD tier.
    std::vector<float> data;
    if (impl_->load_from_ssd(key, data)) {
        impl_->ssd_hits++;

        // Promote to RAM cache.
        while (static_cast<int32_t>(impl_->ram_map.size()) >= impl_->config.ram_cache_capacity) {
            impl_->evict_ram();
        }

        Impl::RamEntry entry;
        entry.key = key;
        entry.data = std::move(data);
        entry.hits = 1;
        entry.pinned = (impl_->usage[key] >= impl_->config.pin_threshold);

        impl_->ram_lru.push_front(std::move(entry));
        impl_->ram_map[key] = impl_->ram_lru.begin();
        out = impl_->ram_lru.front().data;
        return true;
    }

    // SSD cache miss too.
    impl_->ssd_misses++;
    return false;
}

bool TieredExpertStore::write(uint64_t key, const std::vector<float>& data) {
    // Write to SSD tier first.
    std::vector<uint8_t> bytes(data.size() * sizeof(float));
    std::memcpy(bytes.data(), data.data(), bytes.size());

    if (impl_->ssd_tier) {
        impl_->ssd_tier->write(key, bytes.data(), bytes.size());
    }

    // Update RAM cache if present.
    auto it = impl_->ram_map.find(key);
    if (it != impl_->ram_map.end()) {
        it->second->data = data;
        impl_->touch_ram(it->second);
    } else {
        // Add to RAM cache.
        while (static_cast<int32_t>(impl_->ram_map.size()) >= impl_->config.ram_cache_capacity) {
            impl_->evict_ram();
        }

        Impl::RamEntry entry;
        entry.key = key;
        entry.data = data;
        entry.hits = 1;
        entry.pinned = false;

        impl_->ram_lru.push_front(std::move(entry));
        impl_->ram_map[key] = impl_->ram_lru.begin();
    }

    impl_->usage[key]++;
    return true;
}

void TieredExpertStore::prefetch(const std::vector<uint64_t>& keys) {
    // Prefetch into SSD tier cache.
    if (impl_->ssd_tier) {
        impl_->ssd_tier->prefetch(keys);
    }

    // Optionally prefetch into RAM cache.
    if (impl_->config.prefetch_enabled) {
        for (uint64_t key : keys) {
            std::vector<float> data;
            read(key, data);
        }
    }
}

bool TieredExpertStore::promote(uint64_t key) {
    // Check if already in RAM.
    if (impl_->ram_map.find(key) != impl_->ram_map.end()) {
        return true;
    }

    // Load from SSD.
    std::vector<float> data;
    if (!impl_->load_from_ssd(key, data)) {
        return false;
    }

    // Add to RAM cache.
    while (static_cast<int32_t>(impl_->ram_map.size()) >= impl_->config.ram_cache_capacity) {
        impl_->evict_ram();
    }

    Impl::RamEntry entry;
    entry.key = key;
    entry.data = std::move(data);
    entry.hits = 1;
    entry.pinned = true; // Promoted entries are pinned.

    impl_->ram_lru.push_front(std::move(entry));
    impl_->ram_map[key] = impl_->ram_lru.begin();
    return true;
}

bool TieredExpertStore::demote(uint64_t key) {
    auto it = impl_->ram_map.find(key);
    if (it == impl_->ram_map.end()) {
        return false;
    }

    // Demote to SSD.
    impl_->demote_to_ssd(key, it->second->data);

    // Remove from RAM.
    impl_->ram_map.erase(key);
    impl_->ram_lru.erase(it->second);
    return true;
}

bool TieredExpertStore::pin(uint64_t key) {
    auto it = impl_->ram_map.find(key);
    if (it == impl_->ram_map.end()) {
        return false;
    }
    it->second->pinned = true;
    return true;
}

bool TieredExpertStore::unpin(uint64_t key) {
    auto it = impl_->ram_map.find(key);
    if (it == impl_->ram_map.end()) {
        return false;
    }
    it->second->pinned = false;
    return true;
}

void TieredExpertStore::load_usage() {
    if (impl_->usage_file.empty()) return;
    std::ifstream f(impl_->usage_file);
    if (!f) return;

    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        uint32_t layer, idx, part, count;
        char sep;
        if (ss >> layer >> sep >> idx >> sep >> part >> sep >> count) {
            impl_->usage[make_expert_key(layer, idx, part)] = count;
        }
    }
}

void TieredExpertStore::save_usage() {
    if (impl_->usage_file.empty()) return;
    std::ofstream f(impl_->usage_file);
    if (!f) return;

    for (const auto& kv : impl_->usage) {
        uint32_t layer, idx, part;
        unpack_expert_key(kv.first, layer, idx, part);
        f << layer << ':' << idx << ':' << part << ':' << kv.second << '\n';
    }
}

TieredExpertStore::Stats TieredExpertStore::stats() const {
    Stats s;
    s.ram_hits = impl_->ram_hits;
    s.ram_misses = impl_->ram_misses;
    s.ssd_hits = impl_->ssd_hits;
    s.ssd_misses = impl_->ssd_misses;
    s.ram_size = static_cast<int32_t>(impl_->ram_map.size());
    s.ssd_size = impl_->ssd_tier ? impl_->ssd_tier->stats().current_size : 0;
    s.pinned_count = 0;
    for (const auto& entry : impl_->ram_lru) {
        if (entry.pinned) s.pinned_count++;
    }
    return s;
}

void TieredExpertStore::reset_stats() {
    impl_->ram_hits = 0;
    impl_->ram_misses = 0;
    impl_->ssd_hits = 0;
    impl_->ssd_misses = 0;
    if (impl_->ssd_tier) {
        impl_->ssd_tier->reset_stats();
    }
}

SsdTier& TieredExpertStore::ssd_tier() {
    return *impl_->ssd_tier;
}

MirrorManager& TieredExpertStore::mirror_manager() {
    return *impl_->mirror_mgr;
}

bool TieredExpertStore::mirror_active() const {
    return impl_->mirror_mgr && impl_->mirror_mgr->enabled();
}

} // namespace desireeia
