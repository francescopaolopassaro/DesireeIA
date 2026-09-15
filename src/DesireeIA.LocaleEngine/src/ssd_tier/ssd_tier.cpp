// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "ssd_tier.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <list>
#include <mutex>

namespace desireeia {

struct SsdTier::Impl {
    SsdTierConfig config;
    LogFn log;
    mutable std::mutex mtx;

    // LRU cache: most recent at front, oldest at back.
    std::list<SsdBlock> lru_order;
    std::unordered_map<uint64_t, std::list<SsdBlock>::iterator> cache_map;

    // Access timestamps for LRU ordering.
    std::unordered_map<uint64_t, uint64_t> access_times;

    // Statistics.
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;

    // Helper: get current time in nanoseconds.
    static uint64_t now_ns() {
        auto tp = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count());
    }

    // Read raw bytes from the model file at the given offset.
    bool read_from_ssd(const std::string& path, uint64_t offset, size_t size, std::vector<uint8_t>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.seekg(static_cast<std::streamoff>(offset));
        if (!f) return false;
        out.resize(size);
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
        return f.good() || f.eof();
    }

    // Write raw bytes to the model file at the given offset.
    bool write_to_ssd(const std::string& path, uint64_t offset, const uint8_t* data, size_t size) {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f) {
            // Try creating the file if it doesn't exist.
            f = std::fstream(path, std::ios::binary | std::ios::out);
            if (!f) return false;
        }
        f.seekp(static_cast<std::streamoff>(offset));
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        return f.good();
    }

    // Move a block to the front of the LRU list (most recently used).
    void touch(std::list<SsdBlock>::iterator it) {
        lru_order.splice(lru_order.begin(), lru_order, it);
        it->last_access_ns = now_ns();
        it->access_count++;
    }

    // Remove the least recently used unpinned block from cache.
    void evict_one() {
        for (auto it = lru_order.end(); it != lru_order.begin();) {
            --it;
            if (!it->pinned) {
                cache_map.erase(it->key);
                lru_order.erase(it);
                evictions++;
                return;
            }
        }
    }

    // Check if mirror SSD is configured.
    bool has_mirror() const {
        return !config.mirror_path.empty();
    }

    // Log a message if logger is set.
    void log_msg(int32_t level, const char* msg) {
        if (log) {
            log(level, msg);
        }
    }
};

SsdTier::SsdTier(const SsdTierConfig& config, LogFn log)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
    impl_->log = log;
}

SsdTier::~SsdTier() = default;

SsdTier::SsdTier(SsdTier&&) noexcept = default;
SsdTier& SsdTier::operator=(SsdTier&&) noexcept = default;

bool SsdTier::read(uint64_t key, std::vector<uint8_t>& out) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    // Check cache first.
    auto it = impl_->cache_map.find(key);
    if (it != impl_->cache_map.end()) {
        impl_->touch(it->second);
        out = it->second->data;
        impl_->hits++;
        return true;
    }

    // Cache miss: read from SSD.
    impl_->misses++;

    // For now, use the key as an offset into the model file.
    // This is a simplified approach; a real implementation would use
    // a proper mapping from key to file offset.
    const size_t block_size = 4096; // Default block size.
    const uint64_t offset = key * block_size;

    std::vector<uint8_t> data;
    bool read_ok = false;

    // Try primary SSD first.
    if (!impl_->config.model_path.empty()) {
        read_ok = impl_->read_from_ssd(impl_->config.model_path, offset, block_size, data);
    }

    // Fall back to mirror SSD if primary fails.
    if (!read_ok && impl_->has_mirror()) {
        read_ok = impl_->read_from_ssd(impl_->config.mirror_path, offset, block_size, data);
    }

    if (!read_ok) {
        impl_->log_msg(3, "ssd tier: read failed for key");
        return false;
    }

    // Evict if cache is full.
    while (static_cast<int32_t>(impl_->cache_map.size()) >= impl_->config.cache_capacity) {
        impl_->evict_one();
    }

    // Insert into cache.
    SsdBlock block;
    block.key = key;
    block.data = std::move(data);
    block.access_count = 1;
    block.last_access_ns = Impl::now_ns();
    block.pinned = false;

    impl_->lru_order.push_front(std::move(block));
    impl_->cache_map[key] = impl_->lru_order.begin();
    out = impl_->lru_order.front().data;

    return true;
}

bool SsdTier::write(uint64_t key, const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    const size_t block_size = 4096;
    const uint64_t offset = key * block_size;

    // Write to primary SSD.
    bool write_ok = false;
    if (!impl_->config.model_path.empty()) {
        write_ok = impl_->write_to_ssd(impl_->config.model_path, offset, data, size);
    }

    // Write to mirror SSD if enabled.
    if (impl_->has_mirror()) {
        impl_->write_to_ssd(impl_->config.mirror_path, offset, data, size);
    }

    if (!write_ok) {
        impl_->log_msg(3, "ssd tier: write failed for key");
        return false;
    }

    // Update cache with new data.
    auto it = impl_->cache_map.find(key);
    if (it != impl_->cache_map.end()) {
        it->second->data.assign(data, data + size);
        impl_->touch(it->second);
    } else {
        // Evict if cache is full.
        while (static_cast<int32_t>(impl_->cache_map.size()) >= impl_->config.cache_capacity) {
            impl_->evict_one();
        }

        SsdBlock block;
        block.key = key;
        block.data.assign(data, data + size);
        block.access_count = 1;
        block.last_access_ns = Impl::now_ns();
        block.pinned = false;

        impl_->lru_order.push_front(std::move(block));
        impl_->cache_map[key] = impl_->lru_order.begin();
    }

    return true;
}

void SsdTier::prefetch(const std::vector<uint64_t>& keys) {
    std::vector<uint64_t> to_fetch;
    {
        std::lock_guard<std::mutex> lock(impl_->mtx);
        for (uint64_t key : keys) {
            if (impl_->cache_map.find(key) == impl_->cache_map.end()) {
                to_fetch.push_back(key);
            }
        }
    }

    // Read missing blocks (outside lock to allow concurrent reads).
    for (uint64_t key : to_fetch) {
        std::vector<uint8_t> data;
        read(key, data);
    }
}

void SsdTier::evict() {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    while (static_cast<int32_t>(impl_->cache_map.size()) > impl_->config.cache_capacity / 2) {
        impl_->evict_one();
    }
}

bool SsdTier::pin(uint64_t key) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->cache_map.find(key);
    if (it == impl_->cache_map.end()) {
        return false;
    }
    it->second->pinned = true;
    return true;
}

bool SsdTier::unpin(uint64_t key) {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    auto it = impl_->cache_map.find(key);
    if (it == impl_->cache_map.end()) {
        return false;
    }
    it->second->pinned = false;
    return true;
}

SsdTier::Stats SsdTier::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    Stats s;
    s.hits = impl_->hits;
    s.misses = impl_->misses;
    s.evictions = impl_->evictions;
    s.current_size = static_cast<int32_t>(impl_->cache_map.size());
    s.pinned_count = 0;
    for (const auto& block : impl_->lru_order) {
        if (block.pinned) s.pinned_count++;
    }
    return s;
}

void SsdTier::reset_stats() {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->hits = 0;
    impl_->misses = 0;
    impl_->evictions = 0;
}

bool SsdTier::mirror_active() const {
    return impl_->has_mirror();
}

const std::string& SsdTier::model_path() const {
    return impl_->config.model_path;
}

} // namespace desireeia
