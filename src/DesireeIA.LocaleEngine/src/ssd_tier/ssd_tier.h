#ifndef DESIREEIA_SSD_TIER_H
#define DESIREEIA_SSD_TIER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace desireeia {

using LogFn = std::function<void(int32_t, const char*)>;

// Represents a single cached block from SSD storage.
// Each block holds raw byte data and metadata for eviction decisions.
struct SsdBlock {
    uint64_t key;
    std::vector<uint8_t> data;
    uint32_t access_count;
    uint64_t last_access_ns;
    bool pinned;
};

// Configuration for the SSD tier cache.
struct SsdTierConfig {
    // Maximum number of blocks to keep in the in-memory cache.
    int32_t cache_capacity = 1024;
    // Path to the model file on SSD for direct reads.
    std::string model_path;
    // Path to the mirror SSD (empty if single-SSD mode).
    std::string mirror_path;
    // Enable read-ahead prefetching for sequential access patterns.
    bool prefetch_enabled = true;
    // Number of blocks to prefetch ahead during sequential reads.
    int32_t prefetch_depth = 4;
    // Minimum access count before a block is eligible for pinning.
    uint32_t pin_threshold = 4;
};

// SsdTier provides a tiered storage layer for model data.
// It manages an in-memory LRU cache backed by direct SSD reads,
// with optional dual-SSD mirroring for fault tolerance.
//
// Thread safety: all public methods are safe to call from multiple
// threads concurrently. Internal locking protects shared state.
class SsdTier {
public:
    explicit SsdTier(const SsdTierConfig& config, LogFn log = nullptr);
    ~SsdTier();

    // Non-copyable, movable.
    SsdTier(const SsdTier&) = delete;
    SsdTier& operator=(const SsdTier&) = delete;
    SsdTier(SsdTier&&) noexcept;
    SsdTier& operator=(SsdTier&&) noexcept;

    // Read a block by key. Returns true on success.
    // If the block is cached, returns from memory; otherwise reads from SSD.
    bool read(uint64_t key, std::vector<uint8_t>& out);

    // Write a block to both SSD and cache. Used for new expert data.
    bool write(uint64_t key, const uint8_t* data, size_t size);

    // Prefetch a range of keys into the cache.
    // Keys are read sequentially from SSD if not already cached.
    void prefetch(const std::vector<uint64_t>& keys);

    // Evict least-recently-used unpinned entries until cache is below capacity.
    void evict();

    // Pin a block to prevent eviction. Returns false if key not cached.
    bool pin(uint64_t key);

    // Unpin a block to allow future eviction.
    bool unpin(uint64_t key);

    // Get cache statistics for monitoring.
    struct Stats {
        uint64_t hits;
        uint64_t misses;
        uint64_t evictions;
        int32_t current_size;
        int32_t pinned_count;
    };
    Stats stats() const;

    // Reset cache statistics.
    void reset_stats();

    // Check if dual-SSD mirroring is active.
    bool mirror_active() const;

    // Get the configured model path.
    const std::string& model_path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace desireeia

#endif // DESIREEIA_SSD_TIER_H
