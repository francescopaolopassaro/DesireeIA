#ifndef DESIREEIA_TIERED_EXPERT_STORE_H
#define DESIREEIA_TIERED_EXPERT_STORE_H

#include "ssd_tier.h"
#include "mirror_manager.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace desireeia {

using LogFn = std::function<void(int32_t, const char*)>;

// Configuration for the tiered expert store.
struct TieredExpertConfig {
    // Maximum number of experts to keep in RAM cache.
    int32_t ram_cache_capacity = 256;
    // SSD tier cache capacity (in blocks).
    int32_t ssd_cache_capacity = 1024;
    // Minimum access count before an expert is eligible for pinning.
    uint32_t pin_threshold = 4;
    // Enable prefetching of next-layer experts.
    bool prefetch_enabled = true;
    // Number of layers to prefetch ahead.
    int32_t prefetch_depth = 1;
    // Path to the model file on SSD.
    std::string model_path;
    // Path to the mirror SSD (empty to disable).
    std::string mirror_path;
    // Path to the usage file for persistence.
    std::string usage_file;
};

// TieredExpertStore provides a two-tier cache for MoE expert data.
// Hot experts reside in RAM (fastest access), while cold experts
// are stored on SSD with an in-memory block cache.
//
// The store automatically promotes frequently accessed experts to RAM
// and demotes cold experts to SSD, implementing an LRU eviction policy
// with configurable pin thresholds.
//
// Thread safety: all public methods are safe to call from multiple
// threads concurrently.
class TieredExpertStore {
public:
    explicit TieredExpertStore(const TieredExpertConfig& config, LogFn log = nullptr);
    ~TieredExpertStore();

    // Non-copyable, movable.
    TieredExpertStore(const TieredExpertStore&) = delete;
    TieredExpertStore& operator=(const TieredExpertStore&) = delete;
    TieredExpertStore(TieredExpertStore&&) noexcept;
    TieredExpertStore& operator=(TieredExpertStore&&) noexcept;

    // Read expert data by key. Returns true on success.
    // Data is served from RAM cache if available, otherwise from SSD tier.
    bool read(uint64_t key, std::vector<float>& out);

    // Write expert data to the store.
    // Data is written to both RAM cache and SSD tier.
    bool write(uint64_t key, const std::vector<float>& data);

    // Prefetch a list of expert keys into the cache.
    void prefetch(const std::vector<uint64_t>& keys);

    // Promote an expert from SSD tier to RAM cache.
    bool promote(uint64_t key);

    // Demote an expert from RAM cache to SSD tier.
    bool demote(uint64_t key);

    // Pin an expert in RAM to prevent eviction.
    bool pin(uint64_t key);

    // Unpin an expert to allow future eviction.
    bool unpin(uint64_t key);

    // Load usage statistics from the persistence file.
    void load_usage();

    // Save usage statistics to the persistence file.
    void save_usage();

    // Get cache statistics for monitoring.
    struct Stats {
        uint64_t ram_hits;
        uint64_t ram_misses;
        uint64_t ssd_hits;
        uint64_t ssd_misses;
        int32_t ram_size;
        int32_t ssd_size;
        int32_t pinned_count;
    };
    Stats stats() const;

    // Reset statistics.
    void reset_stats();

    // Get access to the underlying SSD tier (for advanced operations).
    SsdTier& ssd_tier();

    // Get access to the mirror manager (for advanced operations).
    MirrorManager& mirror_manager();

    // Check if dual-SSD mirroring is active.
    bool mirror_active() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace desireeia

#endif // DESIREEIA_TIERED_EXPERT_STORE_H
