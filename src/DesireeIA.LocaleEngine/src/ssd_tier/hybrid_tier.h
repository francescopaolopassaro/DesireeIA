// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_HYBRID_TIER_H
#define DESIREEIA_HYBRID_TIER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace desireeia {

using LogFn = std::function<void(int32_t, const char*)>;

// Declared in core/engine.h. Only ever used through a pointer here, so a
// forward declaration keeps this header free of the engine header (which
// itself includes this one).
class ModelReader;

// 3-tier hybrid cache configuration: CPU L3 -> RAM -> SSD.
struct HybridTierConfig {
    // -- RAM tier --
    // Maximum number of expert entries to keep in RAM.
    int32_t ram_capacity = 256;
    // Budget in BYTES for the raw quantized tensor cache — the one on the
    // path the matmul kernels actually take. Counted in bytes rather than
    // entries because tensors in one model span from a few KiB to hundreds of
    // MiB, so an entry count says nothing about memory used. 0 disables it.
    uint64_t ram_bytes_max = 1024ull * 1024ull * 1024ull;
    // Minimum access count before an expert is eligible for pinning in RAM.
    uint32_t ram_pin_threshold = 4;

    // -- SSD tier --
    // Path to the GGUF model file on SSD.
    std::string gguf_path;
    // Maximum number of entries in the SSD block cache (in-memory index
    // over SSD-resident data). Does NOT copy data into RAM; only tracks
    // which blocks are on SSD for fast lookup.
    int32_t ssd_index_capacity = 4096;
    // Enable read-ahead prefetch for sequential layer traversal.
    bool prefetch_enabled = true;
    // Number of layers to prefetch ahead during decode.
    int32_t prefetch_depth = 1;

    // -- Mirror (optional) --
    // Mirror SSD path for fault tolerance (empty = disabled).
    std::string mirror_path;

    // -- Persistence --
    // Path to the usage file for hot-set tracking across sessions.
    std::string usage_file;
};

// Access pattern tracker for a single expert key.
struct ExpertAccess {
    uint64_t key;
    uint32_t total_accesses;
    uint32_t recent_accesses;   // Accesses in the current sliding window.
    uint64_t last_access_ns;
    uint32_t consecutive_layers; // How many consecutive layers accessed this expert.
};

// HybridTier provides a 3-level tiered cache for model data read from GGUF:
//
//   Tier 0 (RAM):  Hot tensors kept in memory, LRU-evicted to SSD.
//   Tier 1 (SSD):  Cold tensors stored in the GGUF file on disk.
//   Tier 2 (Mirror): Optional redundant copy on a second SSD.
//
// The class reads tensor data directly from the GGUF file on SSD,
// caches hot tensors in RAM, and evicts cold tensors back to SSD.
// Works with any tensor type (dense or MoE expert).
//
// Thread safety: all public methods are safe to call from multiple
// threads concurrently.
class HybridTier {
public:
    explicit HybridTier(const HybridTierConfig& config, LogFn log = nullptr);
    ~HybridTier();

    // Non-copyable, movable.
    HybridTier(const HybridTier&) = delete;
    HybridTier& operator=(const HybridTier&) = delete;
    HybridTier(HybridTier&&) noexcept;
    HybridTier& operator=(HybridTier&&) noexcept;

    // Sets the reader used to fetch anything not already cached. Not owned;
    // must outlive this tier.
    //
    // The tier deliberately does NOT parse the model file itself. It used to,
    // with its own header parser, and that parser disagreed with the real one
    // about where tensor data starts (alignment, and older file versions), so
    // every read it served landed at the wrong offset and produced weights
    // that looked plausible but were wrong. Reading through the same reader
    // the rest of the engine uses removes that entire class of bug: there is
    // only one implementation that can be right or wrong, and it is the one
    // already covered by tests.
    void set_source(ModelReader* reader);

    // -- General tensor reads (any GGUF tensor by name) --
    // Read a full tensor by name, dequantized to float.
    // Served from RAM cache if available; otherwise read through the source.
    bool read_tensor(const std::string& name, std::vector<float>& out);

    // Read raw (quantized) tensor bytes by name.
    bool read_tensor_raw(const std::string& name, std::vector<uint8_t>& raw,
                         int& quant_type, uint64_t& ne0, uint64_t& rows);

    // -- MoE expert reads (convenience, builds tensor name internally) --
    bool read_expert(uint32_t layer, uint32_t idx, uint32_t part,
                     std::vector<float>& out);

    // Write tensor data to RAM cache.
    bool write_tensor(const std::string& name, const std::vector<float>& data);

    // Prefetch a list of tensor names into RAM cache.
    void prefetch_tensors(const std::vector<std::string>& names);

    // Prefetch experts for the given layers into RAM cache.
    void prefetch(const std::vector<uint32_t>& layers,
                  const std::vector<uint32_t>& idxs);

    // Pin/unpin a tensor in RAM to prevent/allow eviction.
    bool pin_tensor(const std::string& name);
    bool unpin_tensor(const std::string& name);

    // Legacy MoE convenience methods.
    bool promote(uint32_t layer, uint32_t idx, uint32_t part);
    bool demote(uint32_t layer, uint32_t idx, uint32_t part);
    bool pin(uint32_t layer, uint32_t idx, uint32_t part);
    bool unpin(uint32_t layer, uint32_t idx, uint32_t part);

    // Write expert data (MoE convenience).
    bool write_expert(uint32_t layer, uint32_t idx, uint32_t part,
                      const std::vector<float>& data);

    // -- Persistence --
    void load_usage();
    void save_usage();

    // -- Statistics --
    struct Stats {
        uint64_t ram_hits;
        uint64_t ram_misses;
        uint64_t ssd_reads;
        uint64_t ssd_errors;
        int32_t  ram_size;
        int32_t  pinned_count;
        uint64_t total_bytes_read;
        double   avg_read_ns;
    };
    Stats stats() const;
    void reset_stats();

    // Check if the GGUF file is accessible.
    bool gguf_available() const;

    // Get the model path.
    const std::string& model_path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace desireeia

#endif // DESIREEIA_HYBRID_TIER_H
