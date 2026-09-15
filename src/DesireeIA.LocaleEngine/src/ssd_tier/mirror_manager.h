// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_MIRROR_MANAGER_H
#define DESIREEIA_MIRROR_MANAGER_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace desireeia {

using LogFn = std::function<void(int32_t, const char*)>;

// Configuration for the dual-SSD mirror manager.
struct MirrorConfig {
    // Primary SSD path (required).
    std::string primary_path;
    // Mirror SSD path (empty to disable mirroring).
    std::string mirror_path;
    // Verify reads against both SSDs for data integrity.
    bool verify_on_read = false;
    // Automatically repair a corrupted mirror from the primary.
    bool auto_repair = true;
    // Sync interval in seconds (0 = sync on every write).
    uint32_t sync_interval_seconds = 0;
};

// MirrorManager handles dual-SSD mirroring for fault tolerance.
// It ensures that model data is written to both SSDs and can
// recover from a single SSD failure by reading from the surviving copy.
//
// Thread safety: all public methods are safe to call from multiple
// threads concurrently.
class MirrorManager {
public:
    explicit MirrorManager(const MirrorConfig& config, LogFn log = nullptr);
    ~MirrorManager();

    // Non-copyable, movable.
    MirrorManager(const MirrorManager&) = delete;
    MirrorManager& operator=(const MirrorManager&) = delete;
    MirrorManager(MirrorManager&&) noexcept;
    MirrorManager& operator=(MirrorManager&&) noexcept;

    // Read data from the primary SSD. Falls back to mirror if primary fails.
    // Returns true on success.
    bool read(uint64_t offset, size_t size, std::vector<uint8_t>& out);

    // Write data to both SSDs. Returns true if at least primary write succeeds.
    bool write(uint64_t offset, const uint8_t* data, size_t size);

    // Verify that primary and mirror contain identical data at the given offset.
    // Returns true if they match, false if they differ or either read fails.
    bool verify(uint64_t offset, size_t size);

    // Repair the mirror by copying data from primary to mirror.
    // Returns true on success.
    bool repair(uint64_t offset, size_t size);

    // Full sync: copy the entire model file from primary to mirror.
    // This is a blocking operation for large files.
    bool full_sync();

    // Check if mirroring is enabled.
    bool enabled() const;

    // Get the primary SSD path.
    const std::string& primary_path() const;

    // Get the mirror SSD path.
    const std::string& mirror_path() const;

    // Statistics for monitoring.
    struct Stats {
        uint64_t reads_primary;
        uint64_t reads_mirror;
        uint64_t writes_primary;
        uint64_t writes_mirror;
        uint64_t syncs;
        uint64_t repairs;
        uint64_t verify_errors;
    };
    Stats stats() const;

    // Reset statistics.
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace desireeia

#endif // DESIREEIA_MIRROR_MANAGER_H
