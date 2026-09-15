// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "mirror_manager.h"
#include <cstring>
#include <fstream>
#include <mutex>

namespace desireeia {

struct MirrorManager::Impl {
    MirrorConfig config;
    LogFn log;
    mutable std::mutex mtx;
    Stats stats;

    Impl() : stats{} {}

    // Read raw bytes from a file at the given offset.
    bool read_file(const std::string& path, uint64_t offset, size_t size, std::vector<uint8_t>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.seekg(static_cast<std::streamoff>(offset));
        if (!f) return false;
        out.resize(size);
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(size));
        return f.good() || f.eof();
    }

    // Write raw bytes to a file at the given offset.
    bool write_file(const std::string& path, uint64_t offset, const uint8_t* data, size_t size) {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f) {
            f = std::fstream(path, std::ios::binary | std::ios::out);
            if (!f) return false;
        }
        f.seekp(static_cast<std::streamoff>(offset));
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        return f.good();
    }

    // Get file size.
    int64_t file_size(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return -1;
        return f.tellg();
    }

    void log_msg(int32_t level, const char* msg) {
        if (log) {
            log(level, msg);
        }
    }
};

MirrorManager::MirrorManager(const MirrorConfig& config, LogFn log)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
    impl_->log = log;
}

MirrorManager::~MirrorManager() = default;

MirrorManager::MirrorManager(MirrorManager&&) noexcept = default;
MirrorManager& MirrorManager::operator=(MirrorManager&&) noexcept = default;

bool MirrorManager::read(uint64_t offset, size_t size, std::vector<uint8_t>& out) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    // Try primary first.
    if (!impl_->config.primary_path.empty()) {
        if (impl_->read_file(impl_->config.primary_path, offset, size, out)) {
            impl_->stats.reads_primary++;

            // Optional verification against mirror.
            if (impl_->config.verify_on_read && enabled()) {
                std::vector<uint8_t> mirror_data;
                if (impl_->read_file(impl_->config.mirror_path, offset, size, mirror_data)) {
                    if (std::memcmp(out.data(), mirror_data.data(), size) != 0) {
                        impl_->stats.verify_errors++;
                        impl_->log_msg(3, "mirror verify mismatch on read");

                        // Auto-repair mirror from primary.
                        if (impl_->config.auto_repair) {
                            impl_->write_file(impl_->config.mirror_path, offset, out.data(), size);
                            impl_->stats.repairs++;
                        }
                    }
                }
            }

            return true;
        }
    }

    // Fall back to mirror.
    if (enabled()) {
        if (impl_->read_file(impl_->config.mirror_path, offset, size, out)) {
            impl_->stats.reads_mirror++;

            // Auto-repair primary from mirror.
            if (impl_->config.auto_repair) {
                impl_->write_file(impl_->config.primary_path, offset, out.data(), size);
                impl_->stats.repairs++;
                impl_->log_msg(4, "repaired primary from mirror");
            }

            return true;
        }
    }

    return false;
}

bool MirrorManager::write(uint64_t offset, const uint8_t* data, size_t size) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    bool primary_ok = false;
    bool mirror_ok = false;

    // Write to primary.
    if (!impl_->config.primary_path.empty()) {
        primary_ok = impl_->write_file(impl_->config.primary_path, offset, data, size);
        if (primary_ok) {
            impl_->stats.writes_primary++;
        }
    }

    // Write to mirror.
    if (enabled()) {
        mirror_ok = impl_->write_file(impl_->config.mirror_path, offset, data, size);
        if (mirror_ok) {
            impl_->stats.writes_mirror++;
        }
    }

    return primary_ok;
}

bool MirrorManager::verify(uint64_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    if (!enabled()) return false;

    std::vector<uint8_t> primary_data, mirror_data;
    if (!impl_->read_file(impl_->config.primary_path, offset, size, primary_data)) {
        return false;
    }
    if (!impl_->read_file(impl_->config.mirror_path, offset, size, mirror_data)) {
        return false;
    }

    bool match = (primary_data.size() == mirror_data.size()) &&
                 (std::memcmp(primary_data.data(), mirror_data.data(), size) == 0);
    if (!match) {
        impl_->stats.verify_errors++;
    }
    return match;
}

bool MirrorManager::repair(uint64_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    if (!enabled()) return false;

    std::vector<uint8_t> primary_data;
    if (!impl_->read_file(impl_->config.primary_path, offset, size, primary_data)) {
        return false;
    }

    bool ok = impl_->write_file(impl_->config.mirror_path, offset, primary_data.data(), size);
    if (ok) {
        impl_->stats.repairs++;
    }
    return ok;
}

bool MirrorManager::full_sync() {
    std::lock_guard<std::mutex> lock(impl_->mtx);

    if (!enabled()) return false;

    int64_t size = impl_->file_size(impl_->config.primary_path);
    if (size <= 0) return false;

    const size_t chunk_size = 4 * 1024 * 1024; // 4 MB chunks.
    std::vector<uint8_t> buffer(chunk_size);

    for (int64_t offset = 0; offset < size; offset += chunk_size) {
        size_t to_read = static_cast<size_t>(std::min(static_cast<int64_t>(chunk_size), size - offset));
        if (!impl_->read_file(impl_->config.primary_path, static_cast<uint64_t>(offset), to_read, buffer)) {
            return false;
        }
        if (!impl_->write_file(impl_->config.mirror_path, static_cast<uint64_t>(offset), buffer.data(), to_read)) {
            return false;
        }
    }

    impl_->stats.syncs++;
    return true;
}

bool MirrorManager::enabled() const {
    return !impl_->config.primary_path.empty() && !impl_->config.mirror_path.empty();
}

const std::string& MirrorManager::primary_path() const {
    return impl_->config.primary_path;
}

const std::string& MirrorManager::mirror_path() const {
    return impl_->config.mirror_path;
}

MirrorManager::Stats MirrorManager::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    return impl_->stats;
}

void MirrorManager::reset_stats() {
    std::lock_guard<std::mutex> lock(impl_->mtx);
    impl_->stats = Stats{};
}

} // namespace desireeia
