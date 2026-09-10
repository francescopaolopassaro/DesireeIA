#include "../core/engine.h"
#include "../ssd_tier/hybrid_tier.h"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace desireeia {

namespace {
constexpr uint32_t PIN_THRESHOLD = 4;
}

ExpertStore::ExpertStore(int32_t cache_count, LogFn log)
    : cache_count_(cache_count > 0 ? cache_count : 256), log_(log) {
    worker_ = std::thread(&ExpertStore::prefetch_worker_loop, this);
}

ExpertStore::~ExpertStore() {
    {
        std::lock_guard<std::mutex> lk(qmtx_);
        worker_stop_ = true;
    }
    qcv_.notify_one();
    if (worker_.joinable()) worker_.join();
}

bool ExpertStore::load_data(uint64_t key, std::vector<float>& out) {
    const uint32_t layer = static_cast<uint32_t>(key >> 48);
    const uint32_t idx = static_cast<uint32_t>((key >> 8) & 0xFFFFFFFFu);
    const ExpertPart part = static_cast<ExpertPart>(key & 0xFFu);

    // Held only across the physical read: this is the one thing that must
    // never run concurrently between the caller thread and the prefetch
    // worker (see the note on reader_mtx_ in engine.h). Never held
    // together with mtx_, so a slow read here never blocks the other
    // thread's cache lookups/inserts.
    std::lock_guard<std::mutex> lk(reader_mtx_);

    // If a ModelReader is available, use it (existing path).
    if (reader_) {
        return reader_->read_expert(layer, idx, part, out);
    }

    // Fallback: use HybridTier for SSD-backed reads.
    if (hybrid_tier_) {
        return hybrid_tier_->read_expert(layer, idx, static_cast<uint32_t>(part), out);
    }

    // Stub path (e.g. isolated ExpertStore unit tests).
    out.assign(1, 0.0f);
    return true;
}

void ExpertStore::insert_locked(uint64_t key, std::vector<float>&& data) {
    if (map_.count(key)) return; // already resident (e.g. raced with fetch())
    if (map_.size() >= static_cast<size_t>(cache_count_)) {
        for (auto it2 = order_.end(); it2 != order_.begin();) {
            --it2;
            if (!it2->pinned) {
                map_.erase(it2->key);
                order_.erase(it2);
                break;
            }
        }
    }
    Entry e{key, std::move(data), 0, false};
    if (pin_enabled_ && usage_[key] >= PIN_THRESHOLD) {
        e.pinned = true;
    }
    order_.push_front(std::move(e));
    map_[key] = order_.begin();
}

bool ExpertStore::fetch(uint32_t layer, uint32_t idx, ExpertPart part, std::vector<float>& out) {
    uint64_t key = make_key(layer, idx, part);

    {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            order_.splice(order_.begin(), order_, it->second);
            it->second->hits++;
            if (!it->second->pinned && pin_enabled_ && it->second->hits >= PIN_THRESHOLD) {
                it->second->pinned = true;
                if (log_) {
                    log_(5, "expert pinned");
                }
            }
            out = it->second->data;
            return true;
        }
        usage_[key]++;
    }

    // Miss: read without holding mtx_ (see load_data's own lock), so the
    // prefetch worker can keep inserting other keys while this blocks.
    std::vector<float> data;
    if (!load_data(key, data)) {
        return false;
    }

    std::lock_guard<std::mutex> lk(mtx_);
    // The worker may have inserted this exact key while we were reading
    // it ourselves (both wanted the same expert at once) — re-check.
    auto it2 = map_.find(key);
    if (it2 != map_.end()) {
        out = it2->second->data;
        return true;
    }
    insert_locked(key, std::move(data));
    out = map_[key]->data;
    return true;
}

void ExpertStore::prefetch_async(uint32_t layer, const std::vector<uint32_t>& idxs) {
    {
        std::lock_guard<std::mutex> lk(qmtx_);
        pending_layer_ = layer;
        pending_idxs_ = idxs;
        has_pending_ = true;
    }
    qcv_.notify_one();
}

void ExpertStore::prefetch_worker_loop() {
    static const ExpertPart parts[3] = { ExpertPart::Gate, ExpertPart::Up, ExpertPart::Down };
    for (;;) {
        uint32_t layer;
        std::vector<uint32_t> idxs;
        {
            std::unique_lock<std::mutex> lk(qmtx_);
            qcv_.wait(lk, [&] { return has_pending_ || worker_stop_; });
            if (worker_stop_) return;
            layer = pending_layer_;
            idxs = std::move(pending_idxs_);
            pending_idxs_.clear();
            has_pending_ = false;
        }

        for (uint32_t idx : idxs) {
            for (ExpertPart part : parts) {
                uint64_t key = make_key(layer, idx, part);
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    if (map_.count(key)) continue; // already resident
                }
                // A fresher request may have arrived while we were
                // reading the previous one — finish this key (partial
                // work is never wasted, it's a real cache entry either
                // way) but stop issuing MORE reads for the stale request.
                std::vector<float> data;
                if (!load_data(key, data)) continue;
                std::lock_guard<std::mutex> lk(mtx_);
                insert_locked(key, std::move(data));
            }
            std::lock_guard<std::mutex> lk(qmtx_);
            if (has_pending_) break; // newer request queued, abandon the rest of this one
        }
    }
}

void ExpertStore::fetch_union(const std::vector<ExpertRequest>& reqs,
                              std::vector<float>& out_buffer,
                              std::vector<const float*>& out_ptrs) {
    out_buffer.clear();
    out_ptrs.clear();
    out_ptrs.reserve(reqs.size());

    std::vector<uint64_t> seen;
    for (const auto& r : reqs) {
        uint64_t key = make_key(r.layer, r.idx, r.part);
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
            out_ptrs.push_back(nullptr);
            continue;
        }
        seen.push_back(key);

        std::vector<float> data;
        if (fetch(r.layer, r.idx, r.part, data)) {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = map_.find(key);
            const float* ptr = it != map_.end() ? it->second->data.data() : nullptr;
            out_ptrs.push_back(ptr);
        } else {
            out_ptrs.push_back(nullptr);
        }
    }
}

void ExpertStore::prefetch_layer(uint32_t layer, const std::vector<uint32_t>& idxs) {
    static const ExpertPart parts[3] = { ExpertPart::Gate, ExpertPart::Up, ExpertPart::Down };
    for (int32_t d = 0; d < prefetch_depth_; ++d) {
        uint32_t lyr = layer + static_cast<uint32_t>(d);
        for (uint32_t idx : idxs) {
            for (ExpertPart part : parts) {
                uint64_t key = make_key(lyr, idx, part);
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    if (map_.count(key)) continue;
                }
                std::vector<float> data;
                if (load_data(key, data)) {
                    std::lock_guard<std::mutex> lk(mtx_);
                    insert_locked(key, std::move(data));
                }
            }
        }
    }
}

void ExpertStore::load_usage() {
    if (usage_file_.empty()) return;
    std::ifstream f(usage_file_);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        uint32_t layer;
        uint32_t idx;
        uint32_t part;
        uint32_t count;
        char sep;
        if (ss >> layer >> sep >> idx >> sep >> part >> sep >> count) {
            usage_[make_key(layer, idx, static_cast<ExpertPart>(part))] = count;
        }
    }
}

void ExpertStore::save_usage() {
    if (usage_file_.empty()) return;
    std::ofstream f(usage_file_);
    if (!f) return;
    for (const auto& kv : usage_) {
        uint32_t layer = static_cast<uint32_t>(kv.first >> 48);
        uint32_t idx = static_cast<uint32_t>((kv.first >> 8) & 0xFFFFFFFFu);
        uint32_t part = static_cast<uint32_t>(kv.first & 0xFFu);
        f << layer << ':' << idx << ':' << part << ':' << kv.second << '\n';
    }
}

}