#include "../core/engine.h"
#include <algorithm>
#include <fstream>
#include <sstream>

namespace desireeia {

namespace {
constexpr uint32_t PIN_THRESHOLD = 4;
}

ExpertStore::ExpertStore(int32_t cache_count, LogFn log)
    : cache_count_(cache_count > 0 ? cache_count : 256), log_(log) {
}

bool ExpertStore::load_data(uint64_t key, std::vector<float>& out) {
    if (!reader_) {
        // Nessun reader collegato (es. test unitari su ExpertStore isolato):
        // comportamento stub preservato per non rompere quel path.
        out.assign(1, 0.0f);
        return true;
    }
    const uint32_t layer = static_cast<uint32_t>(key >> 48);
    const uint32_t idx = static_cast<uint32_t>((key >> 8) & 0xFFFFFFFFu);
    const ExpertPart part = static_cast<ExpertPart>(key & 0xFFu);
    return reader_->read_expert(layer, idx, part, out);
}

bool ExpertStore::fetch(uint32_t layer, uint32_t idx, ExpertPart part, std::vector<float>& out) {
    uint64_t key = make_key(layer, idx, part);

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

    std::vector<float> data;
    if (!load_data(key, data)) {
        return false;
    }

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
    out = order_.begin()->data;
    return true;
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
                if (map_.find(key) != map_.end()) {
                    continue;
                }
                std::vector<float> data;
                if (load_data(key, data)) {
                    Entry e{key, std::move(data), 0, false};
                    order_.push_front(std::move(e));
                    map_[key] = order_.begin();
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