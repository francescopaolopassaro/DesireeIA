#include "engine.h"
#include <cctype>
#include <cstring>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace desireeia {

// Reads DESIREEIA_SSD_TIER and returns true when it names a mode.
//
// This exists so the tier can be forced on or off for a benchmark run without
// rebuilding or editing caller code. It applies even to a plan the caller
// filled in completely: a caller-supplied plan is the normal case, and an
// override that only worked on the auto-planned path would be an override that
// never fires in practice.
bool env_ssd_tier_override(int32_t& mode) {
#if defined(_WIN32)
    char buf[32] = {0};
    const DWORD len = GetEnvironmentVariableA("DESIREEIA_SSD_TIER", buf, sizeof(buf));
    const char* tier = (len > 0 && len < sizeof(buf)) ? buf : nullptr;
#else
    const char* tier = std::getenv("DESIREEIA_SSD_TIER");
#endif
    if (!tier || tier[0] == '\0') return false;

    std::string m = tier;
    for (auto& c : m) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (m == "off" || m == "0") {
        mode = DESIREEIA_SSD_TIER_OFF;
    } else if (m == "always" || m == "2" || m == "force") {
        mode = DESIREEIA_SSD_TIER_ALWAYS;
    } else if (m == "auto" || m == "1") {
        mode = DESIREEIA_SSD_TIER_AUTO;
    } else {
        return false;
    }
    return true;
}

desireeia_plan build_plan(const desireeia_hw_info& hw, const std::string& model_path) {
    desireeia_plan p;
    std::memset(&p, 0, sizeof(p));

    p.format = DESIREEIA_FORMAT_UNKNOWN;
    if (!model_path.empty()) {
        std::string lower = model_path;
        for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower.find(".gguf") != std::string::npos) {
            p.format = DESIREEIA_FORMAT_GGUF;
        } else if (lower.find(".safetensors") != std::string::npos ||
                   lower.find("config.json") != std::string::npos) {
            p.format = DESIREEIA_FORMAT_SAFETENSORS;
        }
    }

    p.backend = DESIREEIA_BACKEND_CPU;
    if (hw.axelera_device_count > 0) {
        p.backend = DESIREEIA_BACKEND_AXELERA;
    } else if (hw.cuda_device_count > 0) {
        p.backend = DESIREEIA_BACKEND_CUDA;
    } else if (hw.intel_gpu_count > 0) {
        p.backend = DESIREEIA_BACKEND_INTEL;
    }
#if defined(__APPLE__)
    if (hw.has_metal) {
        p.backend = DESIREEIA_BACKEND_METAL;
    }
#endif

    p.dense_quant = DESIREEIA_QUANT_Q8_0;
    p.expert_quant = DESIREEIA_QUANT_Q4_K;

    p.n_threads = hw.cpu_threads > 0 ? hw.cpu_threads : 4;

    uint64_t budget = hw.ram_free_mb > 0 ? hw.ram_free_mb : 8192;
    if (budget > 64ull * 1024) {
        budget = 64ull * 1024;
    }
    p.ram_budget_mb = budget;

    uint64_t cache = budget / 8;
    if (cache > 4096) {
        cache = 4096;
    }
    if (cache < 512) {
        cache = 512;
    }
    p.expert_cache_count = static_cast<int32_t>(cache);

    p.expert_prefetch_enabled = 1;
    p.kv_compression_enabled = 1;
    p.expert_pin_enabled = 1;
    p.expert_prefetch_depth = 1;
    p.batch_union_enabled = 1;
    p.dual_ssd_enabled = 0;

    // AUTO: engage the tier only for a model that does not fit the RAM budget
    // once the system and runtime reserves are taken out (see the planner in
    // core/ctx.cpp). For a model that fits, this is identical to OFF — nothing
    // is constructed and nothing sits on the tensor read path — so the common
    // case pays nothing for the tier being available.
    //
    // This was OFF for a while because the tier parsed the model file with its
    // own header reader, which disagreed with the real one about where tensor
    // data starts and served weights from the wrong offset. It now reads
    // through the same reader as the rest of the engine, and the selftest
    // compares the two byte-for-byte, so AUTO is safe to be the default again.
    p.ssd_tier_mode = DESIREEIA_SSD_TIER_AUTO;
    // 0 = derive from ram_budget_mb at load time.
    p.ssd_tier_cache_mb = 0;

#if defined(_WIN32)
    if (GetEnvironmentVariableA("DESIREEIA_MODEL_MIRROR", nullptr, 0) > 0) {
        p.dual_ssd_enabled = 1;
    }
#else
    if (std::getenv("DESIREEIA_MODEL_MIRROR") != nullptr) {
        p.dual_ssd_enabled = 1;
    }
#endif
    env_ssd_tier_override(p.ssd_tier_mode);

    return p;
}

}
