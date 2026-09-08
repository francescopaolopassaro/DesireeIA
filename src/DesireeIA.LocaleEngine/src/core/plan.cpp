#include "engine.h"
#include <cctype>
#include <cstring>
#include <cstdlib>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace desireeia {

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

#if defined(_WIN32)
    if (GetEnvironmentVariableA("DESIREEIA_MODEL_MIRROR", nullptr, 0) > 0) {
        p.dual_ssd_enabled = 1;
    }
#else
    if (std::getenv("DESIREEIA_MODEL_MIRROR") != nullptr) {
        p.dual_ssd_enabled = 1;
    }
#endif

    return p;
}

}
