// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_PROFILE_H
#define DESIREEIA_PROFILE_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

namespace desireeia {

// Minimal profiler, always active (the overhead of one clock read + one
// atomic add per matmul call is negligible compared to the cost of the
// call itself: ~180 matmuls/token, a couple of clock reads each is not a
// bottleneck). Used to answer "where does the time go" with real data,
// instead of continuing to guess optimizations one at a time (see
// docs/engine_gap_analysis.md for attempts already discarded for lack of
// concrete data).
struct ProfileCounters {
    std::atomic<int64_t> ns_quantize_act{0}; // quantize_q8_0 of the activation (single-thread part of every kernel)
    std::atomic<int64_t> ns_q4k_compute{0};  // parallel_rows inside matmul_q4_k (after quantization)
    std::atomic<int64_t> ns_q6k_compute{0};
    std::atomic<int64_t> ns_q5k_compute{0};
    std::atomic<int64_t> ns_q40_compute{0};
    std::atomic<int64_t> ns_q80_compute{0};
    std::atomic<int64_t> ns_f32_compute{0};  // matmul_f32 (dequantized fallback, dense_forward.cpp)
    // "Legacy" formats (Q4_1/Q5_0/Q5_1, a single 32-block, rare in modern
    // GGUF files that prefer K-quants): one shared counter, not one per
    // format, to avoid bloating the profiler for lightly-used paths.
    std::atomic<int64_t> ns_legacy_compute{0};
    // Q2_K/Q3_K/Q8_K: shared "kquant2" counter to avoid further bloating
    // the profiler for formats less common than Q4_K/Q5_K/Q6_K.
    std::atomic<int64_t> ns_kquant2_compute{0};
    std::atomic<int64_t> calls_q4k{0};
    std::atomic<int64_t> calls_q6k{0};
    std::atomic<int64_t> calls_q5k{0};
    std::atomic<int64_t> calls_q40{0};
    std::atomic<int64_t> calls_q80{0};
    std::atomic<int64_t> calls_legacy{0};
    std::atomic<int64_t> calls_kquant2{0};
    std::atomic<int64_t> calls_f32{0};
    // Matrices computed in a fused dispatch (Q/K/V together, gate/up
    // together). Counted separately, not inside q4k/q6k: a fused group can
    // mix the two formats, and adding it to one or the other would make
    // the profile lie about exactly the entry being optimized.
    std::atomic<int64_t> ns_fused_compute{0};
    std::atomic<int64_t> calls_fused{0};

    // --- CUDA backend ---
    // Added because the end-to-end profile only said "the time is
    // somewhere else": the counters above wrap the CPU kernels, so the
    // CUDA path all landed in the leftover bucket (ns_serial_gap) with no
    // way to know either which of the two paths was running or how much
    // it cost. Deliberately kept separate: "resident" uses weights already
    // on VRAM, "upload" reloads them on every call (fallback). If the
    // latter aren't zero while the weight cache is active, there's a bug
    // in the dispatch, not a kernel problem.
    std::atomic<int64_t> ns_cuda_resident{0};
    std::atomic<int64_t> calls_cuda_resident{0};
    std::atomic<int64_t> ns_cuda_upload{0};
    std::atomic<int64_t> calls_cuda_upload{0};
    // Attention + output projection executed on device.
    std::atomic<int64_t> ns_cuda_attn{0};
    std::atomic<int64_t> calls_cuda_attn{0};

    // --- Serial/parallel transitions (thread pool) ---
    // Decode continuously alternates parallel regions (the matmuls) and
    // single-thread serial stretches (norm, RoPE, attention, softmax,
    // residuals). These counters measure how often this happens and how
    // much it costs, instead of inferring it: this is the current
    // hypothesis for the 2.1x factor between the isolated kernel's
    // bandwidth (48-55 GB/s) and the one inside the engine (22 GB/s). See
    // docs/engine_gap_analysis.md.
    std::atomic<int64_t> n_dispatch{0};       // how many parallel regions
    std::atomic<int64_t> ns_serial_gap{0};    // time BETWEEN the end of one dispatch and the start of the next
    std::atomic<int64_t> ns_dispatch_wait{0}; // time at the barrier after the caller finished its own slices
    std::atomic<int64_t> ns_dispatch_total{0};// total time inside run_raw

    // --- Serial phases of the forward pass (a single thread while workers sleep) ---
    // Used to know WHERE the serial stretch goes, instead of guessing: the
    // first attempt (reordering the attention loops) fixed a real
    // locality flaw but barely moved the needle, a sign that the bulk was
    // elsewhere.
    std::atomic<int64_t> ns_ser_norm{0};  // rms_norm (attn/ffn/post/qk) + residuals
    std::atomic<int64_t> ns_ser_rope{0};  // RoPE + KV cache write
    std::atomic<int64_t> ns_ser_attn{0};  // QK scores, softmax, V combination
    std::atomic<int64_t> ns_ser_act{0};   // gelu/silu and the FFN's gate*up product
};

inline ProfileCounters& profile_counters() {
    static ProfileCounters pc;
    return pc;
}

// RAII stopwatch: in its destructor, adds the elapsed nanoseconds to the
// given atomic counter.
class ScopedTimer {
public:
    explicit ScopedTimer(std::atomic<int64_t>& target)
        : target_(target), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        target_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count(),
                           std::memory_order_relaxed);
    }
private:
    std::atomic<int64_t>& target_;
    std::chrono::steady_clock::time_point start_;
};

inline std::string profile_dump_string() {
    auto& pc = profile_counters();
    auto ms = [](int64_t ns) { return (double) ns / 1e6; };
    char buf[2048];
    std::snprintf(buf, sizeof(buf),
        "quantize_act=%.2fms  q4k=%.2fms(%lldc)  q5k=%.2fms(%lldc)  q6k=%.2fms(%lldc)  "
        "q4_0=%.2fms(%lldc)  q8_0=%.2fms(%lldc)  legacy=%.2fms(%lldc)  kquant2=%.2fms(%lldc)  f32_fallback=%.2fms(%lldc)"
        "\n  fuso=%.2fms(%lldc)"
        "\n  dispatch=%lld  in_dispatch=%.2fms  barriera=%.2fms  seriale_fra_dispatch=%.2fms"
        "\n  cuda: resident=%.2fms(%lldc)  upload=%.2fms(%lldc)  attn=%.2fms(%lldc)"
        "\n  seriale: norm=%.2fms  rope=%.2fms  attn=%.2fms  act=%.2fms",
        ms(pc.ns_quantize_act.load(std::memory_order_relaxed)),
        ms(pc.ns_q4k_compute.load(std::memory_order_relaxed)), (long long) pc.calls_q4k.load(std::memory_order_relaxed),
        ms(pc.ns_q5k_compute.load(std::memory_order_relaxed)), (long long) pc.calls_q5k.load(std::memory_order_relaxed),
        ms(pc.ns_q6k_compute.load(std::memory_order_relaxed)), (long long) pc.calls_q6k.load(std::memory_order_relaxed),
        ms(pc.ns_q40_compute.load(std::memory_order_relaxed)), (long long) pc.calls_q40.load(std::memory_order_relaxed),
        ms(pc.ns_q80_compute.load(std::memory_order_relaxed)), (long long) pc.calls_q80.load(std::memory_order_relaxed),
        ms(pc.ns_legacy_compute.load(std::memory_order_relaxed)), (long long) pc.calls_legacy.load(std::memory_order_relaxed),
        ms(pc.ns_kquant2_compute.load(std::memory_order_relaxed)), (long long) pc.calls_kquant2.load(std::memory_order_relaxed),
        ms(pc.ns_f32_compute.load(std::memory_order_relaxed)), (long long) pc.calls_f32.load(std::memory_order_relaxed),
        ms(pc.ns_fused_compute.load(std::memory_order_relaxed)), (long long) pc.calls_fused.load(std::memory_order_relaxed),
        (long long) pc.n_dispatch.load(std::memory_order_relaxed),
        ms(pc.ns_dispatch_total.load(std::memory_order_relaxed)),
        ms(pc.ns_dispatch_wait.load(std::memory_order_relaxed)),
        ms(pc.ns_serial_gap.load(std::memory_order_relaxed)),
        ms(pc.ns_cuda_resident.load(std::memory_order_relaxed)), (long long) pc.calls_cuda_resident.load(std::memory_order_relaxed),
        ms(pc.ns_cuda_upload.load(std::memory_order_relaxed)), (long long) pc.calls_cuda_upload.load(std::memory_order_relaxed),
        ms(pc.ns_cuda_attn.load(std::memory_order_relaxed)), (long long) pc.calls_cuda_attn.load(std::memory_order_relaxed),
        ms(pc.ns_ser_norm.load(std::memory_order_relaxed)),
        ms(pc.ns_ser_rope.load(std::memory_order_relaxed)),
        ms(pc.ns_ser_attn.load(std::memory_order_relaxed)),
        ms(pc.ns_ser_act.load(std::memory_order_relaxed)));
    return std::string(buf);
}

inline void profile_reset() {
    auto& pc = profile_counters();
    pc.ns_quantize_act = 0; pc.ns_q4k_compute = 0; pc.ns_q6k_compute = 0; pc.ns_q5k_compute = 0;
    pc.ns_q40_compute = 0; pc.ns_q80_compute = 0; pc.ns_legacy_compute = 0; pc.ns_kquant2_compute = 0;
    pc.ns_f32_compute = 0;
    pc.calls_q4k = 0; pc.calls_q6k = 0; pc.calls_q5k = 0; pc.calls_q40 = 0; pc.calls_q80 = 0;
    pc.calls_legacy = 0; pc.calls_kquant2 = 0; pc.calls_f32 = 0;
    pc.ns_fused_compute = 0; pc.calls_fused = 0;
    pc.ns_cuda_resident = 0; pc.calls_cuda_resident = 0;
    pc.ns_cuda_upload = 0; pc.calls_cuda_upload = 0;
    pc.ns_cuda_attn = 0; pc.calls_cuda_attn = 0;
    pc.n_dispatch = 0; pc.ns_serial_gap = 0; pc.ns_dispatch_wait = 0; pc.ns_dispatch_total = 0;
    pc.ns_ser_norm = 0; pc.ns_ser_rope = 0; pc.ns_ser_attn = 0; pc.ns_ser_act = 0;
}

}

#endif
