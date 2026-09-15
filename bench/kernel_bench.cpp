// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

// Microbenchmark of the quantized kernels.
//
// Why this exists: end-to-end measurements with the CLI on this laptop
// drift by 30% between one batch and the next due to thermal throttling
// (with the same binary), while being stable within 5% inside the same
// batch. Comparing two kernel variants measured minutes apart is therefore
// meaningless — a "10% improvement" is within the drift. Here only the
// kernel is measured, in a few seconds, and the BEST of N repetitions is
// reported: best-of is robust against throttling and interference (both can
// only slow things down, never speed them up).
//
// The dimensions mimic the two matmuls that dominate gemma3-4b's decode
// (measured with the profiler: q4k ~62%, q6k ~31% of the time).
#include "../src/DesireeIA.LocaleEngine/src/core/engine.h"
#include "../src/DesireeIA.LocaleEngine/src/quant/quant.h"
#include "../src/DesireeIA.LocaleEngine/src/core/thread_pool.h"

#include <algorithm>
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace desireeia;

namespace {

double bench(const char* name, size_t rows, size_t cols, size_t bytes_per_row,
             int reps, const std::function<void()>& fn) {
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    const double gb = (double) rows * (double) bytes_per_row / 1e9;
    printf("%-12s rows=%-7zu cols=%-6zu %8.2f ms  %6.2f GB/s\n",
           name, rows, cols, best * 1000.0, gb / best);
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 7;
    // argv[2] = total parallelism. Must be set BEFORE the first matmul,
    // because the global pool is lazily constructed only once. Needed
    // because the default (all logical threads) oversubscribes the physical
    // cores, and with busy-waiting the oversubscription is devastating.
    const int threads = argc > 2 ? atoi(argv[2]) : 0;
    if (threads > 0) ThreadPool::set_thread_override((size_t) threads);
    printf("threads=%s\n", threads > 0 ? argv[2] : "default");

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    // TWO REGIMES, and the difference between them is the whole point of
    // this tool.
    //
    // Mistake made on 2026-09-07 and fixed here: the first version used only
    // the small matrix (2560x10240 = 14.7 MB) reused on every repetition. On
    // this CPU the L3 is 24 MB, so the weights stayed HOT IN CACHE and the
    // benchmark measured a regime that never exists in the real engine:
    // during decode each tensor is read exactly once, streamed from DRAM.
    // Result: a change measured at +67% "in cache" turned out to be neutral
    // end-to-end. A benchmark that lies is worse than no benchmark.
    //
    // - "L3"   : working set that fits in cache. Useful for isolating the
    //            pure COMPUTE cost of the kernel (instructions, dependencies).
    // - "DRAM" : working set much larger than L3. This is the real decode
    //            regime, and the only one that predicts end-to-end behavior.
    // ALWAYS look at the DRAM number to decide whether a change should be kept.
    struct Regime { const char* tag; size_t rows; size_t cols; };
    const Regime regimes[] = {
        { "L3",   2560,   10240 },   // ~15 MB  (ffn_down of gemma3-4b)
        { "DRAM", 200000, 2560  },   // ~288 MB (over 10x the L3)
    };

    for (const Regime& rg : regimes) {
        const size_t n_super = rg.cols / QK_K;
        std::vector<float> x(rg.cols);
        for (auto& v : x) v = fdist(rng);
        std::vector<float> y(rg.rows);

        {
            std::vector<uint8_t> w(rg.rows * n_super * sizeof(block_q4_K));
            for (auto& b : w) b = (uint8_t) byte_dist(rng);
            bench((std::string("q4_K/") + rg.tag).c_str(), rg.rows, rg.cols,
                  n_super * sizeof(block_q4_K), reps,
                  [&] { matmul_q4_k(w.data(), rg.rows, rg.cols, x.data(), y.data()); });
        }
        {
            std::vector<uint8_t> w(rg.rows * n_super * sizeof(block_q6_K));
            for (auto& b : w) b = (uint8_t) byte_dist(rng);
            bench((std::string("q6_K/") + rg.tag).c_str(), rg.rows, rg.cols,
                  n_super * sizeof(block_q6_K), reps,
                  [&] { matmul_q6_k(w.data(), rg.rows, rg.cols, x.data(), y.data()); });
        }
    }
    // "MANY" REGIME: same total bytes as the DRAM regime, but split into
    // many small calls instead of one big one — i.e. the real decode
    // pattern, where each token does ~240 matmuls on distinct tensors of a
    // few MB each. Comparing this with q4_K/DRAM isolates EXACTLY the
    // per-call cost (thread pool dispatch, worker wake-up, barrier),
    // separating it from the cost of compute and memory.
    // Real q4_K tensor shapes from a gemma3-4b layer, with a number of calls
    // chosen to keep the total volume well beyond the L3 (~290 MB). Needed
    // because the cost can depend on the SHAPE (short rows = more dispatch
    // per byte), not just the volume.
    struct ManyCase { const char* tag; size_t rows; size_t cols; size_t calls; };
    const ManyCase many_cases[] = {
        { "kv1024",   1024, 2560, 200 },  // wk / wv
        { "q2048",    2048, 2560, 100 },  // wq / wo
        { "ffn10240", 10240, 2560, 20 },  // ffn_gate / ffn_up (56% of the work)
    };
    for (const ManyCase& mc : many_cases) {
        const size_t n_calls = mc.calls;
        const size_t rows = mc.rows, cols = mc.cols;
        const size_t n_super = cols / QK_K;
        const size_t row_bytes = n_super * sizeof(block_q4_K);

        std::vector<float> x(cols);
        for (auto& v : x) v = fdist(rng);
        std::vector<float> y(rows);
        std::vector<std::vector<uint8_t>> ws(n_calls);
        for (auto& w : ws) {
            w.resize(rows * row_bytes);
            for (auto& b : w) b = (uint8_t) byte_dist(rng);
        }
        // Same calls, but with the activation quantized ONCE outside the
        // loop (matmul_q4_k_pq). The difference between MANY and MANY_PQ is
        // exactly the per-call cost of quantize_act_q8k_rep and the three
        // vector allocations that matmul_q4_k does internally.
        {
            std::vector<int8_t> xq;
            std::vector<float> xs;
            std::vector<int32_t> xsum;
            quantize_act_q8k_rep(x.data(), cols, xq, xs, xsum);
            bench((std::string("q4_K/") + mc.tag).c_str(), rows * n_calls, cols, row_bytes, reps, [&] {
                for (size_t i = 0; i < n_calls; ++i) {
                    matmul_q4_k_pq(ws[i].data(), rows, cols,
                                   xq.data(), xs.data(), xsum.data(), y.data());
                }
            });
        }
    }

#ifdef DESIREEIA_CUDA_ENABLED
    // CUDA backend: where does the per-call time go?
    //
    // The engine's profiler does NOT instrument the CUDA path (its counters
    // wrap CPU kernel dispatch), so the end-to-end numbers only said "the
    // time is somewhere else". Here the single call is measured at
    // INCREASING sizes: if the per-call time stays roughly constant, the
    // bottleneck is fixed latency (submission + sync per call, which no
    // faster kernel can remove); if it scales with volume, it's the
    // kernel/bandwidth. Distinguishing the two cases decides whether it's
    // worth optimizing the kernel or whether the architecture needs to
    // change instead (device-resident activations, no round-trip per
    // matvec).
    printf("\n--- CUDA Q8_0: latency vs bandwidth (weights already on device) ---\n");
    {
        struct CudaCase { const char* tag; size_t rows; size_t cols; };
        const CudaCase cuda_cases[] = {
            { "tiny",   256,  2048 },
            { "small",  2048, 2048 },
            { "medium", 4096, 2048 },
            { "large",  11008, 2048 },
        };
        for (const CudaCase& cc : cuda_cases) {
            const size_t nb = cc.cols / 32;
            const size_t row_bytes = nb * sizeof(block_q8_0);
            std::vector<uint8_t> w(cc.rows * row_bytes);
            for (auto& b : w) b = (uint8_t) byte_dist(rng);
            std::vector<float> x(cc.cols);
            for (auto& v : x) v = fdist(rng);
            std::vector<float> y(cc.rows);

            void* d_qs = nullptr;
            void* d_scale = nullptr;
            if (!matmul_q8_0_cuda_upload_weights(w.data(), cc.rows, cc.cols, &d_qs, &d_scale)) {
                printf("  %-8s weight upload FAILED\n", cc.tag);
                continue;
            }
            // 50 calls per measurement: a single call is dominated by
            // driver warm-up and says nothing about the steady-state regime.
            const size_t n_calls = 50;
            const double best = bench((std::string("cuda/") + cc.tag).c_str(),
                                       cc.rows * n_calls, cc.cols, row_bytes, reps, [&] {
                for (size_t i = 0; i < n_calls; ++i) {
                    matmul_q8_0_cuda_resident(d_qs, d_scale, cc.rows, cc.cols, x.data(), y.data());
                }
            });
            printf("             -> %.1f us per call\n", best * 1e6 / (double) n_calls);
            cuda_free_device(d_qs);
            cuda_free_device(d_scale);
        }

        // CPU reference on the same shape, to have the comparison within
        // the same measurement batch (no thermal drift between the two
        // numbers).
        {
            const size_t rows = 2048, cols = 2048;
            const size_t nb = cols / 32;
            const size_t row_bytes = nb * sizeof(block_q8_0);
            std::vector<uint8_t> w(rows * row_bytes);
            for (auto& b : w) b = (uint8_t) byte_dist(rng);
            std::vector<float> x(cols);
            for (auto& v : x) v = fdist(rng);
            std::vector<float> y(rows);
            const size_t n_calls = 50;
            const double best = bench("cpu/small", rows * n_calls, cols, row_bytes, reps, [&] {
                for (size_t i = 0; i < n_calls; ++i) {
                    matmul_q8_0(w.data(), rows, cols, x.data(), y.data());
                }
            });
            printf("             -> %.1f us per call\n", best * 1e6 / (double) n_calls);
        }
    }
#endif
    return 0;
}
