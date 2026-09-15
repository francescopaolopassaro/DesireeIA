// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "thread_pool.h"
#include "profile.h"
#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define DESIREEIA_HAS_PAUSE 1
#endif

namespace desireeia {

// "I'm spinning idle" hint for the core. On x86 it's _mm_pause (reduces
// power draw and frees resources for the other SMT thread on the same
// physical core); elsewhere it falls back to yield. The project has to
// run on Windows, Linux and macOS, so no x86 intrinsics without a guard.
static inline void desireeia_cpu_relax() {
#if defined(DESIREEIA_HAS_PAUSE)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

size_t& ThreadPool::desired_threads() {
    static size_t v = 0; // 0 = not configured, use the hardware default
    return v;
}

void ThreadPool::set_thread_override(int n) {
    if (n > 0) desired_threads() = (size_t) n;
}

ThreadPool& ThreadPool::global() {
    static const size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t override_n = desired_threads();
    // n = the TOTAL desired parallelism. The calling thread participates
    // in the work (it acts as worker 0), so only n-1 additional workers
    // get created: earlier the caller just sat yielding while it waited,
    // wasting a core, so "4 threads" actually meant 4 workers plus 1 idle
    // caller.
    size_t n_total = override_n > 0 ? override_n : hw;

    // NEVER occupy every logical processor. Measured on this machine (22
    // logical) with the MANY regime of bench/kernel_bench.cpp, which
    // reproduces the ~240 matmuls per token of real decode:
    //     20 threads -> 5.51 ms     22 threads -> 46.67 ms
    // i.e. an 8.5x collapse between 20 and 22. With every logical
    // processor occupied by a pool thread actively spinning, the OS has
    // nowhere to schedule anything else: it only takes one worker being
    // descheduled for the barrier to stay blocked for an entire
    // scheduling quantum, and this happens on EVERY dispatch.
    // The cap also applies to an explicit override: asking for 22 threads
    // on 22 logical processors isn't a performance choice, it's a way to
    // hurt yourself.
    //
    // The cliff is sharp and sits between 20 and 21 out of 22 logical
    // (5.51 ms -> 44.47 ms), so leaving just one free isn't enough: TWO
    // are left free. The thread that dispatches, the logger, and the .NET
    // runtime all need to be able to run without contending a core with a
    // spinning worker.
    const size_t cap = hw > 4 ? hw - 2 : (hw > 1 ? hw - 1 : 1);
    if (n_total > cap) n_total = cap;

    // n_total is the TOTAL parallelism: the calling thread participates
    // in the work (it acts as worker 0), so only n_total-1 additional
    // workers get created.
    const size_t n = n_total > 1 ? n_total - 1 : 0;
    // Allocated once and never destroyed (see the comment in
    // thread_pool.h): no join() on the workers during process/DLL teardown.
    static ThreadPool* pool = new ThreadPool(n);
    return *pool;
}

ThreadPool::ThreadPool(size_t n_workers) {
    n_workers_ = n_workers;
    for (size_t i = 0; i < n_workers; ++i) {
        std::thread t([this, i] { worker_loop(i); });
        t.detach();
    }
}

void ThreadPool::run_raw(size_t total, TaskFn fn, void* ctx,
                         size_t grain_override, size_t min_parallel) {
    if (total == 0) return;
    if (n_workers_ == 0 || total < min_parallel) {
        fn(ctx, 0, total);
        return;
    }

    // Measures serial/parallel transitions. The time between the end of
    // the previous dispatch and the start of this one is exactly the
    // stretch where only one thread runs while the others are idle.
    auto& pc = profile_counters();
    const auto t_enter = std::chrono::steady_clock::now();
    if (have_last_end_) {
        pc.ns_serial_gap.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_enter - last_end_).count(),
            std::memory_order_relaxed);
    }
    pc.n_dispatch.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(mtx_);
        task_fn_ = fn;
        task_ctx_ = ctx;
        total_ = total;
        // Slices small enough to balance well (~8 per worker) but not so
        // small that the atomic fetch_add weighs on the useful work.
        //
        // ROUNDED TO MULTIPLES OF 16: the output y is a float array, and a
        // 64-byte cache line holds 16 of them. With 8-row slices, every
        // line of y was being written by TWO different threads, and every
        // line bounced back and forth between cores on every write (false
        // sharing). Aligning the slices to the cache line means every
        // line of y belongs to a single thread.
        if (grain_override > 0) {
            grain_ = grain_override;
        } else {
            constexpr size_t kFloatsPerLine = 16;
            size_t g = std::max<size_t>(kFloatsPerLine, total / ((n_workers_ + 1) * 8));
            g = (g + kFloatsPerLine - 1) / kFloatsPerLine * kFloatsPerLine;
            grain_ = g;
        }
        next_.store(0, std::memory_order_relaxed);
        pending_.store(n_workers_, std::memory_order_release);
        generation_.fetch_add(1, std::memory_order_release);
    }
    cv_start_.notify_all();

    // The caller also does its share of the work instead of waiting idle.
    run_chunks();

    // From here on the caller has finished its own useful work: everything
    // that follows is pure synchronization cost (waiting for the workers
    // to reach the barrier), and it's the figure to compare against the
    // real work.
    const auto t_barrier = std::chrono::steady_clock::now();
    while (pending_.load(std::memory_order_acquire) != 0) {
        desireeia_cpu_relax();
    }
    const auto t_done = std::chrono::steady_clock::now();
    pc.ns_dispatch_wait.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_done - t_barrier).count(),
        std::memory_order_relaxed);
    pc.ns_dispatch_total.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_done - t_enter).count(),
        std::memory_order_relaxed);
    last_end_ = t_done;
    have_last_end_ = true;
}

void ThreadPool::run_chunks() {
    for (;;) {
        const size_t start = next_.fetch_add(grain_, std::memory_order_relaxed);
        if (start >= total_) break;
        task_fn_(task_ctx_, start, std::min(total_, start + grain_));
    }
}

void ThreadPool::worker_loop(size_t worker_id) {
    // Tried a short spin-wait before cv_start_.wait() (like ggml's
    // optional polling): measured WORSE on this machine (21 workers doing
    // _mm_pause() in parallel compete for hyperthreading core resources,
    // slowing down the real work) — 5-6 tok/s instead of 9.6-10 with plain
    // condition_variable. Removed: the measured data outweighs the
    // theoretical expectation (see docs/engine_gap_analysis.md).
    (void) worker_id; // slices no longer depend on the id: they're dynamic

    // How many rounds of active waiting before actually going to sleep.
    // Decode does ~240 matmuls per token, so the next dispatch almost
    // always arrives within a few microseconds: sleeping and waking up
    // every time cost more than the work itself. Measured with the
    // "MANY" regime of bench/kernel_bench.cpp: the same total bytes split
    // into 200 small calls yielded 7.6 GB/s against 53.7 GB/s in a
    // single large call, i.e. ~166 us of pure overhead per matmul.
    //
    // Historical note: a spin had already been tried and measured WORSE.
    // That version, however, spun with 21 workers (more than the physical
    // cores, so two workers per SMT core stealing resources from each
    // other) and still needed the mutex to read the task. Here the spin
    // is bounded, uses pause, and crucially does NOT take the mutex: the
    // generation is read from an atomic. If the spin expires, it goes
    // back to sleeping on the condition variable, so an idle pool doesn't
    // burn CPU.
    // Spin SHORT, deliberately. Tried extending it to 32768 rounds (~350
    // us, enough to cover the serial work between one matmul and the next
    // and so never fall asleep during a token): measured WORSE, 8.73
    // tok/s against 9.84. Wake-up latency was therefore not the dominant
    // cost; actively-waiting workers instead steal SMT resources and
    // turbo budget from the thread doing the serial work at that moment,
    // and on a 15-core laptop, spinning workers hold down everyone's
    // frequencies. Only a few rounds remain, useful only to catch
    // back-to-back dispatches (wq/wk/wv, ffn_gate/ffn_up) without going
    // through the OS.
    constexpr int kSpinRounds = 512;

    uint64_t local_gen = 0;
    for (;;) {
        int spins = 0;
        for (;;) {
            if (generation_.load(std::memory_order_acquire) != local_gen) break;
            if (++spins >= kSpinRounds) {
                std::unique_lock<std::mutex> lk(mtx_);
                // The predicate re-checks generation_ under the mutex: if
                // run_raw published while we were about to fall asleep,
                // wait returns immediately and the wake-up isn't missed.
                cv_start_.wait(lk, [this, local_gen] {
                    return stop_ || generation_.load(std::memory_order_acquire) != local_gen;
                });
                if (stop_) return;
                break;
            }
            desireeia_cpu_relax();
        }
        local_gen = generation_.load(std::memory_order_acquire);

        // task_fn_/task_ctx_ are neither copied nor read under the mutex:
        // they're written by run_raw() before incrementing generation_ and
        // stay valid until run_raw() returns, which only happens once
        // pending_ has gone back to zero, i.e. after the line below.
        run_chunks();

        pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}




