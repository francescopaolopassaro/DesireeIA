// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_THREAD_POOL_H
#define DESIREEIA_THREAD_POOL_H

#include "desireeia/abi.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace desireeia {

#ifndef DESIREEIA_INTERNAL
#if defined(_WIN32) && defined(DESIREEIA_BUILD)
#define DESIREEIA_INTERNAL __declspec(dllexport)
#else
#define DESIREEIA_INTERNAL
#endif
#endif

// Persistent thread pool for matmul kernels: threads are created once and
// reused for the whole process lifetime, instead of spawning/joining on
// every matmul call.
//
// A single global pool (ThreadPool::global()), shared across every
// model/context open in the process.
//
// Synchronization is deliberately asymmetric, to keep the surface area
// for concurrency bugs small:
// - waking workers: mutex + condition_variable (one thread notifies, the
//   workers wait — the simple direction, one notifier, N waiters).
// - waiting for completion: an atomic counter with a spin-wait (no
//   condition_variable on the "N threads notify, one thread waits" side,
//   which is the trickier direction to get right). The cost is CPU cycles
//   spent spinning (short: microseconds to milliseconds), not correctness.
//
// parallel_for(total, fn) is synchronous: it splits [0,total) into static
// chunks across the workers (the calling thread stays free to do other
// things rather than compute its own slice — easier to reason about
// correctly, at a small throughput cost versus having it also compute),
// and only returns once the work is done. Consecutive calls never overlap.
// The global pool is NEVER destroyed (see global()): on Windows, process
// shutdown starts unloading DLLs and freezing threads before static
// object destructors run — a destructor that calls join() on the workers
// at that point can hang forever (the target thread never terminates
// cleanly because the OS already froze it). That's not a problem: at
// process exit the OS reclaims every thread's resources regardless.
class DESIREEIA_INTERNAL ThreadPool {
public:
    static ThreadPool& global();

    // Sets the desired worker count BEFORE the first call to global()
    // (which lazily constructs the singleton, once per process). If never
    // called, or called after global() has already been constructed, this
    // has no effect — the pool stays whatever it already was. Called from
    // desireeia_create with the plan's resolved thread count.
    // NOT inline, and not defined in the header: the backing variable must
    // exist as a SINGLE copy, the one inside the DLL. Defining a `static
    // size_t v` here inside an inline function would give every module
    // that includes this header its own copy: an executable linking the
    // DLL would set its own copy while the pool read the DLL's, and the
    // override silently ignored. Found via bench/kernel_bench.cpp, which
    // measured 56 GB/s "at 1 thread" — above this machine's single-thread
    // memory bandwidth, so impossible: the default was always running
    // instead of the configured value.
    static void set_thread_override(int n);
    static size_t& desired_threads();

    // The task is passed as a function pointer + context, NOT as
    // std::function: constructing one on every dispatch meant a heap
    // allocation for every matmul (~240 per token) right on the critical
    // path, and inside the mutex-protected section to boot.
    using TaskFn = void (*)(void*, size_t, size_t);

    template <typename Fn>
    void parallel_for(size_t total, Fn&& fn) {
        using F = typename std::decay<Fn>::type;
        F local(std::forward<Fn>(fn));
        run_raw(total, [](void* ctx, size_t a, size_t b) {
            (*static_cast<F*>(ctx))(a, b);
        }, &local);
    }

    // Variant for a few large, independent units (see parallel_units in
    // engine.h): one unit per slice, no minimum threshold.
    template <typename Fn>
    void parallel_for_units(size_t total, Fn&& fn) {
        using F = typename std::decay<Fn>::type;
        F local(std::forward<Fn>(fn));
        run_raw(total, [](void* ctx, size_t a, size_t b) {
            (*static_cast<F*>(ctx))(a, b);
        }, &local, /*grain_override=*/1, /*min_parallel=*/2);
    }

    size_t worker_count() const { return n_workers_; }

private:
    explicit ThreadPool(size_t n_workers);
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // grain_override = 0 -> slices computed automatically and aligned to
    // the cache line; > 0 -> slices of that exact size.
    void run_raw(size_t total, TaskFn fn, void* ctx,
                 size_t grain_override = 0, size_t min_parallel = 64);
    void worker_loop(size_t worker_id);

    std::mutex mtx_;
    std::condition_variable cv_start_;

    void run_chunks();

    // NOT copied by the workers: previously every worker did `fn = task_`
    // under the mutex on every dispatch, i.e. one std::function copy
    // (with allocation) per worker per dispatch — with ~239 matmuls per
    // token and 16 workers that's ~3800 allocations per token, all
    // serialized on the mutex. Written by run_raw() before incrementing
    // generation_ and valid until run_raw() returns (which only happens
    // at pending_==0), so the workers read them without copying and
    // without taking the mutex.
    TaskFn task_fn_ = nullptr;
    void* task_ctx_ = nullptr;
    size_t total_ = 0;
    size_t n_workers_ = 0;

    // DYNAMIC chunking: workers take slices from an atomic counter instead
    // of receiving a static slice computed from worker_id. With static
    // division every dispatch ended up waiting on the slowest (straggler),
    // and it only took one core running slower — from hyperthreading,
    // another process, or throttling — to drag out the entire matmul.
    // Measured: scaling was 1.37x from 4 to 8 threads and only 1.18x from
    // 8 to 16.
    std::atomic<size_t> next_{0};
    size_t grain_ = 1;

    // For the profiler only: the timestamp of the end of the last
    // dispatch, to measure the serial stretch that separates it from the
    // next one. Written and read only by the calling thread (parallel
    // regions never overlap), so no synchronization is needed.
    std::chrono::steady_clock::time_point last_end_{};
    bool have_last_end_ = false;
    // Atomic so the workers can check it in a brief spin-wait BEFORE
    // entering cv_start_.wait() (reduces wake-up latency in the common
    // case: the next dispatch almost always arrives within a few
    // microseconds). The definitive read of task_/total_ still happens
    // under the mutex in worker_loop; the spin is just a heuristic to
    // avoid the cost of an OS wait/wake when it isn't needed.
    std::atomic<uint64_t> generation_{0};
    std::atomic<size_t> pending_{0};
    bool stop_ = false; // never set (the pool is never stopped): see global().
};

}

#endif
