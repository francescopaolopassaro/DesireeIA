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

    // La task viene passata come puntatore a funzione + contesto, NON come
    // std::function: costruirne una a ogni dispatch significava
    // un'allocazione sull'heap per ogni matmul (~240 a token) proprio sul
    // percorso critico, per giunta dentro la sezione col mutex.
    using TaskFn = void (*)(void*, size_t, size_t);

    template <typename Fn>
    void parallel_for(size_t total, Fn&& fn) {
        using F = typename std::decay<Fn>::type;
        F local(std::forward<Fn>(fn));
        run_raw(total, [](void* ctx, size_t a, size_t b) {
            (*static_cast<F*>(ctx))(a, b);
        }, &local);
    }

    // Variante per poche unita' grosse e indipendenti (vedi parallel_units in
    // engine.h): una unita' per fetta, nessuna soglia minima.
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

    // grain_override = 0 -> fette calcolate automaticamente e allineate alla
    // cache line; > 0 -> fette di quella dimensione esatta.
    void run_raw(size_t total, TaskFn fn, void* ctx,
                 size_t grain_override = 0, size_t min_parallel = 64);
    void worker_loop(size_t worker_id);

    std::mutex mtx_;
    std::condition_variable cv_start_;

    void run_chunks();

    // NON copiate dai worker: prima ogni worker faceva `fn = task_` sotto
    // mutex a ogni dispatch, cioe' una copia di std::function (con
    // allocazione) per worker per dispatch — con ~239 matmul per token e 16
    // worker sono ~3800 allocazioni a token, tutte serializzate sul mutex.
    // Scritte da run_raw() prima di incrementare generation_ e valide finche'
    // run_raw() non ritorna (cosa che avviene solo a pending_==0), quindi i
    // worker le leggono senza copiare e senza prendere il mutex.
    TaskFn task_fn_ = nullptr;
    void* task_ctx_ = nullptr;
    size_t total_ = 0;
    size_t n_workers_ = 0;

    // Chunking DINAMICO: i worker prendono fette da un contatore atomico
    // invece di ricevere una fetta statica calcolata da worker_id. Con la
    // divisione statica ogni dispatch finiva col piu' lento (straggler), e
    // bastava un core rallentato — da hyperthreading, da un altro processo o
    // da throttling — per allungare l'intera matmul. Misurato: lo scaling
    // era 1,37x da 4 a 8 thread e solo 1,18x da 8 a 16.
    std::atomic<size_t> next_{0};
    size_t grain_ = 1;

    // Solo per il profiler: istante di fine dell'ultimo dispatch, per misurare
    // il tratto seriale che lo separa dal successivo. Scritti e letti solo dal
    // thread chiamante (le regioni parallele non si sovrappongono mai), quindi
    // non serve sincronizzazione.
    std::chrono::steady_clock::time_point last_end_{};
    bool have_last_end_ = false;
    // Atomico per permettere ai worker di controllarlo in un breve spin-wait
    // PRIMA di entrare in cv_start_.wait() (riduce la latenza di risveglio
    // nel caso comune: la dispatch successiva arriva quasi sempre entro
    // pochi microsecondi). La lettura definitiva di task_/total_ resta
    // comunque sotto mutex in worker_loop, lo spin e' solo un'euristica per
    // evitare il costo di un wait/wake col sistema operativo quando non
    // serve.
    std::atomic<uint64_t> generation_{0};
    std::atomic<size_t> pending_{0};
    bool stop_ = false; // mai impostato (il pool non viene mai fermato): vedi global().
};

}

#endif
