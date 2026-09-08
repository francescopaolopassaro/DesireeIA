#include "thread_pool.h"
#include "profile.h"
#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define DESIREEIA_HAS_PAUSE 1
#endif

namespace desireeia {

// Hint "sto girando a vuoto" per il core. Su x86 e' _mm_pause (riduce il
// consumo e libera risorse all'altro thread SMT dello stesso core fisico);
// altrove si ripiega su yield. Il progetto deve girare su Windows, Linux e
// macOS, quindi niente intrinseche x86 senza guardia.
static inline void desireeia_cpu_relax() {
#if defined(DESIREEIA_HAS_PAUSE)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

size_t& ThreadPool::desired_threads() {
    static size_t v = 0; // 0 = non configurato, usa il default hardware
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

    // MAI occupare tutti i processori logici. Misurato su questa macchina
    // (22 logici) col regime MANY di bench/kernel_bench.cpp, che riproduce
    // le ~240 matmul per token del decode reale:
    //     20 thread -> 5,51 ms     22 thread -> 46,67 ms
    // cioe' un crollo di 8,5x fra 20 e 22. Con ogni processore logico
    // occupato da un thread del pool in attesa attiva, il sistema operativo
    // non ha dove schedulare nient'altro: basta che un worker venga tolto
    // dalla CPU perche' la barriera resti bloccata per un intero quanto di
    // scheduling, e questo accade a OGNI dispatch.
    // Il tetto vale anche per un override esplicito: chiedere 22 thread su 22
    // logici non e' una scelta di prestazioni, e' un modo per farsi male.
    //
    // Il precipizio e' netto e sta fra 20 e 21 su 22 logici (5,51 ms -> 44,47
    // ms), quindi non basta lasciarne libero uno: se ne lasciano DUE. Il
    // thread che manda in esecuzione, il logger e il runtime .NET devono
    // poter girare senza contendere un core a un worker in spin.
    const size_t cap = hw > 4 ? hw - 2 : (hw > 1 ? hw - 1 : 1);
    if (n_total > cap) n_total = cap;

    // n_total is the TOTAL parallelism: the calling thread participates
    // in the work (it acts as worker 0), so only n_total-1 additional
    // workers get created.
    const size_t n = n_total > 1 ? n_total - 1 : 0;
    // Allocato una volta e mai distrutto (vedi commento in thread_pool.h):
    // niente join() dei worker durante lo smontaggio del processo/DLL.
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

    // Misura delle transizioni seriale/parallelo. Il tempo fra la fine del
    // dispatch precedente e l'inizio di questo e' esattamente il tratto in cui
    // gira un solo thread mentre gli altri sono fermi.
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
        // Fette abbastanza piccole da bilanciare (~8 per worker) ma non tanto
        // da far pesare la fetch_add atomica sul lavoro utile.
        //
        // ARROTONDATE A MULTIPLI DI 16: l'uscita y e' un array di float, e
        // una cache line da 64 byte ne contiene 16. Con fette da 8 righe ogni
        // linea di y veniva scritta da DUE thread diversi, e ogni linea
        // rimbalzava avanti e indietro fra i core a ogni scrittura (false
        // sharing). Allineando le fette alla cache line ogni linea di y
        // appartiene a un solo thread.
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

    // Il chiamante lavora anche lui invece di attendere a vuoto.
    run_chunks();

    // Da qui in poi il chiamante ha finito il proprio lavoro utile: tutto
    // quello che segue e' puro costo di sincronizzazione (attesa che i worker
    // arrivino alla barriera), ed e' la voce da confrontare col lavoro vero.
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
    // Provato uno spin-wait breve prima di cv_start_.wait() (come il
    // polling opzionale di ggml): misurato PEGGIO su questa macchina (21
    // worker che fanno _mm_pause() in parallelo competono per le risorse
    // dei core con hyperthreading, rallentando il lavoro vero) â€” 5-6 tok/s
    // invece di 9.6-10 col solo condition_variable. Rimosso: il dato
    // misurato conta piu' dell'aspettativa teorica (vedi
    // docs/engine_gap_analysis.md).
    (void) worker_id; // le fette non dipendono piu' dall'id: sono dinamiche

    // Quanti giri di attesa attiva prima di addormentarsi davvero. Il decode
    // fa ~240 matmul per token, quindi il dispatch successivo arriva quasi
    // sempre entro pochi microsecondi: dormire e risvegliarsi ogni volta
    // costava piu' del lavoro stesso. Misurato con il regime "MANY" di
    // bench/kernel_bench.cpp: gli stessi byte totali spezzati in 200
    // chiamate piccole rendevano 7,6 GB/s contro 53,7 GB/s in una sola
    // chiamata grande, cioe' ~166 us di puro overhead per matmul.
    //
    // Nota storica: uno spin era gia' stato provato e misurato PEGGIO. Quella
    // versione pero' spinnava con 21 worker (oltre i core fisici, quindi due
    // worker per core SMT che si rubavano le risorse a vicenda) e comunque
    // richiedeva il mutex per leggere la task. Qui lo spin e' limitato, usa
    // pause, e soprattutto NON prende il mutex: la generazione si legge da un
    // atomico. Se lo spin scade si torna a dormire sulla condition variable,
    // cosi' un pool inattivo non brucia CPU.
    // Spin BREVE, deliberatamente. Provato ad allungarlo a 32768 giri (~350
    // us, abbastanza da coprire il lavoro seriale fra una matmul e l'altra e
    // quindi da non addormentarsi mai durante un token): misurato PEGGIO,
    // 8,73 tok/s contro 9,84. La latenza di risveglio non era dunque il costo
    // dominante; i worker in attesa attiva rubano invece risorse SMT e budget
    // di turbo al thread che in quel momento fa il lavoro seriale, e su un
    // portatile 15 core che spinnano tengono giu' le frequenze di tutti.
    // Restano pochi giri, utili solo a catturare i dispatch consecutivi
    // (wq/wk/wv, ffn_gate/ffn_up) senza passare dal sistema operativo.
    constexpr int kSpinRounds = 512;

    uint64_t local_gen = 0;
    for (;;) {
        int spins = 0;
        for (;;) {
            if (generation_.load(std::memory_order_acquire) != local_gen) break;
            if (++spins >= kSpinRounds) {
                std::unique_lock<std::mutex> lk(mtx_);
                // Il predicato ricontrolla generation_ sotto mutex: se run_raw
                // ha pubblicato mentre stavamo per addormentarci, wait ritorna
                // subito e non si perde il risveglio.
                cv_start_.wait(lk, [this, local_gen] {
                    return stop_ || generation_.load(std::memory_order_acquire) != local_gen;
                });
                if (stop_) return;
                break;
            }
            desireeia_cpu_relax();
        }
        local_gen = generation_.load(std::memory_order_acquire);

        // task_fn_/task_ctx_ non vengono copiate ne' lette sotto mutex: sono
        // scritte da run_raw() prima di incrementare generation_ e restano
        // valide finche' run_raw() non ritorna, cosa che accade solo quando
        // pending_ e' tornato a zero, cioe' dopo la riga sotto.
        run_chunks();

        pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}




