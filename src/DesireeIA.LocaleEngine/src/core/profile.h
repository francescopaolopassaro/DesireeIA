#ifndef DESIREEIA_PROFILE_H
#define DESIREEIA_PROFILE_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

namespace desireeia {

// Profiler minimo, sempre attivo (l'overhead di una lettura di clock +
// una somma atomica per chiamata matmul e' trascurabile rispetto al costo
// della chiamata stessa: ~180 matmul/token, un paio di clock read ciascuna
// non un collo di bottiglia). Serve a rispondere con dati reali a "dove va
// il tempo", invece di continuare a ipotizzare ottimizzazioni una alla
// volta (vedi docs/engine_gap_analysis.md per i tentativi gia' scartati
// per mancanza di dati concreti).
struct ProfileCounters {
    std::atomic<int64_t> ns_quantize_act{0}; // quantize_q8_0 dell'attivazione (parte single-thread di ogni kernel)
    std::atomic<int64_t> ns_q4k_compute{0};  // parallel_rows dentro matmul_q4_k (dopo la quantizzazione)
    std::atomic<int64_t> ns_q6k_compute{0};
    std::atomic<int64_t> ns_q5k_compute{0};
    std::atomic<int64_t> ns_q40_compute{0};
    std::atomic<int64_t> ns_q80_compute{0};
    std::atomic<int64_t> ns_f32_compute{0};  // matmul_f32 (fallback dequantizzato, dense_forward.cpp)
    // Formati "legacy" (Q4_1/Q5_0/Q5_1, un solo blocco da 32, rari nei GGUF
    // moderni che preferiscono i K-quant): un contatore condiviso, non uno
    // per formato, per non gonfiare il profiler per path poco usati.
    std::atomic<int64_t> ns_legacy_compute{0};
    // Q2_K/Q3_K/Q8_K: contatore condiviso "kquant2" per non gonfiare
    // ulteriormente il profiler per formati meno comuni di Q4_K/Q5_K/Q6_K.
    std::atomic<int64_t> ns_kquant2_compute{0};
    std::atomic<int64_t> calls_q4k{0};
    std::atomic<int64_t> calls_q6k{0};
    std::atomic<int64_t> calls_q5k{0};
    std::atomic<int64_t> calls_q40{0};
    std::atomic<int64_t> calls_q80{0};
    std::atomic<int64_t> calls_legacy{0};
    std::atomic<int64_t> calls_kquant2{0};
    std::atomic<int64_t> calls_f32{0};
    // Matrici calcolate in dispatch fuso (Q/K/V insieme, gate/up insieme).
    // Contate a parte e non dentro q4k/q6k: un gruppo fuso puo' mescolare i
    // due formati, e sommarlo all'uno o all'altro renderebbe il profilo
    // bugiardo proprio sulla voce che si sta cercando di ottimizzare.
    std::atomic<int64_t> ns_fused_compute{0};
    std::atomic<int64_t> calls_fused{0};

    // --- Transizioni seriale/parallelo (thread pool) ---
    // Il decode alterna in continuazione regioni parallele (le matmul) e
    // tratti seriali su un solo thread (norm, RoPE, attenzione, softmax,
    // residui). Questi contatori servono a misurare quante volte accade e
    // quanto costa, invece di dedurlo: e' l'ipotesi corrente per il fattore
    // 2,1 fra la banda del kernel isolato (48-55 GB/s) e quella dentro il
    // motore (22 GB/s). Vedi docs/engine_gap_analysis.md.
    std::atomic<int64_t> n_dispatch{0};       // quante regioni parallele
    std::atomic<int64_t> ns_serial_gap{0};    // tempo FRA la fine di un dispatch e l'inizio del successivo
    std::atomic<int64_t> ns_dispatch_wait{0}; // tempo alla barriera dopo che il chiamante ha finito le sue fette
    std::atomic<int64_t> ns_dispatch_total{0};// tempo totale dentro run_raw

    // --- Fasi seriali del forward (un solo thread mentre i worker dormono) ---
    // Servono a sapere DOVE va il tratto seriale, invece di indovinare: il
    // primo tentativo (ordine dei cicli dell'attenzione) ha corretto un vero
    // difetto di localita' ma ha spostato appena l'ago, segno che il grosso
    // era altrove.
    std::atomic<int64_t> ns_ser_norm{0};  // rms_norm (attn/ffn/post/qk) + residui
    std::atomic<int64_t> ns_ser_rope{0};  // RoPE + scrittura KV cache
    std::atomic<int64_t> ns_ser_attn{0};  // punteggi QK, softmax, combinazione V
    std::atomic<int64_t> ns_ser_act{0};   // gelu/silu e prodotto gate*up della FFN
};

inline ProfileCounters& profile_counters() {
    static ProfileCounters pc;
    return pc;
}

// Cronometro RAII: nel distruttore aggiunge i nanosecondi trascorsi al
// contatore atomico indicato.
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
    pc.n_dispatch = 0; pc.ns_serial_gap = 0; pc.ns_dispatch_wait = 0; pc.ns_dispatch_total = 0;
    pc.ns_ser_norm = 0; pc.ns_ser_rope = 0; pc.ns_ser_attn = 0; pc.ns_ser_act = 0;
}

}

#endif
