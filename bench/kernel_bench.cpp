// Microbenchmark dei kernel quantizzati.
//
// Perche' esiste: le misure end-to-end con la CLI su questo portatile
// derivano del 30% fra un batch e l'altro per throttling termico (a parita'
// di binario), pur essendo stabili al 5% dentro lo stesso batch. Confrontare
// due varianti di kernel misurandole a minuti di distanza e' quindi privo di
// significato — un "miglioramento" del 10% e' dentro la deriva. Qui si
// misura il solo kernel, in pochi secondi, e si riporta il MIGLIORE di N
// ripetizioni: il best-of e' robusto rispetto a throttling e interferenze
// (entrambi possono solo rallentare, mai accelerare).
//
// Le dimensioni imitano i due matmul che dominano il decode di gemma3-4b
// (misurato col profiler: q4k ~62%, q6k ~31% del tempo).
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
    // argv[2] = parallelismo totale. Va fissato PRIMA della prima matmul,
    // perche' il pool globale si costruisce pigramente una volta sola.
    // Serve perche' il default (tutti i thread logici) sovraccarica i core
    // fisici, e con l'attesa attiva l'oversubscription e' devastante.
    const int threads = argc > 2 ? atoi(argv[2]) : 0;
    if (threads > 0) ThreadPool::set_thread_override((size_t) threads);
    printf("threads=%s\n", threads > 0 ? argv[2] : "default");

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    // DUE REGIMI, e la differenza fra i due e' il punto di questo strumento.
    //
    // Errore commesso il 2026-09-07 e corretto qui: la prima versione usava
    // solo la matrice piccola (2560x10240 = 14,7 MB) riusata a ogni
    // ripetizione. Su questa CPU la L3 e' 24 MB, quindi i pesi restavano
    // CALDI IN CACHE e il benchmark misurava un regime che nel motore vero
    // non esiste mai: durante il decode ogni tensore si legge una volta
    // sola, in streaming dalla DRAM. Risultato: una modifica misurata +67%
    // "in cache" si e' rivelata neutra end-to-end. Un benchmark che mente e'
    // peggio di nessun benchmark.
    //
    // - "L3"   : working set che sta in cache. Utile per isolare il costo di
    //            CALCOLO puro del kernel (istruzioni, dipendenze).
    // - "DRAM" : working set molto piu' grande della L3. E' il regime del
    //            decode reale, ed e' l'unico che predice l'end-to-end.
    // Guardare SEMPRE il numero DRAM per decidere se una modifica va tenuta.
    struct Regime { const char* tag; size_t rows; size_t cols; };
    const Regime regimes[] = {
        { "L3",   2560,   10240 },   // ~15 MB  (ffn_down di gemma3-4b)
        { "DRAM", 200000, 2560  },   // ~288 MB (oltre 10x la L3)
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
    // REGIME "MANY": stessi byte totali del regime DRAM, ma spezzati in molte
    // chiamate piccole invece di una grande — cioe' il pattern reale del
    // decode, dove ogni token fa ~240 matmul su tensori distinti di pochi MB.
    // Confrontare questo con q4_K/DRAM isola ESATTAMENTE il costo per
    // chiamata (dispatch del thread pool, risveglio dei worker, barriera)
    // separandolo dal costo del calcolo e della memoria.
    // Forme reali dei tensori q4_K di un layer gemma3-4b, con un numero di
    // chiamate scelto per tenere il volume totale ben oltre la L3 (~290 MB).
    // Serve perche' il costo puo' dipendere dalla FORMA (righe corte = piu'
    // dispatch per byte), non solo dal volume.
    struct ManyCase { const char* tag; size_t rows; size_t cols; size_t calls; };
    const ManyCase many_cases[] = {
        { "kv1024",   1024, 2560, 200 },  // wk / wv
        { "q2048",    2048, 2560, 100 },  // wq / wo
        { "ffn10240", 10240, 2560, 20 },  // ffn_gate / ffn_up (56% del lavoro)
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
        // Stesse chiamate, ma con l'attivazione quantizzata UNA volta
        // fuori dal ciclo (matmul_q4_k_pq). La differenza fra MANY e MANY_PQ
        // e' esattamente il costo per chiamata di quantize_act_q8k_rep e
        // delle tre allocazioni di vettori che matmul_q4_k fa al suo interno.
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
    return 0;
}
