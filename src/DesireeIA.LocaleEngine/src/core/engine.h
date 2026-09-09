#ifndef DESIREEIA_ENGINE_H
#define DESIREEIA_ENGINE_H

#include "desireeia/abi.h"
#include "ssd_tier/hybrid_tier.h"
#include "thread_pool.h"
#include <algorithm>
#include <functional>
#include <list>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace desireeia {

#if defined(_WIN32) && defined(DESIREEIA_BUILD)
#define DESIREEIA_INTERNAL __declspec(dllexport)
#else
#define DESIREEIA_INTERNAL
#endif

// Divide `rows` righe indipendenti fra i worker del pool di thread globale
// (vedi thread_pool.h) e chiama fn(start,end) su ciascun sotto-intervallo.
// Usato dai kernel di matmul: ogni riga di output e' indipendente dalle
// altre (stesso W/x in lettura, y[start..end) in scrittura esclusiva),
// quindi la parallelizzazione non cambia il risultato, solo come il lavoro
// e' distribuito. I thread sono persistenti (creati una sola volta per
// processo): niente overhead di spawn/join a ogni chiamata matmul.
template <typename Fn>
void parallel_rows(size_t rows, Fn&& fn) {
    ThreadPool::global().parallel_for(rows, std::forward<Fn>(fn));
}

// Come parallel_rows, ma per un numero PICCOLO di unita' grosse e
// indipendenti — tipicamente le teste dell'attenzione (8 o 16), dove ogni
// unita' vale millisecondi.
//
// parallel_rows non va bene in quel caso per due motivi, entrambi corretti
// qui: esegue in linea sotto le 64 unita' (soglia pensata per non pagare il
// dispatch su matmul minuscole), e arrotonda le fette a 16 per allineare le
// scritture di y[] alla cache line — con 8 teste finirebbero tutte in
// un'unica fetta, cioe' su un thread solo. Qui la fetta e' una unita' e non
// c'e' falso sharing perche' ogni testa scrive un blocco separato di
// head_dim float.
template <typename Fn>
void parallel_units(size_t units, Fn&& fn) {
    ThreadPool::global().parallel_for_units(units, std::forward<Fn>(fn));
}

using LogFn = std::function<void(int32_t, const char*)>;

desireeia_error probe_hardware(desireeia_hw_info& out);
desireeia_plan build_plan(const desireeia_hw_info& hw, const std::string& model_path);

// Applies DESIREEIA_SSD_TIER=off|auto|always to `mode`, returning true when
// the variable was set to something recognised. Lets a benchmark force the
// tier either way without touching caller code.
bool env_ssd_tier_override(int32_t& mode);

struct ModelMeta {
    desireeia_format format = DESIREEIA_FORMAT_UNKNOWN;
    std::string arch;
    uint32_t n_vocab = 0;
    uint32_t n_ctx = 0;
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;
    std::string path;
    bool expert_model = false;
};

// Vocabolario e metadati tokenizer letti dai KV del formato (es. GGUF
// tokenizer.ggml.*). tokenizer_tag e' il valore grezzo dichiarato dal file
// (es. il tag standard di terze parti per SentencePiece); la mappatura su un
// algoritmo interno avviene altrove (vedi core/arch_tags.h) per rispettare
// la regola di denominazione.
struct VocabData {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int32_t> token_type;
    std::vector<std::string> merges;
    std::string tokenizer_tag;
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t unk_id = -1;
    int32_t pad_id = -1;
    bool add_bos = true;
};

// Un esperto MoE non e' un blob unico ma 3 matrici distinte (gate/up/down
// del suo FFN, stessa struttura del FFN denso ma una per esperto). ExpertPart
// seleziona quale delle 3 leggere/cachare.
enum class ExpertPart : uint32_t { Gate = 0, Up = 1, Down = 2 };

class ModelReader {
public:
    virtual ~ModelReader() = default;
    virtual bool open(const std::string& path, ModelMeta& meta) = 0;
    virtual bool read_tensor(const std::string& name, std::vector<float>& out) = 0;
    virtual bool read_expert(uint32_t layer, uint32_t idx, ExpertPart part, std::vector<float>& out) = 0;
    virtual bool meta_u32(const std::string& key, uint32_t& out) {
        (void)key; (void)out;
        return false;
    }
    virtual bool meta_f32(const std::string& key, float& out) {
        (void)key; (void)out;
        return false;
    }
    virtual bool meta_str(const std::string& key, std::string& out) {
        (void)key; (void)out;
        return false;
    }
    virtual bool read_vocab(VocabData& out) {
        (void)out;
        return false;
    }
    // Legge i byte del tensore cosi' come sono su disco (ancora
    // quantizzati, nessuna dequantizzazione), piu' tipo e shape. Serve ai
    // kernel di matmul che operano direttamente sul formato quantizzato
    // (es. Q4_0 fuso con attivazioni int8, vedi core/matmul.cpp) invece di
    // passare per il dequant-a-float di read_tensor. Non tutti i reader lo
    // implementano (default: fallisce, il chiamante ricade su read_tensor).
    virtual bool read_tensor_raw(const std::string& name, std::vector<uint8_t>& raw,
                                  int& quant_type, uint64_t& ne0, uint64_t& rows) {
        (void) name; (void) raw; (void) quant_type; (void) ne0; (void) rows;
        return false;
    }
};

class KvCache {
public:
    KvCache(uint32_t n_layers, uint32_t n_kv, bool compressed);
    void push(const std::vector<float>& keys, const std::vector<float>& vals);
    bool compressed() const { return compressed_; }
    uint64_t bytes() const;
private:
    uint32_t n_layers_;
    uint32_t n_kv_;
    bool compressed_;
    std::vector<std::vector<float>> keys_;
    std::vector<std::vector<float>> vals_;
};

struct ExpertRequest {
    uint32_t layer;
    uint32_t idx;
    ExpertPart part;
};

class ExpertStore {
public:
    ExpertStore(int32_t cache_count, LogFn log);
    // Il reader non e' posseduto: deve restare valido per tutta la vita
    // di ExpertStore (in ctx.cpp entrambi vivono dentro lo stesso EngineState).
    void set_reader(ModelReader* reader) { reader_ = reader; }
    void set_hybrid_tier(HybridTier* tier) { hybrid_tier_ = tier; }
    bool fetch(uint32_t layer, uint32_t idx, ExpertPart part, std::vector<float>& out);
    void fetch_union(const std::vector<ExpertRequest>& reqs, std::vector<float>& out_buffer,
                     std::vector<const float*>& out_ptrs);
    void prefetch_layer(uint32_t layer, const std::vector<uint32_t>& idxs);
    void set_prefetch_depth(int32_t depth) { prefetch_depth_ = depth > 0 ? depth : 1; }
    void set_pin_enabled(bool on) { pin_enabled_ = on; }
    void set_usage_file(const std::string& path) { usage_file_ = path; }
    void load_usage();
    void save_usage();
private:
    struct Entry {
        uint64_t key;
        std::vector<float> data;
        uint32_t hits;
        bool pinned;
    };
    // layer (16 bit) | idx (32 bit) | part (2 bit): margini ampi per non
    // collidere su modelli reali (migliaia di layer/esperti, 3 sole parti).
    static uint64_t make_key(uint32_t layer, uint32_t idx, ExpertPart part) {
        return (static_cast<uint64_t>(layer) << 48) | (static_cast<uint64_t>(idx) << 8) |
               static_cast<uint64_t>(part);
    }
    bool load_data(uint64_t key, std::vector<float>& out);
    ModelReader* reader_ = nullptr;
    HybridTier* hybrid_tier_ = nullptr;
    int32_t cache_count_;
    int32_t prefetch_depth_ = 1;
    bool pin_enabled_ = true;
    std::string usage_file_;
    LogFn log_;
    std::list<Entry> order_;
    std::unordered_map<uint64_t, std::list<Entry>::iterator> map_;
    std::unordered_map<uint64_t, uint32_t> usage_;
};

DESIREEIA_INTERNAL ModelReader* make_gguf_reader();
DESIREEIA_INTERNAL ModelReader* make_st_reader();

// HybridModelReader wraps an existing ModelReader with HybridTier SSD caching.
// All read_tensor/read_tensor_raw calls are intercepted: if the tensor is
// already in the SSD-tier RAM cache, it is served from there; otherwise it
// is read through the underlying reader AND cached for future reads.
// Takes ownership of the inner reader (deletes it on destruction).
class HybridModelReader : public ModelReader {
public:
    HybridModelReader(ModelReader* inner, HybridTier* tier)
        : inner_(inner), tier_(tier) {}
    ~HybridModelReader() override { delete inner_; }

    bool open(const std::string& path, ModelMeta& meta) override {
        return inner_->open(path, meta);
    }

    bool read_tensor(const std::string& name, std::vector<float>& out) override {
        if (tier_ && tier_->gguf_available()) {
            if (tier_->read_tensor(name, out)) return true;
        }
        return inner_->read_tensor(name, out);
    }

    bool read_expert(uint32_t layer, uint32_t idx, ExpertPart part,
                     std::vector<float>& out) override {
        if (tier_ && tier_->gguf_available()) {
            return tier_->read_expert(layer, idx, static_cast<uint32_t>(part), out);
        }
        return inner_->read_expert(layer, idx, part, out);
    }

    bool meta_u32(const std::string& key, uint32_t& out) override {
        return inner_->meta_u32(key, out);
    }
    bool meta_f32(const std::string& key, float& out) override {
        return inner_->meta_f32(key, out);
    }
    bool meta_str(const std::string& key, std::string& out) override {
        return inner_->meta_str(key, out);
    }
    bool read_vocab(VocabData& out) override {
        return inner_->read_vocab(out);
    }
    bool read_tensor_raw(const std::string& name, std::vector<uint8_t>& raw,
                          int& quant_type, uint64_t& ne0, uint64_t& rows) override {
        if (tier_ && tier_->gguf_available()) {
            if (tier_->read_tensor_raw(name, raw, quant_type, ne0, rows)) return true;
        }
        return inner_->read_tensor_raw(name, raw, quant_type, ne0, rows);
    }

private:
    ModelReader* inner_;
    HybridTier* tier_;
};

namespace vision { struct VisionGGUFContext; }

struct EngineState {
    ModelMeta meta;
    std::vector<int32_t> tokens;
    KvCache* kv = nullptr;
    ExpertStore* experts = nullptr;
    HybridTier* hybrid = nullptr;
    ModelReader* reader = nullptr;
    desireeia_plan plan;
    LogFn log;
    std::string last_error;
    std::string usage_file;

    // Vision encoder for multimodal models (LLaVA, MiniCPM-V, etc.)
    // Loaded from the same GGUF file when clip.vision.* metadata is present.
    vision::VisionGGUFContext* vision = nullptr;
};

desireeia_ctx* engine_create(const char* model_path, const desireeia_plan& plan, LogFn log);
void engine_destroy(desireeia_ctx* ctx);
bool engine_predict(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens, int32_t& out_token);
bool engine_next_token(desireeia_ctx* ctx, int32_t& out_token);
size_t engine_context_size(const desireeia_ctx* ctx);
bool engine_tokenize(desireeia_ctx* ctx, const std::string& text, bool add_bos, std::vector<int32_t>& out_ids);
bool engine_token_piece(desireeia_ctx* ctx, int32_t id, std::string& out);
// which: 0=BOS 1=EOS 2=UNK 3=PAD (desireeia_special_token in abi.h).
bool engine_special_token_id(const desireeia_ctx* ctx, int which, int32_t& out_id);
// true se `id` e' un token di fine generazione (EOS, <end_of_turn>, ...).
bool engine_is_eog_token(const desireeia_ctx* ctx, int32_t id);

// BERT (encoder-only, ArchKind::Bert): embedding per token dell'intera
// sequenza in una sola chiamata (nessuno stato fra chiamate, a differenza
// di engine_predict/engine_next_token). out_embd riempito con
// n_tokens*embd_dim float; out_embd_dim riceve la larghezza embedding del
// modello (per permettere al chiamante di trovare i confini per-token).
// false se il modello caricato non e' un encoder BERT (DESIREEIA_ERR_NOT_SUPPORTED
// nell'ABI) o se ctx/tokens non sono validi.
bool engine_embed(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens,
                   std::vector<float>& out_embd, uint32_t& out_embd_dim);

// Campionamento (vedi core/sampler.h e desireeia_sampling in abi.h).
bool engine_set_sampling(desireeia_ctx* ctx, const desireeia_sampling& params);
bool engine_get_sampling(const desireeia_ctx* ctx, desireeia_sampling& out);

// Template di chat (vedi core/chat_template.h). Applica il formato
// rilevato al caricamento del modello ai messaggi passati, producendo il
// testo da tokenizzare. Ritorna false solo se ctx e' invalido: un
// formato non riconosciuto ricade comunque su ChatML (mai un errore).
bool engine_apply_chat_template(const desireeia_ctx* ctx,
                                const char** roles, const char** contents, size_t n_messages,
                                bool add_assistant, std::string& out);

DESIREEIA_INTERNAL int quantized_matmul(const std::vector<float>& a, const std::vector<float>& b,
                     size_t M, size_t K, size_t N, std::vector<float>& out);

DESIREEIA_INTERNAL void quantize_q8_0(const float* src, size_t n, std::vector<int8_t>& q,
                   std::vector<float>& scales, std::vector<uint8_t>& signs);

// y = W * x, with W as raw Q4_0 rows (not dequantized: 18 bytes per
// 32-weight block) and x quantized internally into int8 Q8_0-style blocks
// (activation), int4xint8 dot product. Avoids ever materializing W in
// float. rows = W's rows (output dimension), cols = columns (input
// dimension, must be a multiple of 32). y must have room for `rows`
// elements.
DESIREEIA_INTERNAL int matmul_q4_0(const uint8_t* q4_data, size_t rows, size_t cols,
                                 const float* x, float* y);

// Come matmul_q4_0 ma per Q8_0: nessun nibble da spacchettare, i pesi sono
// gia' int8 (blocchi da 32, 1 scala fp16 per blocco) — dot diretto con
// l'attivazione anch'essa quantizzata Q8_0, stesso schema. Aggiunta perche'
// scoperta via profiler (Fase 9): i tensori Q8_0 (es. Qwen2.5-Coder in
// Q8_0) ricadevano sul fallback float (matmul_f32, ~3x piu' lento),
// nessun kernel dedicato esisteva. cols deve essere multiplo di 32.
DESIREEIA_INTERNAL int matmul_q8_0(const uint8_t* q8_data, size_t rows, size_t cols,
                                 const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q8_0_batch(const uint8_t* q8_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);

// "Legacy" GGUF formats (a single 32-weight block, no super-block):
// Q4_1 (4-bit, asymmetric: y=q*d+m), Q5_0 (5-bit via a 32-bit qh field,
// symmetric: y=(q-16)*d), Q5_1 (5-bit via qh, asymmetric: y=q*d+m). Rare
// in modern GGUF files (K-quants are preferred) but a legitimate part of
// the format — no reason to leave them stuck on the float fallback if a
// tensor uses one of them.
DESIREEIA_INTERNAL int matmul_q4_1(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q4_1_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);
DESIREEIA_INTERNAL int matmul_q5_0(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q5_0_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);
DESIREEIA_INTERNAL int matmul_q5_1(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q5_1_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);

// Q8_K: super-blocco da 256, un'unica scala float per blocco (no fp16, no
// min), pesi gia' int8 — il piu' semplice dei K-quant.
DESIREEIA_INTERNAL int matmul_q8_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q8_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);

// Q2_K: super-blocco da 256, pesi a 2 bit, scala+min a 4 bit per
// sotto-blocco da 16 (asimmetrico, stesso principio scale/min di Q4_K ma
// granularita' piu' fine e senza il trucco a 6 bit di get_scale_min_k4).
DESIREEIA_INTERNAL int matmul_q2_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q2_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);

// Q3_K: super-blocco da 256, pesi a 3 bit (2 bit da qs + 1 bit da hmask,
// -4 se il bit hmask e' spento), scala a 6 bit per sotto-blocco da 16
// (simmetrico, offset -32, nessun termine min). Formula di unpacking delle
// scale (12 byte -> 16 valori a 6 bit) verificata contro
// dequantize_row_q3_K in quant.cpp, non reinventata.
DESIREEIA_INTERNAL int matmul_q3_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q3_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);

// Come matmul_q4_0 ma per il formato super-block Q4_K (blocchi da 256
// pesi, scale+min a 6 bit per sotto-blocco da 32, quantizzazione
// asimmetrica). cols deve essere multiplo di 256.
DESIREEIA_INTERNAL int matmul_q4_k(const uint8_t* q4k_data, size_t rows, size_t cols,
                                 const float* x, float* y);

// Come matmul_q4_k ma per il formato super-block Q6_K (256 pesi/blocco,
// quantizzazione simmetrica a 6 bit, scala int8 per sotto-blocco da 16,
// niente termine "min"). cols deve essere multiplo di 256.
DESIREEIA_INTERNAL int matmul_q6_k(const uint8_t* q6k_data, size_t rows, size_t cols,
                                 const float* x, float* y);

// Come matmul_q4_k ma per il formato super-block Q5_K (256 pesi/blocco,
// scale+min a 6 bit per sotto-blocco da 32 come Q4_K, ma pesi a 5 bit:
// 4 bit da qs + 1 bit alto da qh). cols deve essere multiplo di 256.
DESIREEIA_INTERNAL int matmul_q5_k(const uint8_t* q5k_data, size_t rows, size_t cols,
                                 const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q5_k_batch(const uint8_t* q5k_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);

// Quantizzazione attivazione stile Q8_K (una scala float per super-blocco
// da 256, vedi la nota in matmul.cpp) condivisa da Q4_K/Q5_K/Q6_K:
// esposta per permettere a dense_forward.cpp di quantizzare UNA VOLTA
// l'attivazione in ingresso a un layer e riusarla per tutte le matrici
// K-quant di quel layer che leggono dalla stessa attivazione (wq/wk/wv/wo,
// ffn_gate/ffn_up), invece di farlo ridondantemente in ogni matmul_qX_k.
DESIREEIA_INTERNAL void quantize_q8_k_super(const float* x, size_t cols, std::vector<int8_t>& q, std::vector<float>& dscale);

// Quantizzazione Q8_K dell'attivazione (una scala ogni 256) esposta con gli
// array per-32 attesi dai kernel: la scala e' replicata nelle 8 caselle del
// super-blocco. E' il prerequisito dell'accumulazione intera in
// matmul_q4_k_core — vedi il commento esteso sulla definizione in matmul.cpp.
DESIREEIA_INTERNAL void quantize_act_q8k_rep(const float* x, size_t cols, std::vector<int8_t>& q,
                                          std::vector<float>& xscale32, std::vector<int32_t>& xsum32);

// Varianti "pre-quantizzate" (Fase "elimina ri-quantizzazione ridondante",
// 2026-09-07): come matmul_q4_k/matmul_q6_k ma prendono l'attivazione
// gia' quantizzata (quantize_q8_k_super) invece di quantizzarla al loro
// interno. xsum (solo per Q4_K, che ha il termine "min") e' la somma
// delle attivazioni quantizzate per sotto-blocco da 32, calcolabile con
// lo stesso schema gia' usato in matmul_q4_k.
// One matrix in a fused group (see matmul_fused_pq). All jobs in a group share
// the same pre-quantized activation and write to disjoint outputs.
enum class FusedPqFormat { Q4_K, Q6_K };

struct FusedPqJob {
    FusedPqFormat  format = FusedPqFormat::Q4_K;
    const uint8_t* data   = nullptr;
    size_t         rows   = 0;
    size_t         cols   = 0;
    const int8_t*  xq     = nullptr;
    const float*   xscale = nullptr;
    const int32_t* xsum   = nullptr;  // Q4_K only; ignored for Q6_K.
    float*         y      = nullptr;
};

// True when every job is a shape and format the fused path can handle, and
// there are at least two of them (one job has nothing to fuse with).
DESIREEIA_INTERNAL bool fused_pq_supported(const FusedPqJob* jobs, size_t n_jobs);

// Computes all the jobs in a SINGLE parallel dispatch instead of one each.
// Independent matrices that read the same activation — Q/K/V, or FFN gate/up —
// have no reason to be separate parallel regions, and every region ends with a
// barrier where all threads wait for the slowest. Fewer regions means fewer
// chances for a descheduled worker to stall the rest.
DESIREEIA_INTERNAL int matmul_fused_pq(const FusedPqJob* jobs, size_t n_jobs);

DESIREEIA_INTERNAL int matmul_q4_k_pq(const uint8_t* q4k_data, size_t rows, size_t cols,
                                    const int8_t* xq, const float* dscale, const int32_t* xsum, float* y);
DESIREEIA_INTERNAL int matmul_q6_k_pq(const uint8_t* q6k_data, size_t rows, size_t cols,
                                    const int8_t* xq, const float* dscale, float* y);

// ============================================================
// Vision / Multimodal support
// ============================================================

// Load vision encoder from the GGUF file's clip.vision.* metadata.
// Called automatically during engine_create when vision metadata is detected.
// Returns true if a vision encoder was loaded.
bool engine_load_vision(desireeia_ctx* ctx);

// Encode an image into embeddings using the loaded vision encoder.
// out_embd receives the projected embeddings suitable for text model injection.
// Returns true on success.
bool engine_encode_image(desireeia_ctx* ctx, const DesireeAIImage& image,
                         std::vector<float>& out_embd, uint32_t& out_dim);

// Get the number of tokens the vision encoder produces per image.
int32_t engine_vision_token_count(desireeia_ctx* ctx);

// Resolved image placeholder token id (from the model's vocabulary), or -1.
int32_t engine_vision_image_token(desireeia_ctx* ctx);

// Check if the loaded model has a vision encoder.
bool engine_has_vision(desireeia_ctx* ctx);

// Multimodal prefill: run the model on a token stream where the image
// placeholder token appears once per vision embedding vector, replacing each
// occurrence's embedding with its vector from embd. See the note in ctx.cpp.
bool engine_predict_vision(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens,
                           const float* embd, size_t n_embd, int32_t image_token,
                           int32_t& out_token);

// Varianti "batch" (Fase 8, prefill): come le matmul_qX_k sopra ma con
// n_tok colonne di attivazione invece di una sola. x e' n_tok blocchi
// contigui da `cols` float (un token dopo l'altro), y e' n_tok blocchi
// contigui da `rows` float. Il guadagno rispetto a chiamare la versione a
// singola colonna n_tok volte: i byte grezzi del peso (e il loro decode
// nibble->int8) vengono letti/decodificati UNA SOLA volta per riga e
// riusati per tutte le colonne, invece di essere ricaricati dalla
// memoria/decodificati n_tok volte. Matematicamente identico alla stessa
// chiamata ripetuta n_tok volte (nessun riordino fra colonne diverse).
DESIREEIA_INTERNAL int matmul_q4_0_batch(const uint8_t* q4_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);
DESIREEIA_INTERNAL int matmul_q4_k_batch(const uint8_t* q4k_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);
DESIREEIA_INTERNAL int matmul_q6_k_batch(const uint8_t* q6k_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);

}

#endif
