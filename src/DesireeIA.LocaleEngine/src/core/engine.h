// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_ENGINE_H
#define DESIREEIA_ENGINE_H

#include "desireeia/abi.h"
#include "ssd_tier/hybrid_tier.h"
#include "thread_pool.h"
#include <algorithm>
#include <condition_variable>
#include <functional>
#include <list>
#include <mutex>
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

// Splits `rows` independent rows across the global thread pool's workers
// (see thread_pool.h) and calls fn(start,end) on each sub-range. Used by
// the matmul kernels: every output row is independent of the others (same
// W/x read, exclusive write to y[start..end)), so parallelizing doesn't
// change the result, only how the work is distributed. The threads are
// persistent (created once per process): no spawn/join overhead on every
// matmul call.
template <typename Fn>
void parallel_rows(size_t rows, Fn&& fn) {
    ThreadPool::global().parallel_for(rows, std::forward<Fn>(fn));
}

// Like parallel_rows, but for a SMALL number of large, independent units —
// typically attention heads (8 or 16), where each unit is worth
// milliseconds.
//
// parallel_rows doesn't fit that case, for two reasons, both correct here:
// it runs inline below 64 units (a threshold meant to avoid paying dispatch
// cost on tiny matmuls), and it rounds slices up to 16 to align y[]
// writes to the cache line — with 8 heads they'd all land in a single
// slice, i.e. on one thread only. Here the slice is one unit, and there's
// no false sharing because each head writes a separate block of head_dim
// floats.
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

// Vocabulary and tokenizer metadata read from the format's KV pairs (e.g.
// GGUF tokenizer.ggml.*). tokenizer_tag is the raw value declared by the
// file (e.g. the standard third-party tag for SentencePiece); the mapping
// onto an internal algorithm happens elsewhere (see core/arch_tags.h) to
// respect the naming rule.
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

// A MoE expert isn't a single blob but 3 distinct matrices (gate/up/down of
// its FFN, same structure as the dense FFN but one per expert). ExpertPart
// selects which of the 3 to read/cache.
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
    // Small numeric arrays from the metadata (bool/int arrays), e.g. a
    // per-layer flag list. Only short arrays are kept by the reader: the
    // huge tokenizer arrays have their own dedicated path and never reach
    // this accessor. Every element is widened to uint32_t, which covers
    // the bool and small-integer element types these keys actually use.
    virtual bool meta_u32_array(const std::string& key, std::vector<uint32_t>& out) {
        (void)key; (void)out;
        return false;
    }
    virtual bool read_vocab(VocabData& out) {
        (void)out;
        return false;
    }
    // Reads the tensor's bytes exactly as they are on disk (still
    // quantized, no dequantization), plus type and shape. Used by the
    // matmul kernels that operate directly on the quantized format (e.g.
    // Q4_0 fused with int8 activations, see core/matmul.cpp) instead of
    // going through read_tensor's dequant-to-float path. Not every reader
    // implements it (default: fails, the caller falls back to read_tensor).
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
    ~ExpertStore();
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;

    // The reader is not owned: it must stay valid for ExpertStore's whole
    // lifetime (in ctx.cpp both live inside the same EngineState).
    void set_reader(ModelReader* reader) { reader_ = reader; }
    void set_hybrid_tier(HybridTier* tier) { hybrid_tier_ = tier; }
    bool fetch(uint32_t layer, uint32_t idx, ExpertPart part, std::vector<float>& out);
    void fetch_union(const std::vector<ExpertRequest>& reqs, std::vector<float>& out_buffer,
                     std::vector<const float*>& out_ptrs);
    void prefetch_layer(uint32_t layer, const std::vector<uint32_t>& idxs);

    // Real overlap (unlike prefetch_layer above, which nobody ever calls
    // today): queues the indices requested for `layer` to a dedicated
    // background worker, which loads them (Gate/Up/Down) while the caller
    // keeps computing. Non-blocking. Only one pending slot: a more recent
    // request replaces the previous one if it hasn't started yet (only the
    // freshest prediction counts). Safe to call together with
    // fetch()/fetch_union() from the same "main" thread while the worker
    // runs: the cache structures share mtx_, the physical reads share
    // reader_mtx_ (never two concurrent reads on the same reader, but the
    // metadata cache and the caller's computation never wait on the
    // worker's I/O).
    void prefetch_async(uint32_t layer, const std::vector<uint32_t>& idxs);

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
    // layer (16 bit) | idx (32 bit) | part (2 bit): wide margins to avoid
    // collisions on real models (thousands of layers/experts, only 3 parts).
    static uint64_t make_key(uint32_t layer, uint32_t idx, ExpertPart part) {
        return (static_cast<uint64_t>(layer) << 48) | (static_cast<uint64_t>(idx) << 8) |
               static_cast<uint64_t>(part);
    }
    // Reads (real I/O) WITHOUT holding mtx_: only reader_mtx_, so it never
    // blocks the other thread's cache operations during a slow read. Each
    // caller re-acquires mtx_ to insert the result into map_/order_ after
    // returning.
    bool load_data(uint64_t key, std::vector<float>& out);
    // Inserts (or updates) `key` into map_/order_ with LRU eviction that
    // skips pinned entries — the same policy as fetch()'s "miss" branch,
    // factored out here because both fetch() and the prefetch_async worker
    // need it. The caller must already hold mtx_.
    void insert_locked(uint64_t key, std::vector<float>&& data);
    void prefetch_worker_loop();

    ModelReader* reader_ = nullptr;
    HybridTier* hybrid_tier_ = nullptr;
    int32_t cache_count_;
    int32_t prefetch_depth_ = 1;
    bool pin_enabled_ = true;
    std::string usage_file_;
    LogFn log_;

    // Protects map_/order_/usage_. Never held during a read (I/O).
    std::mutex mtx_;
    std::list<Entry> order_;
    std::unordered_map<uint64_t, std::list<Entry>::iterator> map_;
    std::unordered_map<uint64_t, uint32_t> usage_;

    // Serializes the physical reads (reader_/hybrid_tier_) between the
    // calling thread and the prefetch worker: no concurrent read on the
    // same reader, so GgufReader/HybridTier's internal thread-safety
    // doesn't need to be verified/guaranteed. The gain is the overlap
    // between the worker's I/O and the caller's computation, not parallel
    // I/O between the two of them.
    std::mutex reader_mtx_;

    // Background worker for prefetch_async: only one pending slot.
    std::thread worker_;
    std::mutex qmtx_;
    std::condition_variable qcv_;
    bool worker_stop_ = false;
    bool has_pending_ = false;
    uint32_t pending_layer_ = 0;
    std::vector<uint32_t> pending_idxs_;
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
// true if `id` is an end-of-generation token (EOS, <end_of_turn>, ...).
bool engine_is_eog_token(const desireeia_ctx* ctx, int32_t id);

// BERT (encoder-only, ArchKind::Bert): per-token embedding for the whole
// sequence in a single call (no state between calls, unlike
// engine_predict/engine_next_token). out_embd filled with
// n_tokens*embd_dim floats; out_embd_dim receives the model's embedding
// width (so the caller can find the per-token boundaries).
// false if the loaded model is not a BERT encoder (DESIREEIA_ERR_NOT_SUPPORTED
// in the ABI) or if ctx/tokens are not valid.
bool engine_embed(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens,
                   std::vector<float>& out_embd, uint32_t& out_embd_dim);

// Sampling (see core/sampler.h and desireeia_sampling in abi.h).
bool engine_set_sampling(desireeia_ctx* ctx, const desireeia_sampling& params);
bool engine_get_sampling(const desireeia_ctx* ctx, desireeia_sampling& out);

// LoRA adapters (Recover-LoRA). Forwards to IForwardEngine::load_lora/
// clear_lora — DESIREEIA_ERR_NOT_SUPPORTED-equivalent (returns false) for
// models with no generative forward engine (BERT encoders) or one that
// doesn't implement it (see forward_iface.h's default).
bool engine_load_lora(desireeia_ctx* ctx, const char* lora_gguf_path, float scale, std::string& err);
bool engine_clear_lora(desireeia_ctx* ctx);

// Prerouter routing prediction. Same not-supported-by-default contract as
// the LoRA calls above (see forward_iface.h).
bool engine_load_prerouter(desireeia_ctx* ctx, const char* path, std::string& err);
bool engine_clear_prerouter(desireeia_ctx* ctx);
bool engine_set_prerouter_heuristic(desireeia_ctx* ctx, bool enabled);

// Chat template (see core/chat_template.h). Applies the format detected
// at model load time to the messages passed in, producing the text to
// tokenize. Returns false only if ctx is invalid: an unrecognized format
// still falls back to ChatML (never an error).
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

// Like matmul_q4_0 but for Q8_0: no nibble to unpack, the weights are
// already int8 (32-wide blocks, 1 fp16 scale per block) — direct dot
// product with the activation, also quantized Q8_0, same scheme. Added
// because it was discovered via the profiler (Phase 9): Q8_0 tensors
// (e.g. Qwen2.5-Coder in Q8_0) were falling back to the float path
// (matmul_f32, ~3x slower), no dedicated kernel existed. cols must be a
// multiple of 32.
DESIREEIA_INTERNAL int matmul_q8_0(const uint8_t* q8_data, size_t rows, size_t cols,
                                 const float* x, float* y);

#ifdef DESIREEIA_CUDA_ENABLED
// CUDA backend (see docs/CUDAPiano.md and src/cuda/backend_cuda.cu).
// matmul_q8_0_cuda: same contract/same math as matmul_q8_0, executed on
// device, uploading the ENTIRE weight matrix on every call — measured 28x
// slower than the CPU (see log in CUDAPiano.md), kept only as a fallback
// for tensors outside the persistent weight cache.
DESIREEIA_INTERNAL int matmul_q8_0_cuda(const uint8_t* q8_data, size_t rows, size_t cols,
                                      const float* x, float* y);

// Device-resident Q8_0 weights: uploaded ONCE (when the tensor enters the
// persistent weight cache, see DenseForward::load_matrix), then reused for
// the whole session by matmul_q8_0_cuda_resident instead of being
// reloaded on every token. Returns false if the device allocation/upload
// fails (out_d_qs/out_d_scale stay nullptr); the pointers must be freed
// with cuda_free_device when the tensor leaves the cache.
DESIREEIA_INTERNAL bool matmul_q8_0_cuda_upload_weights(const uint8_t* q8_data, size_t rows, size_t cols,
                                                       void** out_d_qs, void** out_d_scale);

// Formats the CUDA backend can decode directly in VRAM, kept as small
// integers so the header does not have to depend on MatVecFormat.
#define DESIREEIA_CUDA_FMT_Q4_0 1
#define DESIREEIA_CUDA_FMT_Q4_1 2
#define DESIREEIA_CUDA_FMT_Q5_0 3
#define DESIREEIA_CUDA_FMT_Q5_1 4
#define DESIREEIA_CUDA_FMT_Q4_K 5
#define DESIREEIA_CUDA_FMT_Q5_K 6
#define DESIREEIA_CUDA_FMT_Q6_K 7

// Uploads a weight matrix in whatever representation its format wants:
// K-quants keep their native packed blocks (out_d_scale stays null), Q8_0
// goes through matmul_q8_0_cuda_upload_weights above.
DESIREEIA_INTERNAL bool cuda_upload_weights(int format, const uint8_t* data, size_t rows, size_t cols,
                                           void** out_d_qs, void** out_d_scale);
DESIREEIA_INTERNAL int matmul_kquant_cuda_resident(int format, const void* d_w, size_t rows, size_t cols,
                                                  const float* x, float* y);
DESIREEIA_INTERNAL void cuda_free_device(void* p);
DESIREEIA_INTERNAL void cuda_backend_shutdown();
DESIREEIA_INTERNAL int matmul_q8_0_cuda_resident(const void* d_qs, const void* d_scale, size_t rows, size_t cols,
                                                const float* x, float* y);

// Group of matvecs that share the SAME activation (Q/K/V read attn_norm's
// output, gate/up read ffn_norm's). Running them one at a time costs a
// synchronization each: here the activation is quantized and uploaded
// ONCE, n kernels are launched on the same stream, and there's ONLY ONE
// synchronization at the end. With 252 matvecs per token, the fixed
// per-call latency (~40 us, measured) is one of the dominant costs, so
// reducing the NUMBER of synchronizations matters more than making the
// single kernel faster.
struct CudaQ80Job {
    const void* d_qs;   // weights already on device (MatVec::cuda_qs)
    const void* d_scale;
    size_t rows;
    float* y;           // host destination
};
DESIREEIA_INTERNAL int matmul_q8_0_cuda_resident_group(const CudaQ80Job* jobs, size_t n,
                                                      size_t cols, const float* x);

// Gated FFN block entirely on device: up and gate, activation (act_gelu=0
// -> silu, 1 -> gelu tanh, the same two variants as the CPU path), Q8_0
// requantization of the intermediate, and the down projection. One single
// H2D, one D2H, one synchronization. The intermediate (n_ff elements)
// never comes back to the host: it's born and dies inside the FFN.
DESIREEIA_INTERNAL int matmul_q8_0_cuda_ffn_gated(const void* d_up_qs, const void* d_up_scale,
                                                 const void* d_gate_qs, const void* d_gate_scale,
                                                 const void* d_down_qs, const void* d_down_scale,
                                                 size_t n_ff, size_t n_embd,
                                                 const float* x, float* out, int act_gelu);

// --- KV cache on device (mirror of k_cache_/v_cache_) ---
// Same layout as the host, [layer][pos][kv_head][head_dim]: the device
// copy is written at the single existing write point
// (DenseForward::write_kv_cache), so it can never diverge regardless of
// which path (prefill, decode, CPU fallback) produced k and v.
DESIREEIA_INTERNAL bool cuda_kv_cache_reserve(size_t total_bytes);
DESIREEIA_INTERNAL bool cuda_kv_cache_upload(const void* k_host, const void* v_host,
                                            size_t total_bytes);
DESIREEIA_INTERNAL bool cuda_kv_cache_write(size_t byte_off, const void* k, const void* v,
                                           size_t bytes);

// Causal attention (GQA) + output projection in a single episode: q goes
// up once, attention reads the KV cache already on device and its output
// feeds wo's matvec without returning to the host. Only proj comes down.
// Arguments for the fused attention episode: Q/K/V projections, biases,
// RoPE, KV-cache write, attention and output projection, all on device,
// with a single host synchronisation. Weight pointers are the resident
// device copies (MatVec::cuda_qs / cuda_scale). Bias pointers may be null.
struct CudaQkvAttnArgs {
    const void* wq_qs; const void* wq_scale;
    const void* wk_qs; const void* wk_scale;
    const void* wv_qs; const void* wv_scale;
    const void* wo_qs; const void* wo_scale;
    const float* bq; const float* bk; const float* bv;   // optional
    const float* attn_in;      // host, n_embd (already normalised)
    const float* rope_cache;   // host, n_rot; null disables RoPE
    float* proj;               // host, n_embd (output)
    uint8_t* host_k_row;       // host KV cache slot for this position
    uint8_t* host_v_row;       // kept in sync with the device copy
    size_t n_embd; size_t q_dim; size_t kv_dim;
    uint32_t n_head; uint32_t n_head_kv; uint32_t heads_per_kv;
    uint32_t head_dim; uint32_t n_rot;
    size_t kv_layer_off;       // byte offset of this layer in the KV cache
    uint32_t cc_start; uint32_t pos;
};
DESIREEIA_INTERNAL int cuda_qkv_attention_out(const CudaQkvAttnArgs& args);

// A whole dense layer on device, captured once per layer as a CUDA graph:
// norms, Q/K/V, RoPE, KV write, attention, output projection, both
// residuals and the gated feed-forward. Only x goes up and only the new x
// comes down. One submission and one synchronisation per layer.
struct CudaLayerArgs {
    const void* wq_qs; const void* wq_scale;
    const void* wk_qs; const void* wk_scale;
    const void* wv_qs; const void* wv_scale;
    const void* wo_qs; const void* wo_scale;
    const void* wgate_qs; const void* wgate_scale;
    const void* wup_qs; const void* wup_scale;
    const void* wdown_qs; const void* wdown_scale;
    // Optional per-head attention gate (one matrix ROW per head). Null
    // when the architecture has none, which is every architecture but the
    // hybrid sliding-window one.
    const void* wag_qs; const void* wag_scale;
    const float* bq; const float* bk; const float* bv;   // optional
    const float* attn_norm_w; const float* ffn_norm_w;
    // Optional per-layer norms: QK-norm on Q and K before RoPE, and the
    // sandwich norms applied to the attention and FFN outputs before their
    // residuals. Null when the architecture does not use them.
    const float* q_norm_w; const float* k_norm_w;
    const float* post_attn_norm_w; const float* post_ffn_norm_w;
    const float* x;            // host, n_embd (layer input)
    const float* rope_cache;   // host, n_rot; null disables RoPE
    float* x_out;              // host, n_embd (layer output)
    size_t n_embd; size_t q_dim; size_t kv_dim; size_t n_ff;
    uint32_t n_head; uint32_t n_head_kv; uint32_t heads_per_kv;
    uint32_t head_dim; uint32_t n_rot;
    size_t kv_layer_off;
    uint32_t cc_start; uint32_t pos;
    float rms_eps; int act_gelu;
    // Per-matrix device format id (0 = Q8_0, otherwise DESIREEIA_CUDA_FMT_*).
    // Models mix formats — a Q4_K_M file typically has Q4_K for most
    // tensors and Q6_K for a few — so each matrix carries its own.
    int fmt_q; int fmt_k; int fmt_v; int fmt_o;
    int fmt_gate; int fmt_up; int fmt_down; int fmt_ag;
    // x is uploaded only for the first layer of a token and downloaded
    // only after the last: in between it stays on device, so the layers
    // chain with a single synchronisation per token.
    int upload_x; int download_x;
    // Set when this layer's pos/cc_start and RoPE table must be sent:
    // once per token for a uniform model, per layer when sliding-window
    // layers give different windows or RoPE bases.
    int upload_dyn;
};
DESIREEIA_INTERNAL int cuda_layer_forward(const CudaLayerArgs& args);

DESIREEIA_INTERNAL int cuda_attention_out(const void* d_wo_qs, const void* d_wo_scale,
                                         const float* q, uint32_t n_head, uint32_t heads_per_kv,
                                         uint32_t head_dim, uint32_t kv_dim,
                                         size_t layer_off, uint32_t cc_start, uint32_t pos,
                                         size_t n_embd, size_t q_dim, float* proj,
                                         int kv_quantized, size_t row_bytes);
#endif

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

// Q8_K: 256-wide super-block, a single float scale per block (no fp16, no
// min term), weights already int8 — the simplest of the K-quants.
DESIREEIA_INTERNAL int matmul_q8_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q8_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);

// Q2_K: 256-wide super-block, 2-bit weights, 4-bit scale+min per 16-wide
// sub-block (asymmetric, same scale/min principle as Q4_K but finer
// granularity and without Q4_K's 6-bit get_scale_min_k4 trick).
DESIREEIA_INTERNAL int matmul_q2_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q2_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);

// Q3_K: 256-wide super-block, 3-bit weights (2 bits from qs + 1 bit from
// hmask, -4 when the hmask bit is off), 6-bit scale per 16-wide sub-block
// (symmetric, offset -32, no min term). The scale-unpacking formula (12
// bytes -> 16 6-bit values) was verified against dequantize_row_q3_K in
// quant.cpp, not reinvented.
DESIREEIA_INTERNAL int matmul_q3_k(const uint8_t* data, size_t rows, size_t cols, const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q3_k_batch(const uint8_t* data, size_t rows, size_t cols, const float* x, size_t n_tok, float* y);

// Like matmul_q4_0 but for the Q4_K super-block format (256-weight
// blocks, 6-bit scale+min per 32-wide sub-block, asymmetric
// quantization). cols must be a multiple of 256.
DESIREEIA_INTERNAL int matmul_q4_k(const uint8_t* q4k_data, size_t rows, size_t cols,
                                 const float* x, float* y);

// Like matmul_q4_k but for the Q6_K super-block format (256 weights per
// block, symmetric 6-bit quantization, int8 scale per 16-wide sub-block,
// no "min" term). cols must be a multiple of 256.
DESIREEIA_INTERNAL int matmul_q6_k(const uint8_t* q6k_data, size_t rows, size_t cols,
                                 const float* x, float* y);

// Like matmul_q4_k but for the Q5_K super-block format (256 weights per
// block, 6-bit scale+min per 32-wide sub-block like Q4_K, but 5-bit
// weights: 4 bits from qs + 1 high bit from qh). cols must be a multiple
// of 256.
DESIREEIA_INTERNAL int matmul_q5_k(const uint8_t* q5k_data, size_t rows, size_t cols,
                                 const float* x, float* y);
DESIREEIA_INTERNAL int matmul_q5_k_batch(const uint8_t* q5k_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);

// Q8_K-style activation quantization (one float scale per 256-wide
// super-block, see the note in matmul.cpp) shared by Q4_K/Q5_K/Q6_K:
// exposed so dense_forward.cpp can quantize a layer's input activation
// ONCE and reuse it for all of that layer's K-quant matrices that read
// from the same activation (wq/wk/wv/wo, ffn_gate/ffn_up), instead of
// doing it redundantly inside every matmul_qX_k call.
DESIREEIA_INTERNAL void quantize_q8_k_super(const float* x, size_t cols, std::vector<int8_t>& q, std::vector<float>& dscale);

// Q8_K activation quantization (one scale every 256) exposed with the
// per-32 arrays expected by the kernels: the scale is replicated across
// the super-block's 8 slots. This is the prerequisite for the integer
// accumulation in matmul_q4_k_core — see the extended comment on its
// definition in matmul.cpp.
DESIREEIA_INTERNAL void quantize_act_q8k_rep(const float* x, size_t cols, std::vector<int8_t>& q,
                                          std::vector<float>& xscale32, std::vector<int32_t>& xsum32);

// "Pre-quantized" variants (Phase "eliminate redundant re-quantization",
// 2026-09-07): like matmul_q4_k/matmul_q6_k but take the activation
// already quantized (quantize_q8_k_super) instead of quantizing it
// internally. xsum (Q4_K only, which has the "min" term) is the sum of
// the quantized activations per 32-wide sub-block, computable with the
// same scheme already used in matmul_q4_k.
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

// "Batch" variants (Phase 8, prefill): like the matmul_qX_k above but
// with n_tok activation columns instead of just one. x is n_tok
// contiguous blocks of `cols` floats (one token after another), y is
// n_tok contiguous blocks of `rows` floats. The gain over calling the
// single-column version n_tok times: the raw weight bytes (and their
// nibble->int8 decode) are read/decoded ONCE per row and reused for every
// column, instead of being reloaded from memory/decoded n_tok times.
// Mathematically identical to the same call repeated n_tok times (no
// reordering across different columns).
DESIREEIA_INTERNAL int matmul_q4_0_batch(const uint8_t* q4_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);
DESIREEIA_INTERNAL int matmul_q4_k_batch(const uint8_t* q4k_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);
DESIREEIA_INTERNAL int matmul_q6_k_batch(const uint8_t* q6k_data, size_t rows, size_t cols,
                                       const float* x, size_t n_tok, float* y);

}

#endif
