// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "engine.h"
#include "desireeia/abi.h"
#include "core/arch_tags.h"
#include "core/chat_template.h"
#include "core/jinja.h"
#include "core/sampler.h"
#include "core/tool_grammar.h"
#include "core/thread_pool.h"
#include "models/dense_forward.h"
#include "models/ssm_forward.h"
#include "models/bert_forward.h"
#include "vision/vision_gguf.h"
#include "ssd_tier/hybrid_tier.h"
#include "tokenizer/tokenizer.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <deque>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <set>
#include <system_error>

namespace desireeia {

struct EngineContext {
    EngineState st;
    // Forward engine: DenseForward for attention-based architectures,
    // SsmForward for Mamba2 (see ArchKind::Mamba2 and the extended note on
    // core/forward_iface.h). The two classes share no state: only the
    // pointer is unified behind the minimal interface.
    IForwardEngine* gf = nullptr;
    // BERT (encoder-only, ArchKind::Bert): does NOT implement IForwardEngine —
    // it doesn't generate a "next token", has no logits over a vocabulary, and
    // has no KV cache to reset. Separate field, mutually exclusive with gf
    // (a model is either generative or an encoder, never both here).
    BertForward* bert = nullptr;
    Tokenizer tok;
    bool has_tok = false;
    int32_t last_token = -1;
    bool has_session = false;
    std::mutex mtx;
    // desireeia_cancel: set from any thread, read by the forward engine at
    // layer boundaries; cleared when a predict starts. hooks points at it.
    std::atomic<bool> cancel{false};
    StepHooks hooks;

    // Phase 0 (real inference engine): special token ids read from the
    // model's vocabulary, persisted here (VocabData is local to
    // engine_create) so desireeia_special_token_id can expose them to
    // whoever generates (to stop at EOS) or builds a chat template.
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t unk_id = -1;
    int32_t pad_id = -1;

    // The full set of END-OF-GENERATION (EOG) tokens, not just `eos_id`.
    // Chat models end their turn with dedicated tokens other than <eos> —
    // gemma uses <end_of_turn>, for example. Stopping only on eos_id left
    // the model repeating <end_of_turn> forever after it had already
    // answered correctly.
    std::set<int32_t> eog_ids;

    // Sampling. Greedy by default (temperature 0), i.e. the behavior that
    // existed before and that the end-to-end tests expect; becomes real
    // sampling as soon as the caller raises the temperature.
    Sampler sampler;

    // Token-level tool-call constraint (core/tool_grammar.h) and the
    // decoded piece of every vocabulary entry it filters on, built once on
    // first use (the mask needs every piece at every constrained step).
    ToolCallConstraint tool;
    std::vector<std::string> tool_pieces;
    std::vector<uint8_t> tool_is_eog;
    std::vector<uint8_t> tool_plain;   // piece is safe inside any JSON string

    // Chat prompt format, detected at load time (see engine_create): from
    // the GGUF's chat_template when present, otherwise from the
    // per-architecture default.
    ChatTemplateKind chat_template = ChatTemplateKind::Unknown;
    // The model's own Jinja template, compiled at load (core/jinja.h):
    // rendered first, with chat_template above as the fallback when it is
    // absent or raises.
    std::shared_ptr<jinja::Template> jinja_template;

    // Prompt Lookup Decoding: full history of tokens already in the KV
    // cache (prompt plus confirmed generations), used to look up an
    // n-gram continuation to verify in one batch; a queue of tokens
    // already checked/accepted but not yet returned to the caller
    // (desireeia_next_token returns them one at a time).
    std::vector<int32_t> history;
    std::vector<float> decode_logits;   // engine_next_token's logits, reused across tokens
    std::vector<int32_t> cand_ids;      // ... or only its largest ones (step_candidates)
    std::vector<float> cand_vals;
    std::deque<int32_t> pending;

    // Conversation session (KV prefix reuse). `kv_tokens` is EXACTLY the
    // token sequence whose K/V are in the cache right now (prompt plus every
    // generated token already fed back through step()); `kv_valid` says
    // whether that correspondence still holds. At the next engine_predict
    // the new prompt is compared with kv_tokens: the common prefix stays in
    // the cache and only the new tail is prefilled. A chat that re-sends its
    // whole history every turn (the normal chat-API pattern) then costs one
    // prefill of the NEW messages instead of the whole conversation again.
    // Invalidated by anything that makes the cached K/V no longer a function
    // of the token ids alone: image embeddings, LoRA changes, a failed step.
    std::vector<int32_t> kv_tokens;
    bool kv_valid = false;
    // How many leading kv_tokens got their K/V from a PREFILL (batched step).
    // Tokens appended by decode steps are computed by the single-token path,
    // whose floating-point results differ slightly from the batched one:
    // reusing them is correct but not bit-identical to a full prefill (greedy
    // can pick a different word at a near tie — measured on gemma3-4b). The
    // default "exact" mode therefore reuses only prefilled positions: output
    // identical to a full prefill, and in a chat that is still the whole
    // history except the last answer (which the app re-tokenizes anyway).
    size_t kv_prefilled = 0;
    // 0 = off (always full prefill), 1 = exact (default), 2 = also reuse
    // decoded tokens (faster, not bit-identical).
    int session_mode = 1;
    size_t last_reused = 0;

    // Pipelined decode (CUDA): while the caller holds the token returned
    // last, the step that consumes it is already running and has drawn the
    // next token on the device. `pipe` says such a step is in flight;
    // its input is last_token, its draw landed in readback slot pipe_slot,
    // and pipe_rng is the generator before that draw. drain_pipeline()
    // undoes it (cache position, generator) before anything else runs.
    bool pipe = false;
    uint32_t pipe_slot = 0;
    std::mt19937 pipe_rng;
};

#ifdef DESIREEIA_CUDA_ENABLED
// Contexts alive in the process: the device state of the pipelined decode
// (token, history, readback slots) is shared, so it is used only while a
// single context exists.
static std::atomic<int> g_live_contexts{0};
#endif

// Stops a pipelined decode: waits for the step in flight and rewinds what
// it consumed (the cache position of last_token, the generator draw), so
// the context is exactly where a non-pipelined decode would have left it.
static void drain_pipeline(EngineContext* c) {
#ifdef DESIREEIA_CUDA_ENABLED
    if (!c || !c->pipe) return;
    int32_t discarded = -1;
    cuda_sample_wait(c->pipe_slot, discarded);
    DenseForward* df = dynamic_cast<DenseForward*>(c->gf);
    if (df && df->cache_len() > 0) df->truncate_cache(df->cache_len() - 1);
    c->sampler.set_rng_state(c->pipe_rng);
    c->pipe = false;
#else
    (void) c;
#endif
}

// Length of the prefix of `tokens` that can be kept from the cache. At least
// one token is always left to prefill: the logits for the next token come
// from the last processed position, and the cache does not keep them.
static size_t reusable_prefix(const EngineContext* c, const int32_t* tokens, size_t n_tokens) {
    if (c->session_mode == 0 || !c->kv_valid || n_tokens < 2) return 0;
    const DenseForward* df = dynamic_cast<const DenseForward*>(c->gf);
    if (!df) return 0;   // recurrent state (SSM) cannot be rewound to a position
    size_t cached = c->session_mode == 2 ? c->kv_tokens.size() : c->kv_prefilled;
    size_t limit = std::min({cached, c->kv_tokens.size(), n_tokens - 1, df->cache_len()});
    size_t n = 0;
    while (n < limit && c->kv_tokens[n] == tokens[n]) ++n;
    return n;
}

static void invalidate_session(EngineContext* c) {
    c->kv_tokens.clear();
    c->kv_prefilled = 0;
    c->kv_valid = false;
}

namespace {
// Looks for the LATEST occurrence (most recent = most likely for repeated
// local patterns, e.g. code) of the sequence [last kNgramLen-1 tokens of
// history, last_token] elsewhere in history, and returns what followed it
// as a candidate continuation (up to max_draft tokens). No draft model
// involved: this is the "Prompt Lookup Decoding" technique (n-gram),
// useful especially when the output repeats material already seen in the
// context (typical of code/structured documents). If nothing is found,
// returns false and the caller falls back to normal single-token decode
// (no regression: the extra cost is one linear scan over history).
bool find_ngram_continuation(const std::vector<int32_t>& history, int32_t last_token,
                              size_t ngram_len, size_t max_draft,
                              std::vector<int32_t>& out_continuation) {
    std::vector<int32_t> ext = history;
    ext.push_back(last_token);
    if (ext.size() < ngram_len + 1) return false;
    const size_t trivial_start = ext.size() - ngram_len;
    for (size_t start = trivial_start; start-- > 0; ) {
        bool match = true;
        for (size_t j = 0; j < ngram_len; ++j) {
            if (ext[start + j] != ext[trivial_start + j]) { match = false; break; }
        }
        if (match) {
            const size_t cont_start = start + ngram_len;
            for (size_t k = 0; k < max_draft && cont_start + k < ext.size() - 1; ++k) {
                out_continuation.push_back(ext[cont_start + k]);
            }
            if (!out_continuation.empty()) return true;
        }
    }
    return false;
}
}

namespace {

ModelReader* make_reader(desireeia_format fmt) {
    switch (fmt) {
        case DESIREEIA_FORMAT_GGUF:
            return make_gguf_reader();
        case DESIREEIA_FORMAT_SAFETENSORS:
            return make_st_reader();
        default:
            return nullptr;
    }
}

desireeia_plan resolve_plan(const char* model_path, desireeia_plan plan) {
    bool unconfigured = plan.n_threads <= 0 || plan.ram_budget_mb == 0;
    if (unconfigured) {
        desireeia_hw_info hw;
        std::memset(&hw, 0, sizeof(hw));
        if (probe_hardware(hw) == DESIREEIA_OK) {
            desireeia_plan auto_plan = build_plan(hw, model_path ? model_path : "");
            if (plan.n_threads <= 0) plan.n_threads = auto_plan.n_threads;
            if (plan.ram_budget_mb == 0) plan.ram_budget_mb = auto_plan.ram_budget_mb;
            if (plan.format == DESIREEIA_FORMAT_UNKNOWN) plan.format = auto_plan.format;
            if (plan.expert_cache_count == 0) plan.expert_cache_count = auto_plan.expert_cache_count;
            if (plan.backend == 0) plan.backend = auto_plan.backend;
            if (plan.dense_quant == DESIREEIA_QUANT_F32 && auto_plan.dense_quant != DESIREEIA_QUANT_F32) {
                plan.dense_quant = auto_plan.dense_quant;
            }
            if (plan.expert_quant == DESIREEIA_QUANT_F32 && auto_plan.expert_quant != DESIREEIA_QUANT_F32) {
                plan.expert_quant = auto_plan.expert_quant;
            }
            plan.expert_pin_enabled = auto_plan.expert_pin_enabled;
            plan.expert_prefetch_depth = auto_plan.expert_prefetch_depth;
            plan.batch_union_enabled = auto_plan.batch_union_enabled;
            plan.dual_ssd_enabled = auto_plan.dual_ssd_enabled;
            plan.expert_prefetch_enabled = auto_plan.expert_prefetch_enabled;
            plan.kv_compression_enabled = auto_plan.kv_compression_enabled;
            // Only taken from the auto plan when the caller left the plan
            // unconfigured; a caller that fills the plan in keeps full
            // control of the tier, including turning it off outright.
            plan.ssd_tier_mode = auto_plan.ssd_tier_mode;
            plan.ssd_tier_cache_mb = auto_plan.ssd_tier_cache_mb;
        }
    }
    // Applied last, and regardless of whether the caller configured the plan:
    // it is the escape hatch for measuring the tier, and it would be useless
    // if a filled-in plan could shadow it.
    env_ssd_tier_override(plan.ssd_tier_mode);
    return plan;
}

std::string sidecar_path(const std::string& model_path, const char* suffix) {
    return model_path + suffix;
}

// Size of the model file in bytes, or 0 if it can't be determined.
uint64_t model_file_bytes(const char* path) {
    if (!path || path[0] == '\0') return 0;
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

constexpr uint64_t kMiB = 1024ull * 1024ull;

// What a load is expected to cost in RAM, split into the parts that behave
// differently. Everything is in bytes.
struct TierBudget {
    uint64_t budget = 0;          // What the plan says we may use in total.
    uint64_t system_reserve = 0;  // Left to the OS so the machine stays usable.
    uint64_t runtime_reserve = 0; // KV cache, activations, per-layer scratch.
    uint64_t weight_ceiling = 0;  // What is left over for weights.
    uint64_t model_bytes = 0;     // What the weights actually cost.
    bool     feasible = true;     // False when not even the reserves fit.
};

// Reads an architecture-scoped metadata key ("<arch>.embedding_length" and
// friends), falling back to `fallback` when the file does not carry it.
uint32_t meta_dim(ModelReader* reader, const std::string& arch,
                  const char* suffix, uint32_t fallback) {
    uint32_t v = 0;
    if (reader && reader->meta_u32(arch + suffix, v) && v > 0) return v;
    return fallback;
}

// Estimates what this model needs beyond its weights, and how much of the
// budget that leaves for the weights themselves.
//
// The three parts are estimated separately because they scale differently and
// a single flat percentage gets all three wrong at once. A small model on a
// large machine wants almost no reserve and should stay entirely in RAM; a
// long-context model can spend more on its KV cache than on its weights. The
// old "70% of the budget is for weights" rule was blind to both.
TierBudget plan_tier_budget(const desireeia_plan& plan, const ModelMeta& meta,
                            ModelReader* reader, const char* model_path) {
    TierBudget b;
    b.budget = plan.ram_budget_mb * kMiB;
    b.model_bytes = model_file_bytes(model_path);

    // System reserve: an eighth of the budget, but never less than 512 MiB
    // (below that the machine starts paging under us) and never more than
    // 4 GiB (past that we are just refusing to use RAM we were given).
    b.system_reserve = b.budget / 8;
    if (b.system_reserve < 512 * kMiB) b.system_reserve = 512 * kMiB;
    if (b.system_reserve > 4096 * kMiB) b.system_reserve = 4096 * kMiB;

    const uint32_t layers = meta.n_layers > 0 ? meta.n_layers : 1;
    const uint32_t n_embd = meta_dim(reader, meta.arch, ".embedding_length", 4096);
    const uint32_t n_head = meta_dim(reader, meta.arch, ".attention.head_count", 32);
    const uint32_t n_head_kv =
        meta_dim(reader, meta.arch, ".attention.head_count_kv", n_head);
    const uint32_t head_dim = n_head > 0 ? n_embd / n_head : 128;

    // Context we actually plan to hold. Files routinely advertise a maximum
    // far past what a session uses, and sizing the reserve for 128k context
    // would push every model onto the SSD tier for a cache nobody fills.
    uint32_t n_ctx = meta.n_ctx > 0 ? meta.n_ctx : 4096;
    if (n_ctx > 8192) n_ctx = 8192;

    // K and V, one entry per layer per position: Q8_0 (34 bytes per 32
    // values) when the plan compresses the cache - the default - float32
    // otherwise. The float32 figure was used for both, reserving ~3.8x the
    // real cache and pushing weights off RAM for memory never used.
    const uint64_t kv_values = 2ull * layers * n_ctx * static_cast<uint64_t>(n_head_kv) * head_dim;
    const uint64_t kv_bytes = plan.kv_compression_enabled ? (kv_values * 34 + 31) / 32
                                                          : kv_values * sizeof(float);

    // Activations: a handful of live vectors of width n_embd, plus the logits
    // row over the vocabulary. Small next to the rest, but not nothing for a
    // 256k vocabulary.
    const uint64_t act_bytes =
        (16ull * n_embd + 2ull * (meta.n_vocab > 0 ? meta.n_vocab : 32000)) * sizeof(float);

    // Scratch for dequantizing weights on the way in: two layers' worth, so a
    // layer can be prepared while the previous one is still in use.
    const uint64_t layer_bytes = b.model_bytes / layers;
    const uint64_t scratch_bytes = 2ull * layer_bytes;

    b.runtime_reserve = kv_bytes + act_bytes + scratch_bytes;

    const uint64_t reserved = b.system_reserve + b.runtime_reserve;
    if (reserved >= b.budget) {
        b.feasible = false;
        b.weight_ceiling = 0;
    } else {
        b.weight_ceiling = b.budget - reserved;
    }
    return b;
}

// How many BYTES of raw quantized weights the tier may hold in RAM.
//
// This is the tier's real working set. An explicit ssd_tier_cache_mb wins;
// otherwise it gets what the budget leaves for weights after the system and
// runtime reserves, which is by construction the amount that can be held
// without pushing the machine into paging.
uint64_t tier_cache_bytes(const desireeia_plan& plan, const ModelMeta& meta,
                          ModelReader* reader, const char* model_path, LogFn log) {
    uint64_t bytes = plan.ssd_tier_cache_mb * kMiB;
    if (bytes == 0) {
        const TierBudget b = plan_tier_budget(plan, meta, reader, model_path);
        bytes = b.weight_ceiling;
    }
    // Small enough and the tier re-reads the same layer every token, which is
    // worse than not caching at all because it also pays the bookkeeping.
    if (bytes < 64 * kMiB) bytes = 64 * kMiB;

    if (log) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "ssd tier: raw weight cache %llu MiB",
                      (unsigned long long) (bytes / kMiB));
        log(5, msg);
    }
    return bytes;
}

// How many tensor slots the tier may hold in RAM.
//
// The tier's cache is counted in entries, not bytes, so the byte budget has
// to be turned into a slot count. Tensors within a model are close enough in
// size that an average works: a transformer layer carries roughly nine of
// them (attention projections, norms, the FFN matrices), which is the divisor
// below. Getting this wrong in either direction is expensive — too few slots
// and every layer is re-read from SSD each token, too many and the cache
// evicts the rest of the process.
int32_t tier_cache_slots(const desireeia_plan& plan, const ModelMeta& meta,
                         ModelReader* reader, const char* model_path, LogFn log) {
    const TierBudget b = plan_tier_budget(plan, meta, reader, model_path);

    // An explicit cache size wins; 0 means "work it out from the budget".
    uint64_t cache_bytes = plan.ssd_tier_cache_mb * kMiB;
    if (cache_bytes == 0) cache_bytes = b.weight_ceiling;
    if (cache_bytes == 0) cache_bytes = 256 * kMiB;

    const uint32_t layers = meta.n_layers > 0 ? meta.n_layers : 1;
    const uint64_t avg_tensor = b.model_bytes / (static_cast<uint64_t>(layers) * 9ull + 4ull);

    int32_t slots = plan.expert_cache_count > 0 ? plan.expert_cache_count : 256;
    if (avg_tensor > 0) {
        uint64_t n = cache_bytes / avg_tensor;
        if (n < 16) n = 16;              // Below this the tier cannot hold a layer.
        if (n > 65536) n = 65536;
        slots = static_cast<int32_t>(n);
    }

    if (log) {
        char msg[192];
        std::snprintf(msg, sizeof(msg),
                      "ssd tier: cache %llu MiB, avg tensor %llu KiB -> %d slots",
                      (unsigned long long) (cache_bytes / kMiB),
                      (unsigned long long) (avg_tensor / 1024ull), slots);
        log(5, msg);
    }
    return slots;
}

// Decides whether the SSD tier should actually be constructed for this load.
//
// OFF and ALWAYS are literal. AUTO is the interesting one: it engages only
// when the weights do not fit what is left of the budget after the reserves,
// which is exactly the case the tier exists for. A model that fits stays
// entirely on the RAM path and pays nothing for the tier being available.
bool resolve_ssd_tier(const desireeia_plan& plan, const ModelMeta& meta,
                      ModelReader* reader, const char* model_path, LogFn log) {
    switch (plan.ssd_tier_mode) {
        case DESIREEIA_SSD_TIER_OFF:
            return false;
        case DESIREEIA_SSD_TIER_ALWAYS:
            return true;
        default:
            break;
    }

    const TierBudget b = plan_tier_budget(plan, meta, reader, model_path);
    if (b.model_bytes == 0 || b.budget == 0) {
        // Unknown size or no budget: assume it fits rather than forcing
        // everyone onto the slower path over a stat() that failed.
        return false;
    }

    // Reserves alone overrun the budget. Streaming the weights is the only
    // way this load has a chance, so engage the tier and say plainly why —
    // silently continuing on the RAM path here means thrashing later.
    if (!b.feasible) {
        if (log) {
            char msg[224];
            std::snprintf(msg, sizeof(msg),
                          "ssd tier (auto): reserves %llu MiB exceed budget %llu MiB "
                          "(system %llu + runtime %llu) -> streaming weights",
                          (unsigned long long) ((b.system_reserve + b.runtime_reserve) / kMiB),
                          (unsigned long long) (b.budget / kMiB),
                          (unsigned long long) (b.system_reserve / kMiB),
                          (unsigned long long) (b.runtime_reserve / kMiB));
            log(4, msg);
        }
        return true;
    }

    const bool needs_tier = b.model_bytes > b.weight_ceiling;
    if (log) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "ssd tier (auto): weights %llu MiB, budget %llu MiB "
                      "(system %llu, runtime %llu, free for weights %llu) -> %s",
                      (unsigned long long) (b.model_bytes / kMiB),
                      (unsigned long long) (b.budget / kMiB),
                      (unsigned long long) (b.system_reserve / kMiB),
                      (unsigned long long) (b.runtime_reserve / kMiB),
                      (unsigned long long) (b.weight_ceiling / kMiB),
                      needs_tier ? "on" : "off");
        log(5, msg);
    }
    return needs_tier;
}

}

desireeia_ctx* engine_create(const char* model_path, const desireeia_plan& plan, LogFn log) {
    EngineContext* ctx = new EngineContext;
    ctx->st.plan = resolve_plan(model_path, plan);
    // Configure the global thread pool with the plan's thread count
    // BEFORE any step()/matvec call (that's when ThreadPool::global() gets
    // constructed the first time, lazily): this makes it possible to
    // compare throughput at a fixed, known thread count. If already
    // configured in this process (from an earlier create() call), this
    // has no effect — see ThreadPool::set_thread_override.
    ThreadPool::set_thread_override(ctx->st.plan.n_threads);
    ctx->st.log = log;
    ctx->st.meta.path = model_path;
    ctx->st.last_error.clear();
    ctx->st.usage_file = sidecar_path(model_path ? model_path : "", ".desireeia_usage");

    auto* reader = make_reader(ctx->st.plan.format);
    if (!reader) {
        delete ctx;
        return nullptr;
    }

    ModelMeta meta;
    if (!reader->open(model_path, meta)) {
        delete reader;
        delete ctx;
        return nullptr;
    }
    ctx->st.reader = reader;
    ctx->st.meta = meta;

    uint32_t n_kv = meta.n_layers > 0 ? meta.n_layers : 1;
    ctx->st.kv = new KvCache(meta.n_layers, n_kv, ctx->st.plan.kv_compression_enabled != 0);
    ctx->st.experts = new ExpertStore(ctx->st.plan.expert_cache_count, log);
    ctx->st.experts->set_reader(reader);
    ctx->st.experts->set_prefetch_depth(ctx->st.plan.expert_prefetch_depth);
    ctx->st.experts->set_pin_enabled(ctx->st.plan.expert_pin_enabled != 0);
    ctx->st.experts->set_usage_file(ctx->st.usage_file);
    ctx->st.experts->load_usage();

    // SSD tier: opt-in, and in AUTO mode only when the model actually needs
    // it. When it stays off, nothing is constructed and nothing sits on the
    // tensor read path — the RAM/CPU path is byte-for-byte what it was
    // before this feature existed. That matters: the wrapper below adds a
    // lookup to every read_tensor/read_tensor_raw call, which is cheap when
    // the layer weight cache is on (one hit per layer at load) but is paid
    // per token when the model is too big for that cache.
    if (resolve_ssd_tier(ctx->st.plan, meta, reader, model_path, log)) {
        HybridTierConfig hcfg;
        hcfg.gguf_path = model_path ? model_path : "";
        hcfg.usage_file = ctx->st.usage_file;
        hcfg.ram_capacity = tier_cache_slots(ctx->st.plan, meta, reader, model_path, log);
        hcfg.ram_bytes_max = tier_cache_bytes(ctx->st.plan, meta, reader, model_path, log);
        hcfg.prefetch_enabled = ctx->st.plan.expert_prefetch_enabled != 0;
        hcfg.prefetch_depth = ctx->st.plan.expert_prefetch_depth;

        const char* mirror_env = std::getenv("DESIREEIA_MODEL_MIRROR");
        if (mirror_env && mirror_env[0] != '\0') {
            hcfg.mirror_path = mirror_env;
        }

        ctx->st.hybrid = new HybridTier(hcfg, log);
        // The tier reads misses through the real reader rather than parsing
        // the model file itself, so its view of the file is identical to the
        // engine's by construction.
        ctx->st.hybrid->set_source(reader);
        ctx->st.experts->set_hybrid_tier(ctx->st.hybrid);

        // Wrap the reader so every tensor read goes through the tier
        // (read_tensor, read_tensor_raw, read_expert). The wrapper calls the
        // tier first; the tier falls back to `reader` on a miss. No recursion:
        // the tier holds the inner reader, not the wrapper.
        if (ctx->st.hybrid->gguf_available()) {
            auto* hybrid_reader = new HybridModelReader(reader, ctx->st.hybrid);
            ctx->st.reader = hybrid_reader;
            // Point ExpertStore at the wrapped reader too, so MoE expert
            // fetches are served by the same tier.
            ctx->st.experts->set_reader(hybrid_reader);
            if (log) log(5, "ssd tier: enabled, tensor reads served through SSD tier");
        } else if (log) {
            log(4, "ssd tier: requested but model file not readable, staying on RAM path");
        }
    } else if (log) {
        log(5, "ssd tier: off, weights served from the RAM/CPU path");
    }

    VocabData vocab;
    if (reader->read_vocab(vocab)) {
        ctx->has_tok = ctx->tok.load(vocab);
        ctx->bos_id = vocab.bos_id;
        ctx->eos_id = vocab.eos_id;
        ctx->unk_id = vocab.unk_id;
        ctx->pad_id = vocab.pad_id;

        // Build the EOG set by token name: the names below are the ones
        // that mark end-of-turn/end-of-text across the most common model
        // families. eos_id is always included.
        static const char* kEogNames[] = {
            "<end_of_turn>", "<eos>", "</s>", "<|endoftext|>", "<|end_of_text|>",
            "<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<EOT>", "[EOT]", "[EOS]",
            "<end_of_utterance>",
        };
        if (vocab.eos_id >= 0) ctx->eog_ids.insert(vocab.eos_id);
        for (size_t i = 0; i < vocab.tokens.size(); ++i) {
            for (const char* name : kEogNames) {
                if (vocab.tokens[i] == name) {
                    ctx->eog_ids.insert((int32_t) i);
                    break;
                }
            }
        }
        if (log) log(ctx->has_tok ? 5 : 4,
                      ctx->has_tok ? "tokenizer ready" : "tokenizer kind not implemented");
    }

    // Chat template detection: an explicit chat_template in the model's
    // metadata, when present, wins over the per-architecture default,
    // because a single architecture can have several historical format
    // variants (e.g. an older vs. newer prompt convention, or several
    // Mistral variants) that the architecture tag alone can't distinguish.
    {
        std::string tmpl_str;
        if (reader->meta_str("tokenizer.chat_template", tmpl_str) && !tmpl_str.empty()) {
            ctx->chat_template = detect_chat_template(tmpl_str);
            try {
                ctx->jinja_template = std::make_shared<jinja::Template>(tmpl_str);
            } catch (const std::exception& e) {
                if (log) log(4, (std::string("chat template not compiled, using the built-in format: ") + e.what()).c_str());
            }
        }
        if (ctx->chat_template == ChatTemplateKind::Unknown) {
            ctx->chat_template = chat_template_for_arch(detect_arch(meta.arch));
        }
    }

    const ArchKind arch_kind = detect_arch(meta.arch);
    if (meta.format == DESIREEIA_FORMAT_GGUF && arch_kind == ArchKind::Bert) {
        BertForward* bf = new BertForward;
        if (bf->open(*reader, meta, ctx->st.plan.ram_budget_mb)) {
            ctx->bert = bf;
            if (log) log(5, "bert encoder ready");
        } else {
            delete bf;
            if (log) log(3, "bert encoder init failed");
        }
    } else if (meta.format == DESIREEIA_FORMAT_GGUF && arch_kind == ArchKind::Mamba2) {
        SsmForward* sf = new SsmForward;
        if (sf->open(*reader, meta, ctx->st.plan.ram_budget_mb)) {
            ctx->gf = sf;
            ctx->hooks.cancel = &ctx->cancel;
            ctx->gf->set_hooks(&ctx->hooks);
            if (log) {
                log(5, sf->weight_cache_enabled()
                    ? "ssm forward ready (layer weight cache enabled)"
                    : "ssm forward ready (on-demand layer weights)");
            }
        } else {
            delete sf;
            if (log) log(3, "ssm forward init failed");
        }
    } else if (meta.format == DESIREEIA_FORMAT_GGUF && arch_kind != ArchKind::Unknown) {
        DenseForward* df = new DenseForward;
        if (df->open(*reader, meta, arch_kind, ctx->st.plan.ram_budget_mb, ctx->st.experts,
                     ctx->st.plan.kv_compression_enabled != 0, ctx->st.plan.backend)) {
            ctx->gf = df;
            ctx->hooks.cancel = &ctx->cancel;
            ctx->gf->set_hooks(&ctx->hooks);
            if (log) {
                log(5, df->weight_cache_enabled()
                    ? "dense float forward ready (layer weight cache enabled)"
                    : "dense float forward ready (on-demand layer weights)");
            }
        } else {
            delete df;
            if (log) log(3, "dense forward init failed");
        }
    } else if (log) {
        log(3, "model architecture not supported by forward path");
    }

    if (log) {
        log(5, "model configured with tiered expert cache");
    }

    // Vision encoder automatic detection: if the GGUF carries clip.vision.*
    // metadata and vision tensors, load them into ctx->st.vision so the
    // multimodal path (engine_predict_vision / desireeia_vision_encode) works
    // without the caller having to do anything extra.
    {
        const auto ctxp = reinterpret_cast<desireeia_ctx*>(ctx);
        if (engine_load_vision(ctxp) && log) {
            if (log) log(5, "vision encoder loaded from model file");
        }
    }

#ifdef DESIREEIA_CUDA_ENABLED
    g_live_contexts.fetch_add(1);
#endif
    return reinterpret_cast<desireeia_ctx*>(ctx);
}

void engine_destroy(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return;
    {
        std::lock_guard<std::mutex> lk(c->mtx);
        drain_pipeline(c);
    }
#ifdef DESIREEIA_CUDA_ENABLED
    g_live_contexts.fetch_sub(1);
#endif
    if (c->st.experts) c->st.experts->save_usage();
    if (c->st.hybrid) c->st.hybrid->save_usage();
    delete c->gf;
    delete c->bert;
    delete c->st.kv;
    delete c->st.experts;
    delete c->st.hybrid;
    // If HybridModelReader wraps the original reader, it's deleted via
    // HybridModelReader. Otherwise delete the bare reader.
    delete c->st.reader;
    delete c;
}

bool engine_embed(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens,
                   std::vector<float>& out_embd, uint32_t& out_embd_dim) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->bert || !c->st.reader || n_tokens == 0) return false;
    out_embd_dim = c->bert->config().n_embd;
    return c->bert->encode(*c->st.reader, tokens, n_tokens, out_embd);
}


// ---- tool-call constraint -------------------------------------------------

static void tool_build_pieces(EngineContext* c, size_t vocab) {
    c->tool_pieces.assign(vocab, std::string());
    c->tool_is_eog.assign(vocab, 0);
    c->tool_plain.assign(vocab, 0);
    for (size_t id = 0; id < vocab; ++id) {
        if (c->has_tok) c->tok.piece(static_cast<int32_t>(id), c->tool_pieces[id]);
        c->tool_is_eog[id] = c->eog_ids.count(static_cast<int32_t>(id)) ? 1 : 0;
        const std::string& p = c->tool_pieces[id];
        bool plain = !p.empty() && !c->tool_is_eog[id];
        for (unsigned char ch : p) {
            if (ch == '"' || ch == 0x5C || ch < 0x20) { plain = false; break; }
        }
        c->tool_plain[id] = plain;
    }
}

// Masks every token the constraint rejects. If that would mask the whole
// vocabulary the grammar is at a dead end: it gives up for this generation
// rather than forcing an arbitrary token.
static void tool_mask(EngineContext* c, std::vector<float>& logits) {
    if (c->tool_pieces.size() != logits.size()) tool_build_pieces(c, logits.size());
    // A piece whose first byte is rejected is rejected: 256 checks up front
    // skip the full automaton walk for almost the whole vocabulary.
    bool first_ok[256];
    for (int b = 0; b < 256; ++b) first_ok[b] = c->tool.allows(std::string(1, static_cast<char>(b)), false);
    // Inside a free-form string (most of a call: argument values) a plain
    // piece is accepted without walking the automaton at all.
    const bool plain_string = c->tool.in_plain_string();
    std::vector<uint8_t> keep(logits.size(), 0);
    size_t kept = 0;
    for (size_t id = 0; id < logits.size(); ++id) {
        const std::string& piece = c->tool_pieces[id];
        bool ok;
        if (plain_string && c->tool_plain[id]) {
            ok = true;
        } else if (c->tool_is_eog[id]) {
            ok = c->tool.allows(piece, true);
        } else {
            ok = !piece.empty() && first_ok[static_cast<unsigned char>(piece[0])] && c->tool.allows(piece, false);
        }
        keep[id] = ok;
        kept += ok;
    }
    if (kept == 0) {
        c->tool.give_up();
        return;
    }
    for (size_t id = 0; id < logits.size(); ++id) {
        if (!keep[id]) logits[id] = -1e30f;
    }
}

static void tool_accept(EngineContext* c, int32_t token) {
    if (!c->tool.configured()) return;
    if (c->eog_ids.count(token)) {
        c->tool.accept(std::string());
        return;
    }
    std::string piece;
    if (c->has_tok) c->tok.piece(token, piece);
    c->tool.accept(piece);
}

static bool tool_needs_host(const EngineContext* c) {
    return c->tool.configured() && (c->tool.constraining() || c->tool.armed());
}

bool engine_set_tool_constraint(desireeia_ctx* ctx, const std::string& open_tag, const std::string& close_tag,
                                const std::vector<std::string>& tool_names) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(c);
    c->tool.configure(open_tag, close_tag, tool_names);
    return true;
}

bool engine_predict(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens, int32_t& out_token) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (!c->gf || !c->st.reader || n_tokens == 0) return false;
    c->cancel.store(false, std::memory_order_relaxed);

    c->st.tokens.assign(tokens, tokens + n_tokens);

    // Session: keep the K/V of the prefix shared with the previous call and
    // prefill only what is new; with no shared prefix this is the old full
    // reset + prefill.
    const size_t reuse = reusable_prefix(c, tokens, n_tokens);
    if (reuse > 0) static_cast<DenseForward*>(c->gf)->truncate_cache(reuse);
    else c->gf->reset_cache();
    c->last_reused = reuse;

    std::vector<float> logits;
    if (!c->gf->step(*c->st.reader, tokens + reuse, n_tokens - reuse, logits)) {
        c->st.last_error = c->gf->was_cancelled() ? "prefill cancelled" : "dense forward: prefill failed";
        if (!c->gf->was_cancelled() && !c->gf->last_fail().empty()) c->st.last_error += " (" + c->gf->last_fail() + ")";
        if (c->st.log) c->st.log(3, c->st.last_error.c_str());
        c->gf->reset_cache();
        invalidate_session(c);
        return false;
    }
    c->kv_tokens.assign(tokens, tokens + n_tokens);
    // Positions [0, reuse) keep the provenance they had (in exact mode they
    // were prefilled); [reuse, n_tokens) were just prefilled.
    c->kv_prefilled = n_tokens;
    c->kv_valid = true;
    // The history for the repetition penalties is the prompt itself: the
    // first generated token must not repeat what's already written.
    c->history.assign(tokens, tokens + n_tokens);
    c->tool.reset();
    out_token = c->sampler.sample(logits, c->history);
    tool_accept(c, out_token);
    c->last_token = out_token;
    c->has_session = true;
    c->pending.clear();
    return true;
}

// Phase 9 - Prompt Lookup Decoding: verifies up to kMaxDraft candidate
// tokens (found via n-gram repetition in the history generated so far) in
// A SINGLE batched pass (Phase 8), instead of one token at a time.
// Implemented and measured (2026-09-07) on gemma3-4b Q4_K_M with a generic,
// non-repetitive prompt: NEGATIVE RESULT, not enabled by default.
// Real trace (kNgramLen=3, kMaxDraft=4): most rounds find only K=1-2 with a
// very low acceptance rate (accepted=0 or 1 almost always, never more than
// 1 observed). Since the verification cost scales with K (the lm_head,
// ~525MB for gemma3-4b, is still read/decoded once but the dot-product has
// to be computed for every K column), a K=2 round with accepted<=1 gains
// nothing and costs more than a normal single-token decode: measured decode
// throughput 4.42 tok/s vs 7.3 tok/s baseline (worse, not better). The
// technique remains valid in theory for highly repetitive workloads
// (code-edit, RAG that repeats the context), but needs an adaptive gate
// (e.g. tracking the recent acceptance rate and disabling speculation
// when it drops below a threshold) before it can be left on by default:
// not implemented in this session, see Phase 9 in engine_gap_analysis.md.
// The infrastructure below (step() with all_logits, DenseForward::
// truncate_cache, find_ngram_continuation above) remains available and
// tested regardless, for when the gate gets added.
#ifdef DESIREEIA_CUDA_ENABLED
// Pipelined decode: 1 token returned, 0 failure, -1 not applicable (nothing
// done). Each call returns the token the step in flight drew, after
// queueing the next step - which consumes that token straight from the
// device and draws the one after - so the GPU never waits for the host.
static int next_token_pipelined(EngineContext* c, int32_t& out_token) {
    DenseForward* df = dynamic_cast<DenseForward*>(c->gf);
    Sampler::DeviceParams dp{};
    if (!c->pipe && (!df || g_live_contexts.load() != 1 || !df->pipeline_supported() ||
                     !c->sampler.device_params(dp))) {
        return -1;
    }
    if (c->pipe && !c->sampler.device_params(dp)) { drain_pipeline(c); return -1; }
    auto args = [&](int append_input, uint32_t slot) {
        CudaSampleArgs a{};
        a.temperature = dp.temperature; a.top_k = dp.top_k; a.top_p = dp.top_p;
        a.penalty_repeat = dp.penalty_repeat; a.penalty_freq = dp.penalty_freq;
        a.penalty_present = dp.penalty_present; a.penalty_last_n = dp.penalty_last_n;
        a.kk = dp.kk;
        a.u = dp.temperature > 0.0f ? c->sampler.draw_uniform() : 0.0f;
        a.append_input = append_input;
        a.slot = slot;
        return a;
    };
    auto fail = [&](const char* what) {
        c->st.last_error = std::string("dense forward: ") + what;
        if (!c->gf->last_fail().empty()) c->st.last_error += " (" + c->gf->last_fail() + ")";
        if (c->st.log) c->st.log(3, c->st.last_error.c_str());
        c->pipe = false;
        invalidate_session(c);
        return 0;
    };
    uint32_t cur_slot;
    const int32_t tk = c->last_token;
    if (!c->pipe) {
        // First pipelined token: the step consuming last_token (known on the
        // host) draws on the device, then the pipeline starts.
        c->history.push_back(tk);
        if (cuda_sample_history(c->history.data(), c->history.size()) != DESIREEIA_OK) {
            c->history.pop_back();
            return -1;
        }
        const std::mt19937 before = c->sampler.rng_state();
        const CudaSampleArgs a = args(0, 0);
        const int r = df->decode_launch(*c->st.reader, tk, a, c->decode_logits);
        if (r == 0) {                                // ran, but drew nothing: sample on the host
            c->sampler.set_rng_state(before);
            if (c->kv_valid) c->kv_tokens.push_back(tk);
            out_token = c->sampler.sample(c->decode_logits, c->history);
            c->last_token = out_token;
            return 1;
        }
        if (r < 0) { c->history.pop_back(); return fail("decode failed"); }
        if (c->kv_valid) c->kv_tokens.push_back(tk);
        cur_slot = 0;
    } else {
        // The step in flight consumed last_token: it is part of the sequence now.
        c->history.push_back(tk);
        if (c->kv_valid) c->kv_tokens.push_back(tk);
        cur_slot = c->pipe_slot;
        c->pipe = false;
    }
    // Queue the next step: its input is the token being drawn in cur_slot.
    const uint32_t next_slot = (cur_slot + 1) & 3;
    const std::mt19937 before = c->sampler.rng_state();
    const CudaSampleArgs a = args(1, next_slot);
    std::vector<float> unused;
    const int r = df->decode_launch(*c->st.reader, -1, a, unused);
    if (r == 1) {
        c->pipe = true;
        c->pipe_slot = next_slot;
        c->pipe_rng = before;
    } else {
        c->sampler.set_rng_state(before);            // no step in flight: the draw did not happen
        if (r < 0) {
            int32_t ignore = -1;
            cuda_sample_wait(cur_slot, ignore);
            return fail("decode failed");
        }
    }
    int32_t tok = -1;
    if (cuda_sample_wait(cur_slot, tok) != DESIREEIA_OK) return fail("reading the drawn token failed");
    out_token = tok;
    c->last_token = tok;
    return 1;
}
#endif

bool engine_next_token(desireeia_ctx* ctx, int32_t& out_token) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->gf || !c->st.reader || !c->has_session) return false;

#ifdef DESIREEIA_CUDA_ENABLED
    if (!tool_needs_host(c)) {
        const int r = next_token_pipelined(c, out_token);
        if (r == 1) {
            tool_accept(c, out_token);
            // The step already in flight drew the next token unconstrained:
            // undo it so that token is drawn under the mask.
            if (tool_needs_host(c)) drain_pipeline(c);
        }
        if (r >= 0) return r == 1;                  // -1: not applicable, the step-by-step path below
    } else {
        drain_pipeline(c);
    }
#endif
    int32_t tk = c->last_token;
    // Reused across tokens: a whole vocabulary (~1 MB) allocated and zeroed
    // on every token was measurable next to the sampling itself.
    std::vector<float>& logits = c->decode_logits;
    // `tk` (the token just consumed) enters the history BEFORE sampling:
    // it's exactly the immediate repetition that the penalties need to
    // be able to see.
    c->history.push_back(tk);
    // When the sampler can work from the largest logits alone, the engine
    // selects them on the device and only those come back (not the whole
    // vocabulary); the token drawn is the same.
    int got = 0;
    // The mask needs the whole vocabulary, not just the top candidates.
    const bool masking = c->tool.constraining();
    const uint32_t kk = masking ? 0 : c->sampler.candidates_needed(c->history);
    if (kk) got = c->gf->step_candidates(*c->st.reader, &tk, 1, kk, c->cand_ids, c->cand_vals, logits);
    const bool ok = got == 0 ? c->gf->step(*c->st.reader, &tk, 1, logits) : got > 0;
    if (!ok) {
        c->history.pop_back();
        c->st.last_error = "dense forward: decode failed";
        if (!c->gf->last_fail().empty()) c->st.last_error += " (" + c->gf->last_fail() + ")";
        if (c->st.log) c->st.log(3, c->st.last_error.c_str());
        invalidate_session(c);
        return false;
    }
    // The token just consumed is now in the cache too.
    if (c->kv_valid) c->kv_tokens.push_back(tk);
    if (masking && got != 1) tool_mask(c, logits);
    out_token = got == 1 ? c->sampler.sample_candidates(c->cand_ids, c->cand_vals, c->history)
                         : c->sampler.sample(logits, c->history);
    tool_accept(c, out_token);
    c->last_token = out_token;
    return true;
}

bool engine_set_sampling(desireeia_ctx* ctx, const desireeia_sampling& p) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    SamplerParams sp;
    sp.temperature     = p.temperature;
    sp.top_k           = p.top_k;
    sp.top_p           = p.top_p;
    sp.penalty_repeat  = p.penalty_repeat;
    sp.penalty_freq    = p.penalty_freq;
    sp.penalty_present = p.penalty_present;
    sp.penalty_last_n  = p.penalty_last_n;
    sp.seed            = p.seed;
    c->sampler.configure(sp);
    return true;
}

bool engine_get_sampling(const desireeia_ctx* ctx, desireeia_sampling& out) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    const SamplerParams& sp = c->sampler.params();
    out.temperature     = sp.temperature;
    out.top_k           = sp.top_k;
    out.top_p           = sp.top_p;
    out.penalty_repeat  = sp.penalty_repeat;
    out.penalty_freq    = sp.penalty_freq;
    out.penalty_present = sp.penalty_present;
    out.penalty_last_n  = sp.penalty_last_n;
    out.seed            = sp.seed;
    return true;
}

bool engine_load_lora(desireeia_ctx* ctx, const char* lora_gguf_path, float scale, std::string& err) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c || !lora_gguf_path) { err = "invalid argument"; return false; }
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (!c->gf) {
        err = "model has no generative forward engine (LoRA requires a dense/MLA model, not a BERT encoder)";
        return false;
    }
    // Different weights = the cached K/V no longer match the tokens.
    invalidate_session(c);
    if (!c->gf->load_lora(lora_gguf_path, scale, err)) {
        if (c->st.log) c->st.log(3, err.c_str());
        return false;
    }
    return true;
}

bool engine_clear_lora(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (c->gf) c->gf->clear_lora();
    invalidate_session(c);
    return true;
}

bool engine_session_reset(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (c->gf) c->gf->reset_cache();
    invalidate_session(c);
    c->has_session = false;
    c->pending.clear();
    c->last_reused = 0;
    return true;
}

namespace {
constexpr uint32_t kSessionMagic = 0x564B5344;   // "DSKV"
constexpr uint32_t kSessionVersion = 1;

// Paths cross the ABI as UTF-8; on Windows the narrow CRT functions read
// them in the ANSI code page, so a user folder with an accent would fail.
std::FILE* open_utf8(const std::string& path, const char* mode) {
#ifdef _WIN32
    wchar_t wmode[8] = {};
    for (size_t i = 0; mode[i] && i < 7; ++i) wmode[i] = (wchar_t) mode[i];
    return _wfopen(std::filesystem::u8path(path).c_str(), wmode);
#else
    return std::fopen(path.c_str(), mode);
#endif
}

uint64_t fnv1a(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

// Identity of the loaded model: file size, the first 64 KB (GGUF header and
// metadata) and the shape the cache depends on. Cheap enough to compute on
// every save/load, strict enough that a different model - or the same name
// with other weights - never restores a cache it did not produce.
uint64_t model_identity(const EngineContext* c, const DenseForward* df) {
    uint64_t h = 1469598103934665603ULL;
    std::FILE* f = open_utf8(c->st.meta.path, "rb");
    if (f) {
        std::fseek(f, 0, SEEK_END);
        const long long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        h = fnv1a(h, &size, sizeof(size));
        std::vector<uint8_t> head(64 * 1024);
        const size_t got = std::fread(head.data(), 1, head.size(), f);
        h = fnv1a(h, head.data(), got);
        std::fclose(f);
    }
    h = fnv1a(h, c->st.meta.arch.data(), c->st.meta.arch.size());
    const uint64_t shape[4] = { c->st.meta.n_layers, c->st.meta.n_vocab,
                                (uint64_t) df->kv_bytes_per_pos(), df->kv_quantized() ? 1u : 0u };
    return fnv1a(h, shape, sizeof(shape));
}
} // namespace

// Layout of a session image (little-endian, as every supported target):
//   u32 magic "DSKV" | u32 version | u64 model identity | u64 bytes per
//   position per layer | u64 n | i32 tokens[n] | K layer by layer | V same.
constexpr size_t kSessionHeader = 4 + 4 + 8 + 8 + 8;

size_t engine_session_image_size(desireeia_ctx* ctx, size_t n_prefix) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return 0;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    const DenseForward* df = dynamic_cast<const DenseForward*>(c->gf);
    if (!df || df->kv_bytes_per_pos() == 0 || !c->kv_valid || c->kv_prefilled == 0) return 0;
    size_t n = std::min(c->kv_prefilled, df->cache_len());
    if (n_prefix > 0) n = std::min(n, n_prefix);
    return kSessionHeader + n * sizeof(int32_t) + 2 * (size_t) c->st.meta.n_layers * n * df->kv_bytes_per_pos();
}

bool engine_session_serialize(desireeia_ctx* ctx, size_t n_prefix, std::vector<uint8_t>& out, std::string& err) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) { err = "invalid argument"; return false; }
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    DenseForward* df = dynamic_cast<DenseForward*>(c->gf);
    if (!df || df->kv_bytes_per_pos() == 0) { err = "this model's cache cannot be saved"; return false; }
    if (!c->kv_valid || c->kv_prefilled == 0) { err = "no prefilled session to save"; return false; }
    size_t n = std::min(c->kv_prefilled, df->cache_len());
    if (n_prefix > 0) n = std::min(n, n_prefix);

    std::vector<uint8_t> k, v;
    if (!df->export_kv(n, k, v)) { err = "cache export failed"; return false; }
    const uint64_t id = model_identity(c, df);
    const uint64_t bpp = df->kv_bytes_per_pos();
    const uint64_t n64 = n;
    out.resize(kSessionHeader + n * sizeof(int32_t) + k.size() + v.size());
    uint8_t* p = out.data();
    auto put = [&p](const void* src, size_t bytes) { std::memcpy(p, src, bytes); p += bytes; };
    put(&kSessionMagic, 4);
    put(&kSessionVersion, 4);
    put(&id, 8);
    put(&bpp, 8);
    put(&n64, 8);
    put(c->kv_tokens.data(), n * sizeof(int32_t));
    put(k.data(), k.size());
    put(v.data(), v.size());
    return true;
}

bool engine_session_deserialize(desireeia_ctx* ctx, const uint8_t* data, size_t size, size_t& out_tokens,
                                std::string& err) {
    out_tokens = 0;
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c || (!data && size > 0)) { err = "invalid argument"; return false; }
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    DenseForward* df = dynamic_cast<DenseForward*>(c->gf);
    if (!df || df->kv_bytes_per_pos() == 0) { err = "this model's cache cannot be restored"; return false; }
    if (size < kSessionHeader) { err = "truncated session data"; return false; }
    uint32_t magic = 0, version = 0;
    uint64_t id = 0, bpp = 0, n = 0;
    const uint8_t* p = data;
    auto get = [&p](void* dst, size_t bytes) { std::memcpy(dst, p, bytes); p += bytes; };
    get(&magic, 4);
    get(&version, 4);
    get(&id, 8);
    get(&bpp, 8);
    get(&n, 8);
    if (magic != kSessionMagic || version != kSessionVersion) { err = "not a session file of this engine version"; return false; }
    if (id != model_identity(c, df)) { err = "session saved for a different model or cache layout"; return false; }
    if (bpp != df->kv_bytes_per_pos() || n == 0 || n > (1ull << 24)) { err = "session cache layout mismatch"; return false; }
    const size_t layer_bytes = (size_t) c->st.meta.n_layers * n * bpp;
    if (size != kSessionHeader + n * sizeof(int32_t) + 2 * layer_bytes) { err = "truncated session data"; return false; }
    std::vector<int32_t> tokens(n);
    get(tokens.data(), n * sizeof(int32_t));
    const uint8_t* k = p;
    const uint8_t* v = p + layer_bytes;
    if (!df->import_kv(n, k, v)) { err = "cache import failed (memory?)"; return false; }
    c->kv_tokens = std::move(tokens);
    c->kv_prefilled = n;
    c->kv_valid = true;
    c->has_session = false;   // nothing to continue from until the next predict
    c->pending.clear();
    c->last_reused = 0;
    out_tokens = n;
    return true;
}

bool engine_session_save(desireeia_ctx* ctx, const char* path, size_t n_prefix, std::string& err) {
    if (!path) { err = "invalid argument"; return false; }
    std::vector<uint8_t> image;
    if (!engine_session_serialize(ctx, n_prefix, image, err)) return false;
    const std::string tmp = std::string(path) + ".tmp";
    std::FILE* f = open_utf8(tmp, "wb");
    if (!f) { err = "cannot create " + tmp; return false; }
    bool ok = std::fwrite(image.data(), 1, image.size(), f) == image.size();
    ok = (std::fclose(f) == 0) && ok;
    std::error_code ec;
    const auto tmp_p = std::filesystem::u8path(tmp);
    if (!ok) { std::filesystem::remove(tmp_p, ec); err = "write failed: " + tmp; return false; }
    // rename replaces an existing file atomically on every platform
    // (MoveFileEx with REPLACE_EXISTING on Windows).
    std::filesystem::rename(tmp_p, std::filesystem::u8path(path), ec);
    if (ec) { std::filesystem::remove(tmp_p, ec); err = "cannot rename to " + std::string(path); return false; }
    return true;
}

bool engine_session_load(desireeia_ctx* ctx, const char* path, size_t& out_tokens, std::string& err) {
    out_tokens = 0;
    if (!path) { err = "invalid argument"; return false; }
    std::FILE* f = open_utf8(path, "rb");
    if (!f) { err = "cannot open " + std::string(path); return false; }
    std::vector<uint8_t> image;
    uint8_t buf[1 << 16];
    size_t got;
    while ((got = std::fread(buf, 1, sizeof(buf), f)) > 0) image.insert(image.end(), buf, buf + got);
    std::fclose(f);
    return engine_session_deserialize(ctx, image.data(), image.size(), out_tokens, err);
}

void engine_cancel(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (c) c->cancel.store(true, std::memory_order_relaxed);   // no lock: a predict holds it
}

bool engine_was_cancelled(const desireeia_ctx* ctx) {
    const EngineContext* c = reinterpret_cast<const EngineContext*>(ctx);
    return c && c->gf && c->gf->was_cancelled();
}

void engine_set_progress(desireeia_ctx* ctx, void (*cb)(uint64_t, uint64_t, void*), void* user) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return;
    std::lock_guard<std::mutex> lk(c->mtx);
    c->hooks.progress = cb;
    c->hooks.user = user;
}

void engine_set_log(desireeia_ctx* ctx, LogFn log) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return;
    std::lock_guard<std::mutex> lk(c->mtx);
    c->st.log = std::move(log);
}

bool engine_reserve(desireeia_ctx* ctx, size_t n_positions) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    return c->gf && c->gf->reserve_positions(n_positions);
}

bool engine_trim(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (!c->gf) return false;
    c->gf->release_cache();
    invalidate_session(c);
    return true;
}

std::string engine_last_error(const desireeia_ctx* ctx) {
    const EngineContext* c = reinterpret_cast<const EngineContext*>(ctx);
    return c ? c->st.last_error : std::string();
}

bool engine_set_session_reuse(desireeia_ctx* ctx, int mode) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c || mode < 0 || mode > 2) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    c->session_mode = mode;
    if (mode == 0) invalidate_session(c);
    return true;
}

size_t engine_last_reused_tokens(const desireeia_ctx* ctx) {
    const EngineContext* c = reinterpret_cast<const EngineContext*>(ctx);
    return c ? c->last_reused : 0;
}

bool engine_load_prerouter(desireeia_ctx* ctx, const char* path, std::string& err) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c || !path) { err = "invalid argument"; return false; }
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (!c->gf) {
        err = "model has no generative forward engine (prerouter requires a dense/MLA MoE model)";
        return false;
    }
    if (!c->gf->load_prerouter(path, err)) {
        if (c->st.log) c->st.log(3, err.c_str());
        return false;
    }
    return true;
}

bool engine_clear_prerouter(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (c->gf) c->gf->clear_prerouter();
    return true;
}

bool engine_set_prerouter_heuristic(desireeia_ctx* ctx, bool enabled) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (c->gf) c->gf->set_prerouter_heuristic(enabled);
    return true;
}

// Renders the model's Jinja template. The tokenizer adds BOS itself, so a
// leading bos_token the template printed is dropped (it would be doubled).
static bool render_jinja(const EngineContext* c, const jinja::Value& messages, const jinja::Value* tools,
                         bool add_assistant, std::string& out) {
    if (!c->jinja_template) return false;
    std::string bos, eos;
    if (c->has_tok) {
        if (c->bos_id >= 0) c->tok.piece(c->bos_id, bos);
        if (c->eos_id >= 0) c->tok.piece(c->eos_id, eos);
    }
    jinja::Value vars = jinja::Value::object();
    vars.set("messages", messages);
    vars.set("add_generation_prompt", jinja::Value(add_assistant));
    vars.set("bos_token", jinja::Value(bos));
    vars.set("eos_token", jinja::Value(eos));
    if (tools && !tools->is_none() && !tools->is_undefined()) vars.set("tools", *tools);
    try {
        out = c->jinja_template->render(vars);
    } catch (const std::exception& e) {
        if (c->st.log) c->st.log(4, (std::string("chat template failed, using the built-in format: ") + e.what()).c_str());
        return false;
    }
    if (!bos.empty() && out.compare(0, bos.size(), bos) == 0) out.erase(0, bos.size());
    return true;
}

bool engine_apply_chat_template(const desireeia_ctx* ctx,
                                const char** roles, const char** contents, size_t n_messages,
                                bool add_assistant, std::string& out) {
    const EngineContext* c = reinterpret_cast<const EngineContext*>(ctx);
    if (!c) return false;
    std::vector<ChatMessage> chat;
    chat.reserve(n_messages);
    jinja::Value messages = jinja::Value::array();
    for (size_t i = 0; i < n_messages; ++i) {
        chat.push_back({roles[i] ? roles[i] : "", contents[i] ? contents[i] : ""});
        messages.arr().push_back(jinja::Value::object({{"role", jinja::Value(chat.back().role)},
                                                       {"content", jinja::Value(chat.back().content)}}));
    }
    if (render_jinja(c, messages, nullptr, add_assistant, out)) return true;
    out = apply_chat_template(c->chat_template, chat, add_assistant);
    return true;
}

bool engine_apply_chat_template_json(const desireeia_ctx* ctx, const std::string& messages_json,
                                     const std::string& tools_json, bool add_assistant, std::string& out) {
    const EngineContext* c = reinterpret_cast<const EngineContext*>(ctx);
    if (!c) return false;
    jinja::Value messages, tools;
    try {
        messages = jinja::parse_json(messages_json);
        if (!tools_json.empty()) tools = jinja::parse_json(tools_json);
    } catch (const std::exception& e) {
        if (c->st.log) c->st.log(3, (std::string("chat template: bad JSON: ") + e.what()).c_str());
        return false;
    }
    if (!messages.is_array()) return false;
    if (render_jinja(c, messages, tools.is_undefined() ? nullptr : &tools, add_assistant, out)) return true;
    // Built-in formats only know system/user/assistant: tool results become
    // a user turn and assistant tool calls their <tool_call> text.
    std::vector<ChatMessage> chat;
    for (const auto& m : messages.arr()) {
        std::string role = m.get("role").to_string();
        std::string content = m.get("content").is_string() ? m.get("content").str() : std::string();
        if (role == "tool") {
            content = "[Tool result: " + m.get("name").to_string() + "]\n" + content;
            role = "user";
        } else if (role == "assistant" && m.get("tool_calls").is_array()) {
            for (const auto& call : m.get("tool_calls").arr()) {
                const jinja::Value fn = call.get("function");
                jinja::Value obj = jinja::Value::object({{"name", fn.get("name")}, {"arguments", fn.get("arguments")}});
                content += "<tool_call>\n" + obj.to_json() + "\n</tool_call>";
            }
        }
        chat.push_back({role, content});
    }
    out = apply_chat_template(c->chat_template, chat, add_assistant);
    return true;
}

size_t engine_context_size(const desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c) return 0;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    size_t n = 0;
    // c->st.kv is the legacy KvCache (never populated by the current
    // forward path); the real K/V cache lives inside DenseForward.
    if (c->gf) n += (size_t) c->gf->kv_bytes();
    else if (c->st.kv) n += c->st.kv->bytes();
    return n;
}

uint32_t engine_context_length_trained(const desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c) return 0;
    std::lock_guard<std::mutex> lk(c->mtx);
    return c->gf ? c->gf->trained_context_length() : 0;
}

bool engine_tokenize(desireeia_ctx* ctx, const std::string& text, bool add_bos, std::vector<int32_t>& out_ids) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->has_tok) return false;
    out_ids = c->tok.encode(text, add_bos);
    return true;
}

bool engine_token_piece(desireeia_ctx* ctx, int32_t id, std::string& out) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->has_tok) return false;
    return c->tok.piece(id, out);
}

bool engine_is_eog_token(const desireeia_ctx* ctx, int32_t id) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    return c->eog_ids.count(id) != 0;
}

bool engine_special_token_id(const desireeia_ctx* ctx, int which, int32_t& out_id) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->has_tok) return false;
    switch (which) {
        case 0: out_id = c->bos_id; return true;
        case 1: out_id = c->eos_id; return true;
        case 2: out_id = c->unk_id; return true;
        case 3: out_id = c->pad_id; return true;
        default: return false;
    }
}

// ============================================================
// Vision / Multimodal
// ============================================================

bool engine_load_vision(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c || !c->st.reader) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));

    if (c->st.vision) return true;  // Already loaded.

    auto* vision = new vision::VisionGGUFContext;
    if (!vision::vision_load_from_gguf(*vision, *c->st.reader)) {
        delete vision;
        return false;
    }

    // Resolve the image placeholder token id: the GGUF converter usually
    // records the token STRING in clip.vision.image_token; fall back to
    // the standard LLaVA/MiniCPM placeholders when the key is absent.
    // Lookup is exact-string against the model's own vocabulary.
    if (vision->image_token_id <= 0 && c->has_tok) {
        std::string probe;
        if (!c->st.reader->meta_str("clip.vision.image_token", probe) || probe.empty()) {
            probe.clear();
        }
        const std::vector<std::string> candidates = probe.empty()
            ? std::vector<std::string>{"<image>", "<image_pad>", "<img>", "<span>"}
            : std::vector<std::string>{probe};
        for (const std::string& cand : candidates) {
            const int32_t id = c->tok.token_to_id(cand);
            if (id >= 0) { vision->image_token_id = id; break; }
        }
    }

    c->st.vision = vision;
    return true;
}

bool engine_has_vision(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    return c->st.vision && c->st.vision->initialized;
}

int32_t engine_vision_token_count(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c || !c->st.vision) return 0;
    std::lock_guard<std::mutex> lk(c->mtx);
    return vision::vision_get_token_count(*c->st.vision);
}

int32_t engine_vision_image_token(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c || !c->st.vision) return -1;
    std::lock_guard<std::mutex> lk(c->mtx);
    return c->st.vision->image_token_id;
}

bool engine_encode_image(desireeia_ctx* ctx, const DesireeAIImage& image,
                         std::vector<float>& out_embd, uint32_t& out_dim) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c || !c->st.vision || !c->st.vision->initialized) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (!image.data || image.width == 0 || image.height == 0) return false;

    if (!vision::vision_encode_image(*c->st.vision, image, out_embd)) return false;
    out_dim = static_cast<uint32_t>(vision::vision_get_output_dim(*c->st.vision));
    return true;
}

// Predict with multimodal input: the token stream contains the image
// placeholder(s), and `embd`/`n_embd` are the vision encoder outputs
// (one n_embd-sized vector per image placeholder occurrence, in order).
// The forward engine replaces the placeholder token embedding with each
// vector, so the model sees the image as if its patches were tokens.
bool engine_predict_vision(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens,
                           const float* embd, size_t n_embd, int32_t image_token,
                           int32_t& out_token) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    drain_pipeline(const_cast<EngineContext*>(c));
    if (!c->gf || !c->st.reader || n_tokens == 0) return false;
    if (embd == nullptr || n_embd == 0) return false;

    DenseForward* df = dynamic_cast<DenseForward*>(c->gf);
    if (!df) {
        c->st.last_error = "vision predict: model is not a dense forward engine";
        return false;
    }

    // Size sanity: every image placeholder occurrence in the stream must
    // consume exactly n_embd / (embedding dim) vectors. We check a simpler
    // invariant: the override must not underflow.
    const uint32_t n_embd_dim = df->config().n_embd;
    if (n_embd_dim == 0 || n_embd % n_embd_dim != 0) {
        c->st.last_error = "vision predict: embedding buffer misaligned";
        return false;
    }

    // The caller may ask us to resolve the placeholder from the model's
    // vision metadata/vocabulary (see engine_load_vision).
    if (image_token < 0) {
        if (c->st.vision && c->st.vision->image_token_id > 0) {
            image_token = c->st.vision->image_token_id;
        } else {
            c->st.last_error = "vision predict: no image placeholder token known";
            return false;
        }
    }

    df->set_embedding_override(std::vector<float>(embd, embd + n_embd), image_token);

    // Image embeddings are not a function of the token ids: the next text
    // prompt must not reuse these K/V even if its token ids match.
    invalidate_session(c);
    c->last_reused = 0;
    c->st.tokens.assign(tokens, tokens + n_tokens);
    c->gf->reset_cache();
    std::vector<float> logits;
    if (!c->gf->step(*c->st.reader, tokens, n_tokens, logits)) {
        c->st.last_error = "dense forward: vision prefill failed";
        if (!c->gf->last_fail().empty()) c->st.last_error += " (" + c->gf->last_fail() + ")";
        if (c->st.log) c->st.log(3, c->st.last_error.c_str());
        return false;
    }
    c->history.assign(tokens, tokens + n_tokens);
    c->tool.reset();
    out_token = c->sampler.sample(logits, c->history);
    tool_accept(c, out_token);
    c->last_token = out_token;
    c->has_session = true;
    c->pending.clear();
    return true;
}

}