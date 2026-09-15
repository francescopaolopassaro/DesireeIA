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
#include "core/sampler.h"
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
#include <deque>
#include <filesystem>
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

    // Chat prompt format, detected at load time (see engine_create): from
    // the GGUF's chat_template when present, otherwise from the
    // per-architecture default.
    ChatTemplateKind chat_template = ChatTemplateKind::Unknown;

    // Prompt Lookup Decoding: full history of tokens already in the KV
    // cache (prompt plus confirmed generations), used to look up an
    // n-gram continuation to verify in one batch; a queue of tokens
    // already checked/accepted but not yet returned to the caller
    // (desireeia_next_token returns them one at a time).
    std::vector<int32_t> history;
    std::deque<int32_t> pending;
};

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

    // K and V, one entry per layer per position, float32.
    const uint64_t kv_bytes = 2ull * layers * n_ctx *
                              static_cast<uint64_t>(n_head_kv) * head_dim * sizeof(float);

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

    return reinterpret_cast<desireeia_ctx*>(ctx);
}

void engine_destroy(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return;
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

bool engine_predict(desireeia_ctx* ctx, const int32_t* tokens, size_t n_tokens, int32_t& out_token) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->gf || !c->st.reader || n_tokens == 0) return false;

    c->st.tokens.assign(tokens, tokens + n_tokens);

    c->gf->reset_cache();
    std::vector<float> logits;
    if (!c->gf->step(*c->st.reader, tokens, n_tokens, logits)) {
        c->st.last_error = "dense forward: prefill failed";
        if (!c->gf->last_fail().empty()) c->st.last_error += " (" + c->gf->last_fail() + ")";
        if (c->st.log) c->st.log(3, c->st.last_error.c_str());
        return false;
    }
    // The history for the repetition penalties is the prompt itself: the
    // first generated token must not repeat what's already written.
    c->history.assign(tokens, tokens + n_tokens);
    out_token = c->sampler.sample(logits, c->history);
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
bool engine_next_token(desireeia_ctx* ctx, int32_t& out_token) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->gf || !c->st.reader || !c->has_session) return false;

    int32_t tk = c->last_token;
    std::vector<float> logits;
    if (!c->gf->step(*c->st.reader, &tk, 1, logits)) {
        c->st.last_error = "dense forward: decode failed";
        if (!c->gf->last_fail().empty()) c->st.last_error += " (" + c->gf->last_fail() + ")";
        if (c->st.log) c->st.log(3, c->st.last_error.c_str());
        return false;
    }
    // `tk` (the token just consumed) enters the history BEFORE sampling:
    // it's exactly the immediate repetition that the penalties need to
    // be able to see.
    c->history.push_back(tk);
    out_token = c->sampler.sample(logits, c->history);
    c->last_token = out_token;
    return true;
}

bool engine_set_sampling(desireeia_ctx* ctx, const desireeia_sampling& p) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
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
    if (!c->gf) {
        err = "model has no generative forward engine (LoRA requires a dense/MLA model, not a BERT encoder)";
        return false;
    }
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
    if (c->gf) c->gf->clear_lora();
    return true;
}

bool engine_load_prerouter(desireeia_ctx* ctx, const char* path, std::string& err) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c || !path) { err = "invalid argument"; return false; }
    std::lock_guard<std::mutex> lk(c->mtx);
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
    if (c->gf) c->gf->clear_prerouter();
    return true;
}

bool engine_set_prerouter_heuristic(desireeia_ctx* ctx, bool enabled) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (c->gf) c->gf->set_prerouter_heuristic(enabled);
    return true;
}

bool engine_apply_chat_template(const desireeia_ctx* ctx,
                                const char** roles, const char** contents, size_t n_messages,
                                bool add_assistant, std::string& out) {
    const EngineContext* c = reinterpret_cast<const EngineContext*>(ctx);
    if (!c) return false;
    std::vector<ChatMessage> chat;
    chat.reserve(n_messages);
    for (size_t i = 0; i < n_messages; ++i) {
        chat.push_back({roles[i] ? roles[i] : "", contents[i] ? contents[i] : ""});
    }
    out = apply_chat_template(c->chat_template, chat, add_assistant);
    return true;
}

size_t engine_context_size(const desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(const_cast<desireeia_ctx*>(ctx));
    if (!c) return 0;
    std::lock_guard<std::mutex> lk(c->mtx);
    size_t n = 0;
    // c->st.kv is the legacy KvCache (never populated by the current
    // forward path); the real K/V cache lives inside DenseForward.
    if (c->gf) n += (size_t) c->gf->kv_bytes();
    else if (c->st.kv) n += c->st.kv->bytes();
    return n;
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
    out_token = c->sampler.sample(logits, c->history);
    c->last_token = out_token;
    c->has_session = true;
    c->pending.clear();
    return true;
}

}