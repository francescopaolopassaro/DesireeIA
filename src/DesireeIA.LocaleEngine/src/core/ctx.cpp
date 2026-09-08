#include "engine.h"
#include "desireeia/abi.h"
#include "core/arch_tags.h"
#include "core/chat_template.h"
#include "core/sampler.h"
#include "core/thread_pool.h"
#include "models/dense_forward.h"
#include "models/ssm_forward.h"
#include "models/bert_forward.h"
#include "tokenizer/tokenizer.h"
#include <cstring>
#include <deque>
#include <mutex>
#include <set>

namespace desireeia {

struct EngineContext {
    EngineState st;
    // Motore forward: DenseForward per le architetture ad attenzione,
    // SsmForward per Mamba2 (vedi ArchKind::Mamba2 e la nota estesa su
    // core/forward_iface.h). Le due classi non condividono stato: solo il
    // puntatore e' unificato dietro l'interfaccia minima.
    IForwardEngine* gf = nullptr;
    // BERT (encoder-only, ArchKind::Bert): NON implementa IForwardEngine —
    // non genera un "prossimo token", non ha logit su un vocabolario, non
    // ha una KV-cache da resettare. Campo separato, mutuamente esclusivo
    // con gf (un modello e' o generativo o un encoder, mai entrambi qui).
    BertForward* bert = nullptr;
    Tokenizer tok;
    bool has_tok = false;
    int32_t last_token = -1;
    bool has_session = false;
    std::mutex mtx;

    // Fase 0 (motore di inferenza reale): id dei token speciali letti dal
    // vocabolario del modello, persistiti qui (VocabData e' locale a
    // engine_create) cosi' desireeia_special_token_id puo' esporli a chi
    // genera (per fermarsi a EOS) o compone una chat template.
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

    // Campionamento. Di default greedy (temperature 0), cioe' il
    // comportamento che c'era prima e che i test end-to-end si aspettano;
    // diventa campionamento vero appena il chiamante alza la temperatura.
    Sampler sampler;

    // Formato di prompt per la chat, rilevato al caricamento (vedi
    // engine_create): dal chat_template del GGUF se presente, altrimenti
    // dal default per architettura.
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
// Cerca l'ultima occorrenza (piu' recente = piu' probabile per pattern
// locali ripetuti, es. codice) della sequenza [ultimi kNgramLen-1 token di
// history, last_token] altrove in history, e restituisce quello che la
// seguiva come continuazione candidata (fino a max_draft token). Nessun
// modello draft: e' la tecnica "Prompt Lookup Decoding" (n-gram), utile
// soprattutto quando l'output ripete materiale gia' visto nel contesto
// (tipico di codice/documenti strutturati). Se non trova nulla, ritorna
// false e il chiamante ricade sul decode singolo token normale (nessuna
// regressione: costo aggiuntivo di una scansione lineare su history).
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
        }
    }
    return plan;
}

std::string sidecar_path(const std::string& model_path, const char* suffix) {
    return model_path + suffix;
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
        if (df->open(*reader, meta, arch_kind, ctx->st.plan.ram_budget_mb, ctx->st.experts)) {
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

    return reinterpret_cast<desireeia_ctx*>(ctx);
}

void engine_destroy(desireeia_ctx* ctx) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return;
    delete c->gf;
    delete c->bert;
    delete c->st.kv;
    delete c->st.experts;
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
        return false;
    }
    // La storia per le penalita' di ripetizione e' il prompt stesso: il
    // primo token generato non deve ripetere quello che c'e' gia' scritto.
    c->history.assign(tokens, tokens + n_tokens);
    out_token = c->sampler.sample(logits, c->history);
    c->last_token = out_token;
    c->has_session = true;
    c->pending.clear();
    return true;
}

// Fase 9 - Prompt Lookup Decoding: verifica fino a kMaxDraft token candidati
// (trovati per ripetizione n-gram nella storia gia' generata) in UN SOLO
// passo batched (Fase 8), invece di un token alla volta. Implementata e
// misurata (2026-09-07) su gemma3-4b Q4_K_M con un prompt generico non
// ripetitivo: RISULTATO NEGATIVO, non abilitata di default.
// Trace reale (kNgramLen=3, kMaxDraft=4): la maggior parte dei round trova
// solo K=1-2 con tasso di accettazione molto basso (accepted=0 o 1 quasi
// sempre, mai piu' di 1 osservato). Dato che il costo di verifica scala
// con K (il lm_head, ~525MB per gemma3-4b, viene comunque letto/decodificato
// una volta ma il dot-product va calcolato per ogni colonna K), un round
// K=2 con accepted<=1 non guadagna nulla e costa di piu' di un decode
// singolo normale: decode misurato 4.42 tok/s vs 7.3 tok/s baseline (peggio,
// non meglio). La tecnica resta valida in teoria per workload molto
// ripetitivi (code-edit, RAG che ripete il contesto), ma richiede un gate
// adattivo (es. tracciare il tasso di accettazione recente e disabilitare
// la speculazione quando scende sotto una soglia) prima di poter essere
// tenuta accesa di default: non implementato in questa sessione, vedi
// Fase 9 in engine_gap_analysis.md. L'infrastruttura sotto (step() con
// all_logits, DenseForward::truncate_cache, find_ngram_continuation sopra)
// resta comunque disponibile e testata per quando verra' aggiunto il gate.
bool engine_next_token(desireeia_ctx* ctx, int32_t& out_token) {
    EngineContext* c = reinterpret_cast<EngineContext*>(ctx);
    if (!c) return false;
    std::lock_guard<std::mutex> lk(c->mtx);
    if (!c->gf || !c->st.reader || !c->has_session) return false;

    int32_t tk = c->last_token;
    std::vector<float> logits;
    if (!c->gf->step(*c->st.reader, &tk, 1, logits)) {
        c->st.last_error = "dense forward: decode failed";
        return false;
    }
    // `tk` (il token appena consumato) entra nella storia PRIMA di
    // campionare: e' proprio la ripetizione immediata che le penalita'
    // devono poter vedere.
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
    // c->st.kv e' la KvCache legacy (mai popolata dal forward path attuale);
    // la cache K/V reale vive dentro DenseForward.
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

}