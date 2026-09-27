// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_FORWARD_IFACE_H
#define DESIREEIA_FORWARD_IFACE_H

#include <atomic>
#include <cstdint>
#include "core/engine.h"
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

// Minimal interface shared by the forward engines (DenseForward for
// attention-based architectures, SsmForward for Mamba2/SSM): ONLY the
// methods that ctx.cpp actually calls on `EngineContext::gf` (see the
// targeted grep done before introducing this interface — it's not a
// speculative abstraction, it reflects real usage). Each engine remains
// an independent class with its own internal state (positional K/V cache
// for DenseForward, recurrent conv+SSM state for SsmForward): the
// interface exists only to let EngineContext hold a single pointer, not
// to make the two implementations share code (and indeed they share
// nothing).
struct StepHooks {
    const std::atomic<bool>* cancel = nullptr;
    void (*progress)(uint64_t done, uint64_t total, void* user) = nullptr;
    void* user = nullptr;
};

class IForwardEngine {
public:
    virtual ~IForwardEngine() = default;
    virtual void reset_cache() = 0;
    virtual bool step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
                       std::vector<float>& last_logits, std::vector<float>* all_logits = nullptr) = 0;
    // One decoded token whose sampler needs only the kk largest logits:
    // 1 = ids/vals hold them (largest first, lower id first on ties),
    // 2 = the whole vocabulary is in logits, 0 = not attempted (call step),
    // -1 = the step failed.
    virtual int step_candidates(ModelReader& rd, const int32_t* tokens, size_t n_tokens, uint32_t kk,
                                std::vector<int32_t>& ids, std::vector<float>& vals, std::vector<float>& logits) {
        (void) rd; (void) tokens; (void) n_tokens; (void) kk; (void) ids; (void) vals; (void) logits;
        return 0;
    }
    virtual uint64_t kv_bytes() const = 0;
    virtual bool weight_cache_enabled() const = 0;

    // The model's own trained/declared max context length, read from GGUF
    // metadata ("<arch>.context_length") at load time. 0 = unknown (the
    // engine itself has no fixed context cap - the KV cache grows
    // dynamically - this is purely informational, for callers that want to
    // show or budget around the model's intended context window).
    virtual uint32_t trained_context_length() const { return 0; }

    // Diagnostic only: which tensor/check made the last step() call fail, if
    // it did. The ABI reports a bare error code with no detail, and this is
    // the cheapest way to get one without plumbing a full logger through
    // every load path. Default empty — not every engine needs this level of
    // detail yet.
    virtual const std::string& last_fail() const {
        static const std::string none;
        return none;
    }

    // LoRA adapters (Recover-LoRA). Default: not supported — only
    // DenseForward overrides these; BertForward/SsmForward have no
    // matmul-dispatch choke point to hook a delta into (see the override's
    // doc comment in dense_forward.h for what IS covered).
    virtual bool load_lora(const std::string& lora_gguf_path, float scale, std::string& err) {
        (void) lora_gguf_path; (void) scale;
        err = "LoRA adapters are not supported by this model's forward engine";
        return false;
    }
    virtual void clear_lora() {}

    // Prerouter routing prediction. Default: not supported — only
    // DenseForward overrides these (MoE-only mechanism; BertForward/
    // SsmForward have no router to predict ahead of).
    virtual bool load_prerouter(const std::string& path, std::string& err) {
        (void) path;
        err = "prerouter prediction is not supported by this model's forward engine";
        return false;
    }
    virtual void clear_prerouter() {}
    virtual void set_prerouter_heuristic(bool on) { (void) on; }

    // Cancellation and progress of a multi-token prefill (desireeia_cancel,
    // desireeia_set_progress_callback). The hooks object belongs to the
    // context and outlives the engine. Default: ignored.
    virtual void set_hooks(const StepHooks* hooks) { (void) hooks; }
    // Whether the last step() stopped because of a cancellation.
    virtual bool was_cancelled() const { return false; }
    // KV cache sizing (desireeia_reserve_context / desireeia_trim_cache).
    virtual bool reserve_positions(size_t n) { (void) n; return false; }
    virtual void release_cache() { reset_cache(); }
};

}

#endif
