// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_FORWARD_IFACE_H
#define DESIREEIA_FORWARD_IFACE_H

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
class IForwardEngine {
public:
    virtual ~IForwardEngine() = default;
    virtual void reset_cache() = 0;
    virtual bool step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
                       std::vector<float>& last_logits, std::vector<float>* all_logits = nullptr) = 0;
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
};

}

#endif
