#ifndef DESIREEIA_FORWARD_IFACE_H
#define DESIREEIA_FORWARD_IFACE_H

#include "core/engine.h"
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

// Interfaccia minima condivisa dai motori forward (DenseForward per le
// architetture ad attenzione, SsmForward per Mamba2/SSM): SOLO i metodi che
// ctx.cpp chiama davvero su `EngineContext::gf` (vedi grep mirato prima di
// introdurre questa interfaccia — non e' un'astrazione preventiva, riflette
// l'uso reale). Ogni motore resta una classe indipendente con il proprio
// stato interno (cache K/V posizionale per DenseForward, stato ricorrente
// conv+SSM per SsmForward): l'interfaccia esiste solo per permettere a
// EngineContext di tenere un puntatore unico, non per far condividere
// codice fra le due implementazioni (che infatti non condividono nulla).
class IForwardEngine {
public:
    virtual ~IForwardEngine() = default;
    virtual void reset_cache() = 0;
    virtual bool step(ModelReader& rd, const int32_t* tokens, size_t n_tokens,
                       std::vector<float>& last_logits, std::vector<float>* all_logits = nullptr) = 0;
    virtual uint64_t kv_bytes() const = 0;
    virtual bool weight_cache_enabled() const = 0;

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
