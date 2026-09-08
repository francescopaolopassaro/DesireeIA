#ifndef DESIREEIA_FORWARD_IFACE_H
#define DESIREEIA_FORWARD_IFACE_H

#include "core/engine.h"
#include <cstdint>
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
};

}

#endif
