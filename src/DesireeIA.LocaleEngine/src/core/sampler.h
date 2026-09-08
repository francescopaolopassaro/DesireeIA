#ifndef DESIREEIA_SAMPLER_H
#define DESIREEIA_SAMPLER_H

#include "desireeia/abi.h"

#include <cstdint>
#include <random>
#include <vector>

namespace desireeia {

// Sampling parameters. The defaults match common practice for
// interactive generation.
//
// temperature <= 0 significa GREEDY (argmax): deterministico, nessun
// campionamento. Resta il default perche' e' l'unico comportamento che i
// test end-to-end esistenti si aspettano — chi vuole generazione reale
// alza la temperatura esplicitamente.
struct SamplerParams {
    float    temperature     = 0.0f;   // 0 = greedy
    int32_t  top_k           = 40;     // <= 0 = disattivato
    float    top_p           = 0.95f;  // >= 1 = disattivato
    float    penalty_repeat  = 1.0f;   // 1.0 = disattivata
    float    penalty_freq    = 0.0f;
    float    penalty_present = 0.0f;
    int32_t  penalty_last_n  = 64;     // quanti token indietro guardare
    uint32_t seed            = 0;      // 0 = seme da random_device
};

// Campionatore con stato (il generatore pseudo-casuale). Non e' thread-safe:
// un'istanza per contesto di inferenza, protetta dal mutex del contesto.
class Sampler {
public:
    void configure(const SamplerParams& p);
    const SamplerParams& params() const { return params_; }

    // Sceglie il prossimo token dai logit. `history` sono i token gia'
    // presenti nella sequenza (prompt + generati), usati per le penalita'
    // di ripetizione: se ne guardano gli ultimi penalty_last_n.
    //
    // `logits` viene modificato sul posto (penalita' e temperatura).
    int32_t sample(std::vector<float>& logits, const std::vector<int32_t>& history);

private:
    SamplerParams params_;
    std::mt19937  rng_{std::random_device{}()};
    bool          seeded_ = false;

    // Buffer riusati fra chiamate per non riallocare a ogni token: il
    // vocabolario di gemma3 e' 262144 voci, allocarlo a ogni passo sarebbe
    // una frazione non trascurabile del tempo di decode.
    std::vector<int32_t> idx_;
    std::vector<float>   probs_;
};

}

#endif
