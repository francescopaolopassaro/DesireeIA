#ifndef DESIREEIA_CHAT_TEMPLATE_H
#define DESIREEIA_CHAT_TEMPLATE_H

#include "core/engine.h"
#include "core/arch_tags.h"
#include <string>
#include <vector>

namespace desireeia {

// A single chat message: role ("system"/"user"/"assistant") plus text.
struct ChatMessage {
    std::string role;
    std::string content;
};

// Recognized chat prompt formats. Earlier, only the gemma format existed,
// hardcoded in the CLI: a Qwen/Llama/Mistral/... model in chat mode would
// receive the user's raw prompt with no turn markers at all, and produce
// degraded responses.
//
// Coverage: the families most commonly used for local inference as of
// late 2026 (Qwen and most fine-tunes -> ChatML, Llama 2/3/3.1/3.2/3.3/4,
// every historical Mistral/Mixtral variant, Gemma 2/3, Phi 3/4, DeepSeek
// V2/V3/R1, Command-R, ChatGLM3/4, MiniCPM, Zephyr, Falcon3, Exaone3).
// Many smaller, lab-specific formats with marginal adoption for local use
// aren't covered here — listed as a known gap in docs/engine_gap_analysis.md.
// A model using one of those formats falls back to ChatML (the most
// common fallback, often close enough textually) instead of an
// unformatted raw prompt.
enum class ChatTemplateKind {
    Unknown,
    ChatMl,
    Llama2,
    Llama2Sys,
    Llama2SysBos,
    Llama2SysStrip,
    Llama3,
    Llama4,
    MistralV1,
    MistralV3,
    MistralV3Tekken,
    MistralV7,
    MistralV7Tekken,
    Gemma,
    Phi3,
    Phi4,
    DeepSeek2,
    DeepSeek3,
    CommandR,
    ChatGlm3,
    ChatGlm4,
    MiniCpm,
    Zephyr,
    Falcon3,
    Exaone3,
};

// Detects the format from the model file's `tokenizer.chat_template`
// Jinja string, when present (almost every modern conversion embeds it
// from the original tokenizer config). It doesn't interpret the Jinja
// template — instead it looks for distinctive markers in the string,
// which is enough because the Jinja templates from different labs, while
// textually different, almost always reduce to one of a small number of
// de-facto formats.
DESIREEIA_INTERNAL ChatTemplateKind detect_chat_template(const std::string& tmpl);

// Quando il modello non porta un chat_template nei metadati (GGUF piu'
// vecchi, o conversioni che non lo hanno copiato), si ricade su un default
// per architettura: non e' affidabile quanto il template esplicito (una
// stessa architettura puo' avere piu' varianti storiche, es. llama2 vs
// llama3), ma e' sempre meglio del prompt grezzo senza marcatori di turno.
DESIREEIA_INTERNAL ChatTemplateKind chat_template_for_arch(ArchKind arch);

// Applica il formato ai messaggi, producendo il prompt da tokenizzare.
// add_assistant aggiunge il marcatore di apertura del turno assistente
// (per far generare la risposta), com'era gia' fatto a mano per gemma.
DESIREEIA_INTERNAL std::string apply_chat_template(ChatTemplateKind kind,
                                                 const std::vector<ChatMessage>& chat,
                                                 bool add_assistant);

}

#endif
