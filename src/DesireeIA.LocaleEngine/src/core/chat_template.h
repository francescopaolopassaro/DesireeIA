// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

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
    // Spark2_5: same full-width sentence delimiters as the DeepSeek
    // family, but the role markers use ORDINARY pipes and different words
    // (<|System|>/<|User|>/<|Bot|>), so it is a distinct format and not a
    // variant of the two above.
    Spark25,
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

// When the model carries no chat_template in its metadata (older GGUF
// files, or conversions that didn't copy it over), falls back to a
// per-architecture default: not as reliable as an explicit template (the
// same architecture can have several historical variants, e.g. llama2 vs
// llama3), but always better than the raw prompt with no turn markers.
DESIREEIA_INTERNAL ChatTemplateKind chat_template_for_arch(ArchKind arch);

// Applies the format to the messages, producing the prompt to tokenize.
// add_assistant appends the assistant turn's opening marker (to make it
// generate the response), the same way it was already done by hand for
// gemma.
DESIREEIA_INTERNAL std::string apply_chat_template(ChatTemplateKind kind,
                                                 const std::vector<ChatMessage>& chat,
                                                 bool add_assistant);

}

#endif
