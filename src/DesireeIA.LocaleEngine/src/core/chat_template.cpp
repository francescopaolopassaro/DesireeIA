// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "core/chat_template.h"

namespace desireeia {

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) ++a;
    while (b > a && std::isspace((unsigned char) s[b - 1])) --b;
    return s.substr(a, b - a);
}

bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

}

// Priority-ordered detection: some markers (e.g. "<|assistant|>") appear
// in more than one chat format, so the ORDER these checks run in matters.
// Where a format isn't implemented here (see the list in chat_template.h),
// we fall through to the next check instead of returning Unknown right
// away: a model using that format may still match a more generic pattern
// further down the list (many ChatML-compatible fine-tunes, for example).
ChatTemplateKind detect_chat_template(const std::string& tmpl) {
    if (contains(tmpl, "<|im_start|>")) {
        if (contains(tmpl, "<|im_sep|>")) return ChatTemplateKind::Phi4;
        if (contains(tmpl, "<end_of_utterance>")) return ChatTemplateKind::Unknown; // SmolVLM, not covered
        return ChatTemplateKind::ChatMl;
    }
    if (tmpl.rfind("mistral", 0) == 0 || contains(tmpl, "[INST]")) {
        if (contains(tmpl, "[SYSTEM_PROMPT]")) return ChatTemplateKind::MistralV7;
        if (contains(tmpl, "' [INST] ' + system_message") || contains(tmpl, "[AVAILABLE_TOOLS]")) {
            if (contains(tmpl, " [INST]")) return ChatTemplateKind::MistralV1;
            if (contains(tmpl, "\"[INST]\"")) return ChatTemplateKind::MistralV3Tekken;
            return ChatTemplateKind::MistralV3;
        }
        const bool support_system = contains(tmpl, "<<SYS>>");
        const bool bos_in_history = contains(tmpl, "bos_token + '[INST]");
        const bool strip_msg = contains(tmpl, "content.strip()");
        if (strip_msg) return ChatTemplateKind::Llama2SysStrip;
        if (bos_in_history) return ChatTemplateKind::Llama2SysBos;
        if (support_system) return ChatTemplateKind::Llama2Sys;
        return ChatTemplateKind::Llama2;
    }
    if (contains(tmpl, "<|assistant|>") && contains(tmpl, "<|end|>")) return ChatTemplateKind::Phi3;
    if (contains(tmpl, "[gMASK]<sop>")) return ChatTemplateKind::ChatGlm4;
    if (contains(tmpl, "<|assistant|>") && contains(tmpl, "<|user|>")) {
        return contains(tmpl, "</s>") ? ChatTemplateKind::Falcon3 : ChatTemplateKind::Unknown; // GLMEdge, not covered
    }
    if (contains(tmpl, "<|user|>") && contains(tmpl, "<|endoftext|>")) return ChatTemplateKind::Zephyr;
    if (contains(tmpl, "<start_of_turn>")) return ChatTemplateKind::Gemma;
    if (contains(tmpl, "<|START_OF_TURN_TOKEN|>") && contains(tmpl, "<|USER_TOKEN|>")) return ChatTemplateKind::CommandR;
    if (contains(tmpl, "<|start_header_id|>") && contains(tmpl, "<|end_header_id|>")) return ChatTemplateKind::Llama3;
    if (contains(tmpl, "[gMASK]sop")) return ChatTemplateKind::ChatGlm3;
    if (contains(tmpl, "<\xE7\x94\xA8\xE6\x88\xB7>")) return ChatTemplateKind::MiniCpm; // "<用户>"
    if (contains(tmpl, "'Assistant: ' + message['content'] + eos_token")) return ChatTemplateKind::DeepSeek2;
    // The hex escape sequences are split into adjacent string literals
    // wherever the following byte is a valid hex digit (e.g. 'A', 'e'):
    // without the split, the compiler (MSVC treats it as hard error
    // C7744, GCC only as a warning) keeps consuming digits past \x9C,
    // producing an out-of-range value instead of the intended UTF-8 byte.
    if (contains(tmpl, "<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>") &&
        contains(tmpl, "<\xEF\xBD\x9C" "User\xEF\xBD\x9C>") &&
        contains(tmpl, "<\xEF\xBD\x9C" "end\xE2\x96\x81" "of\xE2\x96\x81sentence\xEF\xBD\x9C>")) {
        return ChatTemplateKind::DeepSeek3; // "<｜Assistant｜>" / "<｜User｜>" / "<｜end▁of▁sentence｜>"
    }
    if (contains(tmpl, "[|system|]") && contains(tmpl, "[|assistant|]") && contains(tmpl, "[|endofturn|]")) {
        return ChatTemplateKind::Exaone3;
    }
    if (contains(tmpl, "<|header_start|>") && contains(tmpl, "<|header_end|>")) return ChatTemplateKind::Llama4;
    return ChatTemplateKind::Unknown;
}

ChatTemplateKind chat_template_for_arch(ArchKind arch) {
    switch (arch) {
        case ArchKind::Gemma:
        case ArchKind::Gemma3:  return ChatTemplateKind::Gemma;
        case ArchKind::Qwen2:   return ChatTemplateKind::ChatMl;
        // DenseGqa covers a couple of related base-shape architecture tags
        // (see arch_tags.h): the newer template convention is assumed by
        // default, since it's the large majority of published checkpoints
        // in this shape. An older model without a chat_template in its
        // metadata would be formatted incorrectly by this fallback: known
        // gap, still preferable to sending a raw, unformatted prompt.
        case ArchKind::DenseGqa: return ChatTemplateKind::Llama3;
        case ArchKind::Mistral: return ChatTemplateKind::MistralV3;
        default: return ChatTemplateKind::Unknown;
    }
}

std::string apply_chat_template(ChatTemplateKind kind, const std::vector<ChatMessage>& chat, bool add_ass) {
    std::string out;
    auto app = [&](const std::string& s) { out += s; };

    switch (kind) {
    case ChatTemplateKind::ChatMl:
        for (const auto& m : chat) app("<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n");
        if (add_ass) app("<|im_start|>assistant\n");
        break;

    case ChatTemplateKind::Phi4:
        for (const auto& m : chat) app("<|im_start|>" + m.role + "<|im_sep|>" + m.content + "<|im_end|>");
        if (add_ass) app("<|im_start|>assistant<|im_sep|>");
        break;

    case ChatTemplateKind::Phi3:
        for (const auto& m : chat) app("<|" + m.role + "|>\n" + m.content + "<|end|>\n");
        if (add_ass) app("<|assistant|>\n");
        break;

    case ChatTemplateKind::Falcon3:
        for (const auto& m : chat) app("<|" + m.role + "|>\n" + m.content + "\n");
        if (add_ass) app("<|assistant|>\n");
        break;

    case ChatTemplateKind::Zephyr:
        for (const auto& m : chat) app("<|" + m.role + "|>\n" + m.content + "<|endoftext|>\n");
        if (add_ass) app("<|assistant|>\n");
        break;

    case ChatTemplateKind::Gemma: {
        // google/gemma-*-it. Gemma has no "system" role: a system
        // message's content is appended to the next user message instead
        // of being dropped (otherwise the system instructions would be
        // silently lost). "assistant" becomes "model" in the turn marker.
        std::string system_prompt;
        for (const auto& m : chat) {
            if (m.role == "system") { system_prompt += trim(m.content); continue; }
            const std::string role = (m.role == "assistant") ? "model" : m.role;
            app("<start_of_turn>" + role + "\n");
            if (!system_prompt.empty() && role != "model") {
                app(system_prompt + "\n\n");
                system_prompt.clear();
            }
            app(trim(m.content) + "<end_of_turn>\n");
        }
        if (add_ass) app("<start_of_turn>model\n");
        break;
    }

    case ChatTemplateKind::Llama3:
        for (const auto& m : chat) {
            app("<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + trim(m.content) + "<|eot_id|>");
        }
        if (add_ass) app("<|start_header_id|>assistant<|end_header_id|>\n\n");
        break;

    case ChatTemplateKind::Llama4:
        for (const auto& m : chat) {
            app("<|header_start|>" + m.role + "<|header_end|>\n\n" + trim(m.content) + "<|eot|>");
        }
        if (add_ass) app("<|header_start|>assistant<|header_end|>\n\n");
        break;

    case ChatTemplateKind::Llama2:
    case ChatTemplateKind::Llama2Sys:
    case ChatTemplateKind::Llama2SysBos:
    case ChatTemplateKind::Llama2SysStrip: {
        const bool support_system = kind != ChatTemplateKind::Llama2;
        const bool bos_in_history = kind == ChatTemplateKind::Llama2SysBos;
        const bool strip_msg = kind == ChatTemplateKind::Llama2SysStrip;
        bool is_inside_turn = true; // skips the initial BOS (added elsewhere by the tokenizer)
        app("[INST] ");
        for (const auto& m : chat) {
            const std::string content = strip_msg ? trim(m.content) : m.content;
            if (!is_inside_turn) {
                is_inside_turn = true;
                app(bos_in_history ? "<s>[INST] " : "[INST] ");
            }
            if (m.role == "system") {
                app(support_system ? ("<<SYS>>\n" + content + "\n<</SYS>>\n\n") : (content + "\n"));
            } else if (m.role == "user") {
                app(content + " [/INST]");
            } else {
                app(content + "</s>");
                is_inside_turn = false;
            }
        }
        break;
    }

    case ChatTemplateKind::MistralV7:
    case ChatTemplateKind::MistralV7Tekken: {
        const char* sp = kind == ChatTemplateKind::MistralV7 ? " " : "";
        for (const auto& m : chat) {
            if (m.role == "system") app(std::string("[SYSTEM_PROMPT]") + sp + m.content + "[/SYSTEM_PROMPT]");
            else if (m.role == "user") app(std::string("[INST]") + sp + m.content + "[/INST]");
            else app(std::string(sp) + m.content + "</s>");
        }
        break;
    }

    case ChatTemplateKind::MistralV1:
    case ChatTemplateKind::MistralV3:
    case ChatTemplateKind::MistralV3Tekken: {
        const std::string leading = kind == ChatTemplateKind::MistralV1 ? " " : "";
        const std::string trailing = kind == ChatTemplateKind::MistralV3Tekken ? "" : " ";
        const bool trim_ass = kind == ChatTemplateKind::MistralV3;
        bool is_inside_turn = false;
        for (const auto& m : chat) {
            if (!is_inside_turn) { app(leading + "[INST]" + trailing); is_inside_turn = true; }
            if (m.role == "system") {
                app(m.content + "\n\n");
            } else if (m.role == "user") {
                app(m.content + leading + "[/INST]");
            } else {
                app(trailing + (trim_ass ? trim(m.content) : m.content) + "</s>");
                is_inside_turn = false;
            }
        }
        break;
    }

    case ChatTemplateKind::CommandR:
        for (const auto& m : chat) {
            if (m.role == "system") app("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" + trim(m.content) + "<|END_OF_TURN_TOKEN|>");
            else if (m.role == "user") app("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + trim(m.content) + "<|END_OF_TURN_TOKEN|>");
            else if (m.role == "assistant") app("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" + trim(m.content) + "<|END_OF_TURN_TOKEN|>");
        }
        if (add_ass) app("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
        break;

    case ChatTemplateKind::ChatGlm3:
        app("[gMASK]sop");
        for (const auto& m : chat) app("<|" + m.role + "|>\n " + m.content);
        if (add_ass) app("<|assistant|>");
        break;

    case ChatTemplateKind::ChatGlm4:
        app("[gMASK]<sop>");
        for (const auto& m : chat) app("<|" + m.role + "|>\n" + m.content);
        if (add_ass) app("<|assistant|>\n");
        break;

    case ChatTemplateKind::MiniCpm:
        for (const auto& m : chat) {
            if (m.role == "user") app("<\xE7\x94\xA8\xE6\x88\xB7>" + trim(m.content) + "<AI>"); // "<用户>"
            else app(trim(m.content));
        }
        break;

    case ChatTemplateKind::DeepSeek2:
        for (const auto& m : chat) {
            if (m.role == "system") app(m.content + "\n\n");
            else if (m.role == "user") app("User: " + m.content + "\n\n");
            else if (m.role == "assistant") app("Assistant: " + m.content + "<\xEF\xBD\x9C" "end\xE2\x96\x81" "of\xE2\x96\x81sentence\xEF\xBD\x9C>");
        }
        if (add_ass) app("Assistant:");
        break;

    case ChatTemplateKind::DeepSeek3:
        for (const auto& m : chat) {
            if (m.role == "system") app(m.content + "\n\n");
            else if (m.role == "user") app("<\xEF\xBD\x9C" "User\xEF\xBD\x9C>" + m.content);
            else if (m.role == "assistant") app("<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>" + m.content + "<\xEF\xBD\x9C" "end\xE2\x96\x81" "of\xE2\x96\x81sentence\xEF\xBD\x9C>");
        }
        if (add_ass) app("<\xEF\xBD\x9C" "Assistant\xEF\xBD\x9C>");
        break;

    case ChatTemplateKind::Exaone3:
        for (const auto& m : chat) {
            if (m.role == "system") app("[|system|]" + trim(m.content) + "[|endofturn|]\n");
            else if (m.role == "user") app("[|user|]" + trim(m.content) + "\n");
            else if (m.role == "assistant") app("[|assistant|]" + trim(m.content) + "[|endofturn|]\n");
        }
        break;

    case ChatTemplateKind::Unknown:
    default:
        // No recognized format: falls back to ChatML. It's the most
        // common fallback (a great many fine-tunes adopt it even without
        // declaring it explicitly), and it's still better than the raw
        // prompt with no turn markers that existed before this phase.
        for (const auto& m : chat) app("<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n");
        if (add_ass) app("<|im_start|>assistant\n");
        break;
    }
    return out;
}

}
