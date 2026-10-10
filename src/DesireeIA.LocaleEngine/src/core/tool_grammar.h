// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#pragma once

// Token-level constraint for tool calls.
//
// Prompt-based tool calling breaks easily on small models: a missing "}",
// stray tag text inside the JSON, a tool name that doesn't exist (all seen
// with a 4B model). Here the engine itself watches the generated text and,
// once the open tag (e.g. "<tool_call>") appears, only lets the sampler
// pick tokens that keep the output a valid prefix of
//
//     <open> {"name": "<one of the declared tools>", "arguments": {...}} <close>
//
// then only an end-of-generation token. Outside a tool call nothing is
// constrained, so plain answers are unaffected.
//
// The automaton works on bytes, so it is independent of the tokenizer:
// a token is allowed when feeding its decoded piece, byte by byte, keeps
// the state valid.

#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {

class ToolCallConstraint {
public:
    // Empty open_tag disables the constraint.
    void configure(std::string open_tag, std::string close_tag, std::vector<std::string> tool_names);
    bool configured() const { return !open_tag_.empty(); }

    // Back to "scanning for the open tag" (start of a generation).
    void reset();

    // The decoded piece of the token actually emitted.
    void accept(const std::string& piece);

    // True while the next token must be filtered (inside a call, or after
    // it waiting for end of generation).
    bool constraining() const { return st_.phase != Phase::Scanning; }
    // True when the text generated so far ends with a proper prefix of the
    // open tag: the next token may enter a call, so it should be drawn on
    // the host where the mask can apply from the very next step.
    bool armed() const;

    // Whether emitting a token with this piece is allowed now. is_eog tells
    // whether the token ends the generation.
    bool allows(const std::string& piece, bool is_eog) const;

    // True inside a free-form JSON string (argument values, nested keys)
    // with no escape pending: there any piece without '"', '\\' or control
    // bytes is accepted as is, which lets the mask skip the automaton for
    // almost the whole vocabulary.
    bool in_plain_string() const;

    // Called when every token would be masked (a dead end the grammar can't
    // leave): stop constraining instead of forcing garbage.
    void give_up();

private:
    enum class Phase : uint8_t { Scanning, Json, Close, Finished };
    enum class Mode : uint8_t {
        Value, ObjKeyOrEnd, ObjKey, Colon, ObjCommaOrEnd, ArrValueOrEnd, ArrCommaOrEnd,
        String, Number, Literal, Done
    };
    struct State {
        Phase phase = Phase::Scanning;
        Mode mode = Mode::Value;
        std::string stack;         // '{' / '[' (SSO: no heap for normal nesting)
        bool in_key = false;       // the String being read is an object key
        bool esc = false;
        int hex_left = 0;
        std::string literal_rest;  // remaining chars of true/false/null
        std::string key;           // current top-level key
        std::string top_key;       // key whose value is being read at depth 1
        std::string name;          // value of "name" so far
        bool reading_name = false;
        size_t close_pos = 0;      // chars of close_tag matched
        std::string tail;          // recent text, for open-tag detection
        bool started = false;      // first non-space char of the JSON seen
    };

    bool step(State& s, unsigned char c) const;
    bool after_value(State& s) const;
    bool name_prefix_ok(const std::string& name, bool closing) const;

    std::string open_tag_, close_tag_;
    std::vector<std::string> names_;
    State st_;
};

} // namespace desireeia
