// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#pragma once

// Token-level constraint that makes the generated text a document valid
// against a JSON schema, from the first token to the end of generation.
//
// The schema is compiled into a small tree of nodes; the generated text is
// followed byte by byte by a stack automaton (one stack per live
// alternative, so anyOf / oneOf / type lists need no backtracking). A token
// is allowed when feeding its decoded piece keeps at least one stack alive,
// which makes the mask independent of the tokenizer.
//
// Supported: type (also as a list), properties / required /
// additionalProperties, items / minItems / maxItems, minLength / maxLength,
// enum / const, anyOf / oneOf, a single-entry allOf, $ref to "#" and to
// "#/$defs/..." or "#/definitions/...", and "any JSON value" for a missing
// type. Not enforced (accepted in the schema, ignored): pattern, format,
// minimum / maximum, uniqueItems, prefixItems. An object that lists
// "properties" accepts only those keys unless additionalProperties says
// otherwise (the safe reading for model output); more than 64 properties in
// one object are refused.
//
// Whitespace between tokens is allowed up to a few characters in a row, so
// the model can keep its usual formatting without ever looping on blanks.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace desireeia {

// The pieces of a vocabulary as a byte trie. Asking "which tokens keep the document valid" by running the automaton over every
// token is 100k-260k walks a step; walking the trie instead advances the automaton once per shared prefix and drops a whole
// branch the moment the automaton dies on it, so a structural position (where almost nothing is allowed) costs a few thousand
// steps, not the vocabulary.
class PieceTrie {
public:
    // pieces[i] is the text of token i. Tokens whose `include[i]` is 0 (or any empty piece) are left out.
    void build(const std::vector<std::string>& pieces, const std::vector<uint8_t>& include);
    bool empty() const { return nodes_.size() <= 1; }
private:
    friend class JsonSchemaConstraint;
    struct Node {
        int32_t first_child = -1, next_sibling = -1;
        int32_t token = -1;          // head of the list of tokens that end here
        uint8_t byte = 0;
    };
    std::vector<Node> nodes_ = std::vector<Node>(1);
    std::vector<int32_t> token_next_;   // per token: the next token that ends at the same node
};

class JsonSchemaConstraint {
public:
    // Empty text turns the constraint off. "json" (or "true" / "{}") means
    // any JSON value. Returns false and fills `error` when the schema
    // cannot be read or uses something unsupported.
    bool configure(const std::string& schema_json, std::string& error);
    bool configured() const { return !nodes_.empty(); }

    // Start of a generation: back to the beginning of the document.
    void reset();

    // The decoded piece of the token actually emitted (empty for an
    // end-of-generation token).
    void accept(const std::string& piece);

    // True while tokens must be filtered (from reset to the end, unless the
    // grammar gave up).
    bool constraining() const { return configured() && !gave_up_; }
    bool armed() const { return false; }

    // Whether emitting a token with this piece is allowed now. An
    // end-of-generation token is allowed only when the document is complete.
    bool allows(const std::string& piece, bool is_eog) const;

    // True inside a free-form string of the document with no length limit:
    // any piece without '"', '\\' or control bytes is accepted as is.
    bool in_plain_string() const;

    // Called when every token would be masked: stop constraining instead of
    // forcing garbage.
    void give_up() { gave_up_ = true; }

    // A key for "the same set of allowed tokens": equal keys give the same
    // mask. Counters that cannot change what is allowed (the length of a string
    // with no length limit, the item count of an unbounded array) are left out,
    // so the repetitive parts of a document hit the same entry again and again.
    uint64_t state_key() const;

    // Marks in `keep` (one byte per token, already sized, zeroed by the caller) every token of `trie` whose piece keeps the document
    // valid from here; the result is the same set as asking allows() for each token. End-of-generation tokens are not in the trie.
    void mask(const PieceTrie& trie, std::vector<uint8_t>& keep) const;

private:
    struct SNode {
        enum Kind : uint8_t { Any, Object, Array, String, Number, Integer, Lits, Union };
        Kind kind = Any;
        // Object: keys sorted by their encoded text (no quotes)
        std::vector<std::string> keys;
        std::vector<int> prop;
        uint64_t req_mask = 0;
        bool extra_ok = false;
        int extra_node = -1;          // schema of the extra keys' values, -1 = any
        // Array
        int items = -1;
        int min_items = 0, max_items = -1;
        // String
        int min_len = 0, max_len = -1;
        // Lits: sorted, unique compact JSON texts
        std::vector<std::string> lits;
        // Union
        std::vector<int> alts;
    };

    struct Frame {
        uint8_t kind = 0;
        uint8_t st = 0;
        int16_t hex = 0;
        int32_t node = 0;
        int32_t a = 0, b = 0, c = 0;
        uint64_t used = 0;
    };

    struct Config {
        std::vector<Frame> stack;
        uint8_t ws = 0;
    };

    // compile
    int compile(const void* schema, const void* root, int depth, std::string& error);
    int add_node(SNode n);
    int lits_node(std::vector<std::string> lits);

    // run
    void step(const Config& in, unsigned char ch, std::vector<Config>& out) const;
    void start_value(Config c, int node, unsigned char ch, std::vector<Config>& out) const;
    bool can_end(const Config& c) const;
    bool run_piece(const std::string& piece, std::vector<Config>& cur) const;
    void walk(const PieceTrie& trie, int32_t node, const std::vector<Config>& cur, std::vector<uint8_t>& keep) const;

    std::vector<SNode> nodes_;
    int root_ = 0;
    int any_obj_ = 0, any_arr_ = 0, any_str_ = 0, any_lits_ = 0, any_num_ = 0, any_value_ = 0;
    std::map<std::string, int> ref_memo_;
    std::vector<Config> cfgs_;
    bool gave_up_ = false;
};

} // namespace desireeia
