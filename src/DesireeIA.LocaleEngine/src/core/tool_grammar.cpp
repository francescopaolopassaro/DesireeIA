// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "core/tool_grammar.h"

#include <cctype>

namespace desireeia {

namespace {

bool is_ws(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

bool prefix_of(const std::string& p, const char* word) {
    return std::string(word).compare(0, p.size(), p) == 0;
}

} // namespace

void ToolCallConstraint::configure(std::string open_tag, std::string close_tag, std::vector<std::string> tool_names) {
    open_tag_ = std::move(open_tag);
    close_tag_ = std::move(close_tag);
    names_ = std::move(tool_names);
    reset();
}

void ToolCallConstraint::reset() { st_ = State{}; }

void ToolCallConstraint::give_up() {
    st_ = State{};
    // Stay off until the next generation: re-arming on the same text would
    // only hit the same dead end again.
    st_.phase = Phase::Scanning;
    st_.tail.clear();
}

bool ToolCallConstraint::armed() const {
    if (!configured() || st_.phase != Phase::Scanning) return false;
    for (size_t k = 1; k < open_tag_.size() && k <= st_.tail.size(); ++k) {
        if (st_.tail.compare(st_.tail.size() - k, k, open_tag_, 0, k) == 0) return true;
    }
    return false;
}

bool ToolCallConstraint::name_prefix_ok(const std::string& name, bool closing) const {
    if (names_.empty()) return closing ? !name.empty() : true;
    for (const auto& n : names_) {
        if (closing ? n == name : n.compare(0, name.size(), name) == 0) return true;
    }
    return false;
}

bool ToolCallConstraint::after_value(State& s) const {
    if (s.stack.empty()) {
        s.mode = Mode::Done;
        s.phase = close_tag_.empty() ? Phase::Finished : Phase::Close;
        s.close_pos = 0;
        return true;
    }
    s.mode = s.stack.back() == '{' ? Mode::ObjCommaOrEnd : Mode::ArrCommaOrEnd;
    return true;
}

bool ToolCallConstraint::step(State& s, unsigned char c) const {
    if (s.phase == Phase::Finished) return false;
    if (s.phase == Phase::Close) {
        if (s.close_pos == 0 && is_ws(c)) return true;
        if (c != static_cast<unsigned char>(close_tag_[s.close_pos])) return false;
        if (++s.close_pos == close_tag_.size()) s.phase = Phase::Finished;
        return true;
    }

    if (!s.started) {
        if (is_ws(c)) return true;
        if (c != '{') return false;  // a tool call is one JSON object
        s.started = true;
        s.stack.push_back('{');
        s.mode = Mode::ObjKeyOrEnd;
        return true;
    }

    const size_t depth = s.stack.size();
    switch (s.mode) {
        case Mode::Value: {
            if (is_ws(c)) return true;
            const bool top = depth == 1;
            if (top && s.top_key == "name" && c != '"') return false;       // name is a string
            if (top && s.top_key == "arguments" && c != '{') return false;  // arguments is an object
            switch (c) {
                case '{': s.stack.push_back('{'); s.mode = Mode::ObjKeyOrEnd; return true;
                case '[': s.stack.push_back('['); s.mode = Mode::ArrValueOrEnd; return true;
                case '"':
                    s.mode = Mode::String;
                    s.in_key = false;
                    s.reading_name = top && s.top_key == "name";
                    if (s.reading_name) s.name.clear();
                    return true;
                case 't': s.mode = Mode::Literal; s.literal_rest = "rue"; return true;
                case 'f': s.mode = Mode::Literal; s.literal_rest = "alse"; return true;
                case 'n': s.mode = Mode::Literal; s.literal_rest = "ull"; return true;
                default:
                    if (c == '-' || std::isdigit(c)) { s.mode = Mode::Number; return true; }
                    return false;
            }
        }
        case Mode::ObjKeyOrEnd:
            if (is_ws(c)) return true;
            if (c == '}') {
                if (depth == 1) return false;  // the call object needs its name
                s.stack.pop_back();
                return after_value(s);
            }
            [[fallthrough]];
        case Mode::ObjKey:
            if (is_ws(c)) return true;
            if (c != '"') return false;
            s.mode = Mode::String;
            s.in_key = true;
            s.key.clear();
            return true;
        case Mode::String:
            if (s.hex_left > 0) {
                if (!std::isxdigit(c)) return false;
                --s.hex_left;
                return true;
            }
            if (s.esc) {
                s.esc = false;
                if (c == 'u') { s.hex_left = 4; return true; }
                if (std::string("\"\\/bfnrt").find(static_cast<char>(c)) == std::string::npos) return false;
                if (s.reading_name) return false;  // tool names are plain identifiers
                return true;
            }
            if (c == '\\') { s.esc = true; return true; }
            if (c < 0x20) return false;
            if (c == '"') {
                if (s.in_key) {
                    if (depth == 1 && s.key != "name" && s.key != "arguments") return false;
                    s.mode = Mode::Colon;
                    return true;
                }
                if (s.reading_name) {
                    if (!name_prefix_ok(s.name, true)) return false;
                    s.reading_name = false;
                }
                return after_value(s);
            }
            if (s.in_key && depth == 1) {
                s.key.push_back(static_cast<char>(c));
                return prefix_of(s.key, "name") || prefix_of(s.key, "arguments");
            }
            if (s.in_key) return true;
            if (s.reading_name) {
                s.name.push_back(static_cast<char>(c));
                return name_prefix_ok(s.name, false);
            }
            return true;
        case Mode::Colon:
            if (is_ws(c)) return true;
            if (c != ':') return false;
            if (depth == 1) s.top_key = s.key;
            s.mode = Mode::Value;
            return true;
        case Mode::ObjCommaOrEnd:
            if (is_ws(c)) return true;
            if (c == ',') { s.mode = Mode::ObjKey; return true; }
            if (c == '}') {
                if (depth == 1 && !name_prefix_ok(s.name, true)) return false;
                s.stack.pop_back();
                return after_value(s);
            }
            return false;
        case Mode::ArrValueOrEnd:
            if (is_ws(c)) return true;
            if (c == ']') { s.stack.pop_back(); return after_value(s); }
            s.mode = Mode::Value;
            return step(s, c);
        case Mode::ArrCommaOrEnd:
            if (is_ws(c)) return true;
            if (c == ',') { s.mode = Mode::Value; return true; }
            if (c == ']') { s.stack.pop_back(); return after_value(s); }
            return false;
        case Mode::Number:
            if (std::isdigit(c) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') return true;
            if (!after_value(s)) return false;
            return step(s, c);  // the delimiter that ended the number
        case Mode::Literal:
            if (s.literal_rest.empty() || c != static_cast<unsigned char>(s.literal_rest[0])) return false;
            s.literal_rest.erase(0, 1);
            return s.literal_rest.empty() ? after_value(s) : true;
        case Mode::Done:
            return false;
    }
    return false;
}

void ToolCallConstraint::accept(const std::string& piece) {
    if (!configured()) return;
    if (st_.phase == Phase::Scanning) {
        st_.tail += piece;
        const size_t at = st_.tail.find(open_tag_);
        if (at == std::string::npos) {
            if (st_.tail.size() > 2 * open_tag_.size()) st_.tail.erase(0, st_.tail.size() - open_tag_.size());
            return;
        }
        const std::string rest = st_.tail.substr(at + open_tag_.size());
        st_ = State{};
        st_.phase = Phase::Json;
        for (unsigned char c : rest) {
            if (!step(st_, c)) { give_up(); return; }
        }
        return;
    }
    for (unsigned char c : piece) {
        if (!step(st_, c)) { give_up(); return; }
    }
}

bool ToolCallConstraint::in_plain_string() const {
    return st_.phase == Phase::Json && st_.mode == Mode::String && !st_.esc && st_.hex_left == 0 &&
           !st_.reading_name && !(st_.in_key && st_.stack.size() == 1);
}

bool ToolCallConstraint::allows(const std::string& piece, bool is_eog) const {
    switch (st_.phase) {
        case Phase::Scanning: return true;
        case Phase::Finished: return is_eog;
        default: break;
    }
    if (is_eog || piece.empty()) return false;
    State s = st_;
    for (unsigned char c : piece) {
        if (!step(s, c)) return false;
    }
    return true;
}

} // namespace desireeia
