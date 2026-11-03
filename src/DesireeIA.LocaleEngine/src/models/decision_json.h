// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_DECISION_JSON_H
#define DESIREEIA_DECISION_JSON_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace desireeia {
namespace djson {

// Small self-contained JSON DOM used by the decision-model backend only
// (config files, tokenizer.json, safetensors headers, request/response
// payloads). Differences from the minimal parser used by the safetensors
// reader: full string escapes (\uXXXX with surrogate pairs), booleans as a
// distinct type, numbers keep their source text (so a state object can be
// re-serialized without altering the digits), and objects keep key order.
struct Value {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;            // string payload, or the raw text of a number
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    const Value* find(const std::string& key) const {
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
    bool is_str() const { return type == Str; }
    bool is_num() const { return type == Num; }
    bool is_arr() const { return type == Arr; }
    bool is_obj() const { return type == Obj; }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s), i_(0) {}

    bool parse(Value& out) {
        skip_ws();
        if (!parse_value(out, 0)) return false;
        skip_ws();
        return i_ == s_.size();
    }

private:
    const std::string& s_;
    size_t i_;

    void skip_ws() {
        while (i_ < s_.size() && (unsigned char) s_[i_] <= ' ') i_++;
    }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xF0 | (cp >> 18));
            out += (char) (0x80 | ((cp >> 12) & 0x3F));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
    }

    bool hex4(uint32_t& v) {
        if (i_ + 4 > s_.size()) return false;
        v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_ + (size_t) k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return false;
        }
        i_ += 4;
        return true;
    }

    bool parse_string(std::string& out) {
        if (i_ >= s_.size() || s_[i_] != '"') return false;
        i_++;
        out.clear();
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == '"') { i_++; return true; }
            if (c != '\\') { out += c; i_++; continue; }
            i_++;
            if (i_ >= s_.size()) return false;
            char e = s_[i_++];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() &&
                        s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        const size_t save = i_;
                        i_ += 2;
                        uint32_t lo = 0;
                        if (hex4(lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            i_ = save;
                        }
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: return false;
            }
        }
        return false;
    }

    bool parse_value(Value& v, int depth) {
        if (depth > 128) return false;
        skip_ws();
        if (i_ >= s_.size()) return false;
        const char c = s_[i_];
        if (c == '{') return parse_obj(v, depth);
        if (c == '[') return parse_arr(v, depth);
        if (c == '"') { v.type = Value::Str; return parse_string(v.str); }
        if (s_.compare(i_, 4, "null") == 0) { v.type = Value::Null; i_ += 4; return true; }
        if (s_.compare(i_, 4, "true") == 0) { v.type = Value::Bool; v.b = true; i_ += 4; return true; }
        if (s_.compare(i_, 5, "false") == 0) { v.type = Value::Bool; v.b = false; i_ += 5; return true; }
        return parse_num(v);
    }

    bool parse_num(Value& v) {
        const size_t start = i_;
        while (i_ < s_.size()) {
            const char c = s_[i_];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') i_++;
            else break;
        }
        if (i_ == start) return false;
        v.type = Value::Num;
        v.str = s_.substr(start, i_ - start);
        v.num = std::strtod(v.str.c_str(), nullptr);
        return true;
    }

    bool parse_arr(Value& v, int depth) {
        v.type = Value::Arr;
        i_++;
        skip_ws();
        if (i_ < s_.size() && s_[i_] == ']') { i_++; return true; }
        for (;;) {
            Value item;
            if (!parse_value(item, depth + 1)) return false;
            v.arr.push_back(std::move(item));
            skip_ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { i_++; continue; }
            if (s_[i_] == ']') { i_++; return true; }
            return false;
        }
    }

    bool parse_obj(Value& v, int depth) {
        v.type = Value::Obj;
        i_++;
        skip_ws();
        if (i_ < s_.size() && s_[i_] == '}') { i_++; return true; }
        for (;;) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (i_ >= s_.size() || s_[i_] != ':') return false;
            i_++;
            Value val;
            if (!parse_value(val, depth + 1)) return false;
            v.obj.emplace_back(std::move(key), std::move(val));
            skip_ws();
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { i_++; continue; }
            if (s_[i_] == '}') { i_++; return true; }
            return false;
        }
    }
};

// Escapes a string for output as a JSON string literal. Non-ASCII bytes pass
// through unchanged (UTF-8 output); only quotes, backslashes and control
// characters are escaped, which is the layout the reference serializer uses
// when it is asked not to force ASCII.
inline void write_string(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    out += '"';
}

// Re-serializes a value the way the reference implementation does when it
// turns a structured state into text: ", " between items, ": " between key
// and value, keys in source order, numbers as written in the source.
inline void write_value(std::string& out, const Value& v) {
    switch (v.type) {
        case Value::Null: out += "null"; break;
        case Value::Bool: out += v.b ? "true" : "false"; break;
        case Value::Num: out += v.str; break;
        case Value::Str: write_string(out, v.str); break;
        case Value::Arr:
            out += '[';
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) out += ", ";
                write_value(out, v.arr[i]);
            }
            out += ']';
            break;
        case Value::Obj:
            out += '{';
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) out += ", ";
                write_string(out, v.obj[i].first);
                out += ": ";
                write_value(out, v.obj[i].second);
            }
            out += '}';
            break;
    }
}

inline std::string fmt_double(double x) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.9g", x);
    return buf;
}

}
}

#endif
