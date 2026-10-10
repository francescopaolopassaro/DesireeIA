// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "core/jinja.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <sstream>
#include <unordered_map>

namespace desireeia {
namespace jinja {

// ===========================================================================
// Value
// ===========================================================================

bool Value::truthy() const {
    switch (t_) {
        case Type::Undefined: case Type::None: return false;
        case Type::Bool: return b_;
        case Type::Int: return i_ != 0;
        case Type::Float: return f_ != 0.0;
        case Type::String: return !s_->empty();
        case Type::Array: return !a_->empty();
        case Type::Object: return !o_->empty();
        case Type::Callable: return true;
    }
    return false;
}

static std::string float_repr(double f) {
    if (std::isnan(f)) return "nan";
    if (std::isinf(f)) return f > 0 ? "inf" : "-inf";
    char buf[40];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, f);
        if (std::strtod(buf, nullptr) == f) break;
    }
    std::string s = buf;
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return s;
}

std::string Value::to_string() const {
    switch (t_) {
        case Type::Undefined: return "";
        case Type::None: return "None";
        case Type::Bool: return b_ ? "True" : "False";
        case Type::Int: return std::to_string(i_);
        case Type::Float: return float_repr(f_);
        case Type::String: return *s_;
        default: return repr();
    }
}

std::string Value::repr() const {
    switch (t_) {
        case Type::String: {
            std::string out = "'";
            for (char c : *s_) {
                if (c == '\'') out += "\\'";
                else if (c == '\\') out += "\\\\";
                else if (c == '\n') out += "\\n";
                else out.push_back(c);
            }
            return out + "'";
        }
        case Type::Array: {
            std::string out = "[";
            for (size_t i = 0; i < a_->size(); ++i) out += (i ? ", " : "") + (*a_)[i].repr();
            return out + "]";
        }
        case Type::Object: {
            std::string out = "{";
            bool first = true;
            for (const auto& [k, v] : *o_) {
                out += (first ? "'" : ", '") + k + "': " + v.repr();
                first = false;
            }
            return out + "}";
        }
        case Type::Callable: return "<function>";
        default: return to_string();
    }
}

static void json_string(std::string& out, const std::string& s) {
    out.push_back('"');
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
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    out.push_back('"');
}

std::string Value::to_json(int indent, int depth) const {
    return to_json_sep(indent, depth, indent >= 0 ? "," : ", ", ": ");
}

std::string Value::to_json_sep(int indent, int depth, const std::string& item_sep, const std::string& key_sep) const {
    std::string out;
    const std::string nl = indent >= 0 ? "\n" + std::string(static_cast<size_t>(indent * (depth + 1)), ' ') : "";
    const std::string nl_end = indent >= 0 ? "\n" + std::string(static_cast<size_t>(indent * depth), ' ') : "";
    switch (t_) {
        case Type::Undefined: case Type::None: return "null";
        case Type::Bool: return b_ ? "true" : "false";
        case Type::Int: return std::to_string(i_);
        case Type::Float: return std::isfinite(f_) ? float_repr(f_) : "null";
        case Type::String: json_string(out, *s_); return out;
        case Type::Array:
            if (a_->empty()) return "[]";
            out = "[";
            for (size_t i = 0; i < a_->size(); ++i) {
                if (i) out += item_sep;
                out += nl + (*a_)[i].to_json_sep(indent, depth + 1, item_sep, key_sep);
            }
            return out + nl_end + "]";
        case Type::Object: {
            if (o_->empty()) return "{}";
            out = "{";
            bool first = true;
            for (const auto& [k, v] : *o_) {
                if (!first) out += item_sep;
                first = false;
                out += nl;
                json_string(out, k);
                out += key_sep + v.to_json_sep(indent, depth + 1, item_sep, key_sep);
            }
            return out + nl_end + "}";
        }
        case Type::Callable: return "null";
    }
    return "null";
}

Value Value::get(const std::string& key) const {
    if (t_ != Type::Object) return Value();
    for (const auto& [k, v] : *o_) {
        if (k == key) return v;
    }
    return Value();
}

void Value::set(const std::string& key, Value v) const {
    if (t_ != Type::Object) throw Error("cannot set attribute on a non-object");
    for (auto& [k, existing] : *o_) {
        if (k == key) { existing = std::move(v); return; }
    }
    o_->emplace_back(key, std::move(v));
}

bool Value::operator==(const Value& o) const {
    if (is_number() && o.is_number()) {
        if (t_ == Type::Int && o.t_ == Type::Int) return i_ == o.i_;
        return as_float() == o.as_float();
    }
    if ((t_ == Type::Bool) != (o.t_ == Type::Bool)) {
        if (t_ == Type::Bool && o.is_number()) return as_int() == o.as_int() && o.as_float() == o.as_int();
        if (o.t_ == Type::Bool && is_number()) return o == *this;
        return false;
    }
    if (t_ != o.t_) return (is_undefined() || is_none()) && (o.is_undefined() || o.is_none()) && t_ == o.t_;
    switch (t_) {
        case Type::Undefined: case Type::None: return true;
        case Type::Bool: return b_ == o.b_;
        case Type::String: return *s_ == *o.s_;
        case Type::Array:
            if (a_->size() != o.a_->size()) return false;
            for (size_t i = 0; i < a_->size(); ++i) if ((*a_)[i] != (*o.a_)[i]) return false;
            return true;
        case Type::Object:
            if (o_->size() != o.o_->size()) return false;
            for (const auto& [k, v] : *o_) if (o.get(k) != v) return false;
            return true;
        case Type::Callable: return fn_ == o.fn_;
        default: return false;
    }
}

bool Value::less(const Value& o) const {
    if (is_number() && o.is_number()) return as_float() < o.as_float();
    if (t_ == Type::Bool && o.t_ == Type::Bool) return b_ < o.b_;
    if (is_string() && o.is_string()) return *s_ < *o.s_;
    if (is_array() && o.is_array()) {
        return std::lexicographical_compare(a_->begin(), a_->end(), o.a_->begin(), o.a_->end(),
                                            [](const Value& x, const Value& y) { return x.less(y); });
    }
    throw Error("'<' not supported between these types");
}

// ===========================================================================
// JSON parser
// ===========================================================================

namespace {

struct JsonParser {
    const std::string& s;
    size_t p = 0;

    void ws() { while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p; }
    [[noreturn]] void fail(const char* what) { throw Error(std::string("json: ") + what + " at " + std::to_string(p)); }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out.push_back(static_cast<char>(cp));
        else if (cp < 0x800) { out.push_back(static_cast<char>(0xC0 | (cp >> 6))); out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    uint32_t hex4() {
        if (p + 4 > s.size()) fail("bad \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = s[p++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else fail("bad hex digit");
        }
        return v;
    }

    std::string str() {
        if (s[p] != '"') fail("expected string");
        ++p;
        std::string out;
        while (p < s.size() && s[p] != '"') {
            char c = s[p++];
            if (c != '\\') { out.push_back(c); continue; }
            if (p >= s.size()) fail("bad escape");
            c = s[p++];
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && p + 1 < s.size() && s[p] == '\\' && s[p + 1] == 'u') {
                        p += 2;
                        const uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: out.push_back(c);
            }
        }
        if (p >= s.size()) fail("unterminated string");
        ++p;
        return out;
    }

    Value value() {
        ws();
        if (p >= s.size()) fail("unexpected end");
        const char c = s[p];
        if (c == '{') {
            ++p;
            Value obj = Value::object();
            ws();
            if (p < s.size() && s[p] == '}') { ++p; return obj; }
            for (;;) {
                ws();
                std::string k = str();
                ws();
                if (p >= s.size() || s[p] != ':') fail("expected ':'");
                ++p;
                obj.obj().emplace_back(std::move(k), value());
                ws();
                if (p < s.size() && s[p] == ',') { ++p; continue; }
                if (p < s.size() && s[p] == '}') { ++p; return obj; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p;
            Value arr = Value::array();
            ws();
            if (p < s.size() && s[p] == ']') { ++p; return arr; }
            for (;;) {
                arr.arr().push_back(value());
                ws();
                if (p < s.size() && s[p] == ',') { ++p; continue; }
                if (p < s.size() && s[p] == ']') { ++p; return arr; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') return Value(str());
        if (s.compare(p, 4, "true") == 0) { p += 4; return Value(true); }
        if (s.compare(p, 5, "false") == 0) { p += 5; return Value(false); }
        if (s.compare(p, 4, "null") == 0) { p += 4; return Value::none(); }
        const size_t start = p;
        bool is_float = false;
        while (p < s.size() && (std::isdigit(static_cast<unsigned char>(s[p])) || std::strchr("+-.eE", s[p]))) {
            if (std::strchr(".eE", s[p])) is_float = true;
            ++p;
        }
        if (p == start) fail("unexpected character");
        const std::string num = s.substr(start, p - start);
        return is_float ? Value(std::strtod(num.c_str(), nullptr)) : Value(static_cast<int64_t>(std::strtoll(num.c_str(), nullptr, 10)));
    }
};

} // namespace

Value parse_json(const std::string& text) {
    JsonParser jp{text};
    Value v = jp.value();
    jp.ws();
    if (jp.p != text.size()) jp.fail("trailing characters");
    return v;
}

// ===========================================================================
// Expressions
// ===========================================================================

struct Context;

struct Expr {
    virtual ~Expr() = default;
    virtual Value eval(Context& ctx) const = 0;
};
using ExprPtr = std::shared_ptr<Expr>;

struct Node {
    virtual ~Node() = default;
    virtual void render(Context& ctx, std::string& out) const = 0;
};
using NodePtr = std::shared_ptr<Node>;
using Body = std::vector<NodePtr>;

enum class Flow { Normal, Break, Continue };

struct Context {
    std::vector<std::unordered_map<std::string, Value>> scopes;
    Flow flow = Flow::Normal;
    int depth = 0;

    Value lookup(const std::string& name) const {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return f->second;
        }
        return Value();
    }
    void assign(const std::string& name, Value v) { scopes.back()[name] = std::move(v); }
};

static void render_body(const Body& body, Context& ctx, std::string& out) {
    for (const auto& n : body) {
        n->render(ctx, out);
        if (ctx.flow != Flow::Normal) return;
    }
}

struct ScopeGuard {
    Context& ctx;
    explicit ScopeGuard(Context& c) : ctx(c) {
        if (++ctx.depth > 200) throw Error("template recursion too deep");
        ctx.scopes.emplace_back();
    }
    ~ScopeGuard() { ctx.scopes.pop_back(); --ctx.depth; }
};

// ---- helpers --------------------------------------------------------------

static std::vector<Value> iterate(const Value& v) {
    switch (v.type()) {
        case Value::Type::Array: return v.arr();
        case Value::Type::Object: {
            std::vector<Value> keys;
            for (const auto& [k, _] : v.obj()) keys.emplace_back(k);
            return keys;
        }
        case Value::Type::String: {
            // UTF-8 aware: one item per code point.
            std::vector<Value> chars;
            const std::string& s = v.str();
            for (size_t i = 0; i < s.size();) {
                size_t n = 1;
                const unsigned char c = static_cast<unsigned char>(s[i]);
                if (c >= 0xF0) n = 4; else if (c >= 0xE0) n = 3; else if (c >= 0xC0) n = 2;
                chars.emplace_back(s.substr(i, n));
                i += n;
            }
            return chars;
        }
        case Value::Type::Undefined: case Value::Type::None: return {};
        default: throw Error("value is not iterable");
    }
}

static int64_t length_of(const Value& v) {
    switch (v.type()) {
        case Value::Type::String: return static_cast<int64_t>(iterate(v).size());
        case Value::Type::Array: return static_cast<int64_t>(v.arr().size());
        case Value::Type::Object: return static_cast<int64_t>(v.obj().size());
        case Value::Type::Undefined: return 0;
        default: throw Error("object has no length");
    }
}

static bool contains(const Value& container, const Value& item) {
    switch (container.type()) {
        case Value::Type::String:
            return item.is_string() && container.str().find(item.str()) != std::string::npos;
        case Value::Type::Array:
            for (const auto& x : container.arr()) if (x == item) return true;
            return false;
        case Value::Type::Object:
            return item.is_string() && !container.get(item.str()).is_undefined();
        case Value::Type::Undefined: case Value::Type::None: return false;
        default: throw Error("argument is not iterable");
    }
}

static int64_t norm_index(int64_t i, int64_t n) { return i < 0 ? i + n : i; }

static Value kwarg(const Kwargs& kw, const std::string& name, const Args& args, size_t pos, Value def = Value()) {
    for (const auto& [k, v] : kw) if (k == name) return v;
    if (pos < args.size()) return args[pos];
    return def;
}

static std::string strip_chars(const std::string& s, const Value& chars, bool left, bool right) {
    const std::string set = chars.is_string() ? chars.str() : std::string(" \t\n\r\f\v");
    size_t b = 0, e = s.size();
    if (left) while (b < e && set.find(s[b]) != std::string::npos) ++b;
    if (right) while (e > b && set.find(s[e - 1]) != std::string::npos) --e;
    return s.substr(b, e - b);
}

static std::string replace_all(std::string s, const std::string& from, const std::string& to, int64_t count = -1) {
    if (from.empty()) return s;
    size_t pos = 0;
    while (count != 0 && (pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
        if (count > 0) --count;
    }
    return s;
}

static std::string title_case(const std::string& s) {
    std::string out = s;
    bool start = true;
    for (char& c : out) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (std::isalpha(u)) { c = static_cast<char>(start ? std::toupper(u) : std::tolower(u)); start = false; }
        else start = !std::isalnum(u);
    }
    return out;
}

static Value call_value(const Value& fn, Args args, Kwargs kwargs) {
    if (!fn.is_callable()) throw Error("value is not callable");
    return fn.fn()(args, kwargs);
}

static bool apply_test(const std::string& name, const Value& v, Args& args, Context& ctx);
static Value apply_filter(const std::string& name, const Value& v, Args& args, Kwargs& kw, Context& ctx);

// Python-style % formatting, enough for '%s'/'%d' used in templates.
static std::string percent_format(const std::string& fmt, const Value& arg) {
    std::vector<Value> args = arg.is_array() ? arg.arr() : std::vector<Value>{arg};
    std::string out;
    size_t ai = 0;
    for (size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%' || i + 1 >= fmt.size()) { out.push_back(fmt[i]); continue; }
        const char spec = fmt[++i];
        if (spec == '%') { out.push_back('%'); continue; }
        if (ai >= args.size()) throw Error("not enough arguments for format string");
        const Value& a = args[ai++];
        if (spec == 'd' || spec == 'i') out += std::to_string(a.as_int());
        else if (spec == 'r') out += a.repr();
        else out += a.to_string();
    }
    return out;
}

static Value call_method(const Value& obj, const std::string& name, Args& args, Kwargs& kw) {
    if (obj.is_string()) {
        const std::string& s = obj.str();
        if (name == "strip") return Value(strip_chars(s, kwarg(kw, "chars", args, 0), true, true));
        if (name == "lstrip") return Value(strip_chars(s, kwarg(kw, "chars", args, 0), true, false));
        if (name == "rstrip") return Value(strip_chars(s, kwarg(kw, "chars", args, 0), false, true));
        if (name == "upper") { std::string r = s; for (char& c : r) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c))); return Value(r); }
        if (name == "lower") { std::string r = s; for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); return Value(r); }
        if (name == "title") return Value(title_case(s));
        if (name == "capitalize") {
            std::string r = s;
            for (char& c : r) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (!r.empty()) r[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(r[0])));
            return Value(r);
        }
        if (name == "startswith" || name == "endswith") {
            const Value a = kwarg(kw, "prefix", args, 0);
            std::vector<Value> cands = a.is_array() ? a.arr() : std::vector<Value>{a};
            for (const auto& c : cands) {
                if (!c.is_string()) continue;
                const std::string& x = c.str();
                if (x.size() > s.size()) continue;
                if (name == "startswith" ? s.compare(0, x.size(), x) == 0 : s.compare(s.size() - x.size(), x.size(), x) == 0) return Value(true);
            }
            return Value(false);
        }
        if (name == "split" || name == "rsplit") {
            const Value sep = kwarg(kw, "sep", args, 0);
            const int64_t maxsplit = kwarg(kw, "maxsplit", args, 1, Value(int64_t(-1))).as_int();
            Value out = Value::array();
            if (!sep.is_string()) {
                std::istringstream is(s);
                std::string w;
                while (is >> w) out.arr().emplace_back(w);
                return out;
            }
            const std::string& d = sep.str();
            if (d.empty()) throw Error("empty separator");
            if (name == "split") {
                size_t start = 0, pos;
                int64_t n = 0;
                while ((maxsplit < 0 || n < maxsplit) && (pos = s.find(d, start)) != std::string::npos) {
                    out.arr().emplace_back(s.substr(start, pos - start));
                    start = pos + d.size();
                    ++n;
                }
                out.arr().emplace_back(s.substr(start));
            } else {
                std::vector<std::string> parts;
                size_t end = s.size();
                int64_t n = 0;
                for (;;) {
                    if (maxsplit >= 0 && n >= maxsplit) break;
                    if (end < d.size()) break;
                    const size_t pos = s.rfind(d, end - d.size());
                    if (pos == std::string::npos) break;
                    parts.push_back(s.substr(pos + d.size(), end - pos - d.size()));
                    end = pos;
                    ++n;
                }
                parts.push_back(s.substr(0, end));
                for (auto it = parts.rbegin(); it != parts.rend(); ++it) out.arr().emplace_back(*it);
            }
            return out;
        }
        if (name == "replace") {
            const int64_t count = kwarg(kw, "count", args, 2, Value(int64_t(-1))).as_int();
            return Value(replace_all(s, kwarg(kw, "old", args, 0).to_string(), kwarg(kw, "new", args, 1).to_string(), count));
        }
        if (name == "find") {
            const size_t pos = s.find(kwarg(kw, "sub", args, 0).to_string());
            return Value(pos == std::string::npos ? int64_t(-1) : static_cast<int64_t>(pos));
        }
        if (name == "count") {
            const std::string sub = kwarg(kw, "sub", args, 0).to_string();
            int64_t n = 0;
            for (size_t pos = 0; !sub.empty() && (pos = s.find(sub, pos)) != std::string::npos; pos += sub.size()) ++n;
            return Value(n);
        }
        if (name == "join") {
            std::string out;
            bool first = true;
            for (const auto& x : iterate(kwarg(kw, "iterable", args, 0))) {
                if (!first) out += s;
                first = false;
                out += x.to_string();
            }
            return Value(out);
        }
        if (name == "format") {
            std::string out;
            size_t ai = 0;
            for (size_t i = 0; i < s.size(); ++i) {
                if (s[i] == '{' && i + 1 < s.size() && s[i + 1] == '{') { out.push_back('{'); ++i; continue; }
                if (s[i] == '}' && i + 1 < s.size() && s[i + 1] == '}') { out.push_back('}'); ++i; continue; }
                if (s[i] != '{') { out.push_back(s[i]); continue; }
                const size_t close = s.find('}', i);
                if (close == std::string::npos) throw Error("bad format string");
                const std::string field = s.substr(i + 1, close - i - 1);
                Value v;
                if (field.empty()) v = ai < args.size() ? args[ai++] : Value();
                else if (std::isdigit(static_cast<unsigned char>(field[0]))) { const size_t k = std::stoul(field); v = k < args.size() ? args[k] : Value(); }
                else v = kwarg(kw, field, {}, 0);
                out += v.to_string();
                i = close;
            }
            return Value(out);
        }
        if (name == "isdigit" || name == "isalpha" || name == "isspace" || name == "isalnum") {
            if (s.empty()) return Value(false);
            for (unsigned char c : s) {
                const bool ok = name == "isdigit" ? std::isdigit(c) : name == "isalpha" ? std::isalpha(c)
                              : name == "isspace" ? std::isspace(c) : std::isalnum(c);
                if (!ok) return Value(false);
            }
            return Value(true);
        }
    } else if (obj.is_object()) {
        if (name == "items") {
            Value out = Value::array();
            for (const auto& [k, v] : obj.obj()) out.arr().push_back(Value::array({Value(k), v}));
            return out;
        }
        if (name == "keys") { Value out = Value::array(); for (const auto& [k, _] : obj.obj()) out.arr().emplace_back(k); return out; }
        if (name == "values") { Value out = Value::array(); for (const auto& [_, v] : obj.obj()) out.arr().push_back(v); return out; }
        if (name == "get") {
            const Value v = obj.get(kwarg(kw, "key", args, 0).to_string());
            return v.is_undefined() ? kwarg(kw, "default", args, 1, Value::none()) : v;
        }
        if (name == "update") {
            const Value other = kwarg(kw, "other", args, 0);
            if (other.is_object()) for (const auto& [k, v] : other.obj()) obj.set(k, v);
            for (const auto& [k, v] : kw) obj.set(k, v);
            return Value::none();
        }
        if (name == "pop") {
            auto& o = obj.obj();
            const std::string key = kwarg(kw, "key", args, 0).to_string();
            for (auto it = o.begin(); it != o.end(); ++it) {
                if (it->first == key) { Value v = it->second; o.erase(it); return v; }
            }
            return kwarg(kw, "default", args, 1, Value::none());
        }
    } else if (obj.is_array()) {
        auto& a = obj.arr();
        if (name == "append") { a.push_back(kwarg(kw, "item", args, 0)); return Value::none(); }
        if (name == "extend") { for (const auto& x : iterate(kwarg(kw, "items", args, 0))) a.push_back(x); return Value::none(); }
        if (name == "pop") {
            if (a.empty()) throw Error("pop from empty list");
            const int64_t i = norm_index(kwarg(kw, "index", args, 0, Value(int64_t(-1))).as_int(), static_cast<int64_t>(a.size()));
            if (i < 0 || i >= static_cast<int64_t>(a.size())) throw Error("pop index out of range");
            Value v = a[static_cast<size_t>(i)];
            a.erase(a.begin() + i);
            return v;
        }
        if (name == "index") {
            const Value x = kwarg(kw, "value", args, 0);
            for (size_t i = 0; i < a.size(); ++i) if (a[i] == x) return Value(static_cast<int64_t>(i));
            throw Error("value not in list");
        }
        if (name == "count") {
            const Value x = kwarg(kw, "value", args, 0);
            int64_t n = 0;
            for (const auto& y : a) n += y == x;
            return Value(n);
        }
        if (name == "insert") {
            const int64_t i = std::clamp<int64_t>(norm_index(kwarg(kw, "index", args, 0).as_int(), static_cast<int64_t>(a.size())), 0, static_cast<int64_t>(a.size()));
            a.insert(a.begin() + i, kwarg(kw, "item", args, 1));
            return Value::none();
        }
    }
    throw Error("unknown method '" + name + "'");
}

// ---- expression nodes ---------------------------------------------------

struct Literal : Expr {
    Value v;
    explicit Literal(Value x) : v(std::move(x)) {}
    Value eval(Context&) const override { return v; }
};

struct Name : Expr {
    std::string name;
    explicit Name(std::string n) : name(std::move(n)) {}
    Value eval(Context& ctx) const override { return ctx.lookup(name); }
};

struct ListLit : Expr {
    std::vector<ExprPtr> items;
    Value eval(Context& ctx) const override {
        Value out = Value::array();
        for (const auto& e : items) out.arr().push_back(e->eval(ctx));
        return out;
    }
};

struct DictLit : Expr {
    std::vector<std::pair<ExprPtr, ExprPtr>> items;
    Value eval(Context& ctx) const override {
        Value out = Value::object();
        for (const auto& [k, v] : items) out.set(k->eval(ctx).to_string(), v->eval(ctx));
        return out;
    }
};

struct Attr : Expr {
    ExprPtr obj;
    std::string name;
    Value eval(Context& ctx) const override {
        const Value o = obj->eval(ctx);
        return o.is_object() ? o.get(name) : Value();
    }
};

static Value subscript(const Value& o, const Value& key) {
    if (o.is_object()) return o.get(key.to_string());
    if (o.is_array()) {
        if (!key.is_number()) return Value();
        const int64_t n = static_cast<int64_t>(o.arr().size());
        const int64_t i = norm_index(key.as_int(), n);
        return i >= 0 && i < n ? o.arr()[static_cast<size_t>(i)] : Value();
    }
    if (o.is_string()) {
        const std::vector<Value> chars = iterate(o);
        if (!key.is_number()) return Value();
        const int64_t n = static_cast<int64_t>(chars.size());
        const int64_t i = norm_index(key.as_int(), n);
        return i >= 0 && i < n ? chars[static_cast<size_t>(i)] : Value();
    }
    return Value();
}

struct Subscript : Expr {
    ExprPtr obj, key;
    Value eval(Context& ctx) const override { return subscript(obj->eval(ctx), key->eval(ctx)); }
};

struct Slice : Expr {
    ExprPtr obj, start, stop, step;
    Value eval(Context& ctx) const override {
        const Value o = obj->eval(ctx);
        const bool is_str = o.is_string();
        std::vector<Value> items = is_str ? iterate(o) : o.is_array() ? o.arr() : std::vector<Value>{};
        if (!is_str && !o.is_array()) return Value();
        const int64_t n = static_cast<int64_t>(items.size());
        const int64_t st = step ? step->eval(ctx).as_int() : 1;
        if (st == 0) throw Error("slice step cannot be zero");
        auto bound = [&](const ExprPtr& e, int64_t def) {
            if (!e) return def;
            const Value v = e->eval(ctx);
            if (v.is_none()) return def;
            int64_t i = v.as_int();
            if (i < 0) i += n;
            return st > 0 ? std::clamp<int64_t>(i, 0, n) : std::clamp<int64_t>(i, -1, n - 1);
        };
        const int64_t b = bound(start, st > 0 ? 0 : n - 1);
        const int64_t e = bound(stop, st > 0 ? n : -1);
        std::vector<Value> out;
        for (int64_t i = b; st > 0 ? i < e : i > e; i += st) out.push_back(items[static_cast<size_t>(i)]);
        if (is_str) {
            std::string s;
            for (const auto& c : out) s += c.str();
            return Value(s);
        }
        return Value::array(std::move(out));
    }
};

struct CallArgs {
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, ExprPtr>> kwargs;
    void eval(Context& ctx, Args& a, Kwargs& kw) const {
        for (const auto& e : args) a.push_back(e->eval(ctx));
        for (const auto& [k, e] : kwargs) kw.emplace_back(k, e->eval(ctx));
    }
};

struct Call : Expr {
    ExprPtr callee;
    CallArgs cargs;
    Value eval(Context& ctx) const override {
        Args a;
        Kwargs kw;
        if (auto attr = std::dynamic_pointer_cast<Attr>(callee)) {
            const Value o = attr->obj->eval(ctx);
            const Value member = o.is_object() ? o.get(attr->name) : Value();
            cargs.eval(ctx, a, kw);
            if (member.is_callable()) return call_value(member, std::move(a), std::move(kw));
            return call_method(o, attr->name, a, kw);
        }
        const Value fn = callee->eval(ctx);
        cargs.eval(ctx, a, kw);
        if (fn.is_undefined()) throw Error("call of an undefined function");
        return call_value(fn, std::move(a), std::move(kw));
    }
};

struct FilterExpr : Expr {
    ExprPtr obj;
    std::string name;
    CallArgs cargs;
    Value eval(Context& ctx) const override {
        const Value v = obj->eval(ctx);
        Args a;
        Kwargs kw;
        cargs.eval(ctx, a, kw);
        return apply_filter(name, v, a, kw, ctx);
    }
};

struct TestExpr : Expr {
    ExprPtr obj;
    std::string name;
    std::vector<ExprPtr> args;
    bool negate = false;
    Value eval(Context& ctx) const override {
        const Value v = obj->eval(ctx);
        Args a;
        for (const auto& e : args) a.push_back(e->eval(ctx));
        return Value(apply_test(name, v, a, ctx) != negate);
    }
};

struct Unary : Expr {
    char op;
    ExprPtr e;
    Value eval(Context& ctx) const override {
        const Value v = e->eval(ctx);
        if (op == '!') return Value(!v.truthy());
        if (op == '-') {
            if (v.type() == Value::Type::Float) return Value(-v.as_float());
            if (v.is_number() || v.type() == Value::Type::Bool) return Value(-v.as_int());
            throw Error("bad operand for unary -");
        }
        return v;
    }
};

static Value arith(const std::string& op, const Value& l, const Value& r) {
    if (op == "+") {
        if (l.is_string() && r.is_string()) return Value(l.str() + r.str());
        if (l.is_array() && r.is_array()) { Array a = l.arr(); a.insert(a.end(), r.arr().begin(), r.arr().end()); return Value::array(std::move(a)); }
    }
    if (op == "*") {
        if (l.is_string() && r.is_number()) { std::string s; for (int64_t i = 0; i < r.as_int(); ++i) s += l.str(); return Value(s); }
        if (l.is_array() && r.is_number()) { Array a; for (int64_t i = 0; i < r.as_int(); ++i) a.insert(a.end(), l.arr().begin(), l.arr().end()); return Value::array(std::move(a)); }
    }
    if (op == "%" && l.is_string()) return Value(percent_format(l.str(), r));
    const bool num_l = l.is_number() || l.type() == Value::Type::Bool;
    const bool num_r = r.is_number() || r.type() == Value::Type::Bool;
    if (!num_l || !num_r) throw Error("unsupported operand types for " + op);
    const bool ints = l.type() != Value::Type::Float && r.type() != Value::Type::Float;
    if (op == "/") {
        if (r.as_float() == 0) throw Error("division by zero");
        return Value(l.as_float() / r.as_float());
    }
    if (op == "//") {
        if (r.as_float() == 0) throw Error("division by zero");
        if (ints) { int64_t q = l.as_int() / r.as_int(); if ((l.as_int() % r.as_int() != 0) && ((l.as_int() < 0) != (r.as_int() < 0))) --q; return Value(q); }
        return Value(std::floor(l.as_float() / r.as_float()));
    }
    if (op == "%") {
        if (r.as_float() == 0) throw Error("modulo by zero");
        if (ints) { int64_t m = l.as_int() % r.as_int(); if (m != 0 && ((m < 0) != (r.as_int() < 0))) m += r.as_int(); return Value(m); }
        double m = std::fmod(l.as_float(), r.as_float());
        if (m != 0 && ((m < 0) != (r.as_float() < 0))) m += r.as_float();
        return Value(m);
    }
    if (op == "**") {
        if (ints && r.as_int() >= 0) { int64_t p = 1; for (int64_t i = 0; i < r.as_int(); ++i) p *= l.as_int(); return Value(p); }
        return Value(std::pow(l.as_float(), r.as_float()));
    }
    if (ints) {
        if (op == "+") return Value(l.as_int() + r.as_int());
        if (op == "-") return Value(l.as_int() - r.as_int());
        if (op == "*") return Value(l.as_int() * r.as_int());
    } else {
        if (op == "+") return Value(l.as_float() + r.as_float());
        if (op == "-") return Value(l.as_float() - r.as_float());
        if (op == "*") return Value(l.as_float() * r.as_float());
    }
    throw Error("unknown operator " + op);
}

struct Binary : Expr {
    std::string op;
    ExprPtr l, r;
    Value eval(Context& ctx) const override {
        if (op == "and") { const Value a = l->eval(ctx); return a.truthy() ? r->eval(ctx) : a; }
        if (op == "or") { const Value a = l->eval(ctx); return a.truthy() ? a : r->eval(ctx); }
        const Value a = l->eval(ctx);
        const Value b = r->eval(ctx);
        if (op == "~") return Value(a.to_string() + b.to_string());
        if (op == "==") return Value(a == b);
        if (op == "!=") return Value(a != b);
        if (op == "<") return Value(a.less(b));
        if (op == ">") return Value(b.less(a));
        if (op == "<=") return Value(!b.less(a));
        if (op == ">=") return Value(!a.less(b));
        if (op == "in") return Value(contains(b, a));
        if (op == "not in") return Value(!contains(b, a));
        return arith(op, a, b);
    }
};

struct Ternary : Expr {
    ExprPtr cond, yes, no;
    Value eval(Context& ctx) const override {
        if (cond->eval(ctx).truthy()) return yes->eval(ctx);
        return no ? no->eval(ctx) : Value();
    }
};

// ---- tests and filters ----------------------------------------------------

static bool apply_test(const std::string& name, const Value& v, Args& args, Context& ctx) {
    (void) ctx;
    auto arg = [&](size_t i) { if (i >= args.size()) throw Error("test '" + name + "' needs an argument"); return args[i]; };
    if (name == "defined") return !v.is_undefined();
    if (name == "undefined") return v.is_undefined();
    if (name == "none") return v.is_none();
    if (name == "string") return v.is_string();
    if (name == "number") return v.is_number();
    if (name == "integer") return v.type() == Value::Type::Int;
    if (name == "float") return v.type() == Value::Type::Float;
    if (name == "boolean") return v.type() == Value::Type::Bool;
    if (name == "true") return v.type() == Value::Type::Bool && v.as_bool();
    if (name == "false") return v.type() == Value::Type::Bool && !v.as_bool();
    if (name == "mapping") return v.is_object();
    if (name == "sequence") return v.is_array() || v.is_string() || v.is_object();
    if (name == "iterable") return v.is_array() || v.is_string() || v.is_object();
    if (name == "callable") return v.is_callable();
    if (name == "odd") return v.as_int() % 2 != 0;
    if (name == "even") return v.as_int() % 2 == 0;
    if (name == "divisibleby") return arg(0).as_int() != 0 && v.as_int() % arg(0).as_int() == 0;
    if (name == "equalto" || name == "eq" || name == "==" || name == "sameas") return v == arg(0);
    if (name == "ne" || name == "!=") return v != arg(0);
    if (name == "lt" || name == "lessthan" || name == "<") return v.less(arg(0));
    if (name == "gt" || name == "greaterthan" || name == ">") return arg(0).less(v);
    if (name == "le" || name == "<=") return !arg(0).less(v);
    if (name == "ge" || name == ">=") return !v.less(arg(0));
    if (name == "in") return contains(arg(0), v);
    if (name == "lower") { if (!v.is_string()) return false; for (unsigned char c : v.str()) if (std::isupper(c)) return false; return true; }
    if (name == "upper") { if (!v.is_string()) return false; for (unsigned char c : v.str()) if (std::islower(c)) return false; return true; }
    throw Error("unknown test '" + name + "'");
}

static Value attribute_path(const Value& v, const std::string& path) {
    Value cur = v;
    size_t start = 0;
    while (start <= path.size()) {
        const size_t dot = path.find('.', start);
        const std::string part = path.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
        if (cur.is_array() && !part.empty() && std::isdigit(static_cast<unsigned char>(part[0]))) cur = subscript(cur, Value(static_cast<int64_t>(std::stoll(part))));
        else cur = cur.get(part);
        if (dot == std::string::npos) break;
        start = dot + 1;
    }
    return cur;
}

static Value apply_filter(const std::string& name, const Value& v, Args& args, Kwargs& kw, Context& ctx) {
    if (name == "length" || name == "count") return Value(length_of(v));
    if (name == "string") return Value(v.to_string());
    if (name == "safe" || name == "e" || name == "escape" || name == "forceescape") return name == "safe" ? v : Value(v.to_string());
    if (name == "trim") return Value(strip_chars(v.to_string(), kwarg(kw, "chars", args, 0), true, true));
    if (name == "upper" || name == "lower" || name == "title" || name == "capitalize") {
        Args none; Kwargs nkw;
        return call_method(Value(v.to_string()), name, none, nkw);
    }
    if (name == "default" || name == "d") {
        const Value def = kwarg(kw, "default_value", args, 0, Value(""));
        const bool boolean = kwarg(kw, "boolean", args, 1, Value(false)).truthy();
        return (v.is_undefined() || (boolean && !v.truthy())) ? def : v;
    }
    if (name == "join") {
        const std::string sep = kwarg(kw, "d", args, 0, Value("")).to_string();
        const Value attr = kwarg(kw, "attribute", args, 1);
        std::string out;
        bool first = true;
        for (const auto& x : iterate(v)) {
            if (!first) out += sep;
            first = false;
            out += (attr.is_undefined() ? x : attribute_path(x, attr.to_string())).to_string();
        }
        return Value(out);
    }
    if (name == "first") { const auto items = iterate(v); return items.empty() ? Value() : items.front(); }
    if (name == "last") { const auto items = iterate(v); return items.empty() ? Value() : items.back(); }
    if (name == "list") return Value::array(iterate(v));
    if (name == "reverse") {
        if (v.is_string()) { auto c = iterate(v); std::string s; for (auto it = c.rbegin(); it != c.rend(); ++it) s += it->str(); return Value(s); }
        auto items = iterate(v);
        std::reverse(items.begin(), items.end());
        return Value::array(std::move(items));
    }
    if (name == "int") {
        if (v.is_string()) { try { return Value(static_cast<int64_t>(std::stoll(v.str()))); } catch (...) { return kwarg(kw, "default", args, 0, Value(int64_t(0))); } }
        if (v.is_number() || v.type() == Value::Type::Bool) return Value(v.as_int());
        return kwarg(kw, "default", args, 0, Value(int64_t(0)));
    }
    if (name == "float") {
        if (v.is_string()) { try { return Value(std::stod(v.str())); } catch (...) { return kwarg(kw, "default", args, 0, Value(0.0)); } }
        if (v.is_number() || v.type() == Value::Type::Bool) return Value(v.as_float());
        return kwarg(kw, "default", args, 0, Value(0.0));
    }
    if (name == "abs") return v.type() == Value::Type::Float ? Value(std::fabs(v.as_float())) : Value(std::llabs(v.as_int()));
    if (name == "round") {
        const int64_t prec = kwarg(kw, "precision", args, 0, Value(int64_t(0))).as_int();
        const double m = std::pow(10.0, static_cast<double>(prec));
        return Value(std::round(v.as_float() * m) / m);
    }
    if (name == "replace") {
        const int64_t count = kwarg(kw, "count", args, 2, Value(int64_t(-1))).as_int();
        return Value(replace_all(v.to_string(), kwarg(kw, "old", args, 0).to_string(), kwarg(kw, "new", args, 1).to_string(), count));
    }
    if (name == "tojson") {
        const Value indent = kwarg(kw, "indent", args, 0);
        const int ind = indent.is_number() ? static_cast<int>(indent.as_int()) : -1;
        const Value seps = kwarg(kw, "separators", args, 1);
        if (seps.is_array() && seps.arr().size() == 2) {
            return Value(v.to_json_sep(ind, 0, seps.arr()[0].to_string(), seps.arr()[1].to_string()));
        }
        return Value(v.to_json(ind));
    }
    if (name == "items") {
        Args none; Kwargs nkw;
        if (v.is_undefined()) return Value::array();
        return call_method(v, "items", none, nkw);
    }
    if (name == "dictsort") {
        if (!v.is_object()) throw Error("dictsort needs a mapping");
        Object o = v.obj();
        std::stable_sort(o.begin(), o.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        Value out = Value::array();
        for (auto& [k, x] : o) out.arr().push_back(Value::array({Value(k), x}));
        return out;
    }
    if (name == "unique") {
        Value out = Value::array();
        for (const auto& x : iterate(v)) if (!contains(out, x)) out.arr().push_back(x);
        return out;
    }
    if (name == "sort") {
        auto items = iterate(v);
        const bool rev = kwarg(kw, "reverse", args, 0, Value(false)).truthy();
        const Value attr = kwarg(kw, "attribute", args, 2);
        std::stable_sort(items.begin(), items.end(), [&](const Value& a, const Value& b) {
            const Value x = attr.is_undefined() ? a : attribute_path(a, attr.to_string());
            const Value y = attr.is_undefined() ? b : attribute_path(b, attr.to_string());
            return rev ? y.less(x) : x.less(y);
        });
        return Value::array(std::move(items));
    }
    if (name == "map") {
        Value out = Value::array();
        const Value attr = kwarg(kw, "attribute", {}, 0);
        if (!attr.is_undefined()) {
            const Value def = kwarg(kw, "default", {}, 0);
            for (const auto& x : iterate(v)) {
                Value y = attribute_path(x, attr.to_string());
                out.arr().push_back(y.is_undefined() && !def.is_undefined() ? def : y);
            }
            return out;
        }
        if (args.empty()) throw Error("map needs a filter name or attribute");
        const std::string fname = args[0].to_string();
        Args rest(args.begin() + 1, args.end());
        for (const auto& x : iterate(v)) {
            Args a = rest; Kwargs k;
            out.arr().push_back(apply_filter(fname, x, a, k, ctx));
        }
        return out;
    }
    if (name == "select" || name == "reject") {
        Value out = Value::array();
        for (const auto& x : iterate(v)) {
            bool ok;
            if (args.empty()) ok = x.truthy();
            else { Args a(args.begin() + 1, args.end()); ok = apply_test(args[0].to_string(), x, a, ctx); }
            if (ok == (name == "select")) out.arr().push_back(x);
        }
        return out;
    }
    if (name == "selectattr" || name == "rejectattr") {
        if (args.empty()) throw Error(name + " needs an attribute");
        Value out = Value::array();
        const std::string attr = args[0].to_string();
        for (const auto& x : iterate(v)) {
            const Value y = attribute_path(x, attr);
            bool ok;
            if (args.size() < 2) ok = y.truthy();
            else { Args a(args.begin() + 2, args.end()); ok = apply_test(args[1].to_string(), y, a, ctx); }
            if (ok == (name == "selectattr")) out.arr().push_back(x);
        }
        return out;
    }
    if (name == "indent") {
        const Value w = kwarg(kw, "width", args, 0, Value(int64_t(4)));
        const std::string pad = w.is_string() ? w.str() : std::string(static_cast<size_t>(w.as_int()), ' ');
        const bool first = kwarg(kw, "first", args, 1, Value(false)).truthy();
        const bool blank = kwarg(kw, "blank", args, 2, Value(false)).truthy();
        const std::string s = v.to_string();
        std::string out;
        size_t start = 0;
        bool is_first = true;
        while (start <= s.size()) {
            const size_t nl = s.find('\n', start);
            const std::string line = s.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
            if ((!is_first || first) && (blank || !line.empty())) out += pad;
            out += line;
            if (nl == std::string::npos) break;
            out += "\n";
            start = nl + 1;
            is_first = false;
        }
        return Value(out);
    }
    if (name == "wordcount") {
        std::istringstream is(v.to_string());
        std::string w;
        int64_t n = 0;
        while (is >> w) ++n;
        return Value(n);
    }
    if (name == "batch") {
        const int64_t size = std::max<int64_t>(1, kwarg(kw, "linecount", args, 0).as_int());
        Value out = Value::array();
        for (const auto& x : iterate(v)) {
            if (out.arr().empty() || static_cast<int64_t>(out.arr().back().arr().size()) >= size) out.arr().push_back(Value::array());
            out.arr().back().arr().push_back(x);
        }
        return out;
    }
    if (name == "attr") return v.get(kwarg(kw, "name", args, 0).to_string());
    if (name == "sum") {
        Value total(int64_t(0));
        const Value attr = kwarg(kw, "attribute", args, 0);
        for (const auto& x : iterate(v)) total = arith("+", total, attr.is_undefined() ? x : attribute_path(x, attr.to_string()));
        return total;
    }
    if (name == "min" || name == "max") {
        const auto items = iterate(v);
        if (items.empty()) return Value();
        Value best = items[0];
        for (const auto& x : items) if (name == "min" ? x.less(best) : best.less(x)) best = x;
        return best;
    }
    throw Error("unknown filter '" + name + "'");
}

// ===========================================================================
// Expression lexer / parser
// ===========================================================================

namespace {

struct Tok {
    enum Kind { End, Name, Int, Float, Str, Op } kind = End;
    std::string text;
};

std::vector<Tok> lex_expr(const std::string& s) {
    std::vector<Tok> out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isspace(c)) { ++i; continue; }
        if (std::isalpha(c) || c == '_') {
            size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
            out.push_back({Tok::Name, s.substr(i, j - i)});
            i = j;
            continue;
        }
        if (std::isdigit(c)) {
            size_t j = i;
            bool is_float = false;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
            if (j + 1 < s.size() && s[j] == '.' && std::isdigit(static_cast<unsigned char>(s[j + 1]))) {
                is_float = true;
                ++j;
                while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
            }
            if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
                size_t k = j + 1;
                if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
                if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
                    is_float = true;
                    j = k;
                    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
                }
            }
            std::string num = s.substr(i, j - i);
            num.erase(std::remove(num.begin(), num.end(), '_'), num.end());
            out.push_back({is_float ? Tok::Float : Tok::Int, num});
            i = j;
            continue;
        }
        if (c == '"' || c == '\'') {
            const char q = static_cast<char>(c);
            std::string str;
            size_t j = i + 1;
            while (j < s.size() && s[j] != q) {
                if (s[j] == '\\' && j + 1 < s.size()) {
                    const char e = s[++j];
                    switch (e) {
                        case 'n': str.push_back('\n'); break;
                        case 't': str.push_back('\t'); break;
                        case 'r': str.push_back('\r'); break;
                        case '\\': str.push_back('\\'); break;
                        case '\'': str.push_back('\''); break;
                        case '"': str.push_back('"'); break;
                        case 'u': {
                            if (j + 4 < s.size()) {
                                const uint32_t cp = static_cast<uint32_t>(std::stoul(s.substr(j + 1, 4), nullptr, 16));
                                JsonParser::put_utf8(str, cp);
                                j += 4;
                            }
                            break;
                        }
                        default: str.push_back('\\'); str.push_back(e);
                    }
                    ++j;
                    continue;
                }
                str.push_back(s[j++]);
            }
            if (j >= s.size()) throw Error("unterminated string literal");
            out.push_back({Tok::Str, str});
            i = j + 1;
            continue;
        }
        static const char* ops[] = {"//", "**", "==", "!=", "<=", ">=", "(", ")", "[", "]", "{", "}", ",", ":", ".",
                                    "|", "~", "+", "-", "*", "/", "%", "<", ">", "=", "!"};
        bool matched = false;
        for (const char* op : ops) {
            const size_t n = std::strlen(op);
            if (s.compare(i, n, op) == 0) {
                out.push_back({Tok::Op, op});
                i += n;
                matched = true;
                break;
            }
        }
        if (!matched) throw Error(std::string("unexpected character '") + static_cast<char>(c) + "' in expression");
    }
    out.push_back({Tok::End, ""});
    return out;
}

class ExprParser {
public:
    explicit ExprParser(std::vector<Tok> toks) : t_(std::move(toks)) {}

    bool at_end() const { return t_[p_].kind == Tok::End; }
    const Tok& peek(size_t k = 0) const { return t_[std::min(p_ + k, t_.size() - 1)]; }
    bool is_op(const char* op, size_t k = 0) const { return peek(k).kind == Tok::Op && peek(k).text == op; }
    bool is_name(const char* n, size_t k = 0) const { return peek(k).kind == Tok::Name && peek(k).text == n; }
    void expect_op(const char* op) {
        if (!is_op(op)) throw Error(std::string("expected '") + op + "' but found '" + peek().text + "'");
        ++p_;
    }
    std::string expect_name() {
        if (peek().kind != Tok::Name) throw Error("expected a name but found '" + peek().text + "'");
        return t_[p_++].text;
    }
    bool accept_name(const char* n) { if (is_name(n)) { ++p_; return true; } return false; }
    bool accept_op(const char* op) { if (is_op(op)) { ++p_; return true; } return false; }

    // Full expression, including a bare tuple "a, b" when allowed.
    ExprPtr parse_tuple_or_expr() {
        ExprPtr first = parse_expr();
        if (!is_op(",")) return first;
        auto list = std::make_shared<ListLit>();
        list->items.push_back(first);
        while (accept_op(",")) {
            if (at_end() || is_op(")") || is_op("]")) break;
            list->items.push_back(parse_expr());
        }
        return list;
    }

    // A for-loop iterable: no ternary, since "for x in xs if cond" uses the
    // trailing "if" as the loop filter.
    ExprPtr parse_iterable() { return parse_or(); }

    ExprPtr parse_expr() {
        ExprPtr e = parse_or();
        if (accept_name("if")) {
            auto t = std::make_shared<Ternary>();
            t->yes = e;
            t->cond = parse_or();
            if (accept_name("else")) t->no = parse_expr();
            return t;
        }
        return e;
    }

private:
    ExprPtr bin(std::string op, ExprPtr l, ExprPtr r) {
        auto b = std::make_shared<Binary>();
        b->op = std::move(op);
        b->l = std::move(l);
        b->r = std::move(r);
        return b;
    }

    ExprPtr parse_or() {
        ExprPtr l = parse_and();
        while (accept_name("or")) l = bin("or", l, parse_and());
        return l;
    }
    ExprPtr parse_and() {
        ExprPtr l = parse_not();
        while (accept_name("and")) l = bin("and", l, parse_not());
        return l;
    }
    ExprPtr parse_not() {
        if (is_name("not") && !is_name("in", 1)) {
            ++p_;
            auto u = std::make_shared<Unary>();
            u->op = '!';
            u->e = parse_not();
            return u;
        }
        return parse_compare();
    }
    ExprPtr parse_compare() {
        ExprPtr l = parse_math1();
        for (;;) {
            static const char* cmp[] = {"==", "!=", "<=", ">=", "<", ">"};
            bool done = true;
            for (const char* op : cmp) {
                if (is_op(op)) { ++p_; l = bin(op, l, parse_math1()); done = false; break; }
            }
            if (!done) continue;
            if (accept_name("in")) { l = bin("in", l, parse_math1()); continue; }
            if (is_name("not") && is_name("in", 1)) { p_ += 2; l = bin("not in", l, parse_math1()); continue; }
            break;
        }
        return l;
    }
    ExprPtr parse_math1() {
        ExprPtr l = parse_concat();
        while (is_op("+") || is_op("-")) {
            const std::string op = t_[p_++].text;
            l = bin(op, l, parse_concat());
        }
        return l;
    }
    ExprPtr parse_concat() {
        ExprPtr l = parse_math2();
        while (accept_op("~")) l = bin("~", l, parse_math2());
        return l;
    }
    ExprPtr parse_math2() {
        ExprPtr l = parse_pow();
        while (is_op("*") || is_op("/") || is_op("//") || is_op("%")) {
            const std::string op = t_[p_++].text;
            l = bin(op, l, parse_pow());
        }
        return l;
    }
    ExprPtr parse_pow() {
        ExprPtr l = parse_unary();
        while (accept_op("**")) l = bin("**", l, parse_unary());
        return l;
    }
    ExprPtr parse_unary() {
        if (is_op("-") || is_op("+")) {
            const char op = t_[p_++].text[0];
            auto u = std::make_shared<Unary>();
            u->op = op;
            u->e = parse_unary();
            return u;
        }
        ExprPtr e = parse_postfix(parse_primary());
        return parse_filters(e);
    }

    void parse_call_args(CallArgs& ca) {
        expect_op("(");
        while (!is_op(")")) {
            if (peek().kind == Tok::Name && is_op("=", 1)) {
                std::string k = t_[p_].text;
                p_ += 2;
                ca.kwargs.emplace_back(std::move(k), parse_expr());
            } else {
                ca.args.push_back(parse_expr());
            }
            if (!accept_op(",")) break;
        }
        expect_op(")");
    }

    ExprPtr parse_filters(ExprPtr e) {
        for (;;) {
            if (accept_op("|")) {
                auto f = std::make_shared<FilterExpr>();
                f->obj = e;
                f->name = expect_name();
                if (is_op("(")) parse_call_args(f->cargs);
                e = parse_postfix(f);
                continue;
            }
            if (is_name("is")) {
                ++p_;
                auto t = std::make_shared<TestExpr>();
                t->obj = e;
                t->negate = accept_name("not");
                if (peek().kind == Tok::Name) {
                    t->name = expect_name();
                } else if (peek().kind == Tok::Op) {
                    t->name = t_[p_++].text;  // "is == x" style
                } else {
                    throw Error("expected a test name");
                }
                if (is_op("(")) {
                    CallArgs ca;
                    parse_call_args(ca);
                    t->args = ca.args;
                } else if (!at_end() && (peek().kind == Tok::Str || peek().kind == Tok::Int || peek().kind == Tok::Float ||
                                         (peek().kind == Tok::Name && !is_name("and") && !is_name("or") && !is_name("else") &&
                                          !is_name("if") && !is_name("is") && !is_name("in") && !is_name("not")))) {
                    // "x is divisibleby 3", "x is sameas none"
                    t->args.push_back(parse_primary());
                }
                e = t;
                continue;
            }
            return e;
        }
    }

    ExprPtr parse_postfix(ExprPtr e) {
        for (;;) {
            if (accept_op(".")) {
                auto a = std::make_shared<Attr>();
                a->obj = e;
                if (peek().kind == Tok::Int) {
                    auto s = std::make_shared<Subscript>();
                    s->obj = e;
                    s->key = std::make_shared<Literal>(Value(static_cast<int64_t>(std::stoll(t_[p_++].text))));
                    e = s;
                    continue;
                }
                a->name = expect_name();
                e = a;
                continue;
            }
            if (is_op("[")) {
                ++p_;
                ExprPtr start, stop, step;
                bool is_slice = false;
                if (!is_op(":")) start = parse_expr();
                if (accept_op(":")) {
                    is_slice = true;
                    if (!is_op(":") && !is_op("]")) stop = parse_expr();
                    if (accept_op(":") && !is_op("]")) step = parse_expr();
                }
                expect_op("]");
                if (is_slice) {
                    auto s = std::make_shared<Slice>();
                    s->obj = e; s->start = start; s->stop = stop; s->step = step;
                    e = s;
                } else {
                    auto s = std::make_shared<Subscript>();
                    s->obj = e; s->key = start;
                    e = s;
                }
                continue;
            }
            if (is_op("(")) {
                auto c = std::make_shared<Call>();
                c->callee = e;
                parse_call_args(c->cargs);
                e = c;
                continue;
            }
            return e;
        }
    }

    ExprPtr parse_primary() {
        const Tok& t = peek();
        switch (t.kind) {
            case Tok::Int: ++p_; return std::make_shared<Literal>(Value(static_cast<int64_t>(std::stoll(t.text))));
            case Tok::Float: ++p_; return std::make_shared<Literal>(Value(std::stod(t.text)));
            case Tok::Str: {
                std::string s = t.text;
                ++p_;
                while (peek().kind == Tok::Str) s += t_[p_++].text;  // implicit concatenation
                return std::make_shared<Literal>(Value(s));
            }
            case Tok::Name: {
                ++p_;
                if (t.text == "true" || t.text == "True") return std::make_shared<Literal>(Value(true));
                if (t.text == "false" || t.text == "False") return std::make_shared<Literal>(Value(false));
                if (t.text == "none" || t.text == "None") return std::make_shared<Literal>(Value::none());
                return std::make_shared<Name>(t.text);
            }
            case Tok::Op:
                if (t.text == "(") {
                    ++p_;
                    if (accept_op(")")) return std::make_shared<ListLit>();
                    ExprPtr e = parse_tuple_or_expr();
                    expect_op(")");
                    return e;
                }
                if (t.text == "[") {
                    ++p_;
                    auto l = std::make_shared<ListLit>();
                    while (!is_op("]")) {
                        l->items.push_back(parse_expr());
                        if (!accept_op(",")) break;
                    }
                    expect_op("]");
                    return l;
                }
                if (t.text == "{") {
                    ++p_;
                    auto d = std::make_shared<DictLit>();
                    while (!is_op("}")) {
                        ExprPtr k = parse_expr();
                        expect_op(":");
                        d->items.emplace_back(k, parse_expr());
                        if (!accept_op(",")) break;
                    }
                    expect_op("}");
                    return d;
                }
                break;
            default: break;
        }
        throw Error("unexpected '" + t.text + "' in expression");
    }

    std::vector<Tok> t_;
    size_t p_ = 0;
};

ExprPtr parse_expression(const std::string& src) {
    ExprParser p(lex_expr(src));
    ExprPtr e = p.parse_tuple_or_expr();
    if (!p.at_end()) throw Error("unexpected '" + p.peek().text + "' after expression");
    return e;
}

} // namespace

// ===========================================================================
// Statements
// ===========================================================================

struct TextNode : Node {
    std::string text;
    void render(Context&, std::string& out) const override { out += text; }
};

struct OutputNode : Node {
    ExprPtr e;
    void render(Context& ctx, std::string& out) const override { out += e->eval(ctx).to_string(); }
};

struct IfNode : Node {
    std::vector<std::pair<ExprPtr, Body>> branches;  // null condition = else
    void render(Context& ctx, std::string& out) const override {
        for (const auto& [cond, body] : branches) {
            if (!cond || cond->eval(ctx).truthy()) {
                render_body(body, ctx, out);
                return;
            }
        }
    }
};

static void bind_targets(Context& ctx, const std::vector<std::string>& targets, const Value& item) {
    if (targets.size() == 1) {
        ctx.assign(targets[0], item);
        return;
    }
    const std::vector<Value> parts = iterate(item);
    if (parts.size() != targets.size()) throw Error("cannot unpack loop item");
    for (size_t i = 0; i < targets.size(); ++i) ctx.assign(targets[i], parts[i]);
}

struct ForNode : Node {
    std::vector<std::string> targets;
    ExprPtr iter, filter;
    Body body, else_body;
    void render(Context& ctx, std::string& out) const override {
        const Value src = iter->eval(ctx);
        std::vector<Value> items;
        if (src.is_object() && targets.size() == 2) {
            // "for k, v in dict" is an error in Jinja, but "for k, v in d.items()" is the norm;
            // keep plain dict iteration on keys.
            items = iterate(src);
        } else {
            items = iterate(src);
        }
        if (filter) {
            std::vector<Value> kept;
            ScopeGuard g(ctx);
            for (const auto& it : items) {
                bind_targets(ctx, targets, it);
                if (filter->eval(ctx).truthy()) kept.push_back(it);
            }
            items.swap(kept);
        }
        if (items.empty()) {
            render_body(else_body, ctx, out);
            return;
        }
        ScopeGuard g(ctx);
        const int64_t n = static_cast<int64_t>(items.size());
        for (int64_t i = 0; i < n; ++i) {
            Value loop = Value::object({
                {"index", Value(i + 1)}, {"index0", Value(i)}, {"revindex", Value(n - i)}, {"revindex0", Value(n - i - 1)},
                {"first", Value(i == 0)}, {"last", Value(i == n - 1)}, {"length", Value(n)},
                {"previtem", i > 0 ? items[static_cast<size_t>(i - 1)] : Value()},
                {"nextitem", i + 1 < n ? items[static_cast<size_t>(i + 1)] : Value()},
            });
            loop.set("cycle", Value::callable([i](Args& a, Kwargs&) {
                if (a.empty()) throw Error("cycle needs arguments");
                return a[static_cast<size_t>(i) % a.size()];
            }));
            ctx.assign("loop", loop);
            bind_targets(ctx, targets, items[static_cast<size_t>(i)]);
            render_body(body, ctx, out);
            if (ctx.flow == Flow::Break) { ctx.flow = Flow::Normal; break; }
            if (ctx.flow == Flow::Continue) ctx.flow = Flow::Normal;
        }
    }
};

struct SetNode : Node {
    std::vector<std::string> targets;  // one name, or several for tuple unpacking
    std::string ns_obj, ns_attr;       // "set ns.attr = ..."
    ExprPtr value;
    Body block;                        // "{% set x %}...{% endset %}"
    void render(Context& ctx, std::string&) const override {
        Value v;
        if (value) {
            v = value->eval(ctx);
        } else {
            std::string captured;
            render_body(block, ctx, captured);
            v = Value(captured);
        }
        if (!ns_obj.empty()) {
            const Value target = ctx.lookup(ns_obj);
            if (!target.is_object()) throw Error("'" + ns_obj + "' is not a namespace");
            target.set(ns_attr, v);
            return;
        }
        if (targets.size() == 1) ctx.assign(targets[0], v);
        else bind_targets(ctx, targets, v);
    }
};

struct MacroNode : Node {
    std::string name;
    std::vector<std::pair<std::string, ExprPtr>> params;
    Body body;
    void render(Context& ctx, std::string&) const override {
        const MacroNode* self = this;
        Context* cptr = &ctx;
        ctx.assign(name, Value::callable([self, cptr](Args& args, Kwargs& kw) {
            Context& c = *cptr;
            ScopeGuard g(c);
            for (size_t i = 0; i < self->params.size(); ++i) {
                const auto& [pname, def] = self->params[i];
                Value v;
                bool found = false;
                for (const auto& [k, x] : kw) if (k == pname) { v = x; found = true; }
                if (!found && i < args.size()) { v = args[i]; found = true; }
                if (!found && def) v = def->eval(c);
                c.assign(pname, v);
            }
            std::string out;
            render_body(self->body, c, out);
            c.flow = Flow::Normal;
            return Value(out);
        }));
    }
};

struct FlowNode : Node {
    Flow flow;
    explicit FlowNode(Flow f) : flow(f) {}
    void render(Context& ctx, std::string&) const override { ctx.flow = flow; }
};

struct FilterBlockNode : Node {
    std::string name;
    std::vector<ExprPtr> args;
    Body body;
    void render(Context& ctx, std::string& out) const override {
        std::string inner;
        render_body(body, ctx, inner);
        Args a;
        for (const auto& e : args) a.push_back(e->eval(ctx));
        Kwargs kw;
        out += apply_filter(name, Value(inner), a, kw, ctx).to_string();
    }
};

struct BlockNode : Node {  // generation / anything rendered as-is
    Body body;
    void render(Context& ctx, std::string& out) const override { render_body(body, ctx, out); }
};

// ===========================================================================
// Template lexer (text / {{ }} / {% %} / {# #}) and statement parser
// ===========================================================================

namespace {

struct Segment {
    enum Kind { Text, Expr, Stmt } kind;
    std::string content;
};

std::vector<Segment> split_template(const std::string& src) {
    std::vector<Segment> out;
    size_t i = 0;
    bool strip_next = false;   // "-%}" / "-}}": lstrip the following text
    bool trim_newline = false; // trim_blocks: drop one newline after a block tag
    auto push_text = [&](std::string text, bool rstrip_all, bool lstrip_line) {
        if (strip_next) {
            size_t b = 0;
            while (b < text.size() && std::isspace(static_cast<unsigned char>(text[b]))) ++b;
            text.erase(0, b);
        } else if (trim_newline) {
            if (text.compare(0, 2, "\r\n") == 0) text.erase(0, 2);
            else if (!text.empty() && text[0] == '\n') text.erase(0, 1);
        }
        strip_next = trim_newline = false;
        if (rstrip_all) {
            size_t e = text.size();
            while (e > 0 && std::isspace(static_cast<unsigned char>(text[e - 1]))) --e;
            text.resize(e);
        } else if (lstrip_line) {
            // lstrip_blocks: spaces/tabs between the last newline and the tag
            size_t e = text.size();
            while (e > 0 && (text[e - 1] == ' ' || text[e - 1] == '\t')) --e;
            if (e == 0 || text[e - 1] == '\n') text.resize(e);
        }
        if (!text.empty()) out.push_back({Segment::Text, std::move(text)});
    };

    while (i < src.size()) {
        size_t open = std::string::npos;
        char kind = 0;
        for (size_t j = src.find('{', i); j != std::string::npos; j = src.find('{', j + 1)) {
            if (j + 1 < src.size() && (src[j + 1] == '{' || src[j + 1] == '%' || src[j + 1] == '#')) {
                open = j;
                kind = src[j + 1];
                break;
            }
        }
        if (open == std::string::npos) {
            push_text(src.substr(i), false, false);
            break;
        }
        size_t inner = open + 2;
        bool lstrip = false, keep = false;
        if (inner < src.size() && src[inner] == '-') { lstrip = true; ++inner; }
        else if (inner < src.size() && src[inner] == '+') { keep = true; ++inner; }
        const char* close = kind == '{' ? "}}" : kind == '%' ? "%}" : "#}";
        // The closing delimiter only counts outside string literals: a
        // template may well print '}}' or '%}' itself.
        size_t end = std::string::npos;
        if (kind == '#') {
            end = src.find(close, inner);
        } else {
            char quote = 0;
            for (size_t k = inner; k + 1 < src.size(); ++k) {
                const char ch = src[k];
                if (quote) {
                    if (ch == '\\') ++k;
                    else if (ch == quote) quote = 0;
                    continue;
                }
                if (ch == '"' || ch == '\'') { quote = ch; continue; }
                if (ch == close[0] && src[k + 1] == close[1]) { end = k; break; }
            }
        }
        if (end == std::string::npos) throw Error("unterminated tag");
        bool rstrip = false;
        size_t content_end = end;
        if (content_end > inner && src[content_end - 1] == '-') { rstrip = true; --content_end; }
        else if (content_end > inner && src[content_end - 1] == '+') { --content_end; }
        push_text(src.substr(i, open - i), lstrip, kind != '{' && !keep);
        std::string content = src.substr(inner, content_end - inner);
        i = end + 2;

        if (kind == '%') {
            // raw blocks: everything up to endraw is literal text
            size_t b = 0;
            while (b < content.size() && std::isspace(static_cast<unsigned char>(content[b]))) ++b;
            if (content.compare(b, 3, "raw") == 0 && content.find_first_not_of(" \t\r\n", b + 3) == std::string::npos) {
                size_t search = i;
                for (;;) {
                    const size_t tag = src.find("{%", search);
                    if (tag == std::string::npos) throw Error("unterminated raw block");
                    const size_t tag_end = src.find("%}", tag);
                    if (tag_end == std::string::npos) throw Error("unterminated raw block");
                    std::string t = src.substr(tag + 2, tag_end - tag - 2);
                    t.erase(std::remove_if(t.begin(), t.end(), [](char ch) { return ch == '-' || std::isspace(static_cast<unsigned char>(ch)); }), t.end());
                    if (t == "endraw") {
                        std::string raw = src.substr(i, tag - i);
                        if (rstrip) { size_t s = 0; while (s < raw.size() && std::isspace(static_cast<unsigned char>(raw[s]))) ++s; raw.erase(0, s); }
                        if (src[tag + 2] == '-') { size_t e = raw.size(); while (e > 0 && std::isspace(static_cast<unsigned char>(raw[e - 1]))) --e; raw.resize(e); }
                        if (!raw.empty()) out.push_back({Segment::Text, raw});
                        i = tag_end + 2;
                        rstrip = tag_end > 0 && src[tag_end - 1] == '-';
                        break;
                    }
                    search = tag_end + 2;
                }
                strip_next = rstrip;
                trim_newline = !rstrip;
                continue;
            }
        }
        if (kind == '{') out.push_back({Segment::Expr, content});
        else if (kind == '%') out.push_back({Segment::Stmt, content});
        strip_next = rstrip;
        trim_newline = !rstrip && kind != '{';
    }
    return out;
}

class StmtParser {
public:
    explicit StmtParser(std::vector<Segment> segs) : s_(std::move(segs)) {}

    Body parse_all() {
        Body body = parse_until({});
        if (p_ < s_.size()) throw Error("unexpected '{% " + s_[p_].content + " %}'");
        return body;
    }

private:
    static std::string first_word(const std::string& c) {
        size_t b = 0;
        while (b < c.size() && std::isspace(static_cast<unsigned char>(c[b]))) ++b;
        size_t e = b;
        while (e < c.size() && (std::isalnum(static_cast<unsigned char>(c[e])) || c[e] == '_')) ++e;
        return c.substr(b, e - b);
    }
    static std::string rest_after(const std::string& c, const std::string& word) {
        const size_t at = c.find(word);
        return c.substr(at + word.size());
    }

    // Parses nodes until a statement whose first word is in `stops` (left
    // unconsumed) or the end.
    Body parse_until(const std::vector<std::string>& stops) {
        Body body;
        while (p_ < s_.size()) {
            const Segment& seg = s_[p_];
            if (seg.kind == Segment::Text) {
                auto t = std::make_shared<TextNode>();
                t->text = seg.content;
                body.push_back(t);
                ++p_;
                continue;
            }
            if (seg.kind == Segment::Expr) {
                auto o = std::make_shared<OutputNode>();
                o->e = parse_expression(seg.content);
                body.push_back(o);
                ++p_;
                continue;
            }
            const std::string word = first_word(seg.content);
            if (std::find(stops.begin(), stops.end(), word) != stops.end()) return body;
            ++p_;
            body.push_back(parse_statement(word, seg.content));
        }
        if (!stops.empty()) throw Error("missing {% " + stops.back() + " %}");
        return body;
    }

    void expect_end(const char* word) {
        if (p_ >= s_.size() || first_word(s_[p_].content) != word) throw Error(std::string("expected {% ") + word + " %}");
        ++p_;
    }

    NodePtr parse_statement(const std::string& word, const std::string& content) {
        const std::string rest = rest_after(content, word);
        if (word == "if") {
            auto n = std::make_shared<IfNode>();
            ExprPtr cond = parse_expression(rest);
            for (;;) {
                Body b = parse_until({"elif", "else", "endif"});
                n->branches.emplace_back(cond, std::move(b));
                const std::string w = first_word(s_[p_].content);
                const std::string c = s_[p_].content;
                ++p_;
                if (w == "endif") break;
                if (w == "else") {
                    n->branches.emplace_back(nullptr, parse_until({"endif"}));
                    expect_end("endif");
                    break;
                }
                cond = parse_expression(rest_after(c, "elif"));
            }
            return n;
        }
        if (word == "for") {
            auto n = std::make_shared<ForNode>();
            ExprParser ep(lex_expr(rest));
            n->targets.push_back(ep.expect_name());
            while (ep.accept_op(",")) n->targets.push_back(ep.expect_name());
            if (!ep.accept_name("in")) throw Error("expected 'in' in for loop");
            // iterable: an expression without the trailing "if" filter or "recursive"
            ExprPtr it = ep.parse_iterable();
            n->iter = it;
            if (ep.accept_name("if")) n->filter = ep.parse_expr();
            ep.accept_name("recursive");
            if (!ep.at_end()) throw Error("unexpected '" + ep.peek().text + "' in for loop");
            n->body = parse_until({"else", "endfor"});
            if (first_word(s_[p_].content) == "else") {
                ++p_;
                n->else_body = parse_until({"endfor"});
            }
            expect_end("endfor");
            return n;
        }
        if (word == "set") {
            auto n = std::make_shared<SetNode>();
            const size_t eq = find_assign(rest);
            const std::string lhs = eq == std::string::npos ? rest : rest.substr(0, eq);
            ExprParser lp(lex_expr(lhs));
            std::string first = lp.expect_name();
            if (lp.accept_op(".")) {
                n->ns_obj = first;
                n->ns_attr = lp.expect_name();
            } else {
                n->targets.push_back(first);
                while (lp.accept_op(",")) n->targets.push_back(lp.expect_name());
            }
            if (!lp.at_end()) throw Error("bad set target");
            if (eq == std::string::npos) {
                n->block = parse_until({"endset"});
                expect_end("endset");
            } else {
                n->value = parse_expression(rest.substr(eq + 1));
            }
            return n;
        }
        if (word == "macro") {
            auto n = std::make_shared<MacroNode>();
            ExprParser ep(lex_expr(rest));
            n->name = ep.expect_name();
            ep.expect_op("(");
            while (!ep.is_op(")")) {
                std::string pname = ep.expect_name();
                ExprPtr def;
                if (ep.accept_op("=")) def = ep.parse_expr();
                n->params.emplace_back(std::move(pname), def);
                if (!ep.accept_op(",")) break;
            }
            ep.expect_op(")");
            n->body = parse_until({"endmacro"});
            expect_end("endmacro");
            return n;
        }
        if (word == "break") return std::make_shared<FlowNode>(Flow::Break);
        if (word == "continue") return std::make_shared<FlowNode>(Flow::Continue);
        if (word == "filter") {
            auto n = std::make_shared<FilterBlockNode>();
            ExprParser ep(lex_expr(rest));
            n->name = ep.expect_name();
            if (ep.accept_op("(")) {
                while (!ep.is_op(")")) {
                    n->args.push_back(ep.parse_expr());
                    if (!ep.accept_op(",")) break;
                }
                ep.expect_op(")");
            }
            n->body = parse_until({"endfilter"});
            expect_end("endfilter");
            return n;
        }
        if (word == "generation") {
            auto n = std::make_shared<BlockNode>();
            n->body = parse_until({"endgeneration"});
            expect_end("endgeneration");
            return n;
        }
        if (word == "do") {
            // "{% do list.append(x) %}" (jinja2.ext.do): evaluate, discard
            struct Discard : Expr {
                ExprPtr inner;
                Value eval(Context& c) const override { inner->eval(c); return Value(""); }
            };
            auto n = std::make_shared<OutputNode>();
            auto d = std::make_shared<Discard>();
            d->inner = parse_expression(rest);
            n->e = d;
            return n;
        }
        throw Error("unsupported statement '" + word + "'");
    }

    // Position of the top-level '=' of a set statement (not '==' etc.).
    static size_t find_assign(const std::string& s) {
        int depth = 0;
        char quote = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            const char c = s[i];
            if (quote) { if (c == '\\') ++i; else if (c == quote) quote = 0; continue; }
            if (c == '"' || c == '\'') { quote = c; continue; }
            if (c == '(' || c == '[' || c == '{') ++depth;
            else if (c == ')' || c == ']' || c == '}') --depth;
            else if (c == '=' && depth == 0) {
                const char prev = i ? s[i - 1] : 0;
                const char next = i + 1 < s.size() ? s[i + 1] : 0;
                if (next != '=' && prev != '=' && prev != '!' && prev != '<' && prev != '>') return i;
            }
        }
        return std::string::npos;
    }

    std::vector<Segment> s_;
    size_t p_ = 0;
};

} // namespace

// ===========================================================================
// Template
// ===========================================================================

Template::Template(const std::string& source) {
    StmtParser sp(split_template(source));
    body_ = sp.parse_all();
}

Template::~Template() = default;

std::string Template::render(const Value& context, std::time_t fixed_now) const {
    Context ctx;
    ctx.scopes.emplace_back();
    auto& g = ctx.scopes.back();
    g["range"] = Value::callable([](Args& a, Kwargs&) {
        int64_t start = 0, stop = 0, step = 1;
        if (a.size() == 1) stop = a[0].as_int();
        else if (a.size() >= 2) { start = a[0].as_int(); stop = a[1].as_int(); if (a.size() >= 3) step = a[2].as_int(); }
        if (step == 0) throw Error("range step cannot be zero");
        Value out = Value::array();
        for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) out.arr().emplace_back(i);
        return out;
    });
    g["namespace"] = Value::callable([](Args& a, Kwargs& kw) {
        Value ns = Value::object();
        if (!a.empty() && a[0].is_object()) for (const auto& [k, v] : a[0].obj()) ns.set(k, v);
        for (const auto& [k, v] : kw) ns.set(k, v);
        return ns;
    });
    g["dict"] = Value::callable([](Args& a, Kwargs& kw) {
        Value d = Value::object();
        if (!a.empty() && a[0].is_object()) for (const auto& [k, v] : a[0].obj()) d.set(k, v);
        for (const auto& [k, v] : kw) d.set(k, v);
        return d;
    });
    g["raise_exception"] = Value::callable([](Args& a, Kwargs&) -> Value {
        throw Error("template raised: " + (a.empty() ? std::string("error") : a[0].to_string()));
    });
    g["strftime_now"] = Value::callable([fixed_now](Args& a, Kwargs&) {
        const std::time_t now = fixed_now ? fixed_now : std::time(nullptr);
        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        char buf[128];
        const std::string fmt = a.empty() ? "%Y-%m-%d" : a[0].to_string();
        std::strftime(buf, sizeof(buf), fmt.c_str(), &tm);
        return Value(std::string(buf));
    });
    if (context.is_object()) {
        for (const auto& [k, v] : context.obj()) g[k] = v;
    }
    std::string out;
    render_body(body_, ctx, out);
    return out;
}

} // namespace jinja
} // namespace desireeia
