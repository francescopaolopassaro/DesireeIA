// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "json_grammar.h"
#include "jinja.h"

#include <algorithm>
#include <cstring>

namespace desireeia {

using jinja::Value;

namespace {

enum : uint8_t { K_VALUE, K_OBJ, K_ARR, K_STR, K_NUM, K_LIT };
// Object states
enum : uint8_t { O_OPEN, O_KEY, O_FKEY, O_FKEY_ESC, O_FKEY_HEX, O_COLON, O_COMMA, O_NEXT };
// Array states
enum : uint8_t { A_OPEN, A_COMMA, A_NEXT };
// String states
enum : uint8_t { S_NORM, S_ESC, S_HEX };
// Number states
enum : uint8_t { N_MINUS, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };

constexpr size_t kMaxDepth = 64;
constexpr size_t kMaxConfigs = 48;
constexpr uint8_t kMaxBlanks = 8;

inline bool is_ws(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
inline bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_hex(unsigned char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

std::string compact(const Value& v) { return v.to_json_sep(-1, 0, ",", ":"); }

} // namespace

// ---------------------------------------------------------------- compile --

int JsonSchemaConstraint::add_node(SNode n) {
    nodes_.push_back(std::move(n));
    return static_cast<int>(nodes_.size()) - 1;
}

int JsonSchemaConstraint::lits_node(std::vector<std::string> lits) {
    std::sort(lits.begin(), lits.end());
    lits.erase(std::unique(lits.begin(), lits.end()), lits.end());
    SNode n;
    n.kind = SNode::Lits;
    n.lits = std::move(lits);
    return add_node(std::move(n));
}

bool JsonSchemaConstraint::configure(const std::string& schema_json, std::string& error) {
    nodes_.clear();
    ref_memo_.clear();
    cfgs_.clear();
    gave_up_ = false;
    error.clear();
    size_t b = 0;
    while (b < schema_json.size() && is_ws(static_cast<unsigned char>(schema_json[b]))) ++b;
    if (b == schema_json.size()) return true;                        // off

    // The generic nodes every "any value" position uses.
    {
        SNode o; o.kind = SNode::Object; o.extra_ok = true;          any_obj_ = add_node(o);
        SNode a; a.kind = SNode::Array;                              any_arr_ = add_node(a);
        SNode s; s.kind = SNode::String;                             any_str_ = add_node(s);
        any_lits_ = lits_node({"false", "null", "true"});
        SNode n; n.kind = SNode::Number;                             any_num_ = add_node(n);
        SNode v; v.kind = SNode::Any;                                any_value_ = add_node(v);
    }

    const std::string text = schema_json.substr(b);
    try {
        if (text == "json" || text == "true") {
            SNode any; any.kind = SNode::Any;
            root_ = add_node(any);
        } else {
            Value schema = jinja::parse_json(text);
            root_ = compile(&schema, &schema, 0, error);
            if (root_ < 0) { nodes_.clear(); return false; }
        }
    } catch (const std::exception& e) {
        error = std::string("schema: ") + e.what();
        nodes_.clear();
        return false;
    }
    reset();
    return true;
}

int JsonSchemaConstraint::compile(const void* sp, const void* rp, int depth, std::string& error) {
    const Value& s = *static_cast<const Value*>(sp);
    const Value& root = *static_cast<const Value*>(rp);
    if (depth > 40) { error = "schema: nesting too deep"; return -1; }
    SNode any; any.kind = SNode::Any;
    if (s.type() == Value::Type::Bool) {
        if (!s.as_bool()) { error = "schema: 'false' accepts nothing"; return -1; }
        return add_node(any);
    }
    if (!s.is_object()) { error = "schema: a schema must be an object or a boolean"; return -1; }

    // $ref: "#", "#/$defs/x", "#/definitions/x". The node is registered
    // before its body is compiled, so a recursive schema ends in a loop.
    const Value ref = s.get("$ref");
    if (ref.is_string()) {
        const std::string& r = ref.str();
        auto it = ref_memo_.find(r);
        if (it != ref_memo_.end()) return it->second;
        const Value* target = &root;
        Value holder;
        if (r != "#") {
            if (r.rfind("#/", 0) != 0) { error = "schema: unsupported $ref '" + r + "'"; return -1; }
            Value cur = root;
            size_t pos = 2;
            while (pos <= r.size()) {
                size_t slash = r.find('/', pos);
                const std::string part = r.substr(pos, slash == std::string::npos ? std::string::npos : slash - pos);
                Value nx = cur.get(part);
                if (nx.is_undefined()) { error = "schema: $ref '" + r + "' not found"; return -1; }
                cur = nx;
                if (slash == std::string::npos) break;
                pos = slash + 1;
            }
            holder = cur;
            target = &holder;
        }
        const int idx = add_node(any);
        ref_memo_[r] = idx;
        const int body = compile(target, rp, depth + 1, error);
        if (body < 0) return -1;
        nodes_[idx] = nodes_[body];
        return idx;
    }

    const Value en = s.get("enum");
    if (en.is_array()) {
        std::vector<std::string> l;
        for (const Value& v : en.arr()) l.push_back(compact(v));
        if (l.empty()) { error = "schema: empty enum"; return -1; }
        return lits_node(std::move(l));
    }
    const Value cs = s.get("const");
    if (!cs.is_undefined()) return lits_node({compact(cs)});

    for (const char* key : {"anyOf", "oneOf"}) {
        const Value u = s.get(key);
        if (u.is_array()) {
            SNode n; n.kind = SNode::Union;
            const int idx = add_node(n);
            std::vector<int> alts;
            for (const Value& alt : u.arr()) {
                const int a = compile(&alt, rp, depth + 1, error);
                if (a < 0) return -1;
                alts.push_back(a);
            }
            if (alts.empty()) { error = "schema: empty anyOf"; return -1; }
            nodes_[idx].alts = std::move(alts);
            return idx;
        }
    }
    const Value all = s.get("allOf");
    if (all.is_array()) {
        if (all.arr().size() != 1) { error = "schema: allOf with several schemas is not supported"; return -1; }
        return compile(&all.arr()[0], rp, depth + 1, error);
    }

    Value type = s.get("type");
    if (type.is_array()) {
        SNode n; n.kind = SNode::Union;
        const int idx = add_node(n);
        std::vector<int> alts;
        for (const Value& t : type.arr()) {
            jinja::Object copy = s.obj();
            for (auto& kv : copy) if (kv.first == "type") kv.second = t;
            Value one = Value::object(std::move(copy));
            const int a = compile(&one, rp, depth + 1, error);
            if (a < 0) return -1;
            alts.push_back(a);
        }
        if (alts.empty()) { error = "schema: empty type list"; return -1; }
        nodes_[idx].alts = std::move(alts);
        return idx;
    }

    std::string t = type.is_string() ? type.str() : std::string();
    if (t.empty()) {
        if (!s.get("properties").is_undefined() || !s.get("required").is_undefined()) t = "object";
        else if (!s.get("items").is_undefined()) t = "array";
        else return add_node(any);
    }

    if (t == "null") return lits_node({"null"});
    if (t == "boolean") return lits_node({"false", "true"});
    if (t == "number" || t == "integer") {
        SNode n; n.kind = t == "integer" ? SNode::Integer : SNode::Number;
        return add_node(n);
    }
    if (t == "string") {
        SNode n; n.kind = SNode::String;
        const Value mn = s.get("minLength"), mx = s.get("maxLength");
        if (mn.is_number()) n.min_len = static_cast<int>(mn.as_int());
        if (mx.is_number()) n.max_len = static_cast<int>(mx.as_int());
        return add_node(n);
    }
    if (t == "array") {
        SNode n; n.kind = SNode::Array;
        const Value mn = s.get("minItems"), mx = s.get("maxItems");
        if (mn.is_number()) n.min_items = static_cast<int>(mn.as_int());
        if (mx.is_number()) n.max_items = static_cast<int>(mx.as_int());
        const int idx = add_node(n);
        const Value it = s.get("items");
        if (it.is_object() || it.type() == Value::Type::Bool) {
            const int c = compile(&it, rp, depth + 1, error);
            if (c < 0) return -1;
            nodes_[idx].items = c;
        }
        return idx;
    }
    if (t == "object") {
        SNode n; n.kind = SNode::Object;
        const int idx = add_node(n);
        std::vector<std::pair<std::string, int>> props;
        std::vector<std::string> required;
        const Value req = s.get("required");
        if (req.is_array()) for (const Value& r : req.arr()) if (r.is_string()) required.push_back(r.str());
        const Value pr = s.get("properties");
        bool has_props = false;
        auto enc = [](const std::string& k) {
            std::string q = Value(k).to_json();
            return q.substr(1, q.size() - 2);
        };
        if (pr.is_object()) {
            for (auto& kv : pr.obj()) {
                has_props = true;
                const int c = compile(&kv.second, rp, depth + 1, error);
                if (c < 0) return -1;
                props.emplace_back(enc(kv.first), c);
            }
        }
        for (const std::string& r : required) {
            const std::string e = enc(r);
            bool seen = false;
            for (auto& p : props) if (p.first == e) { seen = true; break; }
            if (!seen) { props.emplace_back(e, add_node(any)); has_props = true; }
        }
        if (props.size() > 64) { error = "schema: an object may list at most 64 properties"; return -1; }
        std::sort(props.begin(), props.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
        uint64_t req_mask = 0;
        std::vector<std::string> keys;
        std::vector<int> pnodes;
        for (size_t i = 0; i < props.size(); ++i) {
            keys.push_back(props[i].first);
            pnodes.push_back(props[i].second);
            for (const std::string& r : required) if (enc(r) == props[i].first) req_mask |= (1ull << i);
        }
        bool extra_ok = !has_props;
        int extra_node = -1;
        const Value ap = s.get("additionalProperties");
        if (ap.type() == Value::Type::Bool) {
            extra_ok = ap.as_bool();
        } else if (ap.is_object()) {
            extra_ok = true;
            extra_node = compile(&ap, rp, depth + 1, error);
            if (extra_node < 0) return -1;
        }
        SNode& o = nodes_[idx];
        o.keys = std::move(keys);
        o.prop = std::move(pnodes);
        o.req_mask = req_mask;
        o.extra_ok = extra_ok;
        o.extra_node = extra_node;
        return idx;
    }
    error = "schema: unknown type '" + t + "'";
    return -1;
}

// -------------------------------------------------------------------- run --

void JsonSchemaConstraint::reset() {
    gave_up_ = false;
    cfgs_.clear();
    if (nodes_.empty()) return;
    Config c;
    Frame f;
    f.kind = K_VALUE;
    f.node = root_;
    c.stack.push_back(f);
    cfgs_.push_back(std::move(c));
}

bool JsonSchemaConstraint::can_end(const Config& c) const {
    if (c.stack.empty()) return true;
    // A bare number or literal at the root has no closing character: it
    // ends where the generation does.
    for (const Frame& f : c.stack) {
        if (f.kind == K_NUM) {
            if (!(f.st == N_ZERO || f.st == N_INT || f.st == N_FRAC || f.st == N_EXP)) return false;
        } else if (f.kind == K_LIT) {
            const SNode& n = nodes_[f.node];
            bool acc = false;
            for (int i = f.a; i < f.b; ++i) if (static_cast<int>(n.lits[i].size()) == f.c) { acc = true; break; }
            if (!acc) return false;
        } else {
            return false;
        }
    }
    return true;
}

void JsonSchemaConstraint::start_value(Config c, int node, unsigned char ch, std::vector<Config>& out) const {
    if (c.stack.size() >= kMaxDepth) return;
    const SNode& n = nodes_[node];
    Frame f;
    switch (n.kind) {
    case SNode::Union:
        for (int alt : n.alts) start_value(c, alt, ch, out);
        return;
    case SNode::Any:
        if (ch == '{') { start_value(std::move(c), any_obj_, ch, out); }
        else if (ch == '[') { start_value(std::move(c), any_arr_, ch, out); }
        else if (ch == '"') { start_value(std::move(c), any_str_, ch, out); }
        else if (ch == '-' || is_digit(ch)) { start_value(std::move(c), any_num_, ch, out); }
        else if (ch == 't' || ch == 'f' || ch == 'n') { start_value(std::move(c), any_lits_, ch, out); }
        return;
    case SNode::Object:
        if (ch != '{') return;
        f.kind = K_OBJ; f.st = O_OPEN; f.node = node;
        c.stack.push_back(f);
        out.push_back(std::move(c));
        return;
    case SNode::Array:
        if (ch != '[') return;
        f.kind = K_ARR; f.st = A_OPEN; f.node = node;
        c.stack.push_back(f);
        out.push_back(std::move(c));
        return;
    case SNode::String:
        if (ch != '"') return;
        f.kind = K_STR; f.st = S_NORM; f.node = node;
        c.stack.push_back(f);
        out.push_back(std::move(c));
        return;
    case SNode::Number:
    case SNode::Integer:
        f.kind = K_NUM; f.node = node;
        if (ch == '-') f.st = N_MINUS;
        else if (ch == '0') f.st = N_ZERO;
        else if (ch >= '1' && ch <= '9') f.st = N_INT;
        else return;
        c.stack.push_back(f);
        out.push_back(std::move(c));
        return;
    case SNode::Lits: {
        f.kind = K_LIT; f.node = node; f.a = 0; f.b = static_cast<int>(n.lits.size()); f.c = 0;
        // consume the first character
        int lo = -1, hi = -1;
        for (int i = f.a; i < f.b; ++i) {
            if (n.lits[i].size() > 0 && static_cast<unsigned char>(n.lits[i][0]) == ch) {
                if (lo < 0) lo = i;
                hi = i + 1;
            }
        }
        if (lo < 0) return;
        f.a = lo; f.b = hi; f.c = 1;
        bool longer = false;
        for (int i = lo; i < hi; ++i) if (static_cast<int>(n.lits[i].size()) > 1) { longer = true; break; }
        if (longer) c.stack.push_back(f);       // otherwise the literal is already complete
        out.push_back(std::move(c));
        return;
    }
    }
}

void JsonSchemaConstraint::step(const Config& in, unsigned char ch, std::vector<Config>& out) const {
    Config c = in;
    // The character may have to be handled by several frames in turn: a
    // number or a literal ends only when the next character shows it, and
    // that character then belongs to the parent.
    for (int guard = 0; guard < 8; ++guard) {
        if (c.stack.empty()) {                  // document complete: blanks only
            if (is_ws(ch) && c.ws < kMaxBlanks) { ++c.ws; out.push_back(std::move(c)); }
            return;
        }
        Frame& f = c.stack.back();
        const SNode& n = nodes_[f.node];
        const bool ws = is_ws(ch);
        switch (f.kind) {
        case K_VALUE: {
            if (ws) {
                if (c.ws >= kMaxBlanks) return;
                ++c.ws;
                out.push_back(std::move(c));
                return;
            }
            const int node = f.node;
            c.stack.pop_back();
            c.ws = 0;
            start_value(std::move(c), node, ch, out);
            return;
        }
        case K_OBJ: {
            auto key_start = [&](Config& cc) -> bool {      // ch is the opening quote of a key
                Frame& o = cc.stack.back();
                const SNode& on = nodes_[o.node];
                bool any_free = false;
                for (size_t i = 0; i < on.keys.size(); ++i) if (!(o.used & (1ull << i))) { any_free = true; break; }
                if (any_free) {
                    o.st = O_KEY; o.a = 0; o.b = static_cast<int>(on.keys.size()); o.c = 0;
                    return true;
                }
                if (on.extra_ok) { o.st = O_FKEY; return true; }
                return false;
            };
            switch (f.st) {
            case O_OPEN:
            case O_NEXT:
                if (ws) {
                    if (c.ws >= kMaxBlanks) return;
                    ++c.ws; out.push_back(std::move(c)); return;
                }
                c.ws = 0;
                if (ch == '"') {
                    if (key_start(c)) out.push_back(std::move(c));
                    return;
                }
                if (ch == '}' && f.st == O_OPEN) {
                    if ((f.used & n.req_mask) != n.req_mask) return;
                    c.stack.pop_back();
                    out.push_back(std::move(c));
                    return;
                }
                return;
            case O_KEY: {
                if (ch == '"') {
                    for (int i = f.a; i < f.b; ++i) {
                        if (!(f.used & (1ull << i)) && static_cast<int>(n.keys[i].size()) == f.c) {
                            f.used |= (1ull << i);
                            f.b = n.prop[i];
                            f.st = O_COLON;
                            out.push_back(std::move(c));
                            return;
                        }
                    }
                    if (n.extra_ok) {                       // a free key that begins like a known one
                        f.b = n.extra_node;
                        f.st = O_COLON;
                        out.push_back(std::move(c));
                    }
                    return;
                }
                int lo = -1, hi = -1;
                for (int i = f.a; i < f.b; ++i) {
                    if (static_cast<int>(n.keys[i].size()) > f.c && static_cast<unsigned char>(n.keys[i][f.c]) == ch) {
                        if (lo < 0) lo = i;
                        hi = i + 1;
                    }
                }
                bool free_left = false;
                for (int i = lo; i >= 0 && i < hi; ++i) if (!(f.used & (1ull << i))) { free_left = true; break; }
                if (lo >= 0 && free_left) {
                    f.a = lo; f.b = hi; ++f.c;
                    out.push_back(std::move(c));
                    return;
                }
                if (n.extra_ok && ch >= 0x20 && ch != '\\') {   // leaves the known keys: a free key
                    f.st = O_FKEY;
                    out.push_back(std::move(c));
                }
                return;
            }
            case O_FKEY:
                if (ch == '"') { f.b = n.extra_node; f.st = O_COLON; out.push_back(std::move(c)); return; }
                if (ch == '\\') { f.st = O_FKEY_ESC; out.push_back(std::move(c)); return; }
                if (ch < 0x20) return;
                out.push_back(std::move(c));
                return;
            case O_FKEY_ESC:
                if (ch == 'u') { f.st = O_FKEY_HEX; f.hex = 4; out.push_back(std::move(c)); return; }
                if (std::strchr("\"\\/bfnrt", ch) && ch) { f.st = O_FKEY; out.push_back(std::move(c)); }
                return;
            case O_FKEY_HEX:
                if (!is_hex(ch)) return;
                if (--f.hex == 0) f.st = O_FKEY;
                out.push_back(std::move(c));
                return;
            case O_COLON:
                if (ws) {
                    if (c.ws >= kMaxBlanks) return;
                    ++c.ws; out.push_back(std::move(c)); return;
                }
                if (ch != ':') return;
                c.ws = 0;
                {
                    const int child = f.b;
                    f.st = O_COMMA;
                    Frame v; v.kind = K_VALUE; v.node = child < 0 ? any_value_ : child;
                    c.stack.push_back(v);
                }
                out.push_back(std::move(c));
                return;
            case O_COMMA:
                if (ws) {
                    if (c.ws >= kMaxBlanks) return;
                    ++c.ws; out.push_back(std::move(c)); return;
                }
                c.ws = 0;
                if (ch == ',') {
                    bool room = n.extra_ok;
                    for (size_t i = 0; i < n.keys.size() && !room; ++i) if (!(f.used & (1ull << i))) room = true;
                    if (!room) return;
                    f.st = O_NEXT;
                    out.push_back(std::move(c));
                    return;
                }
                if (ch == '}') {
                    if ((f.used & n.req_mask) != n.req_mask) return;
                    c.stack.pop_back();
                    out.push_back(std::move(c));
                }
                return;
            }
            return;
        }
        case K_ARR: {
            auto item_value = [&](Config& cc) {
                Frame& a = cc.stack.back();
                a.st = A_COMMA;
                ++a.a;
                Frame v; v.kind = K_VALUE; v.node = nodes_[a.node].items < 0 ? any_value_ : nodes_[a.node].items;
                cc.stack.push_back(v);
            };
            if (ws) {
                if (c.ws >= kMaxBlanks) return;
                ++c.ws; out.push_back(std::move(c)); return;
            }
            c.ws = 0;
            if (f.st == A_OPEN) {
                if (ch == ']') {
                    if (f.a < n.min_items) return;
                    c.stack.pop_back();
                    out.push_back(std::move(c));
                    return;
                }
                if (n.max_items == 0) return;
                item_value(c);
                continue;                       // the same character begins the item
            }
            if (f.st == A_COMMA) {
                if (ch == ',') {
                    if (n.max_items >= 0 && f.a >= n.max_items) return;
                    f.st = A_NEXT;
                    out.push_back(std::move(c));
                    return;
                }
                if (ch == ']') {
                    if (f.a < n.min_items) return;
                    c.stack.pop_back();
                    out.push_back(std::move(c));
                }
                return;
            }
            // A_NEXT
            item_value(c);
            continue;
        }
        case K_STR: {
            if (f.st == S_NORM) {
                if (ch == '"') {
                    if (f.a < n.min_len) return;
                    c.stack.pop_back();
                    out.push_back(std::move(c));
                    return;
                }
                if (ch == '\\') {
                    if (n.max_len >= 0 && f.a >= n.max_len) return;
                    f.st = S_ESC;
                    out.push_back(std::move(c));
                    return;
                }
                if (ch < 0x20) return;
                if ((ch & 0xC0) != 0x80) {                  // a continuation byte adds no character
                    if (n.max_len >= 0 && f.a >= n.max_len) return;
                    ++f.a;
                }
                out.push_back(std::move(c));
                return;
            }
            if (f.st == S_ESC) {
                if (ch == 'u') { f.st = S_HEX; f.hex = 4; ++f.a; out.push_back(std::move(c)); return; }
                if (ch && std::strchr("\"\\/bfnrt", ch)) { f.st = S_NORM; ++f.a; out.push_back(std::move(c)); }
                return;
            }
            if (!is_hex(ch)) return;
            if (--f.hex == 0) f.st = S_NORM;
            out.push_back(std::move(c));
            return;
        }
        case K_NUM: {
            const bool integer = n.kind == SNode::Integer;
            const bool e = ch == 'e' || ch == 'E';
            switch (f.st) {
            case N_MINUS:
                if (ch == '0') f.st = N_ZERO;
                else if (ch >= '1' && ch <= '9') f.st = N_INT;
                else return;
                out.push_back(std::move(c)); return;
            case N_ZERO:
                if (ch == '.' && !integer) { f.st = N_DOT; out.push_back(std::move(c)); return; }
                if (e && !integer) { f.st = N_E; out.push_back(std::move(c)); return; }
                if (is_digit(ch)) return;
                break;
            case N_INT:
                if (is_digit(ch)) { out.push_back(std::move(c)); return; }
                if (ch == '.' && !integer) { f.st = N_DOT; out.push_back(std::move(c)); return; }
                if (e && !integer) { f.st = N_E; out.push_back(std::move(c)); return; }
                break;
            case N_DOT:
                if (!is_digit(ch)) return;
                f.st = N_FRAC;
                out.push_back(std::move(c)); return;
            case N_FRAC:
                if (is_digit(ch)) { out.push_back(std::move(c)); return; }
                if (e) { f.st = N_E; out.push_back(std::move(c)); return; }
                break;
            case N_E:
                if (ch == '+' || ch == '-') { f.st = N_ESIGN; out.push_back(std::move(c)); return; }
                if (!is_digit(ch)) return;
                f.st = N_EXP;
                out.push_back(std::move(c)); return;
            case N_ESIGN:
                if (!is_digit(ch)) return;
                f.st = N_EXP;
                out.push_back(std::move(c)); return;
            case N_EXP:
                if (is_digit(ch)) { out.push_back(std::move(c)); return; }
                break;
            }
            // The number ends here; the character belongs to the parent.
            c.stack.pop_back();
            continue;
        }
        case K_LIT: {
            int lo = -1, hi = -1;
            for (int i = f.a; i < f.b; ++i) {
                if (static_cast<int>(n.lits[i].size()) > f.c && static_cast<unsigned char>(n.lits[i][f.c]) == ch) {
                    if (lo < 0) lo = i;
                    hi = i + 1;
                }
            }
            if (lo >= 0) {
                f.a = lo; f.b = hi; ++f.c;
                bool longer = false;
                for (int i = lo; i < hi; ++i) if (static_cast<int>(n.lits[i].size()) > f.c) { longer = true; break; }
                if (!longer) c.stack.pop_back();
                out.push_back(std::move(c));
                return;
            }
            bool acc = false;
            for (int i = f.a; i < f.b; ++i) if (static_cast<int>(n.lits[i].size()) == f.c) { acc = true; break; }
            if (!acc) return;
            c.stack.pop_back();
            continue;
        }
        }
        return;
    }
}

bool JsonSchemaConstraint::run_piece(const std::string& piece, std::vector<Config>& cur) const {
    std::vector<Config> next;
    for (unsigned char ch : piece) {
        next.clear();
        for (const Config& c : cur) {
            step(c, ch, next);
            if (next.size() > kMaxConfigs) next.resize(kMaxConfigs);
        }
        if (next.empty()) return false;
        cur.swap(next);
    }
    return true;
}

bool JsonSchemaConstraint::allows(const std::string& piece, bool is_eog) const {
    if (!configured() || gave_up_) return true;
    if (is_eog) {
        for (const Config& c : cfgs_) if (can_end(c)) return true;
        return false;
    }
    if (piece.empty()) return false;
    std::vector<Config> cur = cfgs_;
    return run_piece(piece, cur);
}

void JsonSchemaConstraint::accept(const std::string& piece) {
    if (!configured() || gave_up_ || piece.empty()) return;
    std::vector<Config> cur = cfgs_;
    if (!run_piece(piece, cur)) { gave_up_ = true; return; }
    cfgs_.swap(cur);
}

uint64_t JsonSchemaConstraint::state_key() const {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2); h *= 1099511628211ull; };
    mix(gave_up_ ? 1 : 0);
    for (const Config& c : cfgs_) {
        mix(0xC0FFEE + c.stack.size());
        mix(c.ws);
        for (const Frame& f : c.stack) {
            const SNode& n = nodes_[f.node];
            int32_t a = f.a;
            if (f.kind == K_STR && n.max_len < 0 && n.min_len <= 0) a = 0;
            else if (f.kind == K_ARR && n.max_items < 0 && n.min_items <= 0) a = 0;
            mix(f.kind); mix(f.st); mix(static_cast<uint64_t>(f.node));
            mix(static_cast<uint64_t>(static_cast<uint32_t>(a)));
            mix(static_cast<uint64_t>(static_cast<uint32_t>(f.b)));
            mix(static_cast<uint64_t>(static_cast<uint32_t>(f.c)));
            mix(static_cast<uint64_t>(static_cast<uint16_t>(f.hex)));
            mix(f.used);
        }
    }
    return h;
}

bool JsonSchemaConstraint::in_plain_string() const {
    if (!configured() || gave_up_ || cfgs_.size() != 1) return false;
    const Config& c = cfgs_[0];
    if (c.stack.empty()) return false;
    const Frame& f = c.stack.back();
    return f.kind == K_STR && f.st == S_NORM && nodes_[f.node].max_len < 0;
}

} // namespace desireeia
