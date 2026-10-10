// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#pragma once

// Native Jinja interpreter for chat templates.
//
// Every modern model ships its prompt format as a Jinja template in the
// GGUF ("tokenizer.chat_template"). Recognizing a fixed list of formats by
// markers (chat_template.cpp) covers the common families but can never
// cover them all, and it can't express tool roles, tool definitions or
// assistant tool calls the way each model was trained on. Running the
// template itself does.
//
// Coverage is the subset of Jinja2 that Hugging Face chat templates use,
// with HF's environment settings (trim_blocks, lstrip_blocks):
//   statements  if/elif/else, for (tuple unpacking, loop.*, for-if, else),
//               set (incl. namespace attributes and block set), macro,
//               break/continue, raw, filter, generation (passthrough)
//   expressions literals, lists, dicts, attribute/subscript/slice, calls
//               with keyword args, filters, tests, `a if c else b`, and the
//               usual arithmetic/comparison/logic/`in`/`~` operators
//   builtins    range, namespace, dict, raise_exception, strftime_now
//   filters     tojson, length/count, trim, upper/lower/title/capitalize,
//               default, join, first/last, list, string/int/float, replace,
//               items, map, select/reject, selectattr/rejectattr, unique,
//               reverse, sort, abs, round, indent, safe, wordcount, batch
//   methods     str: strip/lstrip/rstrip/split/startswith/endswith/upper/
//               lower/title/replace/format/find/count/join
//               dict: items/keys/values/get   list: append/pop/extend/index
//
// Anything outside that raises jinja::Error; the caller then falls back to
// the marker-based formats, so an exotic template never breaks a load.

#include <cstdint>
#include <ctime>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace desireeia {
namespace jinja {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;  // insertion-ordered, like Python dicts
using Args = std::vector<Value>;
using Kwargs = std::vector<std::pair<std::string, Value>>;
using Function = std::function<Value(Args&, Kwargs&)>;

class Value {
public:
    enum class Type : uint8_t { Undefined, None, Bool, Int, Float, String, Array, Object, Callable };

    Value() = default;
    static Value none() { Value v; v.t_ = Type::None; return v; }
    Value(bool b) : t_(Type::Bool), b_(b) {}
    Value(int v) : t_(Type::Int), i_(v) {}
    Value(int64_t v) : t_(Type::Int), i_(v) {}
    Value(double v) : t_(Type::Float), f_(v) {}
    Value(const char* s) : t_(Type::String), s_(std::make_shared<std::string>(s)) {}
    Value(std::string s) : t_(Type::String), s_(std::make_shared<std::string>(std::move(s))) {}
    static Value array(Array a = {}) { Value v; v.t_ = Type::Array; v.a_ = std::make_shared<Array>(std::move(a)); return v; }
    static Value object(Object o = {}) { Value v; v.t_ = Type::Object; v.o_ = std::make_shared<Object>(std::move(o)); return v; }
    static Value callable(Function f) { Value v; v.t_ = Type::Callable; v.fn_ = std::make_shared<Function>(std::move(f)); return v; }

    Type type() const { return t_; }
    bool is_undefined() const { return t_ == Type::Undefined; }
    bool is_none() const { return t_ == Type::None; }
    bool is_string() const { return t_ == Type::String; }
    bool is_array() const { return t_ == Type::Array; }
    bool is_object() const { return t_ == Type::Object; }
    bool is_number() const { return t_ == Type::Int || t_ == Type::Float; }
    bool is_callable() const { return t_ == Type::Callable; }

    bool as_bool() const { return b_; }
    int64_t as_int() const { return t_ == Type::Float ? static_cast<int64_t>(f_) : t_ == Type::Bool ? b_ : i_; }
    double as_float() const { return t_ == Type::Float ? f_ : static_cast<double>(as_int()); }
    const std::string& str() const { return *s_; }
    Array& arr() const { return *a_; }
    Object& obj() const { return *o_; }
    Function& fn() const { return *fn_; }

    bool truthy() const;
    // Python str(): True/False/None, ints, shortest float repr, containers as repr.
    std::string to_string() const;
    // Python repr() (strings quoted) - what str() of a container prints.
    std::string repr() const;
    // json.dumps with ensure_ascii=False; indent < 0 = single line.
    std::string to_json(int indent = -1, int depth = 0) const;
    // Same, with json.dumps' separators=(item_sep, key_sep).
    std::string to_json_sep(int indent, int depth, const std::string& item_sep, const std::string& key_sep) const;

    // Object member lookup (Undefined when absent or not an object).
    Value get(const std::string& key) const;
    void set(const std::string& key, Value v) const;

    bool operator==(const Value& o) const;
    bool operator!=(const Value& o) const { return !(*this == o); }
    bool less(const Value& o) const;

private:
    Type t_ = Type::Undefined;
    bool b_ = false;
    int64_t i_ = 0;
    double f_ = 0.0;
    std::shared_ptr<std::string> s_;
    std::shared_ptr<Array> a_;
    std::shared_ptr<Object> o_;
    std::shared_ptr<Function> fn_;
};

// JSON text -> Value (objects keep key order). Throws jinja::Error.
Value parse_json(const std::string& text);

struct Node;

class Template {
public:
    // Parses the template once (throws jinja::Error on syntax it doesn't
    // support); render() can then be called many times.
    explicit Template(const std::string& source);
    ~Template();
    Template(const Template&) = delete;
    Template& operator=(const Template&) = delete;

    // `context` is an object of global variables (messages, tools,
    // add_generation_prompt, bos_token, ...).
    // `now` fixes the clock strftime_now() reads (0 = current time); tests
    // pass a constant so date-printing templates render reproducibly.
    std::string render(const Value& context, std::time_t now = 0) const;

private:
    std::vector<std::shared_ptr<Node>> body_;
};

} // namespace jinja
} // namespace desireeia
