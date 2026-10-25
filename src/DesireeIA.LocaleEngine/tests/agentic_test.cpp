// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

// Tests for the agentic layer of the engine, none of which needs a model:
//   - core/tool_grammar   token-level tool-call constraint
//   - core/jinja          chat templates, compared with jinja2's output
//                         (tests/fixtures/chat_templates, see regen.py)
//   - memory/             context memory on SSD: windowed reads, search,
//                         stream appends, write-through edits, re-sync with
//                         the source file, stale-edit refusal, session expiry

#include "core/jinja.h"
#include "core/tool_grammar.h"
#include "core/json_grammar.h"
#include "memory/context_memory.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#ifndef DESIREEIA_FIXTURES_DIR
#error "DESIREEIA_FIXTURES_DIR must point at tests/fixtures"
#endif

namespace fs = std::filesystem;
using namespace desireeia;

static int g_passed = 0, g_failed = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        if (cond) { ++g_passed; }                                                \
        else { ++g_failed; std::printf("  FAIL %s (%s:%d)\n", what, __FILE__, __LINE__); } \
    } while (0)

static std::string slurp(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}


// ---- JSON schema constraint ---------------------------------------------

// Feeds `text` one byte at a time, as one-byte pieces: false at the first byte the schema refuses.
static bool jfeed(JsonSchemaConstraint& c, const std::string& text) {
    for (char ch : text) {
        const std::string piece(1, ch);
        if (!c.allows(piece, false)) return false;
        c.accept(piece);
    }
    return true;
}

static bool jcomplete(const JsonSchemaConstraint& c) { return c.allows(std::string(), true); }

static JsonSchemaConstraint jmake(const char* schema) {
    JsonSchemaConstraint c;
    std::string err;
    const bool ok = c.configure(schema, err);
    CHECK(ok, schema);
    if (!ok) std::printf("    %s\n", err.c_str());
    c.reset();
    return c;
}

static void test_json_grammar() {
    std::printf("--- JSON schema constraint ---\n");
    const char* person = R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},)"
                         R"("tags":{"type":"array","items":{"type":"string"},"maxItems":2},)"
                         R"("role":{"enum":["admin","user"]}},"required":["name","age"]})";
    {
        auto c = jmake(person);
        CHECK(!c.allows("x", false), "a document starts with {");
        CHECK(!jcomplete(c), "no end before the document");
        CHECK(jfeed(c, R"({"name":"Ann","age":31,"tags":["a","b"],"role":"admin"})"), "a valid document is accepted");
        CHECK(jcomplete(c), "end allowed once the document is complete");
        CHECK(!jfeed(c, "x"), "nothing but blanks after the document");
    }
    {
        auto c = jmake(person);
        CHECK(jfeed(c, "{\n  \"age\": 7,\n  \"name\": \"Bo \\\"x\\\" \\u00e9\"\n}"), "any key order, blanks and escapes");
        CHECK(jcomplete(c), "pretty document complete");
    }
    {
        auto c = jmake(person);
        CHECK(jfeed(c, R"({"age":1)") && !c.allows("}", false), "a required key missing blocks the closing brace");
        CHECK(!c.allows(",\"age\"", false), "a key cannot repeat");
        CHECK(jfeed(c, R"(,"name":"z")") && c.allows("}", false), "once required keys are there it can close");
    }
    {
        auto c = jmake(person);
        CHECK(!jfeed(c, R"({"zip")"), "a key outside properties is refused");
    }
    {
        auto c = jmake(person);
        CHECK(!jfeed(c, R"({"name":1)"), "a number where a string is required is refused");
    }
    {
        auto c = jmake(person);
        CHECK(jfeed(c, R"({"age":1)") && !jfeed(c, "."), "an integer takes no decimal point");
    }
    {
        auto c = jmake(person);
        CHECK(!jfeed(c, R"({"tags":["a","b","c")"), "maxItems stops the third element");
    }
    {
        auto c = jmake(person);
        CHECK(!jfeed(c, R"({"role":"adx")"), "an enum value must be one of the listed");
        auto d = jmake(person);
        CHECK(jfeed(d, R"({"role":"us)") && d.allows("er\"", false), "an enum value is completed token by token");
    }
    {
        auto c = jmake(R"({"type":"number"})");
        CHECK(jfeed(c, "-12.5e+3"), "a root number");
        CHECK(jcomplete(c), "a root number ends with the generation");
        auto d = jmake(R"({"type":"number"})");
        CHECK(jfeed(d, "0") && !d.allows("1", false), "no leading zeros");
        auto e = jmake(R"({"type":"number"})");
        CHECK(jfeed(e, "1.") && !jcomplete(e), "a number cut after the point is not complete");
    }
    {
        auto c = jmake(R"({"anyOf":[{"type":"string"},{"type":"null"},{"type":"integer"}]})");
        CHECK(jfeed(c, "null") && jcomplete(c), "anyOf takes the null branch");
        auto d = jmake(R"({"anyOf":[{"type":"string"},{"type":"null"},{"type":"integer"}]})");
        CHECK(jfeed(d, "42") && jcomplete(d), "anyOf takes the integer branch");
        auto e = jmake(R"({"type":["string","null"]})");
        CHECK(jfeed(e, "\"hi\"") && jcomplete(e), "a type list takes the string branch");
        CHECK(!jmake(R"({"type":["string","null"]})").allows("t", false), "a type list refuses true");
    }
    {
        const char* tree = R"({"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},)"
                           R"("kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"]}},"$ref":"#/$defs/node"})";
        auto c = jmake(tree);
        CHECK(jfeed(c, R"({"v":1,"kids":[{"v":2,"kids":[]},{"v":3}]})") && jcomplete(c), "a recursive schema through $ref");
        auto d = jmake(tree);
        CHECK(!jfeed(d, R"({"v":1,"kids":[{"w":2)"), "the recursive node enforces its own keys");
    }
    {
        auto c = jmake("json");
        CHECK(jfeed(c, R"([1,{"a":[true,null,"x"],"b":-2.5e1}, "s"])") && jcomplete(c), "any JSON value");
        auto d = jmake("json");
        CHECK(!jfeed(d, "[1,]"), "a trailing comma is not JSON");
        auto e = jmake("json");
        CHECK(!jfeed(e, "{'a':1}"), "single quotes are not JSON");
    }
    {
        auto c = jmake(R"({"type":"string","minLength":2,"maxLength":3})");
        CHECK(jfeed(c, "\"a") && !c.allows("\"", false), "minLength blocks an early close");
        CHECK(jfeed(c, "bc") && !c.allows("d", false) && c.allows("\"", false), "maxLength stops a fourth character");
    }
    {
        auto c = jmake(R"({"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":true})");
        CHECK(jfeed(c, R"({"a":true,"free":[1,2],"more":"x"})") && jcomplete(c), "extra keys when additionalProperties allows");
    }
    {
        auto c = jmake(person);
        CHECK(c.allows("{\"na", false) && c.allows("{\"age\":", false), "multi-byte pieces are checked byte by byte");
        CHECK(!c.allows("{\"age\":\"", false), "a multi-byte piece that breaks the schema is refused");
    }
    {
        JsonSchemaConstraint c;
        std::string err;
        CHECK(!c.configure(R"({"type":"bogus"})", err) && !err.empty(), "an unknown type is refused with a message");
        CHECK(!c.configure("{not json", err), "text that is not JSON is refused");
        CHECK(c.configure("", err) && !c.configured(), "an empty schema turns it off");
    }
}

// The byte-trie mask must be the very set that asking token by token gives, in every state a document passes through.
static void test_json_trie_mask() {
    std::printf("--- JSON schema mask by trie ---\n");
    std::vector<std::string> vocab;
    for (int ch = 32; ch < 127; ++ch) vocab.emplace_back(1, (char) ch);
    for (const char* t : {"{\"", "\":", "\",", "\"}", "\"]", "{", "}", "[", "]", ",", ":", "true", "false", "null", "tru", "nul", "12", "-1", "3.14", "1e5",
                          "name", "age", "\"name\"", "\"age\"", "tags", " \"", "\n", "\n  ", "  ", "\\n", "\\\"", "\\u00", "\xC3\xA9", "\xC3", "\xA9",
                          "village", "town", "city", "cit", "ity", "Rome", " Rome", "{\"name\":", "\":\"", "\":[", "],", "\"],\"", "0.5,", "10}", "]}", "}}"})
        vocab.emplace_back(t);
    const size_t eog = vocab.size();
    vocab.emplace_back("<eos>");                         // an end-of-generation token: never in the trie, asked separately
    std::vector<uint8_t> include(vocab.size(), 1), plain(vocab.size(), 0);
    include[eog] = 0;
    for (size_t i = 0; i < vocab.size(); ++i) {
        plain[i] = include[i] && !vocab[i].empty();
        for (unsigned char b : vocab[i]) if (b == '"' || b == '\\' || b < 0x20) plain[i] = 0;
    }
    std::vector<uint8_t> special(vocab.size(), 0);
    for (size_t i = 0; i < vocab.size(); ++i) special[i] = include[i] && !plain[i];
    PieceTrie all, spec;
    all.build(vocab, include);
    spec.build(vocab, special);

    const char* schemas[] = {
        R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"},"maxItems":2},"kind":{"enum":["village","town","city"]}},"required":["name","age"]})",
        "json",
        R"({"anyOf":[{"type":"string"},{"type":"null"},{"type":"number"}]})",
    };
    size_t checked = 0, mismatches = 0;
    for (const char* schema : schemas) {
        for (unsigned seed = 1; seed <= 6; ++seed) {
            JsonSchemaConstraint c;
            std::string err;
            if (!c.configure(schema, err)) { CHECK(false, "schema for the trie test"); return; }
            c.reset();
            unsigned rng = seed * 2654435761u;
            for (int step = 0; step < 60; ++step) {
                // the set by asking every token
                std::vector<uint8_t> want(vocab.size(), 0);
                for (size_t i = 0; i < vocab.size(); ++i) want[i] = (i == eog) ? 0 : (c.allows(vocab[i], false) ? 1 : 0);
                // the set by the trie (plain pieces first when inside a plain string, as the engine does)
                std::vector<uint8_t> got(vocab.size(), 0);
                if (c.in_plain_string()) { for (size_t i = 0; i < vocab.size(); ++i) if (plain[i]) got[i] = 1; c.mask(spec, got); }
                else c.mask(all, got);
                for (size_t i = 0; i < vocab.size(); ++i) if (i != eog) { ++checked; if (want[i] != got[i]) ++mismatches; }
                // advance with a pseudo-random allowed token
                std::vector<size_t> ok;
                for (size_t i = 0; i < vocab.size(); ++i) if (want[i]) ok.push_back(i);
                if (ok.empty()) break;
                rng = rng * 1664525u + 1013904223u;
                c.accept(vocab[ok[(rng >> 8) % ok.size()]]);
                if (c.allows(std::string(), true)) break;      // the document is complete
            }
        }
    }
    CHECK(checked > 3000, "the trie test looked at enough tokens");
    if (mismatches) std::printf("    %zu of %zu tokens differ\n", mismatches, checked);
    CHECK(mismatches == 0, "the trie gives exactly the set that token-by-token asking gives");
}

// ---- tool-call constraint -----------------------------------------------

static bool emit(ToolCallConstraint& t, const std::string& text) {
    for (char ch : text) {
        const std::string piece(1, ch);
        if (!t.allows(piece, false)) return false;
        t.accept(piece);
    }
    return true;
}

static void test_tool_grammar() {
    std::printf("--- tool-call constraint ---\n");
    ToolCallConstraint t;
    t.configure("<tool_call>", "</tool_call>", {"memory_search", "memory_read"});
    t.reset();
    CHECK(!t.constraining(), "free text is not constrained");
    CHECK(emit(t, "Sure. <tool") && t.armed(), "partial open tag arms the constraint");
    CHECK(emit(t, "_call>") && t.constraining(), "open tag starts the constraint");
    CHECK(!t.allows("x", false), "a call must start with {");
    CHECK(!t.allows("", true), "no end of generation inside a call");
    CHECK(emit(t, "\n{\"name\": \"memory_"), "name prefix of a declared tool");
    CHECK(!t.allows("w", false), "undeclared tool name rejected");
    CHECK(emit(t, "search\", \"arguments\": {\"handle\": \"sp_1\", \"q\": \"a \\\"b\\\"\", \"k\": 3, \"x\": [true, null, -1.5e3]}"),
          "nested arguments with escapes, numbers, literals, arrays");
    CHECK(!t.allows("<", false), "stray tag text inside the JSON rejected");
    CHECK(emit(t, "}"), "call object closes");
    CHECK(t.allows("</tool_call>", false) && !t.allows("</arg_value>", false), "only the close tag after the JSON");
    t.accept("</tool_call>");
    CHECK(t.allows("", true) && !t.allows("more", false), "only end of generation after the call");

    ToolCallConstraint u;
    u.configure("<tool_call>", "</tool_call>", {"open_file"});
    u.accept("<tool_call>{\"");
    CHECK(!u.allows("foo", false), "unknown top-level key rejected");
    CHECK(u.allows("name\":\"open_file\"", false) && !u.allows("name\":\"open_fil\"", false), "name must be complete");

    ToolCallConstraint v;
    v.configure("<tool_call>", "</tool_call>", {});
    v.accept("<tool_call>{");
    CHECK(!v.allows("}", false), "a call needs its name");
    v.accept("\"name\": \"any\", \"arguments\": {\"s\": \"");
    CHECK(v.in_plain_string() && v.allows("free text 123", false), "plain string fast path");
}

// ---- Jinja ------------------------------------------------------------------

static void test_jinja_fixtures() {
    std::printf("--- jinja chat templates vs jinja2 ---\n");
    const fs::path dir = fs::path(DESIREEIA_FIXTURES_DIR) / "chat_templates";
    const jinja::Value contexts = jinja::parse_json(slurp(dir / "contexts.json"));
    const std::time_t fixed_now = 1767268800;  // must match regen.py
    int templates = 0;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".jinja") continue;
        ++templates;
        const std::string name = entry.path().stem().string();
        std::unique_ptr<jinja::Template> tmpl;
        try {
            tmpl = std::make_unique<jinja::Template>(slurp(entry.path()));
        } catch (const std::exception& e) {
            CHECK(false, (name + " parses: " + e.what()).c_str());
            continue;
        }
        for (const auto& [cname, ctx] : contexts.obj()) {
            const fs::path base = dir / "expected" / (name + "." + cname);
            std::string got, err;
            try {
                got = tmpl->render(ctx, fixed_now);
            } catch (const std::exception& e) {
                err = e.what();
            }
            const std::string label = name + " / " + cname;
            if (fs::exists(fs::path(base.string() + ".error"))) {
                CHECK(!err.empty(), (label + " refuses like jinja2").c_str());
            } else {
                const std::string expected = slurp(fs::path(base.string() + ".txt"));
                CHECK(err.empty() && got == expected, (label + (err.empty() ? " output differs" : " raised: " + err)).c_str());
            }
        }
    }
    CHECK(templates >= 20, "fixture templates found");
}

static void test_jinja_language() {
    std::printf("--- jinja language features ---\n");
    auto render = [](const std::string& src, const std::string& ctx_json = "{}") {
        try {
            return jinja::Template(src).render(jinja::parse_json(ctx_json));
        } catch (const std::exception& e) {
            return std::string("ERROR: ") + e.what();
        }
    };
    CHECK(render("{{ '}}' }}{{ \"%}\" }}") == "}}%}", "delimiters inside string literals");
    CHECK(render("{% set ns = namespace(n=0) %}{% for i in range(4) if i is even %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}") == "2",
          "namespace, range, for-if, tests");
    CHECK(render("{{ x[::-1] }}|{{ x[1:] | join(',') }}|{{ 'a.b.c'.split('.')[-1] }}", "{\"x\": [1, 2, 3]}") == "[3, 2, 1]|2,3|c",
          "slices, join, methods, negative index");
    CHECK(render("{{ d | tojson }}|{{ d | tojson(separators=(',', ':')) }}", "{\"d\": {\"k\": [1, \"è\"]}}") ==
              "{\"k\": [1, \"è\"]}|{\"k\":[1,\"è\"]}", "tojson with separators, no ascii escaping");
    CHECK(render("{% for m in ms | selectattr('role', 'equalto', 'user') %}{{ loop.index }}{{ m.c }}{% endfor %}",
                 "{\"ms\": [{\"role\": \"user\", \"c\": \"a\"}, {\"role\": \"x\", \"c\": \"b\"}, {\"role\": \"user\", \"c\": \"c\"}]}") == "1a2c",
          "selectattr and loop.index");
    CHECK(render("{% macro m(a, b='B') %}[{{ a }}{{ b }}]{% endmacro %}{{ m('x') }}{{ m('y', b='z') }}") == "[xB][yz]", "macros with defaults");
    CHECK(render("{%- if true -%}\n  A  \n{%- endif -%}\n") == "A", "whitespace control");
    CHECK(render("{% if x is not defined and y is none %}ok{% endif %}", "{\"y\": null}") == "ok", "is not defined / is none");
    CHECK(render("{{ raise_exception('nope') }}").rfind("ERROR:", 0) == 0, "raise_exception propagates");
}

// ---- context memory -------------------------------------------------------

static std::string field(const std::string& json, const std::string& key) {
    const jinja::Value v = jinja::parse_json(json).get(key);
    return v.is_string() ? v.str() : v.to_json();
}

static void test_context_memory() {
    std::printf("--- context memory ---\n");
    const fs::path work = fs::temp_directory_path() / ("desireeia-agentic-test-" + std::to_string(std::time(nullptr)));
    fs::create_directories(work);
    const fs::path src = work / "big.py";
    {
        std::ofstream out(src, std::ios::binary);
        for (int i = 0; i < 50000; ++i) out << "def function_" << i << "(x):\r\n    return x * " << i << "  # città\r\n";
        out << "def needle_target():\r\n    return 'old'\r\n";
    }
    const uint32_t total = 100002;
    {
        ContextMemory mem(work / "store", 3.0);
        const std::string stub = mem.put_file("s1", "big.py", src);
        const std::string handle = field(stub, "handle");
        CHECK(field(stub, "total_lines") == std::to_string(total), "file indexed line by line");
        CHECK(stub.size() < 6000, "the model gets a stub, not the file");
        CHECK(mem.put_file("s1", "big.py", src) == stub, "same content gives the identical stub");

        const std::string window = mem.read("s1", handle, 1001, 2);
        CHECK(field(window, "content") == "1001\tdef function_500(x):\n1002\t    return x * 500  # città", "line window read");
        CHECK(field(window, "next_offset") == "1003", "next window cursor");

        const std::string hits = mem.search("s1", "needle_target", "", 3);
        CHECK(hits.find("def needle_target") != std::string::npos, "BM25 search finds the function");

        mem.append("s1", "log", "one\n");
        const std::string log = mem.append("s1", "log", "two\nthree\n");
        CHECK(field(log, "total_lines") == "3" && field(log, "version") == "2", "stream appends");

        const std::string edit = mem.replace_lines("s1", handle, total, total, "    return 'new'");
        CHECK(field(edit, "ok") == "true" && field(edit, "written_to_source") == "true", "edit applied and written through");
        const std::string disk = slurp(src);
        const std::string tail = "    return 'new'\r\n";
        CHECK(disk.size() > tail.size() && disk.compare(disk.size() - tail.size(), tail.size(), tail) == 0,
              "edit on disk, CRLF kept");

        { std::ofstream out(src, std::ios::binary | std::ios::app); out << "# external change\r\n"; }
        const std::string resynced = mem.read("s1", handle, total + 1, 1);
        CHECK(field(resynced, "content") == std::to_string(total + 1) + "\t# external change", "outside change picked up");

        { std::ofstream out(src, std::ios::binary | std::ios::app); out << "x = 1\r\n"; }
        const std::string stale = mem.replace_lines("s1", handle, 1, 1, "zzz");
        CHECK(stale.find("\"error\"") != std::string::npos, "edit on stale line numbers refused");

        std::this_thread::sleep_for(std::chrono::milliseconds(3500));
        mem.sweep();
        CHECK(!fs::exists(work / "store" / "s1"), "idle session deleted");
        CHECK(fs::exists(src), "the source file is never touched by expiry");
    }
    std::error_code ec;
    fs::remove_all(work, ec);
}

int main() {
    test_tool_grammar();
    test_json_grammar();
    test_json_trie_mask();
    test_jinja_language();
    test_jinja_fixtures();
    test_context_memory();
    std::printf("passed=%d failed=%d\n%s\n", g_passed, g_failed, g_failed ? "agentic test FAIL" : "agentic test PASS");
    return g_failed ? 1 : 0;
}
