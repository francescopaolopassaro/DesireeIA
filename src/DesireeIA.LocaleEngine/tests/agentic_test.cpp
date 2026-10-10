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
        CHECK(disk.size() > 20 && disk.compare(disk.size() - 17, 17, "    return 'new'\r\n") == 0, "edit on disk, CRLF kept");

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
    test_jinja_language();
    test_jinja_fixtures();
    test_context_memory();
    std::printf("passed=%d failed=%d\n%s\n", g_passed, g_failed, g_failed ? "agentic test FAIL" : "agentic test PASS");
    return g_failed ? 1 : 0;
}
