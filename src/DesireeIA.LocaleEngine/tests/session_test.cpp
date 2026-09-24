// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.
//
// Conversation session (KV prefix reuse) against a real model.
//
// Turn 1: prompt P1 -> generate G1. Turn 2: P2 = P1 + G1 + a new user
// message. In the default exact mode desireeia_predict(P2) must take |P1|
// tokens (the prefilled positions) from the cache, prefill only the rest, AND
// produce exactly the tokens a full prefill of P2 produces (greedy sampling,
// deterministic). Mode 2 also reuses G1 (decoded): more reuse, not bit-exact.
//
// Usage: desireeia_session_test <model.gguf>
// Without a model path argument the test is skipped (exit code 0).

#include "desireeia/abi.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int32_t> tokenize(desireeia_ctx* ctx, const std::string& text, bool bos) {
    size_t n = 0;
    desireeia_tokenize(ctx, text.c_str(), bos ? 1 : 0, nullptr, 0, &n);
    std::vector<int32_t> ids(n);
    if (n) desireeia_tokenize(ctx, text.c_str(), bos ? 1 : 0, ids.data(), ids.size(), &n);
    ids.resize(n);
    return ids;
}

static std::string chat_prompt(desireeia_ctx* ctx, std::vector<const char*> roles,
                               std::vector<const char*> contents) {
    std::vector<char> buf(1 << 16);
    size_t len = 0;
    desireeia_apply_chat_template(ctx, roles.data(), contents.data(), roles.size(), 1,
                                  buf.data(), buf.size(), &len);
    return std::string(buf.data(), strnlen(buf.data(), buf.size()));
}

// Generates up to max_new tokens after `prompt` (first one included).
static std::vector<int32_t> generate(desireeia_ctx* ctx, const std::vector<int32_t>& prompt,
                                     int max_new, double* prefill_ms) {
    std::vector<int32_t> out;
    int32_t tok = -1;
    auto t0 = std::chrono::steady_clock::now();
    if (desireeia_predict(ctx, prompt.data(), prompt.size(), &tok) != DESIREEIA_OK) return out;
    auto t1 = std::chrono::steady_clock::now();
    if (prefill_ms) *prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    for (int i = 0; i < max_new; ++i) {
        int32_t eog = 0;
        desireeia_is_eog_token(ctx, tok, &eog);
        if (eog) break;
        out.push_back(tok);
        if (desireeia_next_token(ctx, &tok) != DESIREEIA_OK) break;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("session_test: no model path, skipped\n");
        return 0;
    }
    const char* model = argv[1];

    desireeia_hw_info hw;
    std::memset(&hw, 0, sizeof(hw));
    desireeia_probe_hw(&hw);
    desireeia_plan plan;
    std::memset(&plan, 0, sizeof(plan));
    desireeia_make_plan(&hw, model, nullptr, &plan);
    desireeia_ctx* ctx = nullptr;
    if (desireeia_create(model, &plan, nullptr, nullptr, &ctx) != DESIREEIA_OK || !ctx) {
        std::printf("FAIL: cannot load %s\n", model);
        return 1;
    }

    int failed = 0;

    // Turn 1.
    std::string p1 = chat_prompt(ctx, {"user"}, {"My name is Francesco and my cat is called Luna. Say hello."});
    auto t1 = tokenize(ctx, p1, true);
    double pf1 = 0;
    auto g1 = generate(ctx, t1, 24, &pf1);
    size_t reused1 = desireeia_last_reused_tokens(ctx);
    std::printf("turn 1: prompt %zu tokens, generated %zu, reused %zu, prefill %.0f ms\n",
                t1.size(), g1.size(), reused1, pf1);
    if (reused1 != 0) { std::printf("FAIL: turn 1 must not reuse anything\n"); failed++; }

    // Turn 2 = turn 1 + its answer + a follow-up, built at the TOKEN level so
    // the shared prefix is exact (a chat app re-tokenizing text can share a
    // bit less; the reuse is then partial, never wrong).
    std::string follow = chat_prompt(ctx, {"user"}, {"What is my cat called?"});
    auto tf = tokenize(ctx, follow, false);
    std::vector<int32_t> t2 = t1;
    t2.insert(t2.end(), g1.begin(), g1.end());
    t2.insert(t2.end(), tf.begin(), tf.end());

    double pf2 = 0;
    auto g2_reuse = generate(ctx, t2, 24, &pf2);
    size_t reused2 = desireeia_last_reused_tokens(ctx);
    std::printf("turn 2 (session): prompt %zu tokens, reused %zu, prefill %.0f ms\n", t2.size(), reused2, pf2);
    // Exact mode (default): only the prefilled positions are reused, i.e. the
    // whole turn-1 prompt; the turn-1 answer (decoded) is prefilled again.
    if (reused2 != t1.size()) {
        std::printf("FAIL: exact mode should reuse exactly %zu tokens\n", t1.size());
        failed++;
    }

    // Same turn 2 from scratch.
    desireeia_session_reset(ctx);
    double pf3 = 0;
    auto g2_full = generate(ctx, t2, 24, &pf3);
    size_t reused3 = desireeia_last_reused_tokens(ctx);
    std::printf("turn 2 (full):    prompt %zu tokens, reused %zu, prefill %.0f ms\n", t2.size(), reused3, pf3);
    if (reused3 != 0) { std::printf("FAIL: after reset nothing may be reused\n"); failed++; }

    if (g2_reuse != g2_full) {
        std::printf("FAIL: session output differs from full prefill\n  session:");
        for (auto t : g2_reuse) std::printf(" %d", t);
        std::printf("\n  full:   ");
        for (auto t : g2_full) std::printf(" %d", t);
        std::printf("\n");
        failed++;
    } else {
        std::printf("session output == full prefill (%zu tokens)\n", g2_full.size());
    }

    // Prefix built by prefill only (no decode steps): here the cached K/V were
    // computed by exactly the same batched path a full prefill uses, so the
    // output must be identical. This isolates the reuse mechanism (positions,
    // truncation) from batch-vs-decode numerics.
    desireeia_session_reset(ctx);
    {
        int32_t tok = -1;
        desireeia_predict(ctx, t1.data(), t1.size(), &tok);   // cache = t1, nothing decoded
    }
    std::vector<int32_t> t3 = t1;
    t3.insert(t3.end(), tf.begin(), tf.end());
    auto g3_reuse = generate(ctx, t3, 24, nullptr);
    size_t reused4 = desireeia_last_reused_tokens(ctx);
    desireeia_session_reset(ctx);
    auto g3_full = generate(ctx, t3, 24, nullptr);
    std::printf("prefill-only prefix: reused %zu, session %s full\n", reused4, g3_reuse == g3_full ? "==" : "!=");
    if (reused4 != t1.size()) { std::printf("FAIL: expected %zu reused tokens\n", t1.size()); failed++; }
    if (g3_reuse != g3_full) { std::printf("FAIL: prefill-only reuse differs from full prefill\n"); failed++; }

    // Mode 2: the decoded tokens are reused too (faster, not bit-identical, so
    // only the amount of reuse is checked, not the output).
    desireeia_session_reset(ctx);
    desireeia_set_session_reuse(ctx, 2);
    generate(ctx, t1, 24, nullptr);
    generate(ctx, t2, 1, nullptr);
    size_t reused5 = desireeia_last_reused_tokens(ctx);
    std::printf("mode 2: reused %zu of %zu\n", reused5, t2.size());
    if (reused5 <= t1.size()) { std::printf("FAIL: mode 2 should also reuse decoded tokens\n"); failed++; }
    if (desireeia_set_session_reuse(ctx, 3) == DESIREEIA_OK) { std::printf("FAIL: mode 3 must be rejected\n"); failed++; }

    // Reuse disabled: back to the old behavior.
    desireeia_set_session_reuse(ctx, 0);
    generate(ctx, t2, 1, nullptr);
    if (desireeia_last_reused_tokens(ctx) != 0) { std::printf("FAIL: reuse disabled but tokens reused\n"); failed++; }

    desireeia_destroy(ctx);
    std::printf(failed == 0 ? "session_test: OK\n" : "session_test: %d FAILED\n", failed);
    return failed == 0 ? 0 : 1;
}
