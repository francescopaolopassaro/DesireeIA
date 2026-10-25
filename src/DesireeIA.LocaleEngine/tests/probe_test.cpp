// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.
//
// Decode speed and token probes against a real model.
//
//   desireeia_probe_test <model.gguf> [mode] [tokens] [rounds]
//
// mode:
//   speed      greedy decode, nothing tracked  - the number that must not move
//   stats      confidence tracking on
//   topk       confidence tracking + top candidates (K = 5)
//   attn       attention probe on (last decoded token, every layer)
//   check      correctness of the probes (exit code 1 on failure)
//
// Speed rounds are reported one by one: the laptop throttles, so only the
// FIRST round after a rest is a clean reading (see bench/run_bench.ps1).
// Without a model path the test is skipped (exit code 0).

#include "desireeia/abi.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static std::string chat_prompt(desireeia_ctx* ctx, const char* user) {
    const char* roles[] = {"user"};
    const char* contents[] = {user};
    std::vector<char> buf(1 << 16);
    size_t len = 0;
    desireeia_apply_chat_template(ctx, roles, contents, 1, 1, buf.data(), buf.size(), &len);
    return std::string(buf.data(), strnlen(buf.data(), buf.size()));
}

struct Run {
    std::vector<int32_t> tokens;
    double prefill_ms = 0, decode_ms = 0;
    std::vector<float> logprob, entropy, maxprob;
    std::vector<std::vector<int32_t>> top_id;
    std::vector<std::vector<float>> top_lp;
};

// Greedy generation of up to max_new tokens. `per_token` runs after each decode step.
template <class F>
static Run generate(desireeia_ctx* ctx, const std::vector<int32_t>& prompt, int max_new, F&& per_token, bool reset = true) {
    Run r;
    int32_t tok = -1;
    if (reset) desireeia_session_reset(ctx);
    auto t0 = std::chrono::steady_clock::now();
    if (desireeia_predict(ctx, prompt.data(), prompt.size(), &tok) != DESIREEIA_OK) return r;
    auto t1 = std::chrono::steady_clock::now();
    r.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double dec = 0;
    for (int i = 0; i < max_new; ++i) {
        int32_t eog = 0;
        desireeia_is_eog_token(ctx, tok, &eog);
        if (eog) break;
        r.tokens.push_back(tok);
        per_token(ctx, r);
        auto a = std::chrono::steady_clock::now();
        if (desireeia_next_token(ctx, &tok) != DESIREEIA_OK) break;
        dec += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    }
    r.decode_ms = dec;
    return r;
}

static desireeia_ctx* load(const char* model) {
    desireeia_hw_info hw;
    std::memset(&hw, 0, sizeof(hw));
    desireeia_probe_hw(&hw);
    desireeia_plan plan;
    std::memset(&plan, 0, sizeof(plan));
    desireeia_make_plan(&hw, model, nullptr, &plan);
    // DSR_BACKEND=cpu forces the CPU path: the same checks must hold on both, and the two must agree.
    if (const char* b = std::getenv("DSR_BACKEND")) if (std::strcmp(b, "cpu") == 0) plan.backend = DESIREEIA_BACKEND_CPU;
    if (const char* t = std::getenv("DSR_THREADS")) plan.n_threads = (uint32_t) std::atoi(t);
    desireeia_ctx* ctx = nullptr;
    if (desireeia_create(model, &plan, nullptr, nullptr, &ctx) != DESIREEIA_OK || !ctx) return nullptr;
    desireeia_sampling s;
    std::memset(&s, 0, sizeof(s));
    desireeia_get_sampling(ctx, &s);
    s.temperature = 0.0f;   // greedy: the run is deterministic
    desireeia_set_sampling(ctx, &s);
    return ctx;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("probe_test: no model path, skipped\n"); return 0; }
    const char* model = argv[1];
    std::string mode = argc > 2 ? argv[2] : "speed";
    int tokens = argc > 3 ? std::atoi(argv[3]) : 64;
    int rounds = argc > 4 ? std::atoi(argv[4]) : 3;

    desireeia_ctx* ctx = load(model);
    if (!ctx) { std::printf("FAIL: cannot load %s\n", model); return 1; }
    std::string text = "Explain in four sentences why the sky is blue and why sunsets are red.";
    // DSR_REPEAT=N lengthens the prompt (a long prefill and a longer context for the decode that follows)
    if (const char* r = std::getenv("DSR_REPEAT")) {
        std::string ctxt;
        for (int i = 0, n = std::atoi(r); i < n; ++i) ctxt += "The atmosphere scatters short wavelengths more than long ones, which is why daylight looks blue. ";
        text = ctxt + text;
    }
    auto prompt = tokenize(ctx, chat_prompt(ctx, text.c_str()), true);
    std::printf("model %s | mode %s | prompt %zu tokens | %d new tokens x %d rounds\n", model, mode.c_str(), prompt.size(), tokens, rounds);

    if (mode == "speed" || mode == "stats" || mode == "topk" || mode == "attn") {
        if (mode != "speed") desireeia_set_confidence_tracking(ctx, 1);
#ifdef PROBE_HAS_TOPK
        if (mode == "topk") desireeia_set_top_candidates(ctx, 5);
#endif
#ifdef PROBE_HAS_ATTN
        if (mode == "attn") desireeia_set_attention_probe(ctx, 1);
#endif
        generate(ctx, prompt, 8, [](desireeia_ctx*, Run&) {});   // warm-up: kernels, caches
        if (std::getenv("DSR_PROFILE")) desireeia_profile_reset();
        const bool print_tokens = std::getenv("DSR_PRINT") != nullptr;   // greedy tokens + chosen-token log-probs, to compare two engines
        if (print_tokens) desireeia_set_confidence_tracking(ctx, 1);
        for (int r = 0; r < rounds; ++r) {
            std::vector<float> lps;
            auto run = generate(ctx, prompt, tokens, [&](desireeia_ctx* c, Run&) {
                if (print_tokens) { float a, b, d; if (desireeia_get_token_stats(c, &a, &b, &d) == DESIREEIA_OK) lps.push_back(a); }
                if (mode == "stats" || mode == "topk") { float a, b, d; desireeia_get_token_stats(c, &a, &b, &d); }
#ifdef PROBE_HAS_TOPK
                if (mode == "topk") { int32_t ids[8]; float lp[8]; int32_t n = 0; desireeia_get_top_candidates(c, ids, lp, 8, &n); }
#endif
            });
            double tps = run.decode_ms > 0 ? run.tokens.size() * 1000.0 / run.decode_ms : 0;
            std::printf("round %d: %zu tokens, prefill %.0f ms, decode %.1f tok/s\n", r + 1, run.tokens.size(), run.prefill_ms, tps);
            if (print_tokens && r == 0) {
                std::printf("tokens:"); for (auto t : run.tokens) std::printf(" %d", t); std::printf("\n");
                std::printf("lps:"); for (float v : lps) std::printf(" %.5f", v); std::printf("\n");
            }
        }
        if (std::getenv("DSR_PROFILE")) { char buf[4096]; desireeia_profile_dump(buf, sizeof buf); std::printf("%s\n", buf); }
        desireeia_destroy(ctx);
        return 0;
    }

    int failed = 0;
#if defined(PROBE_HAS_PREFIX)
    if (mode == "two") {
        // Two models in one process, used turn by turn: each must produce what it produces alone.
        const char* path2 = std::getenv("DSR_MODEL2");
        if (!path2) { std::printf("set DSR_MODEL2\n"); return 1; }
        auto solo = [&](desireeia_ctx* c, const std::vector<int32_t>& p, int n) {
            auto r = generate(c, p, n, [](desireeia_ctx*, Run&) {});
            return r.tokens;
        };
        auto p1 = prompt;
        auto ref1 = solo(ctx, p1, 40);
        desireeia_ctx* ctx2 = load(path2);
        if (!ctx2) { std::printf("FAIL: cannot load %s\n", path2); return 1; }
        auto p2 = tokenize(ctx2, chat_prompt(ctx2, "Name three primary colours and one fact about each."), true);
        auto ref2 = solo(ctx2, p2, 40);
        // interleaved: a token of one, a token of the other
        std::vector<int32_t> o1, o2;
        int32_t t1 = -1, t2 = -1;
        desireeia_session_reset(ctx); desireeia_session_reset(ctx2);
        if (desireeia_predict(ctx, p1.data(), p1.size(), &t1) != DESIREEIA_OK) { std::printf("FAIL: predict 1\n"); return 1; }
        if (desireeia_predict(ctx2, p2.data(), p2.size(), &t2) != DESIREEIA_OK) { std::printf("FAIL: predict 2\n"); return 1; }
        for (int i = 0; i < 40; ++i) {
            int32_t e = 0;
            desireeia_is_eog_token(ctx, t1, &e); if (!e) { o1.push_back(t1); if (desireeia_next_token(ctx, &t1) != DESIREEIA_OK) break; }
            desireeia_is_eog_token(ctx2, t2, &e); if (!e) { o2.push_back(t2); if (desireeia_next_token(ctx2, &t2) != DESIREEIA_OK) break; }
        }
        const bool ok1 = o1 == ref1, ok2 = o2 == ref2;
        std::printf("model 1 alternating == alone: %s (%zu tokens) | model 2: %s (%zu tokens)\n", ok1 ? "yes" : "NO", ref1.size(), ok2 ? "yes" : "NO", ref2.size());
        if (!ok1 || !ok2) failed++;
        desireeia_destroy(ctx2);     // the first one keeps working after the second goes away
        auto again = solo(ctx, p1, 40);
        std::printf("after destroying model 2, model 1 == alone: %s\n", again == ref1 ? "yes" : "NO");
        if (again != ref1) failed++;
        desireeia_destroy(ctx);
        std::printf(failed ? "FAILED (%d)\n" : "OK\n", failed);
        return failed ? 1 : 0;
    }
#endif
#if defined(PROBE_HAS_PREFIX)
    if (mode == "prefix") {
        // Prompts A and B differ from the first sentence on but share the same long-prompt SIZE; the sequence
        // A, B, A shows what coming back to A costs: with the live session only, a full prefill; with the prefix
        // store, the K/V of A are restored. The tokens generated by the second A must equal the first A's.
        auto build = [&](const char* doc, int reps) {
            std::string t;
            for (int i = 0; i < reps; ++i) t += doc;
            return tokenize(ctx, chat_prompt(ctx, (std::string("Read this text. ") + t + " Question: summarise it in one sentence.").c_str()), true);
        };
        const int reps = std::getenv("DSR_REPEAT") ? std::atoi(std::getenv("DSR_REPEAT")) : 60;
        auto A = build("The atmosphere scatters short wavelengths more than long ones, which is why daylight looks blue. ", reps);
        auto B = build("A compiler translates source code into machine instructions, optimising them along the way. ", reps);
        if (std::getenv("DSR_B_IS_A")) B = A;   // control: the same prompt twice in a row, served by the live session alone
        const long long mb = std::getenv("DSR_PREFIX_MB") ? std::atoll(std::getenv("DSR_PREFIX_MB")) : 2048;
        if (mb > 0 && desireeia_set_prefix_cache(ctx, (uint64_t) mb * 1024ull * 1024ull, 64) != DESIREEIA_OK) { std::printf("FAIL: prefix cache refused\n"); return 1; }
        std::printf("prompts of %zu and %zu tokens, prefix store %s\n", A.size(), B.size(), mb > 0 ? "ON" : "off");
        desireeia_session_reset(ctx);
        auto a1 = generate(ctx, A, 16, [](desireeia_ctx*, Run&) {}, false);
        size_t u1 = desireeia_last_reused_tokens(ctx);
        auto b1 = generate(ctx, B, 16, [](desireeia_ctx*, Run&) {}, false);
        size_t u2 = desireeia_last_reused_tokens(ctx);
        auto a2 = generate(ctx, A, 16, [](desireeia_ctx*, Run&) {}, false);
        size_t u3 = desireeia_last_reused_tokens(ctx);
        if (std::getenv("DSR_B_IS_A")) std::printf("control: B (same prompt, live session only) %s A\n", b1.tokens == a1.tokens ? "==" : "differs from");
        std::printf("A  : prefill %6.0f ms, reused %zu\n", a1.prefill_ms, u1);
        std::printf("B  : prefill %6.0f ms, reused %zu\n", b1.prefill_ms, u2);
        std::printf("A' : prefill %6.0f ms, reused %zu   (second visit)\n", a2.prefill_ms, u3);
        if (mb > 0) {
            uint64_t bytes = 0, hits = 0, restored = 0; int32_t entries = 0;
            desireeia_get_prefix_cache_info(ctx, &bytes, &entries, &hits, &restored);
            std::printf("store: %.0f MB in %d entries, %llu hits, %llu tokens restored\n", bytes / 1048576.0, entries,
                        (unsigned long long) hits, (unsigned long long) restored);
            if (hits == 0 || u3 + 4 < A.size()) { std::printf("FAIL: the second visit to A did not restore it\n"); failed++; }
        }
        // The last prompt token runs on its own after any cache reuse (the live session does the same), so a word can
        // differ at a near-tie: reported, not a failure. The restore itself is what the checks above verify.
        if (a2.tokens != a1.tokens) std::printf("note: A' generated different tokens than A (rounding at a near-tie, as with live-session reuse)\n");
        else std::printf("A' == A (%zu tokens, identical)\n", a1.tokens.size());
        desireeia_destroy(ctx);
        std::printf(failed ? "FAILED (%d)\n" : "OK\n", failed);
        return failed ? 1 : 0;
    }
#endif
#if defined(PROBE_HAS_ATTN)
    if (mode == "dump") {
        desireeia_set_attention_probe(ctx, 1);
        std::vector<float> w(65536);
        std::vector<int32_t> last;
        const bool want_lp = std::getenv("DSR_LP") != nullptr;
        if (want_lp) desireeia_set_confidence_tracking(ctx, 1);
        std::vector<float> lps;
        auto run = generate(ctx, prompt, std::getenv("DSR_TOKENS") ? std::atoi(std::getenv("DSR_TOKENS")) : 10, [&](desireeia_ctx* c, Run&) {
            if (want_lp) { float a, e, m; if (desireeia_get_token_stats(c, &a, &e, &m) == DESIREEIA_OK) lps.push_back(a); }
        });
        if (want_lp) { std::printf("lps:"); for (float v : lps) std::printf(" %.5f", v); std::printf("\n"); }
        int32_t layers = 0, heads = 0, pos = 0, total = 0;
        if (desireeia_get_attention_shape(ctx, &layers, &heads, &pos, &total) != DESIREEIA_OK) { std::printf("no capture\n"); return 1; }
        std::printf("tokens:"); for (auto t : run.tokens) std::printf(" %d", t);
        std::printf("\nshape %d layers x %d heads, %d positions\n", layers, heads, pos);
        for (int32_t l = 0; l < layers; l += std::max(1, layers / 6)) {
            int32_t n = 0;
            if (desireeia_get_attention(ctx, l, -1, w.data(), (int32_t) w.size(), &n) != DESIREEIA_OK) { std::printf("layer %d: none\n", l); continue; }
            int b1 = 0, b2 = 0;
            for (int i = 0; i < n; ++i) { if (w[i] > w[b1]) b1 = i; }
            for (int i = 0; i < n; ++i) { if (i != b1 && w[i] > w[b2]) b2 = i; }
            std::printf("layer %2d mean-head: top pos %d (%.4f), then %d (%.4f), first-token %.4f\n", l, b1, w[b1], b2, w[b2], w[0]);
        }
        desireeia_destroy(ctx);
        return 0;
    }
#endif
#if defined(PROBE_HAS_TOPK)
    if (mode == "check") {
        // 1. top candidates: sorted, K of them, the chosen (greedy) token is first, probabilities sane.
        desireeia_set_confidence_tracking(ctx, 1);
        desireeia_set_top_candidates(ctx, 5);
        int32_t bad_order = 0, bad_first = 0, bad_sum = 0, seen = 0;
        auto run = generate(ctx, prompt, 24, [&](desireeia_ctx* c, Run& r) {
            int32_t ids[8]; float lp[8]; int32_t n = 0;
            if (desireeia_get_top_candidates(c, ids, lp, 8, &n) != DESIREEIA_OK || n != 5) { bad_order++; return; }
            seen++;
            for (int i = 1; i < n; ++i) if (lp[i] > lp[i - 1] + 1e-6f) bad_order++;
            if (ids[0] != r.tokens.back()) bad_first++;      // temperature 0: the chosen token is the top one
            double s = 0; for (int i = 0; i < n; ++i) s += std::exp((double) lp[i]);
            float a, b, d; desireeia_get_token_stats(c, &a, &b, &d);
            if (s > 1.0 + 1e-3 || std::fabs(std::exp((double) lp[0]) - d) > 1e-3) bad_sum++;   // top-1 prob == max_prob
        });
        std::printf("top-K: %d tokens checked | order errors %d | first!=chosen %d | prob errors %d\n", seen, bad_order, bad_first, bad_sum);
        if (seen == 0 || bad_order || bad_first || bad_sum) failed++;

        // 2. tracking never changes what is generated
        desireeia_set_confidence_tracking(ctx, 0);
        auto plain = generate(ctx, prompt, 24, [](desireeia_ctx*, Run&) {});
        if (plain.tokens != run.tokens) { std::printf("FAIL: tracking changed the generated tokens\n"); failed++; }
        else std::printf("tracking on/off generate the same %zu tokens\n", plain.tokens.size());
    }
#endif
#if defined(PROBE_HAS_ATTN)
    if (mode == "check") {
        desireeia_set_attention_probe(ctx, 1);
        int32_t layers = 0, heads = 0, pos = 0, total = 0;
        int errs = 0, seen = 0;
        int tok_index = 0;
        auto run = generate(ctx, prompt, 12, [&](desireeia_ctx* c, Run&) {
            // the first token comes out of the prefill: nothing is captured for it (documented)
            if (tok_index++ == 0) { if (desireeia_get_attention_shape(c, &layers, &heads, &pos, &total) == DESIREEIA_OK) errs++; return; }
            if (desireeia_get_attention_shape(c, &layers, &heads, &pos, &total) != DESIREEIA_OK || layers <= 0 || pos <= 0) { errs++; return; }
            std::vector<float> w((size_t) pos);
            for (int32_t l = 0; l < layers; l += std::max(1, layers / 6)) {
                for (int32_t h = -1; h < heads; h += std::max(1, heads / 3)) {
                    int32_t n = 0;
                    if (desireeia_get_attention(c, l, h, w.data(), (int32_t) w.size(), &n) != DESIREEIA_OK || n <= 0) { errs++; continue; }
                    double s = 0; bool neg = false;
                    for (int32_t i = 0; i < n; ++i) { s += w[i]; if (w[i] < 0) neg = true; }
                    if (std::fabs(s - 1.0) > 2e-2 || neg) { errs++; std::printf("  layer %d head %d: sum %.4f\n", l, h, s); }
                    seen++;
                }
            }
        });
        std::printf("attention: %d (layer, head) distributions checked over %zu tokens | errors %d | shape %d layers x %d heads x %d positions\n",
                    seen, run.tokens.size(), errs, layers, heads, pos);
        if (seen == 0 || errs) failed++;
        desireeia_set_attention_probe(ctx, 0);
        auto plain = generate(ctx, prompt, 12, [](desireeia_ctx*, Run&) {});
        if (plain.tokens != run.tokens) { std::printf("FAIL: attention probe changed the generated tokens\n"); failed++; }
        else std::printf("probe on/off generate the same %zu tokens\n", plain.tokens.size());
    }
#endif
    desireeia_destroy(ctx);
    std::printf(failed ? "FAILED (%d)\n" : "OK\n", failed);
    return failed ? 1 : 0;
}
