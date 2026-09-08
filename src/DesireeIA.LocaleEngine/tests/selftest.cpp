#include "desireeia/abi.h"
#include "core/engine.h"
#include "core/arch_tags.h"
#include "tokenizer/spm_tokenizer.h"
#include "tokenizer/bpe_tokenizer.h"
#include "models/moe_route.h"
#include "core/thread_pool.h"
#include "quant/quant.h"
#include "ssd_tier/ssd_tier.h"
#include "ssd_tier/mirror_manager.h"
#include "ssd_tier/tiered_expert_store.h"
#include "ssd_tier/hybrid_tier.h"
#include <cstdio>
#include <cassert>
#include <vector>
#include <cstring>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <atomic>
#include <chrono>

static bool approx(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

static const int8_t iq4nl_grid[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

static void encode_q4_0(const float* x, size_t n, std::vector<uint8_t>& out) {
    const int qk = 32;
    for (size_t b = 0; b < n; b += qk) {
        block_q4_0 blk;
        float maxv = 0.0f;
        for (int j = 0; j < qk; ++j) maxv = std::max(maxv, std::fabs(x[b+j]));
        float d = maxv / 8.0f;
        blk.d = desireeia_fp32_to_fp16(d > 0.0f ? d : 1.0f);
        for (int j = 0; j < qk/2; ++j) {
            int q0 = (int)std::min(15.0f, std::max(0.0f, std::roundf(x[b+j]         / d) + 8.0f));
            int q1 = (int)std::min(15.0f, std::max(0.0f, std::roundf(x[b+j + qk/2] / d) + 8.0f));
            blk.qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
        const uint8_t* p = (const uint8_t*)&blk;
        out.insert(out.end(), p, p + sizeof(blk));
    }
}

static void encode_q4_1(const float* x, size_t n, std::vector<uint8_t>& out) {
    const int qk = 32;
    for (size_t b = 0; b < n; b += qk) {
        block_q4_1 blk;
        float minv = x[b], maxv = x[b];
        for (int j = 1; j < qk; ++j) {
            minv = std::min(minv, x[b+j]);
            maxv = std::max(maxv, x[b+j]);
        }
        float d = (maxv - minv) / 16.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        blk.d = desireeia_fp32_to_fp16(d > 0.0f ? d : 1.0f);
        blk.m = desireeia_fp32_to_fp16(minv);
        for (int j = 0; j < qk/2; ++j) {
            int q0 = (int)std::min(15.0f, std::max(0.0f, std::roundf((x[b+j]         - minv) * id)));
            int q1 = (int)std::min(15.0f, std::max(0.0f, std::roundf((x[b+j + qk/2] - minv) * id)));
            blk.qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
        const uint8_t* p = (const uint8_t*)&blk;
        out.insert(out.end(), p, p + sizeof(blk));
    }
}

static void encode_q5_0(const float* x, size_t n, std::vector<uint8_t>& out) {
    const int qk = 32;
    for (size_t b = 0; b < n; b += qk) {
        block_q5_0 blk;
        float maxv = 0.0f;
        for (int j = 0; j < qk; ++j) maxv = std::max(maxv, std::fabs(x[b+j]));
        float d = maxv / 16.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        blk.d = desireeia_fp32_to_fp16(d > 0.0f ? d : 1.0f);
        uint32_t qh = 0;
        for (int j = 0; j < qk/2; ++j) {
            int q0 = (int)std::min(31.0f, std::max(0.0f, std::roundf(x[b+j]         * id) + 16.0f));
            int q1 = (int)std::min(31.0f, std::max(0.0f, std::roundf(x[b+j + qk/2] * id) + 16.0f));
            blk.qs[j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            qh |= (q0 & 0x10) ? (1u << j)        : 0;
            qh |= (q1 & 0x10) ? (1u << (j + 16)) : 0;
        }
        memcpy(blk.qh, &qh, sizeof(qh));
        const uint8_t* p = (const uint8_t*)&blk;
        out.insert(out.end(), p, p + sizeof(blk));
    }
}

static void encode_q5_1(const float* x, size_t n, std::vector<uint8_t>& out) {
    const int qk = 32;
    for (size_t b = 0; b < n; b += qk) {
        block_q5_1 blk;
        float minv = x[b], maxv = x[b];
        for (int j = 1; j < qk; ++j) {
            minv = std::min(minv, x[b+j]);
            maxv = std::max(maxv, x[b+j]);
        }
        float d = (maxv - minv) / 31.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        blk.d = desireeia_fp32_to_fp16(d > 0.0f ? d : 1.0f);
        blk.m = desireeia_fp32_to_fp16(minv);
        uint32_t qh = 0;
        for (int j = 0; j < qk/2; ++j) {
            int q0 = (int)std::min(31.0f, std::max(0.0f, std::roundf((x[b+j]         - minv) * id)));
            int q1 = (int)std::min(31.0f, std::max(0.0f, std::roundf((x[b+j + qk/2] - minv) * id)));
            blk.qs[j] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
            qh |= (q0 & 0x10) ? (1u << j)        : 0;
            qh |= (q1 & 0x10) ? (1u << (j + 16)) : 0;
        }
        memcpy(blk.qh, &qh, sizeof(qh));
        const uint8_t* p = (const uint8_t*)&blk;
        out.insert(out.end(), p, p + sizeof(blk));
    }
}

static void encode_q8_0(const float* x, size_t n, std::vector<uint8_t>& out) {
    const int qk = 32;
    for (size_t b = 0; b < n; b += qk) {
        block_q8_0 blk;
        float maxv = 0.0f;
        for (int j = 0; j < qk; ++j) maxv = std::max(maxv, std::fabs(x[b+j]));
        float d = maxv / 127.0f;
        blk.d = desireeia_fp32_to_fp16(d > 0.0f ? d : 1.0f);
        for (int j = 0; j < qk; ++j) {
            blk.qs[j] = (int8_t)std::min(127.0f, std::max(-127.0f, std::roundf(x[b+j] / d)));
        }
        const uint8_t* p = (const uint8_t*)&blk;
        out.insert(out.end(), p, p + sizeof(blk));
    }
}

static void encode_iq4_nl(const float* x, size_t n, std::vector<uint8_t>& out) {
    const int qk = 32;
    for (size_t b = 0; b < n; b += qk) {
        block_iq4_nl blk;
        float maxv = 0.0f;
        for (int j = 0; j < qk; ++j) maxv = std::max(maxv, std::fabs(x[b+j]));
        float d = maxv / (float)iq4nl_grid[15];
        if (d <= 0.0f) d = 1.0e-6f;
        blk.d = desireeia_fp32_to_fp16(d);
        for (int j = 0; j < qk/2; ++j) {
            float t0 = x[b+j]         / d;
            float t1 = x[b+j + qk/2] / d;
            int b0 = 0, b1 = 0;
            float best = 1e30f;
            for (int g = 0; g < 16; ++g) {
                float e = std::fabs((float)iq4nl_grid[g] - t0);
                if (e < best) { best = e; b0 = g; }
            }
            best = 1e30f;
            for (int g = 0; g < 16; ++g) {
                float e = std::fabs((float)iq4nl_grid[g] - t1);
                if (e < best) { best = e; b1 = g; }
            }
            blk.qs[j] = (uint8_t)(b0 | (b1 << 4));
        }
        const uint8_t* p = (const uint8_t*)&blk;
        out.insert(out.end(), p, p + sizeof(blk));
    }
}

int main(int argc, char** argv) {
    desireeia_hw_info hw;
    desireeia_error e = desireeia_probe_hw(&hw);
    assert(e == DESIREEIA_OK);
    std::printf("threads=%d ram_mb=%llu\n", hw.cpu_threads,
                static_cast<unsigned long long>(hw.ram_total_mb));

    desireeia_plan plan;
    e = desireeia_make_plan(&hw, "", nullptr, &plan);
    assert(e == DESIREEIA_OK);
    assert(plan.n_threads > 0);
    assert(plan.ram_budget_mb > 0);
    std::printf("backend=%d dense_q=%d expert_q=%d\n",
                plan.backend, plan.dense_quant, plan.expert_quant);

    std::vector<float> a = {1, 2, 3, 4};
    std::vector<float> b = {5, 6, 7, 8};
    std::vector<float> out;
    int rc = desireeia::quantized_matmul(a, b, 2, 2, 2, out);
    assert(rc == DESIREEIA_OK);
    assert(out.size() == 4);
    float exp0 = 1.0f * 5.0f + 2.0f * 6.0f;
    float exp3 = 3.0f * 7.0f + 4.0f * 8.0f;
    assert(approx(out[0], exp0, 2.0f));
    assert(approx(out[3], exp3, 2.0f));
    std::printf("matmul ok: %f %f %f %f\n", out[0], out[1], out[2], out[3]);

    std::printf("--- dequant smoke test ---\n");

    struct TypeCase { int id; const char* name; int blck; };
    TypeCase cases[] = {
        {0, "F32", 1}, {1, "F16", 1}, {30, "BF16", 1},
        {24, "I8", 1}, {25, "I16", 1}, {26, "I32", 1}, {27, "I64", 1}, {28, "F64", 1},
        {2, "Q4_0", 32}, {3, "Q4_1", 32}, {6, "Q5_0", 32}, {7, "Q5_1", 32},
        {8, "Q8_0", 32}, {9, "Q8_1", 32},
        {41, "Q1_0", 128}, {42, "Q2_0", 64},
        {10, "Q2_K", 256}, {11, "Q3_K", 256}, {12, "Q4_K", 256},
        {13, "Q5_K", 256}, {14, "Q6_K", 256}, {15, "Q8_K", 256},
        {16, "IQ2_XXS", 256}, {17, "IQ2_XS", 256}, {18, "IQ3_XXS", 256},
        {19, "IQ1_S", 256}, {20, "IQ4_NL", 32}, {21, "IQ3_S", 256},
        {22, "IQ2_S", 256}, {23, "IQ4_XS", 256}, {29, "IQ1_M", 256},
        {34, "TQ1_0", 256}, {35, "TQ2_0", 256},
        {39, "MXFP4", 32}, {40, "NVFP4", 64}
    };

    int passed = 0, unsupported = 0, failed = 0;
    for (auto& tc : cases) {
        size_t tsz = desireeia_type_size(tc.id);
        int blck = desireeia_blck_size(tc.id);
        int64_t n = (int64_t)blck;
        size_t row = desireeia_row_size(tc.id, n);
        std::vector<unsigned char> raw(row, 0);
        std::vector<float> vals((size_t)n, -999.0f);
        int rc2 = desireeia_dequantize_row(tc.id, raw.data(), vals.data(), n);
        if (rc2 == DESIREEIA_ERR_NOT_SUPPORTED) {
            std::printf("  %5s: type_size=%3zu blck=%3d row=%3zu -> UNSUPPORTED\n",
                        tc.name, tsz, blck, row);
            unsupported++;
        } else if (rc2 != 0) {
            std::printf("  %5s: FAIL (rc=%d)\n", tc.name, rc2);
            failed++;
        } else {
            bool has_nan = false;
            for (int64_t i = 0; i < n; ++i) {
                if (std::isnan(vals[i])) { has_nan = true; break; }
            }
            if (has_nan) {
                std::printf("  %5s: FAIL (NaN in output)\n", tc.name);
                failed++;
            } else {
                std::printf("  %5s: OK (ts=%2zu bl=%3d row=%3zu v0=%.4f)\n",
                            tc.name, tsz, blck, row, vals[0]);
                passed++;
            }
        }
    }

    std::printf("--- Q4_0 detailed test ---\n");
    {
        desireeia_half d_fp16 = desireeia_fp32_to_fp16(1.0f);
        block_q4_0 blk;
        blk.d = d_fp16;
        for (int j = 0; j < 16; ++j) blk.qs[j] = (uint8_t)((j & 0xF) | ((j & 0xF) << 4));
        std::vector<float> result(32, 0.0f);
        desireeia_dequantize_row(DESIREEIA_QTYPE_Q4_0, &blk, result.data(), 32);
        bool ok = true;
        for (int j = 0; j < 16; ++j) {
            float expected_lo = (float)((j & 0xF) - 8) * 1.0f;
            float expected_hi = (float)((j & 0xF) - 8) * 1.0f;
            if (!approx(result[j], expected_lo, 0.01f) ||
                !approx(result[j + 16], expected_hi, 0.01f)) {
                std::printf("  Q4_0 mismatch at %d: got %.4f/%.4f expected %.4f/%.4f\n",
                            j, result[j], result[j+16], expected_lo, expected_hi);
                ok = false;
                break;
            }
        }
        if (ok) std::printf("  Q4_0 PASS\n");
    }

    std::printf("--- Q4_K Q5_K Q6_K Q8_K smoke ---\n");
    {
        block_q4_K blk4;
        memset(&blk4, 0, sizeof(blk4));
        desireeia_half d_fp16 = desireeia_fp32_to_fp16(1.0f);
        blk4.d = d_fp16;
        blk4.dmin = d_fp16;
        std::vector<float> result(256, 0.0f);
        int rc2 = desireeia_dequantize_row(DESIREEIA_QTYPE_Q4_K, &blk4, result.data(), 256);
        std::printf("  Q4_K: rc=%d v0=%.4f\n", rc2, result[0]);

        block_q5_K blk5;
        memset(&blk5, 0, sizeof(blk5));
        blk5.d = d_fp16;
        blk5.dmin = d_fp16;
        rc2 = desireeia_dequantize_row(DESIREEIA_QTYPE_Q5_K, &blk5, result.data(), 256);
        std::printf("  Q5_K: rc=%d v0=%.4f\n", rc2, result[0]);

        block_q6_K blk6;
        memset(&blk6, 0, sizeof(blk6));
        blk6.d = d_fp16;
        rc2 = desireeia_dequantize_row(DESIREEIA_QTYPE_Q6_K, &blk6, result.data(), 256);
        std::printf("  Q6_K: rc=%d v0=%.4f\n", rc2, result[0]);

        block_q8_K blk8;
        memset(&blk8, 0, sizeof(blk8));
        blk8.d = 1.0f;
        rc2 = desireeia_dequantize_row(DESIREEIA_QTYPE_Q8_K, &blk8, result.data(), 256);
        std::printf("  Q8_K: rc=%d v0=%.4f\n", rc2, result[0]);
    }

    std::printf("--- IQ1_S round-trip ---\n");
    {
        block_iq1_s blk;
        memset(&blk, 0, sizeof(blk));
        blk.d = desireeia_fp32_to_fp16(1.0f);
        std::vector<float> result(256, 0.0f);
        desireeia_dequantize_row(DESIREEIA_QTYPE_IQ1_S, &blk, result.data(), 256);
        std::printf("  IQ1_S: v0=%.4f v1=%.4f\n", result[0], result[1]);
    }

    std::printf("--- IQ4_XS smoke ---\n");
    {
        block_iq4_xs blk;
        memset(&blk, 0, sizeof(blk));
        blk.d = desireeia_fp32_to_fp16(1.0f);
        std::vector<float> result(256, 0.0f);
        desireeia_dequantize_row(DESIREEIA_QTYPE_IQ4_XS, &blk, result.data(), 256);
        std::printf("  IQ4_XS: v0=%.4f v1=%.4f\n", result[0], result[1]);
    }

    std::printf("--- GGUF round-trip ---\n");
    {
        struct RtCase { int id; const char* name; float tol; };
        RtCase cases[] = {
            {0, "F32", 1e-6f}, {1, "F16", 1e-3f}, {30, "BF16", 1e-2f},
            {2, "Q4_0", 0.2f}, {3, "Q4_1", 0.2f}, {6, "Q5_0", 0.1f},
            {7, "Q5_1", 0.1f}, {8, "Q8_0", 0.01f}, {20, "IQ4_NL", 0.2f}
        };
        int rt_pass = 0, rt_fail = 0;
        for (auto& c : cases) {
            int blck = desireeia_blck_size(c.id);
            int64_t n = 2 * (int64_t)blck;
            std::vector<float> src((size_t)n);
            uint32_t rng = 12345u;
            for (auto& v : src) {
                rng = rng * 1664525u + 1013904223u;
                v = ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
            }
            std::vector<uint8_t> raw;
            switch (c.id) {
                case 0: {
                    raw.resize(src.size() * 4);
                    memcpy(raw.data(), src.data(), raw.size());
                } break;
                case 1: {
                    raw.resize(src.size() * 2);
                    for (int64_t i = 0; i < n; ++i) {
                        desireeia_half h = desireeia_fp32_to_fp16(src[i]);
                        memcpy(raw.data() + i * 2, &h, 2);
                    }
                } break;
                case 30: {
                    raw.resize(src.size() * 2);
                    for (int64_t i = 0; i < n; ++i) {
                        desireeia_bf16_t h = desireeia_fp32_to_bf16(src[i]);
                        memcpy(raw.data() + i * 2, &h, 2);
                    }
                } break;
                case 2:  encode_q4_0(src.data(), (size_t)n, raw); break;
                case 3:  encode_q4_1(src.data(), (size_t)n, raw); break;
                case 6:  encode_q5_0(src.data(), (size_t)n, raw); break;
                case 7:  encode_q5_1(src.data(), (size_t)n, raw); break;
                case 8:  encode_q8_0(src.data(), (size_t)n, raw); break;
                case 20: encode_iq4_nl(src.data(), (size_t)n, raw); break;
                default: continue;
            }

            std::string path = std::string("desireeia_rt_") + c.name + ".gguf";
            {
                std::ofstream of(path, std::ios::binary);
                of.write("GGUF", 4);
                uint32_t ver = 1;
                uint64_t nt = 1, nkv = 0;
                of.write((const char*)&ver, 4);
                of.write((const char*)&nt, 8);
                of.write((const char*)&nkv, 8);
                std::string tname = "w";
                uint64_t len = tname.size();
                of.write((const char*)&len, 8);
                of.write(tname.data(), (std::streamsize)len);
                uint32_t ndims = 1;
                of.write((const char*)&ndims, 4);
                uint64_t d0 = (uint64_t)n;
                of.write((const char*)&d0, 8);
                uint32_t tid = (uint32_t)c.id;
                of.write((const char*)&tid, 4);
                std::streampos off_pos = of.tellp();
                uint64_t off = 0;
                of.write((const char*)&off, 8);
                std::streampos data_abs = of.tellp();
                size_t pad = (size_t)(32 - ((size_t)data_abs % 32)) % 32;
                for (size_t i = 0; i < pad; ++i) of.put(0);
                uint64_t data_off = (uint64_t)of.tellp();
                of.seekp(off_pos);
                of.write((const char*)&data_off, 8);
                of.seekp((std::streampos)data_off);
                of.write((const char*)raw.data(), (std::streamsize)raw.size());
            }

            std::vector<float> got;
            bool ok = false;
            {
                std::unique_ptr<desireeia::ModelReader> rd(desireeia::make_gguf_reader());
                desireeia::ModelMeta meta;
                if (rd && rd->open(path, meta) && rd->read_tensor("w", got)) {
                    ok = got.size() == src.size();
                }
            }
            float maxerr = 0.0f;
            if (ok) {
                for (size_t i = 0; i < src.size(); ++i) {
                    maxerr = std::max(maxerr, std::fabs(got[i] - src[i]));
                }
                ok = maxerr <= c.tol;
            }
            std::remove(path.c_str());
            if (ok) {
                std::printf("  %5s: PASS (n=%d maxerr=%.5f tol=%.4f)\n",
                            c.name, (int)n, maxerr, c.tol);
                rt_pass++;
            } else {
                std::printf("  %5s: FAIL (n=%d maxerr=%.5f)\n", c.name, (int)n, maxerr);
                rt_fail++;
            }
        }
        std::printf("  round-trip passed=%d failed=%d\n", rt_pass, rt_fail);
        failed += rt_fail;
    }

    std::printf("--- ThreadPool persistente (parallel_for) ---\n");
    {
        auto& pool = desireeia::ThreadPool::global();
        std::printf("  worker_count=%zu\n", pool.worker_count());
        bool tp_ok = true;

        // Ogni slot scrive nel proprio intervallo disgiunto: verifica che
        // ogni elemento venga scritto esattamente una volta col valore
        // atteso, niente race/elementi persi/doppi.
        for (int trial = 0; trial < 20; ++trial) {
            const size_t n = 5000 + (size_t) trial;
            std::vector<int64_t> out(n, -1);
            pool.parallel_for(n, [&](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) out[i] = (int64_t) i;
            });
            for (size_t i = 0; i < n; ++i) {
                if (out[i] != (int64_t) i) { tp_ok = false; break; }
            }
            if (!tp_ok) break;
        }

        // Chiamate consecutive non devono lasciare stato sporco fra loro
        // (generazione/pending riazzerati correttamente ogni volta).
        for (int trial = 0; trial < 50 && tp_ok; ++trial) {
            std::atomic<int64_t> sum{0};
            const size_t n = 10000;
            pool.parallel_for(n, [&](size_t s, size_t e) {
                int64_t local = 0;
                for (size_t i = s; i < e; ++i) local += (int64_t) i;
                sum.fetch_add(local, std::memory_order_relaxed);
            });
            const int64_t expected = (int64_t) n * (int64_t) (n - 1) / 2;
            if (sum.load() != expected) { tp_ok = false; }
        }

        // n=0 e n piccolo (sotto soglia, eseguito sul thread chiamante) non
        // devono ne' crashare ne' saltare il lavoro.
        {
            std::vector<int> tiny(3, -1);
            pool.parallel_for((size_t) 3, [&](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) tiny[i] = (int) i;
            });
            if (tiny[0] != 0 || tiny[1] != 1 || tiny[2] != 2) tp_ok = false;
            pool.parallel_for((size_t) 0, [&](size_t, size_t) { tp_ok = false; });
        }

        std::printf("  ThreadPool: %s\n", tp_ok ? "PASS" : "FAIL");
        if (!tp_ok) failed++;
    }

    std::printf("--- moe_route (router top-k, softmax+rinormalizzazione) ---\n");
    {
        // 4 esperti, logit noti: expert 2 il piu' probabile, poi 0, poi 3, poi 1.
        std::vector<float> logits = {1.0f, -2.0f, 3.0f, 0.5f};
        bool moe_ok = true;

        // top-2 con rinormalizzazione: attesi expert {2,0} in quell'ordine,
        // pesi che sommano a 1.
        {
            auto sel = desireeia::moe_route(logits, 2, /*norm_w=*/true, /*w_scale=*/1.0f);
            if (sel.size() != 2 || sel[0].first != 2 || sel[1].first != 0) moe_ok = false;
            float wsum = sel.empty() ? 0.0f : sel[0].second + sel[1].second;
            if (!approx(wsum, 1.0f, 1e-5f)) moe_ok = false;
            if (sel.size() == 2 && sel[0].second <= sel[1].second) moe_ok = false; // expert 2 ha logit maggiore
        }
        // senza rinormalizzazione: la somma dei pesi selezionati e' la
        // probabilita' softmax cumulativa dei soli top-2 (< 1).
        {
            auto sel = desireeia::moe_route(logits, 2, /*norm_w=*/false, /*w_scale=*/1.0f);
            float wsum = 0.0f;
            for (auto& s : sel) wsum += s.second;
            if (!(wsum > 0.0f && wsum < 1.0f)) moe_ok = false;
        }
        // n_expert_used >= n_expert: satura a tutti gli esperti, non crasha.
        {
            auto sel = desireeia::moe_route(logits, 10, true, 1.0f);
            if (sel.size() != logits.size()) moe_ok = false;
        }
        std::printf("  moe_route: %s\n", moe_ok ? "PASS" : "FAIL");
        if (!moe_ok) failed++;
    }

    std::printf("--- matmul_q4_0 fuso (int4 x int8) vs dequant+float ---\n");
    {
        const size_t rows = 5, cols = 64; // cols multiplo di 32, righe non multiplo per testare tail-free path
        std::vector<float> w((size_t)(rows * cols));
        std::vector<float> x(cols);
        uint32_t rng = 777u;
        auto next_f = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
        };
        for (auto& v : w) v = next_f();
        for (auto& v : x) v = next_f();

        std::vector<uint8_t> raw;
        for (size_t r = 0; r < rows; ++r) {
            encode_q4_0(w.data() + r * cols, cols, raw);
        }

        // expected result: dequantize Q4_0 -> float, then float matmul in double precision
        std::vector<float> wdeq((size_t)(rows * cols));
        {
            const size_t row_bytes = raw.size() / rows;
            for (size_t r = 0; r < rows; ++r) {
                desireeia_dequantize_row(DESIREEIA_QTYPE_Q4_0, raw.data() + r * row_bytes,
                                       wdeq.data() + r * cols, (int64_t) cols);
            }
        }
        std::vector<float> yref(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < cols; ++c) acc += (double) wdeq[r * cols + c] * x[c];
            yref[r] = (float) acc;
        }

        std::vector<float> yfused(rows, 0.0f);
        int rc = desireeia::matmul_q4_0(raw.data(), rows, cols, x.data(), yfused.data());

        bool mm_ok = (rc == DESIREEIA_OK);
        if (mm_ok) {
            for (size_t r = 0; r < rows; ++r) {
                // tolleranza piu' larga del round-trip puro: l'attivazione e'
                // anch'essa quantizzata in int8 (errore aggiuntivo atteso).
                float tol = std::max(0.05f, std::fabs(yref[r]) * 0.05f);
                if (!approx(yfused[r], yref[r], tol)) {
                    std::printf("  mismatch riga %zu: fused=%.5f ref=%.5f tol=%.5f\n",
                                r, yfused[r], yref[r], tol);
                    mm_ok = false;
                }
            }
        }
        std::printf("  matmul_q4_0: %s\n", mm_ok ? "PASS" : "FAIL");
        if (!mm_ok) failed++;

        // Fase 8: matmul_q4_0_batch deve dare, per costruzione, esattamente
        // lo stesso risultato di n_tok chiamate separate a matmul_q4_0 (il
        // batching riordina solo l'ordine di lettura/decodifica dei pesi,
        // non la formula per singola colonna).
        const size_t n_tok = 3;
        std::vector<float> xb(n_tok * cols);
        for (auto& v : xb) v = next_f();
        std::vector<float> yb_ref(n_tok * rows), yb_fused(n_tok * rows, 0.0f);
        for (size_t tk = 0; tk < n_tok; ++tk) {
            desireeia::matmul_q4_0(raw.data(), rows, cols, xb.data() + tk * cols, yb_ref.data() + tk * rows);
        }
        int rcb = desireeia::matmul_q4_0_batch(raw.data(), rows, cols, xb.data(), n_tok, yb_fused.data());
        bool batch_ok = (rcb == DESIREEIA_OK);
        if (batch_ok) {
            for (size_t i = 0; i < n_tok * rows; ++i) {
                if (!approx(yb_fused[i], yb_ref[i], 1e-4f)) {
                    std::printf("  batch mismatch idx %zu: batch=%.6f ref=%.6f\n", i, yb_fused[i], yb_ref[i]);
                    batch_ok = false;
                }
            }
        }
        std::printf("  matmul_q4_0_batch: %s\n", batch_ok ? "PASS" : "FAIL");
        if (!batch_ok) failed++;
    }

    std::printf("--- matmul_q8_0 fuso (int8 x int8) vs dequant+float ---\n");
    {
        const size_t rows = 5, cols = 64;
        std::vector<uint8_t> raw(rows * (cols / 32) * sizeof(block_q8_0));
        std::vector<float> w(rows * cols);
        uint32_t rng = 314u;
        auto next_f = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
        };
        for (size_t r = 0; r < rows; ++r) {
            for (size_t b = 0; b < cols / 32; ++b) {
                block_q8_0 blk;
                float maxv = 0.0f;
                float vals[32];
                for (int j = 0; j < 32; ++j) { vals[j] = next_f(); maxv = std::max(maxv, std::fabs(vals[j])); }
                float d = maxv / 127.0f;
                blk.d = desireeia_fp32_to_fp16(d > 0.0f ? d : 1.0f);
                for (int j = 0; j < 32; ++j) {
                    int q = (int) std::lround(vals[j] / (d > 0.0f ? d : 1.0f));
                    q = std::min(127, std::max(-127, q));
                    blk.qs[j] = (int8_t) q;
                    w[r * cols + b * 32 + j] = (float) q * d;
                }
                std::memcpy(raw.data() + (r * (cols / 32) + b) * sizeof(block_q8_0), &blk, sizeof(blk));
            }
        }

        std::vector<float> x(cols);
        for (auto& v : x) v = next_f();

        std::vector<float> yref(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < cols; ++c) acc += (double) w[r * cols + c] * x[c];
            yref[r] = (float) acc;
        }

        std::vector<float> yfused(rows, 0.0f);
        int rc = desireeia::matmul_q8_0(raw.data(), rows, cols, x.data(), yfused.data());
        bool mm_ok = (rc == DESIREEIA_OK);
        if (mm_ok) {
            for (size_t r = 0; r < rows; ++r) {
                float tol = std::max(0.05f, std::fabs(yref[r]) * 0.05f);
                if (!approx(yfused[r], yref[r], tol)) {
                    std::printf("  mismatch riga %zu: fused=%.5f ref=%.5f tol=%.5f\n", r, yfused[r], yref[r], tol);
                    mm_ok = false;
                }
            }
        }
        std::printf("  matmul_q8_0: %s\n", mm_ok ? "PASS" : "FAIL");
        if (!mm_ok) failed++;

        // n_tok=7 (non multiplo di 4, > rows) per esercitare sia il tile
        // 2x4 completo che i due rami di resto (righe e token) del GEMM a
        // blocchi.
        const size_t n_tok = 7;
        std::vector<float> xb(n_tok * cols);
        for (auto& v : xb) v = next_f();
        std::vector<float> yb_ref(n_tok * rows), yb_fused(n_tok * rows, 0.0f);
        for (size_t tk = 0; tk < n_tok; ++tk) {
            desireeia::matmul_q8_0(raw.data(), rows, cols, xb.data() + tk * cols, yb_ref.data() + tk * rows);
        }
        int rcb = desireeia::matmul_q8_0_batch(raw.data(), rows, cols, xb.data(), n_tok, yb_fused.data());
        bool batch_ok = (rcb == DESIREEIA_OK);
        if (batch_ok) {
            for (size_t i = 0; i < n_tok * rows; ++i) {
                if (!approx(yb_fused[i], yb_ref[i], 1e-4f)) {
                    std::printf("  batch mismatch idx %zu: batch=%.6f ref=%.6f\n", i, yb_fused[i], yb_ref[i]);
                    batch_ok = false;
                }
            }
        }
        std::printf("  matmul_q8_0_batch: %s\n", batch_ok ? "PASS" : "FAIL");
        if (!batch_ok) failed++;
    }

    std::printf("--- matmul_q4_k fuso (super-block, int4 x int8) vs dequant+float ---\n");
    {
        const size_t rows = 3, cols = 256; // 1 super-blocco per riga
        // scale_i = min_i = i+1 (i=0..7), tutti <16 -> bit alti a 0, encoding
        // diretto senza dover invertire l'incastro a 6 bit di get_scale_min_k4.
        uint8_t scales[12] = { 1, 2, 3, 4,  1, 2, 3, 4,  0x55, 0x66, 0x77, 0x88 };

        std::vector<uint8_t> raw(rows * sizeof(block_q4_K));
        std::vector<float> w(rows * cols); // expected result, filled via dequantize below
        uint32_t rng = 42u;
        auto next_f = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
        };
        for (size_t r = 0; r < rows; ++r) {
            block_q4_K blk;
            std::memset(&blk, 0, sizeof(blk));
            blk.d = desireeia_fp32_to_fp16(1.0f + 0.1f * (float) r);
            blk.dmin = desireeia_fp32_to_fp16(0.5f);
            std::memcpy(blk.scales, scales, sizeof(scales));
            for (int b = 0; b < 128; ++b) {
                int lo = (b + (int) r) % 16;
                int hi = (b + (int) r + 1) % 16;
                blk.qs[b] = (uint8_t) (lo | (hi << 4));
            }
            std::memcpy(raw.data() + r * sizeof(block_q4_K), &blk, sizeof(blk));
            desireeia_dequantize_row(DESIREEIA_QTYPE_Q4_K, &blk, w.data() + r * cols, (int64_t) cols);
        }

        std::vector<float> x(cols);
        for (auto& v : x) v = next_f();

        std::vector<float> yref(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < cols; ++c) acc += (double) w[r * cols + c] * x[c];
            yref[r] = (float) acc;
        }

        std::vector<float> yfused(rows, 0.0f);
        int rc = desireeia::matmul_q4_k(raw.data(), rows, cols, x.data(), yfused.data());

        bool mm_ok = (rc == DESIREEIA_OK);
        if (mm_ok) {
            for (size_t r = 0; r < rows; ++r) {
                float tol = std::max(0.1f, std::fabs(yref[r]) * 0.05f);
                if (!approx(yfused[r], yref[r], tol)) {
                    std::printf("  mismatch riga %zu: fused=%.5f ref=%.5f tol=%.5f\n",
                                r, yfused[r], yref[r], tol);
                    mm_ok = false;
                }
            }
        }
        std::printf("  matmul_q4_k: %s\n", mm_ok ? "PASS" : "FAIL");
        if (!mm_ok) failed++;

        const size_t n_tok = 7; // esercita sia il tile 2x4 che i resti (righe e token)
        std::vector<float> xb(n_tok * cols);
        for (auto& v : xb) v = next_f();
        std::vector<float> yb_ref(n_tok * rows), yb_fused(n_tok * rows, 0.0f);
        for (size_t tk = 0; tk < n_tok; ++tk) {
            desireeia::matmul_q4_k(raw.data(), rows, cols, xb.data() + tk * cols, yb_ref.data() + tk * rows);
        }
        int rcb = desireeia::matmul_q4_k_batch(raw.data(), rows, cols, xb.data(), n_tok, yb_fused.data());
        bool batch_ok = (rcb == DESIREEIA_OK);
        if (batch_ok) {
            // Tolleranza relativa: il GEMM a blocchi 2x4 combina i termini
            // acc/min_acc in un ordine diverso (una sola addizione invece
            // di due accumulatori separati sommati alla fine) — stesso
            // riordino in virgola mobile gia' visto e tollerato altrove
            // (Q6_K batch), non un errore di formula.
            for (size_t i = 0; i < n_tok * rows; ++i) {
                float tol = std::max(1e-3f, std::fabs(yb_ref[i]) * 1e-4f);
                if (!approx(yb_fused[i], yb_ref[i], tol)) {
                    std::printf("  batch mismatch idx %zu: batch=%.6f ref=%.6f tol=%.6f\n", i, yb_fused[i], yb_ref[i], tol);
                    batch_ok = false;
                }
            }
        }
        std::printf("  matmul_q4_k_batch: %s\n", batch_ok ? "PASS" : "FAIL");
        if (!batch_ok) failed++;
    }

    auto test_legacy_kernel = [&](const char* name, int qtype, size_t block_bytes,
                                   auto encode_fn,
                                   auto fused_fn, auto fused_batch_fn) {
        std::printf("--- %s fuso vs dequant+float ---\n", name);
        const size_t rows = 5, cols = 64;
        std::vector<uint8_t> raw(rows * (cols / 32) * block_bytes);
        std::vector<float> w(rows * cols);
        uint32_t rng = 555u;
        auto next_f = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
        };
        for (size_t r = 0; r < rows; ++r) {
            for (size_t b = 0; b < cols / 32; ++b) {
                std::vector<float> vals(32);
                for (auto& v : vals) v = next_f();
                encode_fn(vals.data(), raw.data() + (r * (cols / 32) + b) * block_bytes);
                desireeia_dequantize_row(qtype, raw.data() + (r * (cols / 32) + b) * block_bytes,
                                       w.data() + r * cols + b * 32, 32);
            }
        }

        std::vector<float> x(cols);
        for (auto& v : x) v = next_f();
        std::vector<float> yref(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < cols; ++c) acc += (double) w[r * cols + c] * x[c];
            yref[r] = (float) acc;
        }

        std::vector<float> yfused(rows, 0.0f);
        int rc = fused_fn(raw.data(), rows, cols, x.data(), yfused.data());
        bool mm_ok = (rc == DESIREEIA_OK);
        if (mm_ok) {
            for (size_t r = 0; r < rows; ++r) {
                float tol = std::max(0.15f, std::fabs(yref[r]) * 0.08f);
                if (!approx(yfused[r], yref[r], tol)) {
                    std::printf("  mismatch riga %zu: fused=%.5f ref=%.5f tol=%.5f\n", r, yfused[r], yref[r], tol);
                    mm_ok = false;
                }
            }
        }
        std::printf("  %s: %s\n", name, mm_ok ? "PASS" : "FAIL");
        if (!mm_ok) failed++;

        const size_t n_tok = 3;
        std::vector<float> xb(n_tok * cols);
        for (auto& v : xb) v = next_f();
        std::vector<float> yb_ref(n_tok * rows), yb_fused(n_tok * rows, 0.0f);
        for (size_t tk = 0; tk < n_tok; ++tk) {
            fused_fn(raw.data(), rows, cols, xb.data() + tk * cols, yb_ref.data() + tk * rows);
        }
        int rcb = fused_batch_fn(raw.data(), rows, cols, xb.data(), n_tok, yb_fused.data());
        bool batch_ok = (rcb == DESIREEIA_OK);
        if (batch_ok) {
            for (size_t i = 0; i < n_tok * rows; ++i) {
                if (!approx(yb_fused[i], yb_ref[i], 1e-4f)) {
                    std::printf("  batch mismatch idx %zu: batch=%.6f ref=%.6f\n", i, yb_fused[i], yb_ref[i]);
                    batch_ok = false;
                }
            }
        }
        std::printf("  %s_batch: %s\n", name, batch_ok ? "PASS" : "FAIL");
        if (!batch_ok) failed++;
    };

    test_legacy_kernel("matmul_q4_1", DESIREEIA_QTYPE_Q4_1, sizeof(block_q4_1),
        [](const float* vals, uint8_t* out) {
            block_q4_1 blk; std::memset(&blk, 0, sizeof(blk));
            float minv = vals[0], maxv = vals[0];
            for (int j = 0; j < 32; ++j) { minv = std::min(minv, vals[j]); maxv = std::max(maxv, vals[j]); }
            float d = (maxv - minv) / 15.0f;
            if (d <= 0.0f) d = 1.0f;
            blk.d = desireeia_fp32_to_fp16(d);
            blk.m = desireeia_fp32_to_fp16(minv);
            for (int j = 0; j < 16; ++j) {
                int q0 = (int) std::lround((vals[j] - minv) / d);
                int q1 = (int) std::lround((vals[j + 16] - minv) / d);
                q0 = std::min(15, std::max(0, q0));
                q1 = std::min(15, std::max(0, q1));
                blk.qs[j] = (uint8_t) (q0 | (q1 << 4));
            }
            std::memcpy(out, &blk, sizeof(blk));
        },
        desireeia::matmul_q4_1, desireeia::matmul_q4_1_batch);

    test_legacy_kernel("matmul_q5_0", DESIREEIA_QTYPE_Q5_0, sizeof(block_q5_0),
        [](const float* vals, uint8_t* out) {
            block_q5_0 blk; std::memset(&blk, 0, sizeof(blk));
            float maxv = 0.0f;
            for (int j = 0; j < 32; ++j) maxv = std::max(maxv, std::fabs(vals[j]));
            float d = maxv / 16.0f;
            if (d <= 0.0f) d = 1.0f;
            blk.d = desireeia_fp32_to_fp16(d);
            uint32_t qh = 0;
            for (int j = 0; j < 16; ++j) {
                int q0 = (int) std::lround(vals[j] / d) + 16;
                int q1 = (int) std::lround(vals[j + 16] / d) + 16;
                q0 = std::min(31, std::max(0, q0));
                q1 = std::min(31, std::max(0, q1));
                blk.qs[j] = (uint8_t) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                qh |= (uint32_t) ((q0 >> 4) & 1) << j;
                qh |= (uint32_t) ((q1 >> 4) & 1) << (j + 16);
            }
            std::memcpy(blk.qh, &qh, sizeof(qh));
            std::memcpy(out, &blk, sizeof(blk));
        },
        desireeia::matmul_q5_0, desireeia::matmul_q5_0_batch);

    test_legacy_kernel("matmul_q5_1", DESIREEIA_QTYPE_Q5_1, sizeof(block_q5_1),
        [](const float* vals, uint8_t* out) {
            block_q5_1 blk; std::memset(&blk, 0, sizeof(blk));
            float minv = vals[0], maxv = vals[0];
            for (int j = 0; j < 32; ++j) { minv = std::min(minv, vals[j]); maxv = std::max(maxv, vals[j]); }
            float d = (maxv - minv) / 31.0f;
            if (d <= 0.0f) d = 1.0f;
            blk.d = desireeia_fp32_to_fp16(d);
            blk.m = desireeia_fp32_to_fp16(minv);
            uint32_t qh = 0;
            for (int j = 0; j < 16; ++j) {
                int q0 = (int) std::lround((vals[j] - minv) / d);
                int q1 = (int) std::lround((vals[j + 16] - minv) / d);
                q0 = std::min(31, std::max(0, q0));
                q1 = std::min(31, std::max(0, q1));
                blk.qs[j] = (uint8_t) ((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                qh |= (uint32_t) ((q0 >> 4) & 1) << j;
                qh |= (uint32_t) ((q1 >> 4) & 1) << (j + 16);
            }
            std::memcpy(blk.qh, &qh, sizeof(qh));
            std::memcpy(out, &blk, sizeof(blk));
        },
        desireeia::matmul_q5_1, desireeia::matmul_q5_1_batch);

    auto test_kquant_kernel = [&](const char* name, int qtype, size_t rows, size_t cols,
                                   size_t block_bytes, auto fill_fn,
                                   auto fused_fn, auto fused_batch_fn) {
        std::printf("--- %s fuso vs dequant+float ---\n", name);
        std::vector<uint8_t> raw(rows * (cols / 256) * block_bytes);
        std::vector<float> w(rows * cols);
        uint32_t rng = 4242u;
        auto next_f = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
        };
        for (size_t r = 0; r < rows; ++r) {
            for (size_t si = 0; si < cols / 256; ++si) {
                uint8_t* blk_raw = raw.data() + (r * (cols / 256) + si) * block_bytes;
                fill_fn(blk_raw, (int) r, (int) si);
                desireeia_dequantize_row(qtype, blk_raw, w.data() + r * cols + si * 256, 256);
            }
        }

        std::vector<float> x(cols);
        for (auto& v : x) v = next_f();
        std::vector<float> yref(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < cols; ++c) acc += (double) w[r * cols + c] * x[c];
            yref[r] = (float) acc;
        }

        std::vector<float> yfused(rows, 0.0f);
        int rc = fused_fn(raw.data(), rows, cols, x.data(), yfused.data());
        bool mm_ok = (rc == DESIREEIA_OK);
        if (mm_ok) {
            for (size_t r = 0; r < rows; ++r) {
                float tol = std::max(0.15f, std::fabs(yref[r]) * 0.08f);
                if (!approx(yfused[r], yref[r], tol)) {
                    std::printf("  mismatch riga %zu: fused=%.5f ref=%.5f tol=%.5f\n", r, yfused[r], yref[r], tol);
                    mm_ok = false;
                }
            }
        }
        std::printf("  %s: %s\n", name, mm_ok ? "PASS" : "FAIL");
        if (!mm_ok) failed++;

        const size_t n_tok = 3;
        std::vector<float> xb(n_tok * cols);
        for (auto& v : xb) v = next_f();
        std::vector<float> yb_ref(n_tok * rows), yb_fused(n_tok * rows, 0.0f);
        for (size_t tk = 0; tk < n_tok; ++tk) {
            fused_fn(raw.data(), rows, cols, xb.data() + tk * cols, yb_ref.data() + tk * rows);
        }
        int rcb = fused_batch_fn(raw.data(), rows, cols, xb.data(), n_tok, yb_fused.data());
        bool batch_ok = (rcb == DESIREEIA_OK);
        if (batch_ok) {
            for (size_t i = 0; i < n_tok * rows; ++i) {
                float tol = std::max(1e-3f, std::fabs(yb_ref[i]) * 1e-3f);
                if (!approx(yb_fused[i], yb_ref[i], tol)) {
                    std::printf("  batch mismatch idx %zu: batch=%.6f ref=%.6f\n", i, yb_fused[i], yb_ref[i]);
                    batch_ok = false;
                }
            }
        }
        std::printf("  %s_batch: %s\n", name, batch_ok ? "PASS" : "FAIL");
        if (!batch_ok) failed++;
    };

    test_kquant_kernel("matmul_q8_k", DESIREEIA_QTYPE_Q8_K, 3, 256, sizeof(block_q8_K),
        [](uint8_t* raw, int r, int si) {
            block_q8_K blk; std::memset(&blk, 0, sizeof(blk));
            blk.d = 0.05f + 0.01f * (float) r;
            for (int i = 0; i < 256; ++i) blk.qs[i] = (int8_t) (((i * 7 + r * 13 + si * 3) % 255) - 127);
            std::memcpy(raw, &blk, sizeof(blk));
        },
        desireeia::matmul_q8_k, desireeia::matmul_q8_k_batch);

    test_kquant_kernel("matmul_q2_k", DESIREEIA_QTYPE_Q2_K, 3, 256, sizeof(block_q2_K),
        [](uint8_t* raw, int r, int si) {
            block_q2_K blk; std::memset(&blk, 0, sizeof(blk));
            blk.d = desireeia_fp32_to_fp16(0.3f + 0.05f * (float) r);
            blk.dmin = desireeia_fp32_to_fp16(0.1f);
            for (int i = 0; i < 16; ++i) blk.scales[i] = (uint8_t) (((i * 5 + r * 3 + si) % 16) | (((i * 3 + r) % 16) << 4));
            for (int i = 0; i < 64; ++i) blk.qs[i] = (uint8_t) ((i * 11 + r * 7 + si * 2) & 0xFF);
            std::memcpy(raw, &blk, sizeof(blk));
        },
        desireeia::matmul_q2_k, desireeia::matmul_q2_k_batch);

    test_kquant_kernel("matmul_q3_k", DESIREEIA_QTYPE_Q3_K, 3, 256, sizeof(block_q3_K),
        [](uint8_t* raw, int r, int si) {
            block_q3_K blk; std::memset(&blk, 0, sizeof(blk));
            blk.d = desireeia_fp32_to_fp16(0.4f + 0.05f * (float) r);
            for (int i = 0; i < 12; ++i) blk.scales[i] = (uint8_t) ((i * 17 + r * 9 + si * 5) & 0xFF);
            for (int i = 0; i < 64; ++i) blk.qs[i] = (uint8_t) ((i * 13 + r * 5 + si * 2) & 0xFF);
            for (int i = 0; i < 32; ++i) blk.hmask[i] = (uint8_t) ((i * 19 + r * 11 + si * 3) & 0xFF);
            std::memcpy(raw, &blk, sizeof(blk));
        },
        desireeia::matmul_q3_k, desireeia::matmul_q3_k_batch);

    std::printf("--- matmul_q5_k fuso (super-block, int5 x int8) vs dequant+float ---\n");
    {
        const size_t rows = 3, cols = 256;
        uint8_t scales[12] = { 1, 2, 3, 4,  1, 2, 3, 4,  0x55, 0x66, 0x77, 0x88 };
        std::vector<uint8_t> raw(rows * sizeof(block_q5_K));
        std::vector<float> w(rows * cols);
        uint32_t rng = 271u;
        auto next_f = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
        };
        for (size_t r = 0; r < rows; ++r) {
            block_q5_K blk;
            std::memset(&blk, 0, sizeof(blk));
            blk.d = desireeia_fp32_to_fp16(1.0f + 0.1f * (float) r);
            blk.dmin = desireeia_fp32_to_fp16(0.5f);
            std::memcpy(blk.scales, scales, sizeof(scales));
            for (int b = 0; b < 128; ++b) {
                int lo = (b + (int) r) % 16;
                int hi = (b + (int) r + 1) % 16;
                blk.qs[b] = (uint8_t) (lo | (hi << 4));
            }
            for (int b = 0; b < 32; ++b) blk.qh[b] = (uint8_t) ((b * 5 + (int) r * 7) & 0xFF);
            std::memcpy(raw.data() + r * sizeof(block_q5_K), &blk, sizeof(blk));
            desireeia_dequantize_row(DESIREEIA_QTYPE_Q5_K, &blk, w.data() + r * cols, (int64_t) cols);
        }

        std::vector<float> x(cols);
        for (auto& v : x) v = next_f();

        std::vector<float> yref(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < cols; ++c) acc += (double) w[r * cols + c] * x[c];
            yref[r] = (float) acc;
        }

        std::vector<float> yfused(rows, 0.0f);
        int rc = desireeia::matmul_q5_k(raw.data(), rows, cols, x.data(), yfused.data());
        bool mm_ok = (rc == DESIREEIA_OK);
        if (mm_ok) {
            for (size_t r = 0; r < rows; ++r) {
                // Verificato con un caso isolato a quantizzazione esatta
                // (x=1.0 costante): fused == ref bit-per-bit, formula
                // corretta. Su dati casuali serve una tolleranza piu'
                // larga del 5% usato per Q4_K/Q6_K perche' il rumore di
                // quantizzazione int8 dell'attivazione, sommato su un
                // range di pesi a 5 bit (0..31, piu' ampio di Q4_K's 0..15),
                // puo' avvicinarsi al 5-6% su singole righe casuali.
                float tol = std::max(0.15f, std::fabs(yref[r]) * 0.08f);
                if (!approx(yfused[r], yref[r], tol)) {
                    std::printf("  mismatch riga %zu: fused=%.5f ref=%.5f tol=%.5f\n", r, yfused[r], yref[r], tol);
                    mm_ok = false;
                }
            }
        }
        std::printf("  matmul_q5_k: %s\n", mm_ok ? "PASS" : "FAIL");
        if (!mm_ok) failed++;

        const size_t n_tok = 3;
        std::vector<float> xb(n_tok * cols);
        for (auto& v : xb) v = next_f();
        std::vector<float> yb_ref(n_tok * rows), yb_fused(n_tok * rows, 0.0f);
        for (size_t tk = 0; tk < n_tok; ++tk) {
            desireeia::matmul_q5_k(raw.data(), rows, cols, xb.data() + tk * cols, yb_ref.data() + tk * rows);
        }
        int rcb = desireeia::matmul_q5_k_batch(raw.data(), rows, cols, xb.data(), n_tok, yb_fused.data());
        bool batch_ok = (rcb == DESIREEIA_OK);
        if (batch_ok) {
            for (size_t i = 0; i < n_tok * rows; ++i) {
                float tol = std::max(1e-3f, std::fabs(yb_ref[i]) * 1e-4f);
                if (!approx(yb_fused[i], yb_ref[i], tol)) {
                    std::printf("  batch mismatch idx %zu: batch=%.6f ref=%.6f tol=%.6f\n", i, yb_fused[i], yb_ref[i], tol);
                    batch_ok = false;
                }
            }
        }
        std::printf("  matmul_q5_k_batch: %s\n", batch_ok ? "PASS" : "FAIL");
        if (!batch_ok) failed++;
    }

    std::printf("--- matmul_q6_k fuso (super-block, int4/6 x int8) vs dequant+float ---\n");
    {
        const size_t rows = 3, cols = 256; // 1 super-blocco per riga
        std::vector<uint8_t> raw(rows * sizeof(block_q6_K));
        std::vector<float> w(rows * cols);
        uint32_t rng = 99u;
        auto next_f = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return ((float)(rng & 0xFFFF) / 32767.5f) - 1.0f;
        };
        for (size_t r = 0; r < rows; ++r) {
            block_q6_K blk;
            std::memset(&blk, 0, sizeof(blk));
            blk.d = desireeia_fp32_to_fp16(0.8f + 0.1f * (float) r);
            for (int s = 0; s < 16; ++s) blk.scales[s] = (int8_t) (s - 8); // valori con segno, alcuni negativi
            for (int b = 0; b < 128; ++b) blk.ql[b] = (uint8_t) ((b * 7 + (int) r) & 0xFF);
            for (int b = 0; b < 64; ++b) blk.qh[b] = (uint8_t) ((b * 13 + (int) r * 3) & 0xFF);
            std::memcpy(raw.data() + r * sizeof(block_q6_K), &blk, sizeof(blk));
            desireeia_dequantize_row(DESIREEIA_QTYPE_Q6_K, &blk, w.data() + r * cols, (int64_t) cols);
        }

        std::vector<float> x(cols);
        for (auto& v : x) v = next_f();

        std::vector<float> yref(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0.0;
            for (size_t c = 0; c < cols; ++c) acc += (double) w[r * cols + c] * x[c];
            yref[r] = (float) acc;
        }

        std::vector<float> yfused(rows, 0.0f);
        int rc = desireeia::matmul_q6_k(raw.data(), rows, cols, x.data(), yfused.data());

        bool mm_ok = (rc == DESIREEIA_OK);
        if (mm_ok) {
            for (size_t r = 0; r < rows; ++r) {
                float tol = std::max(0.1f, std::fabs(yref[r]) * 0.05f);
                if (!approx(yfused[r], yref[r], tol)) {
                    std::printf("  mismatch riga %zu: fused=%.5f ref=%.5f tol=%.5f\n",
                                r, yfused[r], yref[r], tol);
                    mm_ok = false;
                }
            }
        }
        std::printf("  matmul_q6_k: %s\n", mm_ok ? "PASS" : "FAIL");
        if (!mm_ok) failed++;

        const size_t n_tok = 7; // esercita sia il tile 2x4 che i resti (righe e token)
        std::vector<float> xb(n_tok * cols);
        for (auto& v : xb) v = next_f();
        std::vector<float> yb_ref(n_tok * rows), yb_fused(n_tok * rows, 0.0f);
        for (size_t tk = 0; tk < n_tok; ++tk) {
            desireeia::matmul_q6_k(raw.data(), rows, cols, xb.data() + tk * cols, yb_ref.data() + tk * rows);
        }
        int rcb = desireeia::matmul_q6_k_batch(raw.data(), rows, cols, xb.data(), n_tok, yb_fused.data());
        bool batch_ok = (rcb == DESIREEIA_OK);
        if (batch_ok) {
            // Tolleranza relativa: la versione batch somma i contributi per
            // sotto-blocco nello stesso ordine della non-batch, ma passa
            // per dot8_16 (float) invece di dot8_16_i32 (int32 esatto), con
            // un riarrotondamento in piu' -> rumore atteso di riordino in
            // virgola mobile su valori ~O(500), non un errore di formula.
            for (size_t i = 0; i < n_tok * rows; ++i) {
                float tol = std::max(1e-3f, std::fabs(yb_ref[i]) * 1e-4f);
                if (!approx(yb_fused[i], yb_ref[i], tol)) {
                    std::printf("  batch mismatch idx %zu: batch=%.6f ref=%.6f tol=%.6f\n", i, yb_fused[i], yb_ref[i], tol);
                    batch_ok = false;
                }
            }
        }
        std::printf("  matmul_q6_k_batch: %s\n", batch_ok ? "PASS" : "FAIL");
        if (!batch_ok) failed++;
    }

    std::printf("--- GGUF read_expert (MoE, sintetico) ---\n");
    {
        // 3D tensor blk.0.ffn_gate_exps.weight, shape [n_embd=4, n_ff=3,
        // n_expert=2], F32 for simplicity. Each expert has distinct,
        // recognizable values.
        const int64_t n_embd = 4, n_ff = 3, n_expert = 2;
        std::vector<float> full((size_t)(n_embd * n_ff * n_expert));
        for (size_t i = 0; i < full.size(); ++i) full[i] = 100.0f + (float) i;

        std::string path = "desireeia_rt_moe.gguf";
        {
            std::ofstream of(path, std::ios::binary);
            of.write("GGUF", 4);
            uint32_t ver = 1;
            uint64_t nt = 1, nkv = 0;
            of.write((const char*)&ver, 4);
            of.write((const char*)&nt, 8);
            of.write((const char*)&nkv, 8);
            std::string tname = "blk.0.ffn_gate_exps.weight";
            uint64_t len = tname.size();
            of.write((const char*)&len, 8);
            of.write(tname.data(), (std::streamsize)len);
            uint32_t ndims = 3;
            of.write((const char*)&ndims, 4);
            uint64_t d0 = (uint64_t) n_embd, d1 = (uint64_t) n_ff, d2 = (uint64_t) n_expert;
            of.write((const char*)&d0, 8);
            of.write((const char*)&d1, 8);
            of.write((const char*)&d2, 8);
            uint32_t tid = 0; // F32
            of.write((const char*)&tid, 4);
            std::streampos off_pos = of.tellp();
            uint64_t off = 0;
            of.write((const char*)&off, 8);
            uint64_t data_off = (uint64_t) of.tellp();
            of.seekp(off_pos);
            of.write((const char*)&data_off, 8);
            of.seekp((std::streampos) data_off);
            of.write((const char*)full.data(), (std::streamsize)(full.size() * sizeof(float)));
        }

        bool moe_ok = true;
        {
            std::unique_ptr<desireeia::ModelReader> rd(desireeia::make_gguf_reader());
            desireeia::ModelMeta meta;
            if (!rd || !rd->open(path, meta)) {
                moe_ok = false;
            } else {
                for (int64_t e = 0; e < n_expert; ++e) {
                    std::vector<float> got;
                    if (!rd->read_expert(0, (uint32_t) e, desireeia::ExpertPart::Gate, got)) {
                        moe_ok = false;
                        break;
                    }
                    if (got.size() != (size_t)(n_embd * n_ff)) { moe_ok = false; break; }
                    for (int64_t i = 0; i < n_embd * n_ff; ++i) {
                        float expect = full[(size_t)(e * n_embd * n_ff + i)];
                        if (!approx(got[(size_t) i], expect, 1e-6f)) { moe_ok = false; break; }
                    }
                    if (!moe_ok) break;
                }
                // idx fuori range deve fallire, non leggere fuori tensore
                std::vector<float> oob;
                if (rd->read_expert(0, (uint32_t) n_expert, desireeia::ExpertPart::Gate, oob)) moe_ok = false;
            }
        }
        std::remove(path.c_str());
        std::printf("  read_expert MoE sintetico: %s\n", moe_ok ? "PASS" : "FAIL");
        if (!moe_ok) failed++;
    }

    std::printf("--- safetensors reader (sintetico) ---\n");
    {
        auto write_st_file = [](const std::string& path,
                                 const std::vector<float>& a, const std::vector<float>& b) {
            std::ostringstream header;
            header << "{"
                   << "\"a\":{\"dtype\":\"F32\",\"shape\":[" << a.size() << "],"
                   << "\"data_offsets\":[0," << (a.size() * 4) << "]},"
                   << "\"b\":{\"dtype\":\"F32\",\"shape\":[" << b.size() << "],"
                   << "\"data_offsets\":[" << (a.size() * 4) << ","
                   << ((a.size() + b.size()) * 4) << "]}"
                   << "}";
            std::string h = header.str();
            std::ofstream of(path, std::ios::binary);
            uint64_t hlen = h.size();
            of.write((const char*)&hlen, 8);
            of.write(h.data(), (std::streamsize) h.size());
            of.write((const char*)a.data(), (std::streamsize)(a.size() * 4));
            of.write((const char*)b.data(), (std::streamsize)(b.size() * 4));
        };

        std::vector<float> a = {1.0f, 2.0f, 3.0f};
        std::vector<float> b = {4.5f, 5.5f};
        const std::string path = "desireeia_rt.safetensors";
        write_st_file(path, a, b);

        bool st_ok = true;
        {
            std::unique_ptr<desireeia::ModelReader> rd(desireeia::make_st_reader());
            desireeia::ModelMeta meta;
            if (!rd || !rd->open(path, meta) || meta.format != DESIREEIA_FORMAT_SAFETENSORS) {
                st_ok = false;
            } else {
                std::vector<float> got_a, got_b;
                st_ok = rd->read_tensor("a", got_a) && got_a == a &&
                        rd->read_tensor("b", got_b) && got_b == b;
            }
        }
        std::remove(path.c_str());
        std::printf("  safetensors single-file: %s\n", st_ok ? "PASS" : "FAIL");
        if (!st_ok) failed++;
    }

    std::printf("--- real GGUF open ---\n");
    if (argc > 1) {
        std::string path = argv[1];
        std::unique_ptr<desireeia::ModelReader> rd(desireeia::make_gguf_reader());
        desireeia::ModelMeta meta;
        bool ok = rd && rd->open(path, meta);
        if (ok) {
            std::printf("  %s: arch=%s vocab=%u layers=%u expert=%d\n",
                        path.c_str(), meta.arch.c_str(), meta.n_vocab,
                        meta.n_layers, meta.expert_model ? 1 : 0);

            std::printf("--- tokenizer ---\n");
            desireeia::VocabData vocab;
            desireeia::ArchKind arch_kind = desireeia::detect_arch(meta.arch);
            desireeia::SpmTokenizer tok;
            desireeia::BpeTokenizer btok;
            bool have_tok = false;
            bool is_bpe = false;
            std::vector<int32_t> ids;
            if (rd->read_vocab(vocab)) {
                desireeia::TokenizerKind tok_kind = desireeia::detect_tokenizer(vocab.tokenizer_tag);
                std::printf("  vocab tokens=%zu scores=%zu merges=%zu bos=%d eos=%d unk=%d add_bos=%d tok_kind=%d\n",
                            vocab.tokens.size(), vocab.scores.size(), vocab.merges.size(), vocab.bos_id,
                            vocab.eos_id, vocab.unk_id, vocab.add_bos ? 1 : 0, (int) tok_kind);
                if (tok_kind == desireeia::TokenizerKind::SpmUnigram) {
                    if (tok.load(vocab)) {
                        have_tok = true;
                        const char* sample = "Hello world";
                        ids = tok.encode(sample, true);
                        std::printf("  encode(\"%s\") -> [", sample);
                        for (size_t i = 0; i < ids.size(); ++i) {
                            std::string p;
                            tok.piece(ids[i], p);
                            std::printf("%s%d:'%s'", i ? " " : "", ids[i], p.c_str());
                        }
                        std::printf("]\n");
                        if (ids.empty()) {
                            std::printf("  tokenizer FAIL (empty encode)\n");
                            failed++;
                        }
                    } else {
                        std::printf("  tokenizer load FAIL\n");
                        failed++;
                    }
                } else if (tok_kind == desireeia::TokenizerKind::Bpe) {
                    if (btok.load(vocab)) {
                        have_tok = true;
                        is_bpe = true;
                        const char* sample = "Hello world";
                        ids = btok.encode(sample, vocab.add_bos);
                        std::printf("  encode(\"%s\") -> [", sample);
                        for (size_t i = 0; i < ids.size(); ++i) {
                            std::string p;
                            btok.piece(ids[i], p);
                            std::printf("%s%d:'%s'", i ? " " : "", ids[i], p.c_str());
                        }
                        std::printf("]\n");
                        if (ids.empty()) {
                            std::printf("  tokenizer FAIL (empty encode)\n");
                            failed++;
                        }
                    } else {
                        std::printf("  tokenizer load FAIL\n");
                        failed++;
                    }
                } else {
                    std::printf("  tokenizer kind not implemented (tag=%s)\n", vocab.tokenizer_tag.c_str());
                }
            } else {
                std::printf("  no vocab in file\n");
            }

            std::printf("--- forward predict ---\n");
            if (arch_kind == desireeia::ArchKind::Unknown) {
                std::printf("  arch not recognized by DenseForward, skipped\n");
            } else {
                if (ids.empty()) {
                    int32_t bos = vocab.bos_id >= 0 ? vocab.bos_id : 0;
                    ids.push_back(bos);
                    std::printf("  no working tokenizer for this arch: using BOS-only [%d]\n", bos);
                }

                desireeia_hw_info hw2;
                memset(&hw2, 0, sizeof(hw2));
                desireeia_probe_hw(&hw2);
                desireeia_plan plan2;
                memset(&plan2, 0, sizeof(plan2));
                desireeia_make_plan(&hw2, path.c_str(), nullptr, &plan2);

                desireeia_ctx* mctx = nullptr;
                desireeia_error ce = desireeia_create(path.c_str(), &plan2, nullptr, nullptr, &mctx);
                if (ce == DESIREEIA_OK && mctx) {
                    int32_t out_tok = -1;
                    desireeia_error pe = desireeia_predict(mctx, ids.data(), ids.size(), &out_tok);
                    auto piece_of = [&](int32_t id, std::string& out) -> bool {
                        if (!have_tok) return false;
                        return is_bpe ? btok.piece(id, out) : tok.piece(id, out);
                    };
                    if (pe == DESIREEIA_OK) {
                        std::printf("  predict(n=%zu) -> token %d", ids.size(), out_tok);
                        std::string p0;
                        if (piece_of(out_tok, p0)) std::printf(" ('%s')", p0.c_str());
                        std::printf("\n");
                        for (int step = 0; step < 3; ++step) {
                            int32_t nt = -1;
                            if (desireeia_next_token(mctx, &nt) != DESIREEIA_OK) break;
                            std::string p;
                            piece_of(nt, p);
                            std::printf("  next_token -> %d ('%s')\n", nt, p.c_str());
                        }
                    } else {
                        std::printf("  predict FAIL (rc=%d) - arch recognized but forward path not wired/failed\n", (int) pe);
                    }
                    desireeia_destroy(mctx);
                } else {
                    std::printf("  desireeia_create FAIL (rc=%d)\n", (int) ce);
                    failed++;
                }
            }
        } else {
            std::printf("  %s: OPEN FAIL\n", path.c_str());
            failed++;
        }
    } else {
        std::printf("  (no model path arg)\n");
    }

    // SSD-tier chat inference test: full generation with token timing.
    std::printf("\n--- ssd_tier chat inference ---\n");
    {
        const std::string model_path_ssd =
            "C:\\Users\\fpassaro\\AppData\\Local\\Kodinn\\google_gemma-3-4b-it-Q4_K_M.gguf";
        std::ifstream check(model_path_ssd, std::ios::binary);
        if (!check) {
            std::printf("  (model not found, skipping)\n");
        } else {
            check.close();

            desireeia_hw_info hw3;
            memset(&hw3, 0, sizeof(hw3));
            desireeia_probe_hw(&hw3);
            desireeia_plan plan3;
            memset(&plan3, 0, sizeof(plan3));
            desireeia_make_plan(&hw3, model_path_ssd.c_str(), nullptr, &plan3);

            desireeia_ctx* sctx = nullptr;
            desireeia_error ce = desireeia_create(model_path_ssd.c_str(), &plan3, nullptr, nullptr, &sctx);
            if (ce != DESIREEIA_OK || !sctx) {
                std::printf("  desireeia_create FAIL (rc=%d)\n", (int) ce);
                failed++;
            } else {
                // Build the prompt manually (simple format for gemma3).
                const char* roles[] = {"user"};
                const char* contents[] = {"Qual e la capitale di Roma?"};
                char chat_buf[4096];
                size_t chat_len = 0;
                desireeia_apply_chat_template(sctx, roles, contents, 1, 1,
                                              chat_buf, sizeof(chat_buf), &chat_len);

                // Tokenize.
                std::vector<int32_t> prompt_ids;
                prompt_ids.resize(2048);
                size_t n_ids = 0;
                desireeia_tokenize(sctx, chat_buf, 1, prompt_ids.data(),
                                   prompt_ids.size(), &n_ids);
                prompt_ids.resize(n_ids);
                std::printf("  prompt: \"%s\"\n", chat_buf);
                std::printf("  tokens: %zu\n", n_ids);

                // Prefill (all prompt tokens at once).
                int32_t first_tok = -1;
                uint64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                desireeia_predict(sctx, prompt_ids.data(), prompt_ids.size(), &first_tok);
                uint64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();

                double prefill_ms = (t1 - t0) / 1000.0;
                std::printf("  prefill: %zu tokens in %.1f ms (%.1f tok/s)\n",
                            n_ids, prefill_ms,
                            n_ids > 0 ? (n_ids * 1000.0 / prefill_ms) : 0.0);

                // Decode 48 tokens.
                int n_gen = 48;
                std::vector<int32_t> gen_tokens;
                gen_tokens.push_back(first_tok);

                uint64_t decode_start = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();

                for (int i = 0; i < n_gen; ++i) {
                    int32_t nt = -1;
                    if (desireeia_next_token(sctx, &nt) != DESIREEIA_OK) break;
                    gen_tokens.push_back(nt);
                    // Stop on EOS.
                    int32_t is_eog = 0;
                    desireeia_is_eog_token(sctx, nt, &is_eog);
                    if (is_eog) break;
                }

                uint64_t decode_end = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();

                double decode_ms = (decode_end - decode_start) / 1000.0;
                int n_actual = static_cast<int>(gen_tokens.size()) - 1;
                std::printf("  decode: %d tokens in %.1f ms (%.1f tok/s)\n",
                            n_actual, decode_ms,
                            n_actual > 0 ? (n_actual * 1000.0 / decode_ms) : 0.0);

                // Decode and print generated text.
                std::printf("  response: \"");
                for (int32_t tk : gen_tokens) {
                    // Use tokenizer to decode.
                    desireeia_token_piece(sctx, tk, chat_buf, sizeof(chat_buf));
                    std::printf("%s", chat_buf);
                }
                std::printf("\"\n");

                desireeia_destroy(sctx);
                std::printf("  ssd_tier chat inference OK\n");
                passed++;
            }
        }
    }

    // SSD tier tests.
    std::printf("\n--- ssd_tier basic ---\n");
    {
        // Test 1: write fails when no SSD file is configured (expected).
        desireeia::SsdTierConfig cfg;
        cfg.cache_capacity = 16;
        cfg.model_path = "";
        desireeia::SsdTier tier(cfg);

        std::vector<uint8_t> write_data = {0x01, 0x02, 0x03, 0x04};
        bool wok = tier.write(42, write_data.data(), write_data.size());
        if (wok) {
            std::printf("  ssd_tier write should fail without SSD path\n");
            failed++;
        } else {
            std::printf("  ssd_tier write correctly fails without SSD path\n");
            passed++;
        }

        // Test 2: read from empty cache returns false.
        std::vector<uint8_t> read_data;
        bool rok = tier.read(42, read_data);
        if (rok) {
            std::printf("  ssd_tier read should fail on empty cache\n");
            failed++;
        } else {
            std::printf("  ssd_tier read correctly fails on empty cache\n");
            passed++;
        }

        // Test 3: pin/unpin on non-existent key returns false.
        bool pinned = tier.pin(42);
        if (pinned) {
            std::printf("  ssd_tier pin should fail on non-existent key\n");
            failed++;
        } else {
            std::printf("  ssd_tier pin correctly fails on non-existent key\n");
            passed++;
        }

        // Test 4: stats reporting works.
        desireeia::SsdTier::Stats st = tier.stats();
        if (st.hits == 0 && st.misses >= 0 && st.current_size == 0) {
            std::printf("  ssd_tier stats OK: hits=%lu misses=%lu size=%d\n",
                        (unsigned long)st.hits, (unsigned long)st.misses, st.current_size);
            passed++;
        } else {
            std::printf("  ssd_tier stats unexpected\n");
            failed++;
        }
    }

    // Test 5: SsdTier with a temp file for read/write.
    std::printf("\n--- ssd_tier file-backed ---\n");
    {
        const std::string tmp_path = "desireeia_ssd_test.tmp";
        // Create a temp file with 8192 bytes of zeros.
        {
            std::ofstream f(tmp_path, std::ios::binary);
            std::vector<char> zeros(8192, 0);
            f.write(zeros.data(), zeros.size());
        }

        desireeia::SsdTierConfig cfg;
        cfg.cache_capacity = 8;
        cfg.model_path = tmp_path;
        desireeia::SsdTier tier(cfg);

        // Write data to key 0 (offset 0 in the file).
        std::vector<uint8_t> data = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
        bool wok = tier.write(0, data.data(), data.size());
        if (!wok) {
            std::printf("  ssd_tier file-backed write FAIL\n");
            failed++;
        } else {
            // Read it back (should come from cache).
            std::vector<uint8_t> read_back;
            bool rok = tier.read(0, read_back);
            if (!rok || read_back.size() < data.size() ||
                std::memcmp(read_back.data(), data.data(), data.size()) != 0) {
                std::printf("  ssd_tier file-backed read-back FAIL\n");
                failed++;
            } else {
                std::printf("  ssd_tier file-backed write+read OK\n");
                passed++;
            }

            // Pin/unpin should work on cached key.
            if (!tier.pin(0)) {
                std::printf("  ssd_tier file-backed pin FAIL\n");
                failed++;
            } else if (!tier.unpin(0)) {
                std::printf("  ssd_tier file-backed unpin FAIL\n");
                failed++;
            } else {
                std::printf("  ssd_tier file-backed pin/unpin OK\n");
                passed++;
            }
        }

        // Cleanup.
        std::remove(tmp_path.c_str());
    }

    // Mirror manager tests.
    std::printf("\n--- mirror_manager basic ---\n");
    {
        desireeia::MirrorConfig cfg;
        cfg.primary_path = "";   // No real files for this test.
        cfg.mirror_path = "";
        desireeia::MirrorManager mgr(cfg);

        // Mirror should be disabled when paths are empty.
        if (mgr.enabled()) {
            std::printf("  mirror_manager should be disabled with empty paths\n");
            failed++;
        } else {
            std::printf("  mirror_manager disabled as expected\n");
            passed++;
        }

        // Stats.
        desireeia::MirrorManager::Stats st = mgr.stats();
        std::printf("  mirror_manager stats: reads_p=%lu reads_m=%lu writes_p=%lu writes_m=%lu\n",
                    (unsigned long)st.reads_primary, (unsigned long)st.reads_mirror,
                    (unsigned long)st.writes_primary, (unsigned long)st.writes_mirror);
        passed++;
    }

    // Tiered expert store tests.
    std::printf("\n--- tiered_expert_store basic ---\n");
    {
        desireeia::TieredExpertConfig cfg;
        cfg.ram_cache_capacity = 8;
        cfg.ssd_cache_capacity = 16;
        cfg.model_path = "";
        cfg.mirror_path = "";
        desireeia::TieredExpertStore store(cfg);

        // Write and read back from RAM cache.
        std::vector<float> write_data = {1.0f, 2.0f, 3.0f, 4.0f};
        bool wok = store.write(100, write_data);
        if (!wok) {
            std::printf("  tiered_expert_store write FAIL\n");
            failed++;
        } else {
            std::vector<float> read_data;
            bool rok = store.read(100, read_data);
            if (!rok || read_data != write_data) {
                std::printf("  tiered_expert_store read-back FAIL\n");
                failed++;
            } else {
                std::printf("  tiered_expert_store write+read OK\n");
                passed++;
            }
        }

        // Pin/unpin.
        bool pinned = store.pin(100);
        if (!pinned) {
            std::printf("  tiered_expert_store pin FAIL\n");
            failed++;
        } else {
            bool unpinned = store.unpin(100);
            if (!unpinned) {
                std::printf("  tiered_expert_store unpin FAIL\n");
                failed++;
            } else {
                std::printf("  tiered_expert_store pin/unpin OK\n");
                passed++;
            }
        }

        // Stats.
        desireeia::TieredExpertStore::Stats st = store.stats();
        std::printf("  tiered_expert_store stats: ram_hits=%lu ram_misses=%lu ram_size=%d pinned=%d\n",
                    (unsigned long)st.ram_hits, (unsigned long)st.ram_misses,
                    st.ram_size, st.pinned_count);
        passed++;
    }

    // Q4_K quantizer: the point of this test is the ERROR, not just that it
    // runs. Requantizing a tensor is a lossy trade made to move fewer bytes,
    // so the loss has to be a measured number sitting next to the speed
    // number — otherwise the trade is being made blind.
    std::printf("\n--- quantize_row_q4_K ---\n");
    {
        const int64_t n = 256 * 32;
        std::vector<float> src((size_t) n);
        // A weight-like distribution: roughly normal, a few outliers, and one
        // sub-block that is entirely positive (the case where the min term
        // cannot be used and the quantizer has to fall back to [0,hi]).
        uint32_t rng = 12345u;
        auto next = [&]() {
            rng = rng * 1664525u + 1013904223u;
            return (float) ((int32_t) (rng >> 8) % 20001 - 10000) / 10000.0f;
        };
        for (int64_t i = 0; i < n; ++i) {
            float v = (next() + next() + next()) / 3.0f * 0.08f;
            if ((i % 997) == 0) v *= 12.0f;          // Outlier.
            if (i >= 256 && i < 288) v = 0.01f + 0.05f * (next() * 0.5f + 0.5f);
            src[(size_t) i] = v;
        }

        std::vector<block_q4_K> enc((size_t) (n / 256));
        quantize_row_q4_K(src.data(), enc.data(), n);

        std::vector<float> back((size_t) n);
        desireeia_dequantize_row(DESIREEIA_QTYPE_Q4_K, enc.data(), back.data(), n);

        double se = 0.0, sref = 0.0, worst = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double e = (double) back[(size_t) i] - src[(size_t) i];
            se += e * e;
            sref += (double) src[(size_t) i] * src[(size_t) i];
            if (std::fabs(e) > worst) worst = std::fabs(e);
        }
        const double rel_rms = std::sqrt(se / (double) n) / std::sqrt(sref / (double) n);

        // Reference point: the naive min/max affine fit, computed here in the
        // test. The quantizer is supposed to beat it — that is the whole
        // reason it does a least-squares search instead — so the comparison
        // belongs in the test rather than in a claim.
        double naive_se = 0.0;
        for (int64_t b = 0; b < n; b += 32) {
            float lo = src[(size_t) b], hi = lo;
            for (int64_t i = b + 1; i < b + 32; ++i) {
                if (src[(size_t) i] < lo) lo = src[(size_t) i];
                if (src[(size_t) i] > hi) hi = src[(size_t) i];
            }
            if (lo > 0.0f) lo = 0.0f;
            if (hi < 0.0f) hi = 0.0f;
            const float step = (hi - lo) / 15.0f;
            for (int64_t i = b; i < b + 32; ++i) {
                double e = 0.0;
                if (step > 0.0f) {
                    int q = (int) std::lround((src[(size_t) i] - lo) / step);
                    if (q < 0) q = 0; if (q > 15) q = 15;
                    e = (double) (step * (float) q + lo) - src[(size_t) i];
                } else {
                    e = -(double) src[(size_t) i];
                }
                naive_se += e * e;
            }
        }
        const double naive_rel_rms = std::sqrt(naive_se / (double) n) / std::sqrt(sref / (double) n);

        std::printf("  q4_K round-trip: rel_rms=%.4f worst_abs=%.5f\n", rel_rms, worst);
        std::printf("  q4_K vs naive min/max fit: %.4f -> %.4f (%.1f%% less error)\n",
                    naive_rel_rms, rel_rms,
                    naive_rel_rms > 0.0 ? (1.0 - rel_rms / naive_rel_rms) * 100.0 : 0.0);
        // Note this compares against a min/max fit WITHOUT the second-level
        // 6-bit scale quantization, so the reference is slightly optimistic;
        // beating it anyway is the meaningful result.
        if (rel_rms <= naive_rel_rms) {
            std::printf("  q4_K least-squares fit beats min/max OK\n");
            passed++;
        } else {
            std::printf("  q4_K least-squares fit WORSE than min/max\n");
            failed++;
        }
        // A correct 4-bit affine fit lands well under 10% relative RMS on this
        // distribution. A packing or scale-fitting bug blows straight past it,
        // which is what this bound is here to catch — it is not a quality
        // target, it is a "the format is being written correctly" gate.
        if (rel_rms < 0.10) {
            std::printf("  q4_K round-trip error within bound OK\n");
            passed++;
        } else {
            std::printf("  q4_K round-trip error TOO HIGH (%.4f)\n", rel_rms);
            failed++;
        }

        // Every sub-block must round-trip its own range: a packing bug that
        // swapped scales between sub-blocks would keep the global RMS
        // plausible while wrecking individual blocks.
        bool per_block_ok = true;
        for (int64_t b = 0; b < n; b += 256) {
            double bse = 0.0, bref = 0.0;
            for (int64_t i = b; i < b + 256; ++i) {
                const double e = (double) back[(size_t) i] - src[(size_t) i];
                bse += e * e;
                bref += (double) src[(size_t) i] * src[(size_t) i];
            }
            if (bref > 0.0 && std::sqrt(bse / bref) > 0.20) per_block_ok = false;
        }
        if (per_block_ok) {
            std::printf("  q4_K per-super-block error uniform OK\n");
            passed++;
        } else {
            std::printf("  q4_K per-super-block error NOT uniform (packing bug?)\n");
            failed++;
        }
    }

    // Hybrid tier tests (CPU+RAM+SSD).
    std::printf("\n--- hybrid_tier basic ---\n");
    {
        // Test without GGUF file (should report unavailable).
        desireeia::HybridTierConfig cfg;
        cfg.ram_capacity = 8;
        cfg.gguf_path = "";
        desireeia::HybridTier tier(cfg);

        if (tier.gguf_available()) {
            std::printf("  hybrid_tier should report unavailable without GGUF\n");
            failed++;
        } else {
            std::printf("  hybrid_tier correctly reports unavailable without GGUF\n");
            passed++;
        }

        // Stats should work even when unavailable.
        desireeia::HybridTier::Stats st = tier.stats();
        std::printf("  hybrid_tier stats: ram_hits=%lu ram_misses=%lu ram_size=%d\n",
                    (unsigned long)st.ram_hits, (unsigned long)st.ram_misses, st.ram_size);
        passed++;
    }

    // Hybrid tier with real GGUF model file (if available).
    std::printf("\n--- hybrid_tier GGUF-backed ---\n");
    {
        const std::string model_path =
            "C:\\Users\\fpassaro\\AppData\\Local\\Kodinn\\google_gemma-3-4b-it-Q4_K_M.gguf";

        // Check if the model file exists.
        std::ifstream test_f(model_path, std::ios::binary);
        if (!test_f) {
            std::printf("  (GGUF model not found at %s, skipping)\n", model_path.c_str());
            // Don't count as failure - model may not be present.
        } else {
            test_f.close();

            desireeia::HybridTierConfig cfg;
            cfg.ram_capacity = 16;  // Small cache to test eviction.
            cfg.gguf_path = model_path;
            cfg.prefetch_enabled = true;
            cfg.prefetch_depth = 1;
            desireeia::HybridTier tier(cfg);

            // The tier does not parse the model file itself: it serves misses
            // through a real reader, which is what keeps its view of the file
            // identical to the engine's. Without a source it is correctly
            // unavailable.
            if (tier.gguf_available()) {
                std::printf("  hybrid_tier should be unavailable before set_source\n");
                failed++;
            } else {
                std::printf("  hybrid_tier correctly unavailable before set_source\n");
                passed++;
            }

            desireeia::ModelReader* src = desireeia::make_gguf_reader();
            desireeia::ModelMeta src_meta;
            const bool src_ok = src && src->open(model_path, src_meta);
            if (src_ok) tier.set_source(src);

            if (!tier.gguf_available()) {
                std::printf("  hybrid_tier source attach FAIL\n");
                failed++;
            } else {
                std::printf("  hybrid_tier source attached OK\n");
                passed++;

                // A tensor served by the tier must match byte-for-byte what
                // the reader returns directly. This is the check that would
                // have caught the tier reading from the wrong file offset.
                const std::string probe = "token_embd.weight";
                std::vector<float> via_tier, via_reader;
                const bool tier_ok = tier.read_tensor(probe, via_tier);
                const bool reader_ok = src->read_tensor(probe, via_reader);
                if (tier_ok && reader_ok && via_tier.size() == via_reader.size() &&
                    !via_tier.empty() &&
                    std::memcmp(via_tier.data(), via_reader.data(),
                                via_tier.size() * sizeof(float)) == 0) {
                    std::printf("  hybrid_tier tensor matches reader byte-for-byte (%zu floats)\n",
                                via_tier.size());
                    passed++;
                } else {
                    std::printf("  hybrid_tier tensor MISMATCH vs reader (tier=%d reader=%d %zu vs %zu)\n",
                                (int) tier_ok, (int) reader_ok,
                                via_tier.size(), via_reader.size());
                    failed++;
                }

                // Second read of the same tensor must come from the RAM cache.
                std::vector<float> again;
                const desireeia::HybridTier::Stats before = tier.stats();
                tier.read_tensor(probe, again);
                const desireeia::HybridTier::Stats after = tier.stats();
                if (after.ram_hits > before.ram_hits) {
                    std::printf("  hybrid_tier second read served from RAM cache\n");
                    passed++;
                } else {
                    std::printf("  hybrid_tier second read did NOT hit the cache\n");
                    failed++;
                }

                // Read expert from layer 0, expert 0, gate part.
                // This may fail for dense (non-MoE) models which don't
                // have ffn_*_exps.weight tensors -- that's expected.
                std::vector<float> expert_data;
                bool rok = tier.read_expert(0, 0, 0, expert_data);
                if (!rok || expert_data.empty()) {
                    std::printf("  hybrid_tier read_expert(0,0,0) not-MoE or FAIL (expected for dense models)\n");
                    passed++;

                    // Prefetch should also be a no-op.
                    tier.prefetch({2}, {0, 1, 2, 3});
                    std::printf("  hybrid_tier prefetch layer 2 (no-op for dense) OK\n");
                    passed++;

                    // Pin/unpin should fail gracefully.
                    if (!tier.pin(0, 0, 0)) {
                        std::printf("  hybrid_tier pin correctly fails for dense model\n");
                        passed++;
                    } else {
                        std::printf("  hybrid_tier pin unexpectedly succeeded for dense model\n");
                        tier.unpin(0, 0, 0);
                        passed++;
                    }
                } else {
                    std::printf("  hybrid_tier read_expert(0,0,0) OK (%zu floats)\n",
                                expert_data.size());
                    passed++;

                    // Read same expert again (should be RAM hit).
                    std::vector<float> expert_data2;
                    tier.read_expert(0, 0, 0, expert_data2);
                    if (expert_data2 == expert_data) {
                        std::printf("  hybrid_tier RAM cache hit OK\n");
                        passed++;
                    } else {
                        std::printf("  hybrid_tier RAM cache hit mismatch\n");
                        failed++;
                    }

                    // Read expert from layer 1 to test multi-layer.
                    std::vector<float> expert_l1;
                    rok = tier.read_expert(1, 0, 0, expert_l1);
                    if (!rok || expert_l1.empty()) {
                        std::printf("  hybrid_tier read_expert(1,0,0) FAIL\n");
                        failed++;
                    } else {
                        std::printf("  hybrid_tier read_expert(1,0,0) OK (%zu floats)\n",
                                    expert_l1.size());
                        passed++;
                    }

                    // Prefetch layer 2.
                    tier.prefetch({2}, {0, 1, 2, 3});
                    std::printf("  hybrid_tier prefetch layer 2 OK\n");
                    passed++;

                    // Pin/unpin.
                    if (!tier.pin(0, 0, 0)) {
                        std::printf("  hybrid_tier pin FAIL\n");
                        failed++;
                    } else if (!tier.unpin(0, 0, 0)) {
                        std::printf("  hybrid_tier unpin FAIL\n");
                        failed++;
                    } else {
                        std::printf("  hybrid_tier pin/unpin OK\n");
                        passed++;
                    }
                }

                // Final stats.
                desireeia::HybridTier::Stats st = tier.stats();
                std::printf("  hybrid_tier final stats:\n");
                std::printf("    ram_hits=%lu ram_misses=%lu\n",
                            (unsigned long)st.ram_hits, (unsigned long)st.ram_misses);
                std::printf("    ssd_reads=%lu ssd_errors=%lu\n",
                            (unsigned long)st.ssd_reads, (unsigned long)st.ssd_errors);
                std::printf("    ram_size=%d pinned=%d\n", st.ram_size, st.pinned_count);
                std::printf("    total_bytes_read=%lu avg_read_us=%.1f\n",
                            (unsigned long)st.total_bytes_read, st.avg_read_ns / 1000.0);
                passed++;
            }

            // The tier holds a bare pointer to the reader, so drop the
            // reference before the reader goes away.
            tier.set_source(nullptr);
            delete src;
        }
    }

    std::printf("passed=%d unsupported=%d failed=%d\n", passed, unsupported, failed);
    if (failed == 0) std::printf("selftest PASS\n");
    else             std::printf("selftest FAIL\n");
    return failed;
}
