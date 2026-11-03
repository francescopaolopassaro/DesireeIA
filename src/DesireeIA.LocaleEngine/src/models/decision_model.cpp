// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "decision_model.h"
#include "decision_json.h"
#include "core/thread_pool.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <filesystem>

namespace desireeia {

namespace {

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

bool file_exists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::u8path(path), ec);
}

bool dir_exists(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(std::filesystem::u8path(path), ec);
}

std::string join(const std::string& dir, const std::string& leaf) {
    if (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) return dir + leaf;
    return dir + "/" + leaf;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

int qtype_id(const std::string& t) {
    if (t == "choice") return 0;
    if (t == "score") return 1;
    if (t == "noul") return 2;
    return -1;
}

const djson::Value* first_of(const djson::Value& o, std::initializer_list<const char*> keys) {
    for (const char* k : keys) {
        if (const djson::Value* v = o.find(k)) return v;
    }
    return nullptr;
}

std::string temp_bucket(int qtype, size_t k) {
    static const char* names[] = { "choice", "score", "noul" };
    const char* size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
    return std::string(names[qtype]) + ":" + size;
}

void add_num(std::string& out, double x) {
    out += std::isfinite(x) ? djson::fmt_double(x) : "0";
}

void add_float_array(std::string& out, const std::vector<float>& v) {
    out += '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ", ";
        add_num(out, v[i]);
    }
    out += ']';
}

}

bool DecisionModel::is_decision_dir(const std::string& path) {
    return dir_exists(path) && file_exists(join(path, "rl_agent_config.json")) &&
           file_exists(join(path, "model.safetensors")) &&
           file_exists(join(path, "encoder/config.json")) &&
           file_exists(join(path, "tokenizer/tokenizer.json"));
}

bool DecisionModel::open(const std::string& dir, std::string& err) {
    std::string text;
    if (!read_file(join(dir, "rl_agent_config.json"), text)) { err = "cannot read rl_agent_config.json"; return false; }
    djson::Value cfg;
    {
        djson::Parser p(text);
        if (!p.parse(cfg) || !cfg.is_obj()) { err = "rl_agent_config.json is not valid JSON"; return false; }
    }
    auto num = [&](const char* k, double dflt) {
        const djson::Value* v = cfg.find(k);
        return (v && v->is_num()) ? v->num : dflt;
    };
    max_len_ = (uint32_t) num("max_len", 512);
    head_max_len_ = (uint32_t) num("head_max_len", 192);
    const uint32_t head_layers = (uint32_t) num("head_layers", 2);
    if (const djson::Value* n = cfg.find("model_name")) name_ = n->str;
    temp_by_type_.assign(3, 1.0f);
    if (const djson::Value* t = cfg.find("temperature")) {
        if (t->is_arr()) {
            for (size_t i = 0; i < 3 && i < t->arr.size(); ++i) temp_by_type_[i] = (float) t->arr[i].num;
        }
    }
    temp_by_bucket_.clear();
    if (const djson::Value* t = cfg.find("temperature_by_options")) {
        if (t->is_obj()) {
            for (const auto& kv : t->obj) temp_by_bucket_[kv.first] = (float) kv.second.num;
        }
    }

    ModernBertConfig ecfg;
    if (!read_file(join(dir, "encoder/config.json"), text)) { err = "cannot read encoder/config.json"; return false; }
    if (!ModernBertEncoder::parse_config(text, ecfg, err)) return false;
    d_ = ecfg.n_embd;

    if (!tok_.load_file(join(dir, "tokenizer/tokenizer.json"))) { err = "cannot load tokenizer.json"; return false; }
    if (!st_.open(join(dir, "model.safetensors"))) { err = "cannot open model.safetensors"; return false; }
    if (!enc_.load(st_, "encoder.", ecfg, err)) return false;

    auto vec = [&](const std::string& n, std::vector<float>& v) {
        if (!st_.has(n) || !st_.read_f32(n, v)) { err = "missing or unreadable tensor: " + n; return false; }
        return true;
    };
    auto mat = [&](const std::string& n, HalfMat& m, uint64_t rows, uint64_t cols) {
        const auto sh = st_.shape(n);
        if (sh.size() != 2 || (rows && sh[0] != rows) || sh[1] != cols) { err = "unexpected shape for " + n; return false; }
        if (!st_.read_half(n, m)) { err = "cannot read tensor: " + n; return false; }
        return true;
    };

    const uint64_t d = d_;
    head_.assign(head_layers, HeadLayer());
    for (uint32_t i = 0; i < head_layers; ++i) {
        const std::string p = "head.layers." + std::to_string(i) + ".";
        HeadLayer& L = head_[i];
        if (!vec(p + "norm1.weight", L.n1w) || !vec(p + "norm1.bias", L.n1b) ||
            !vec(p + "norm2.weight", L.n2w) || !vec(p + "norm2.bias", L.n2b) ||
            !vec(p + "self_attn.in_proj_bias", L.in_proj_b) || !vec(p + "self_attn.out_proj.bias", L.out_proj_b) ||
            !vec(p + "linear1.bias", L.lin1_b) || !vec(p + "linear2.bias", L.lin2_b)) return false;
        if (!mat(p + "self_attn.in_proj_weight", L.in_proj, 3 * d, d) ||
            !mat(p + "self_attn.out_proj.weight", L.out_proj, d, d) ||
            !mat(p + "linear1.weight", L.lin1, 0, d) ||
            !mat(p + "linear2.weight", L.lin2, d, L.lin1.rows)) return false;
    }

    if (!vec("type_emb.weight", type_emb_) || type_emb_.size() != 3 * d) { if (err.empty()) err = "bad type_emb.weight"; return false; }
    std::vector<float> l2b;
    if (!vec("scorer.0.weight", sc_nw_) || !vec("scorer.0.bias", sc_nb_) || !vec("scorer.1.bias", sc_l1_b_) ||
        !vec("scorer.3.weight", sc_l2_w_) || !vec("scorer.3.bias", l2b) || l2b.size() != 1 || sc_l2_w_.size() != d) {
        if (err.empty()) err = "bad scorer tensors";
        return false;
    }
    sc_l2_b_ = l2b[0];
    if (!mat("scorer.1.weight", sc_l1_, d, d)) return false;

    if (!vec("act_head.0.bias", act_l0_b_) || !vec("act_head.2.bias", act_l2_b_)) return false;
    if (!mat("act_head.0.weight", act_l0_, 0, d + 4) || !mat("act_head.2.weight", act_l2_, 0, act_l0_.rows)) return false;
    return true;
}

float DecisionModel::temperature_for(int qtype, size_t k) const {
    auto it = temp_by_bucket_.find(temp_bucket(qtype, k));
    float t = it != temp_by_bucket_.end() ? it->second : temp_by_type_[(size_t) qtype];
    return t > 1e-4f ? t : 1e-4f;
}

bool DecisionModel::build_input(const std::string& state_text, const Question& q, bool truncate_left,
                                std::vector<int32_t>& ids, std::vector<size_t>& markers) const {
    const std::string& mask = tok_.mask_text();
    const int32_t mask_id = tok_.mask_id();
    const int32_t cls = tok_.cls_id();
    const int32_t sep = tok_.sep_id();

    std::vector<int32_t> head_ids = tok_.encode(q.type + " question: " + replace_all(q.instruction, mask, " "));
    std::vector<std::vector<int32_t>> opt_ids;
    for (const std::string& o : q.options) {
        std::vector<int32_t> t = tok_.encode(" " + replace_all(o, mask, " "));
        if (t.size() > 48) t.resize(48);
        t.insert(t.begin(), mask_id);
        opt_ids.push_back(std::move(t));
    }

    long opt_budget = (long) head_max_len_;
    for (const auto& o : opt_ids) opt_budget -= (long) o.size();
    if (opt_budget < 16) {              // too many or too long options: shrink every option evenly
        const long per = std::max<long>(4, ((long) head_max_len_ - 16) / std::max<long>(1, (long) opt_ids.size()));
        opt_budget = (long) head_max_len_;
        for (auto& o : opt_ids) {
            if ((long) o.size() > per) o.resize((size_t) per);
            opt_budget -= (long) o.size();
        }
    }
    const size_t head_keep = (size_t) std::max<long>(8, opt_budget);
    if (head_ids.size() > head_keep) head_ids.resize(head_keep);

    ids.clear();
    markers.clear();
    ids.push_back(cls);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(sep);
    std::vector<size_t> all_markers;
    for (const auto& o : opt_ids) {
        all_markers.push_back(ids.size());
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(sep);

    const long room = std::max<long>(0, (long) max_len_ - (long) ids.size() - 1);
    std::vector<int32_t> st = tok_.encode(replace_all(state_text, mask, " "));
    if (truncate_left) {
        // Mirrors the reference slice st[-room:], where room == 0 keeps
        // everything instead of nothing.
        if (room > 0 && (long) st.size() > room) st.erase(st.begin(), st.end() - room);
    } else if ((long) st.size() > room) {
        st.resize((size_t) room);
    }
    ids.insert(ids.end(), st.begin(), st.end());
    ids.push_back(sep);
    if (ids.size() > max_len_) ids.resize(max_len_);
    for (size_t m : all_markers) {
        if (m < max_len_) markers.push_back(m);
    }
    return markers.size() == q.options.size();
}

bool DecisionModel::run(const std::vector<int32_t>& ids, const std::vector<size_t>& markers, int qtype,
                        Answer& out) const {
    const size_t n = ids.size();
    const size_t d = d_;
    std::vector<float> h;
    if (!enc_.forward(ids.data(), n, h)) return false;

    for (size_t t = 0; t < n; ++t) {
        for (size_t i = 0; i < d; ++i) h[t * d + i] += type_emb_[(size_t) qtype * d + i];
    }

    const size_t H = std::max<size_t>(1, d / 64);
    const size_t hd = d / H;
    const float scale = 1.0f / std::sqrt((float) hd);
    std::vector<float> y(n * d), qkv(n * 3 * d), att(n * d), proj(n * d);
    for (const HeadLayer& L : head_) {
        layer_norm(h.data(), n, d, L.n1w.data(), L.n1b.data(), 1e-5f, y.data());
        linear_f16(y.data(), n, d, L.in_proj, L.in_proj_b.data(), qkv.data());
        ThreadPool::global().parallel_for(n * H, [&](size_t a, size_t b) {
            std::vector<float> sc(n);
            for (size_t idx = a; idx < b; ++idx) {
                const size_t t = idx / H;
                const size_t hh = idx % H;
                const float* q = qkv.data() + t * 3 * d + hh * hd;
                float mx = -1e30f;
                for (size_t k = 0; k < n; ++k) {
                    sc[k] = dot_f32(q, qkv.data() + k * 3 * d + d + hh * hd, hd) * scale;
                    mx = std::max(mx, sc[k]);
                }
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) { sc[k] = std::exp(sc[k] - mx); sum += sc[k]; }
                const float inv = (float) (1.0 / sum);
                float* o = att.data() + t * d + hh * hd;
                for (size_t j = 0; j < hd; ++j) o[j] = 0.0f;
                for (size_t k = 0; k < n; ++k) {
                    const float w = sc[k] * inv;
                    axpy_f32(o, w, qkv.data() + k * 3 * d + 2 * d + hh * hd, hd);
                }
            }
        });
        linear_f16(att.data(), n, d, L.out_proj, L.out_proj_b.data(), proj.data());
        for (size_t i = 0; i < n * d; ++i) h[i] += proj[i];

        layer_norm(h.data(), n, d, L.n2w.data(), L.n2b.data(), 1e-5f, y.data());
        std::vector<float> ffn(n * L.lin1.rows);
        linear_f16(y.data(), n, d, L.lin1, L.lin1_b.data(), ffn.data());
        for (float& v : ffn) v = v > 0.0f ? v : 0.0f;               // ReLU, the default of the reference layer
        linear_f16(ffn.data(), n, L.lin1.rows, L.lin2, L.lin2_b.data(), proj.data());
        for (size_t i = 0; i < n * d; ++i) h[i] += proj[i];
    }

    // scorer at every option marker
    const size_t k = markers.size();
    std::vector<float> m(k * d), mn(k * d), s1(k * d);
    for (size_t i = 0; i < k; ++i) std::copy(h.begin() + (long) (markers[i] * d), h.begin() + (long) ((markers[i] + 1) * d), m.begin() + (long) (i * d));
    layer_norm(m.data(), k, d, sc_nw_.data(), sc_nb_.data(), 1e-5f, mn.data());
    linear_f16(mn.data(), k, d, sc_l1_, sc_l1_b_.data(), s1.data());
    out.logits.assign(k, 0.0f);
    for (size_t i = 0; i < k; ++i) {
        double acc = sc_l2_b_;
        for (size_t j = 0; j < d; ++j) acc += (double) gelu_exact(s1[i * d + j]) * sc_l2_w_[j];
        out.logits[i] = (float) acc;
    }

    // act head: pooled first position + summary of the answer distribution
    std::vector<float> p = out.logits;
    softmax_inplace(p.data(), k);
    const float kk = (float) std::max<size_t>(k, 2);
    double ent = 0.0;
    for (float v : p) ent -= (double) v * std::log(std::max(v, 1e-9f));
    ent /= std::log((double) kk);
    float t1 = 0.0f, t2 = 0.0f;
    for (float v : p) {
        if (v > t1) { t2 = t1; t1 = v; }
        else if (v > t2) t2 = v;
    }
    std::vector<float> feat(d + 4);
    std::copy(h.begin(), h.begin() + (long) d, feat.begin());
    feat[d + 0] = t1;
    feat[d + 1] = t1 - t2;
    feat[d + 2] = (float) ent;
    feat[d + 3] = kk / 255.0f;
    std::vector<float> a0(act_l0_.rows);
    linear_f16(feat.data(), 1, d + 4, act_l0_, act_l0_b_.data(), a0.data());
    for (float& v : a0) v = gelu_exact(v);
    out.act.assign(act_l2_.rows, 0.0f);
    linear_f16(a0.data(), 1, act_l0_.rows, act_l2_, act_l2_b_.data(), out.act.data());
    softmax_inplace(out.act.data(), out.act.size());
    out.n_tokens = n;
    return true;
}

bool DecisionModel::decide(const std::string& request_json, std::string& response_json, std::string& err) {
    djson::Value req;
    {
        djson::Parser p(request_json);
        if (!p.parse(req) || !req.is_obj()) { err = "request is not a valid JSON object"; return false; }
    }
    const djson::Value* state = req.find("state");
    if (!state) { err = "request has no \"state\""; return false; }
    std::string state_text;
    if (state->is_str()) state_text = state->str;
    else djson::write_value(state_text, *state);

    const djson::Value* qs = req.find("questions");
    if (!qs || !qs->is_arr() || qs->arr.empty()) { err = "request has no \"questions\""; return false; }
    bool calibrate = true;
    if (const djson::Value* c = req.find("calibrate")) calibrate = c->type != djson::Value::Bool || c->b;
    bool truncate_left = false;
    if (const djson::Value* c = req.find("truncate_left")) truncate_left = c->type == djson::Value::Bool && c->b;

    response_json = "{\"model\": ";
    djson::write_string(response_json, name_);
    response_json += ", \"answers\": [";

    for (size_t qi = 0; qi < qs->arr.size(); ++qi) {
        const djson::Value& qv = qs->arr[qi];
        const std::string where = "question " + std::to_string(qi) + ": ";
        if (!qv.is_obj()) { err = where + "must be an object"; return false; }

        Question q;
        const djson::Value* tv = first_of(qv, { "type", "t" });
        if (!tv || !tv->is_str() || qtype_id(tv->str) < 0) { err = where + "type must be choice, score or noul"; return false; }
        q.type = tv->str;
        const int qt = qtype_id(q.type);
        if (const djson::Value* iv = first_of(qv, { "instruction", "ins" })) {
            if (iv->is_str()) q.instruction = iv->str;
        }
        const djson::Value* crit = first_of(qv, { "options", "crit", "levels" });

        if (qt == 0) {                       // choice: label -> description
            if (!crit) { err = where + "choice needs \"options\""; return false; }
            if (crit->is_obj()) {
                for (const auto& kv : crit->obj) {
                    q.labels.push_back(kv.first);
                    const std::string desc = kv.second.is_str() ? kv.second.str : std::string();
                    q.options.push_back(desc.empty() ? kv.first : kv.first + ": " + desc);
                }
            } else if (crit->is_arr()) {
                for (const auto& v : crit->arr) {
                    q.labels.push_back(v.str);
                    q.options.push_back(v.str);
                }
            } else { err = where + "\"options\" must be an object or an array"; return false; }
        } else if (qt == 1) {                // score: ordinal levels
            if (!crit || !crit->is_arr()) { err = where + "score needs an \"options\" array"; return false; }
            for (size_t i = 0; i < crit->arr.size(); ++i) {
                q.labels.push_back(std::to_string(i));
                q.options.push_back("level " + std::to_string(i) + ": " + crit->arr[i].str);
            }
        } else {                             // noul: always [false, true]
            std::string f, t;
            const djson::Value* src = (crit && crit->is_obj()) ? crit : &qv;
            if (const djson::Value* v = src->find("false")) f = v->str;
            if (const djson::Value* v = src->find("true")) t = v->str;
            q.labels = { "false", "true" };
            q.options = { "false: " + (f.empty() ? std::string("no, the statement does not hold") : f),
                          "true: " + (t.empty() ? std::string("yes, the statement holds") : t) };
        }
        if (q.options.size() < 2) { err = where + "needs at least two options"; return false; }

        std::vector<int32_t> ids;
        std::vector<size_t> markers;
        if (!build_input(state_text, q, truncate_left, ids, markers)) {
            err = where + "the options do not fit in the model's input window";
            return false;
        }
        Answer a;
        if (!run(ids, markers, qt, a)) { err = where + "encoder failed"; return false; }

        const size_t k = a.logits.size();
        std::vector<float> probs = a.logits;
        if (calibrate) {
            const float T = temperature_for(qt, k);
            for (float& v : probs) v /= T;
        }
        softmax_inplace(probs.data(), k);
        size_t best = 0;
        for (size_t i = 1; i < k; ++i) if (probs[i] > probs[best]) best = i;
        double ent = 0.0;
        for (float v : probs) ent -= (double) v * std::log(std::max(v, 1e-12f));
        const double conf = k < 2 ? 1.0 : 1.0 - ent / std::log((double) k);

        if (qi) response_json += ", ";
        response_json += "{\"type\": ";
        djson::write_string(response_json, q.type);
        response_json += ", \"options\": [";
        for (size_t i = 0; i < q.labels.size(); ++i) {
            if (i) response_json += ", ";
            djson::write_string(response_json, q.labels[i]);
        }
        response_json += "], \"probabilities\": ";
        add_float_array(response_json, probs);
        response_json += ", \"logits\": ";
        add_float_array(response_json, a.logits);
        response_json += ", \"index\": " + std::to_string(best);
        response_json += ", \"choice\": ";
        djson::write_string(response_json, q.labels[best]);
        response_json += ", \"confidence\": ";
        add_num(response_json, conf);
        response_json += ", \"act_probability\": ";
        add_num(response_json, a.act.empty() ? 0.0 : a.act[0]);
        response_json += ", \"act_probabilities\": ";
        add_float_array(response_json, a.act);
        response_json += ", \"n_tokens\": " + std::to_string(a.n_tokens) + "}";
    }
    response_json += "]}";
    return true;
}

}
