// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#include "desireeia/abi.h"
#include "core/engine.h"
#include "core/profile.h"
#include "vision/vision_clip.h"
#include "vision/vision_image.h"

#include <cstring>
#include <exception>
#include <new>
#include <string>

namespace {
    desireeia_log_cb g_log = nullptr;
    void* g_log_user = nullptr;

    void log_msg(int32_t level, const char* msg) {
        if (g_log) {
            g_log(level, msg, g_log_user);
        }
    }

    void emit_impl(const char* file, int32_t line, const char* fn, int32_t level, const char* fmt) {
        (void)file; (void)line; (void)fn;
        log_msg(level, fmt);
    }

    // Every exported function that does real work runs inside ABI_GUARD: a
    // C++ exception must never cross the C ABI. Before this, an allocation
    // failure (e.g. growing the KV cache for a long prompt) or any other
    // exception unwound straight into the host runtime - .NET reported only
    // "External component has thrown an exception", Python could lose the
    // whole process - and the real cause was lost. Now it is logged through
    // the logger and returned as an error code (NO_MEM for bad_alloc).
    // Last error raised on this thread (desireeia_last_error without a ctx).
    thread_local std::string tl_last_error;

    desireeia_error report_exception(const char* fn, const char* what, desireeia_error code) {
        std::string m = std::string(fn) + ": " + what;
        tl_last_error = m;
        log_msg(3, m.c_str());
        return code;
    }
}

#define ABI_GUARD_BEGIN try {
#define ABI_GUARD_END \
    } catch (const std::bad_alloc&) { \
        return report_exception(__func__, "out of memory", DESIREEIA_ERR_NO_MEM); \
    } catch (const std::exception& e) { \
        return report_exception(__func__, e.what(), DESIREEIA_ERR_UNDEFINED); \
    } catch (...) { \
        return report_exception(__func__, "unknown native exception", DESIREEIA_ERR_UNDEFINED); \
    }

#define DESIREEIA_EMIT(level, msg) emit_impl(__FILE__, __LINE__, __func__, (level), (msg))

DESIREEIA_API const char* desireeia_version(void) {
    return "0.1.3";
}

DESIREEIA_API desireeia_error desireeia_set_logger(desireeia_log_cb cb, void* user) {
    g_log = cb;
    g_log_user = user;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_probe_hw(desireeia_hw_info* out) {
    ABI_GUARD_BEGIN
    if (!out) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    return desireeia::probe_hardware(*out);
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_make_plan(const desireeia_hw_info* hw,
                                         const char* model_path,
                                         const desireeia_plan* override,
                                         desireeia_plan* out) {
    ABI_GUARD_BEGIN
    if (!hw || !out) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    desireeia_plan base = desireeia::build_plan(*hw, model_path ? model_path : "");
    if (override) {
        if (override->format != DESIREEIA_FORMAT_UNKNOWN) {
            base.format = override->format;
        }
        if (override->backend != 0) {
            base.backend = override->backend;
        }
        if (override->dense_quant != DESIREEIA_QUANT_F32) {
            base.dense_quant = override->dense_quant;
        }
        if (override->expert_quant != DESIREEIA_QUANT_F32) {
            base.expert_quant = override->expert_quant;
        }
        if (override->n_threads > 0) {
            base.n_threads = override->n_threads;
        }
        if (override->ram_budget_mb > 0) {
            base.ram_budget_mb = override->ram_budget_mb;
        }
        base.expert_cache_count = override->expert_cache_count;
        base.expert_prefetch_enabled = override->expert_prefetch_enabled;
        base.kv_compression_enabled = override->kv_compression_enabled;
        base.expert_pin_enabled = override->expert_pin_enabled;
        base.expert_prefetch_depth = override->expert_prefetch_depth;
        base.batch_union_enabled = override->batch_union_enabled;
        base.dual_ssd_enabled = override->dual_ssd_enabled;
    }
    *out = base;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_create(const char* model_path,
                                      const desireeia_plan* plan,
                                      desireeia_log_cb cb,
                                      void* user,
                                      desireeia_ctx** out) {
    ABI_GUARD_BEGIN
    if (!model_path || !plan || !out) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    // The callback given here is this context's logger. It used to replace
    // the process-wide one, so a second model redirected the first's log.
    // It still becomes the process-wide logger when none is set, as before.
    if (cb && !g_log) {
        g_log = cb;
        g_log_user = user;
    }
    desireeia_ctx* ctx = desireeia::engine_create(model_path, *plan, [cb, user](int32_t l, const char* m) {
        if (cb) cb(l, m, user);
        else log_msg(l, m);
    });
    if (!ctx) {
        return report_exception(__func__, "model could not be loaded (see the log)", DESIREEIA_ERR_IO);
    }
    *out = ctx;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_destroy(desireeia_ctx* ctx) {
    ABI_GUARD_BEGIN
    if (!ctx) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    desireeia::engine_destroy(ctx);
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_predict(desireeia_ctx* ctx,
                                       const int32_t* tokens,
                                       size_t n_tokens,
                                       int32_t* out_token) {
    ABI_GUARD_BEGIN
    if (!ctx || !tokens || !out_token) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    if (!desireeia::engine_predict(ctx, tokens, n_tokens, *out_token)) {
        if (desireeia::engine_was_cancelled(ctx)) return DESIREEIA_ERR_CANCELLED;
        tl_last_error = desireeia::engine_last_error(ctx);
        return DESIREEIA_ERR_UNDEFINED;
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_next_token(desireeia_ctx* ctx, int32_t* out_token) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_token) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    int32_t t = 0;
    if (!desireeia::engine_next_token(ctx, t)) {
        return DESIREEIA_ERR_UNDEFINED;
    }
    *out_token = t;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API size_t desireeia_context_size(const desireeia_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
    return desireeia::engine_context_size(ctx);
}

DESIREEIA_API uint32_t desireeia_context_length_trained(const desireeia_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
    return desireeia::engine_context_length_trained(ctx);
}

DESIREEIA_API desireeia_error desireeia_session_reset(desireeia_ctx* ctx) {
    ABI_GUARD_BEGIN
    if (!ctx) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    return desireeia::engine_session_reset(ctx) ? DESIREEIA_OK : DESIREEIA_ERR_UNDEFINED;
    ABI_GUARD_END
}

namespace {
desireeia_error session_error(const char* fn, const std::string& err) {
    desireeia_error code = DESIREEIA_ERR_IO;
    if (err.find("cannot be saved") != std::string::npos || err.find("cannot be restored") != std::string::npos)
        code = DESIREEIA_ERR_NOT_SUPPORTED;
    else if (err.find("different model") != std::string::npos || err.find("mismatch") != std::string::npos ||
             err.find("not a session file") != std::string::npos || err.find("truncated") != std::string::npos)
        code = DESIREEIA_ERR_PARSE;
    else if (err.find("invalid argument") != std::string::npos)
        code = DESIREEIA_ERR_INVALID_ARG;
    return report_exception(fn, err.c_str(), code);
}
} // namespace

DESIREEIA_API desireeia_error desireeia_session_save(desireeia_ctx* ctx, const char* path, size_t n_prefix) {
    ABI_GUARD_BEGIN
    if (!ctx || !path) return DESIREEIA_ERR_INVALID_ARG;
    std::string err;
    if (desireeia::engine_session_save(ctx, path, n_prefix, err)) return DESIREEIA_OK;
    return session_error(__func__, err);
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_session_save_mem(desireeia_ctx* ctx, size_t n_prefix,
                                                        uint8_t* out, size_t capacity, size_t* out_size) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_size) return DESIREEIA_ERR_INVALID_ARG;
    const size_t need = desireeia::engine_session_image_size(ctx, n_prefix);
    *out_size = need;
    if (!out) return DESIREEIA_OK;
    if (need == 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    if (capacity < need) return DESIREEIA_ERR_INVALID_ARG;
    std::vector<uint8_t> image;
    std::string err;
    if (!desireeia::engine_session_serialize(ctx, n_prefix, image, err)) return session_error(__func__, err);
    if (image.size() > capacity) return DESIREEIA_ERR_INVALID_ARG;   // a predict ran in between
    std::memcpy(out, image.data(), image.size());
    *out_size = image.size();
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_session_load_mem(desireeia_ctx* ctx, const uint8_t* data, size_t size,
                                                        size_t* out_tokens) {
    ABI_GUARD_BEGIN
    if (out_tokens) *out_tokens = 0;
    if (!ctx || !data) return DESIREEIA_ERR_INVALID_ARG;
    std::string err;
    size_t n = 0;
    const bool ok = desireeia::engine_session_deserialize(ctx, data, size, n, err);
    if (out_tokens) *out_tokens = n;
    return ok ? DESIREEIA_OK : session_error(__func__, err);
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_session_load(desireeia_ctx* ctx, const char* path, size_t* out_tokens) {
    ABI_GUARD_BEGIN
    if (!ctx || !path) return DESIREEIA_ERR_INVALID_ARG;
    std::string err;
    size_t n = 0;
    const bool ok = desireeia::engine_session_load(ctx, path, n, err);
    if (out_tokens) *out_tokens = n;
    return ok ? DESIREEIA_OK : session_error(__func__, err);
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_set_session_reuse(desireeia_ctx* ctx, int32_t mode) {
    ABI_GUARD_BEGIN
    if (!ctx || mode < 0 || mode > 2) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    return desireeia::engine_set_session_reuse(ctx, (int) mode) ? DESIREEIA_OK : DESIREEIA_ERR_UNDEFINED;
    ABI_GUARD_END
}

DESIREEIA_API size_t desireeia_last_reused_tokens(const desireeia_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
    return desireeia::engine_last_reused_tokens(ctx);
}

DESIREEIA_API desireeia_error desireeia_tokenize(desireeia_ctx* ctx,
                                        const char* text,
                                        int32_t add_bos,
                                        int32_t* out_ids,
                                        size_t max_ids,
                                        size_t* out_count) {
    ABI_GUARD_BEGIN
    if (!ctx || !text || !out_count) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    std::vector<int32_t> ids;
    if (!desireeia::engine_tokenize(ctx, text, add_bos != 0, ids)) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    *out_count = ids.size();
    if (out_ids && max_ids >= ids.size()) {
        std::memcpy(out_ids, ids.data(), ids.size() * sizeof(int32_t));
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_embed(desireeia_ctx* ctx,
                                     const int32_t* tokens,
                                     size_t n_tokens,
                                     float* out_embd,
                                     size_t out_capacity,
                                     size_t* out_len,
                                     uint32_t* out_embd_dim) {
    ABI_GUARD_BEGIN
    if (!ctx || !tokens || n_tokens == 0 || !out_len || !out_embd_dim) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    std::vector<float> embd;
    uint32_t dim = 0;
    if (!desireeia::engine_embed(ctx, tokens, n_tokens, embd, dim)) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    *out_len = embd.size();
    *out_embd_dim = dim;
    if (out_embd && out_capacity >= embd.size()) {
        std::memcpy(out_embd, embd.data(), embd.size() * sizeof(float));
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_token_piece(desireeia_ctx* ctx,
                                           int32_t id,
                                           char* out_buf,
                                           size_t buf_size) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_buf || buf_size == 0) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    std::string piece;
    if (!desireeia::engine_token_piece(ctx, id, piece)) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    if (piece.size() + 1 > buf_size) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    std::memcpy(out_buf, piece.data(), piece.size());
    out_buf[piece.size()] = '\0';
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_special_token_id(const desireeia_ctx* ctx,
                                                desireeia_special_token which,
                                                int32_t* out_id) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_id) return DESIREEIA_ERR_INVALID_ARG;
    int32_t id = -1;
    if (!desireeia::engine_special_token_id(ctx, (int) which, id)) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    *out_id = id;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_is_eog_token(const desireeia_ctx* ctx,
                                            int32_t id,
                                            int32_t* out_is_eog) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_is_eog) return DESIREEIA_ERR_INVALID_ARG;
    *out_is_eog = desireeia::engine_is_eog_token(ctx, id) ? 1 : 0;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_set_sampling(desireeia_ctx* ctx,
                                            const desireeia_sampling* params) {
    ABI_GUARD_BEGIN
    if (!ctx || !params) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_set_sampling(ctx, *params)) return DESIREEIA_ERR_INVALID_ARG;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_get_sampling(const desireeia_ctx* ctx,
                                            desireeia_sampling* out) {
    ABI_GUARD_BEGIN
    if (!ctx || !out) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_get_sampling(ctx, *out)) return DESIREEIA_ERR_INVALID_ARG;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_load_lora_adapter(desireeia_ctx* ctx,
                                                            const char* lora_gguf_path,
                                                            float scale) {
    ABI_GUARD_BEGIN
    if (!ctx || !lora_gguf_path) return DESIREEIA_ERR_INVALID_ARG;
    std::string err;
    if (!desireeia::engine_load_lora(ctx, lora_gguf_path, scale, err)) {
        if (err.find("not a LoRA adapter") != std::string::npos) return DESIREEIA_ERR_PARSE;
        if (err.find("no generative forward engine") != std::string::npos) return DESIREEIA_ERR_NOT_SUPPORTED;
        return DESIREEIA_ERR_IO;
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_clear_lora_adapters(desireeia_ctx* ctx) {
    ABI_GUARD_BEGIN
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_clear_lora(ctx)) return DESIREEIA_ERR_UNDEFINED;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_load_prerouter(desireeia_ctx* ctx, const char* path) {
    ABI_GUARD_BEGIN
    if (!ctx || !path) return DESIREEIA_ERR_INVALID_ARG;
    std::string err;
    if (!desireeia::engine_load_prerouter(ctx, path, err)) {
        if (err.find("no valid prerouter heads") != std::string::npos) return DESIREEIA_ERR_PARSE;
        if (err.find("no generative forward engine") != std::string::npos ||
            err.find("no MoE experts") != std::string::npos) return DESIREEIA_ERR_NOT_SUPPORTED;
        return DESIREEIA_ERR_IO;
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_clear_prerouter(desireeia_ctx* ctx) {
    ABI_GUARD_BEGIN
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_clear_prerouter(ctx)) return DESIREEIA_ERR_UNDEFINED;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_set_prerouter_heuristic(desireeia_ctx* ctx, int32_t enabled) {
    ABI_GUARD_BEGIN
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_set_prerouter_heuristic(ctx, enabled != 0)) return DESIREEIA_ERR_UNDEFINED;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_apply_chat_template(const desireeia_ctx* ctx,
                                                    const char** roles,
                                                    const char** contents,
                                                    size_t n_messages,
                                                    int add_assistant,
                                                    char* out_buf,
                                                    size_t buf_size,
                                                    size_t* out_len) {
    ABI_GUARD_BEGIN
    if (!ctx || (n_messages > 0 && (!roles || !contents))) return DESIREEIA_ERR_INVALID_ARG;
    std::string result;
    if (!desireeia::engine_apply_chat_template(ctx, roles, contents, n_messages,
                                            add_assistant != 0, result)) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    if (out_len) *out_len = result.size();
    if (out_buf && buf_size > 0) {
        const size_t n = result.size() < buf_size - 1 ? result.size() : buf_size - 1;
        std::memcpy(out_buf, result.data(), n);
        out_buf[n] = '\0';
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API int32_t desireeia_abi_version(void) {
    return DESIREEIA_ABI_VERSION;
}

DESIREEIA_API size_t desireeia_last_error(const desireeia_ctx* ctx, char* buf, size_t buf_size) {
    try {
        std::string m = ctx ? desireeia::engine_last_error(ctx) : std::string();
        if (m.empty()) m = tl_last_error;
        if (buf && buf_size > 0) {
            const size_t n = std::min(m.size(), buf_size - 1);
            std::memcpy(buf, m.data(), n);
            buf[n] = '\0';
        }
        return m.size();
    } catch (...) {
        return 0;
    }
}

DESIREEIA_API desireeia_error desireeia_set_ctx_logger(desireeia_ctx* ctx, desireeia_log_cb cb, void* user) {
    ABI_GUARD_BEGIN
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    desireeia::engine_set_log(ctx, [cb, user](int32_t l, const char* m) {
        if (cb) cb(l, m, user);
        else log_msg(l, m);
    });
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_cancel(desireeia_ctx* ctx) {
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    desireeia::engine_cancel(ctx);
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_set_progress_callback(desireeia_ctx* ctx, desireeia_progress_cb cb, void* user) {
    ABI_GUARD_BEGIN
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    desireeia::engine_set_progress(ctx, cb, user);
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API int32_t desireeia_gpu_count(void) {
#ifdef DESIREEIA_CUDA_ENABLED
    try { return desireeia::cuda_gpu_count(); } catch (...) { return 0; }
#else
    return 0;
#endif
}

DESIREEIA_API desireeia_error desireeia_probe_gpu(int32_t index, desireeia_gpu_info* out) {
    ABI_GUARD_BEGIN
    if (!out) return DESIREEIA_ERR_INVALID_ARG;
    std::memset(out, 0, sizeof(*out));
#ifdef DESIREEIA_CUDA_ENABLED
    uint64_t total = 0, free_b = 0;
    int32_t ma = 0, mi = 0, sms = 0;
    if (!desireeia::cuda_probe_gpu(index, out->name, sizeof(out->name), total, free_b, ma, mi, sms))
        return DESIREEIA_ERR_INVALID_ARG;
    out->total_bytes = total;
    out->free_bytes = free_b;
    out->cc_major = ma;
    out->cc_minor = mi;
    out->multiprocessors = sms;
    return DESIREEIA_OK;
#else
    (void) index;
    return DESIREEIA_ERR_NOT_SUPPORTED;
#endif
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_reserve_context(desireeia_ctx* ctx, size_t n_positions) {
    ABI_GUARD_BEGIN
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    if (desireeia::engine_reserve(ctx, n_positions)) return DESIREEIA_OK;
    return report_exception(__func__, "the KV cache could not be reserved", DESIREEIA_ERR_NO_MEM);
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_trim_cache(desireeia_ctx* ctx) {
    ABI_GUARD_BEGIN
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    return desireeia::engine_trim(ctx) ? DESIREEIA_OK : DESIREEIA_ERR_NOT_SUPPORTED;
    ABI_GUARD_END
}

DESIREEIA_API void desireeia_profile_dump(char* out_buf, size_t buf_size) {
    if (!out_buf || buf_size == 0) return;
    const std::string s = desireeia::profile_dump_string();
    const size_t n = s.size() < buf_size - 1 ? s.size() : buf_size - 1;
    std::memcpy(out_buf, s.data(), n);
    out_buf[n] = '\0';
}

DESIREEIA_API void desireeia_profile_reset(void) {
    desireeia::profile_reset();
}

// ============================================================
// Vision Module - ABI Functions
// ============================================================

DESIREEIA_API desireeia_error desireeia_load_image(const char* path,
                                                   int32_t expected_channels,
                                                   DesireeAIImage* out) {
    ABI_GUARD_BEGIN
    if (!path || !out) return DESIREEIA_ERR_INVALID_ARG;

    auto buf = desireeia::vision::load_image_from_file(path, expected_channels);
    if (buf.empty()) return DESIREEIA_ERR_IO;

    out->width = buf.width;
    out->height = buf.height;
    out->channels = buf.channels;
    out->data = new uint8_t[buf.data.size()];
    std::memcpy(out->data, buf.data.data(), buf.data.size());
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API void desireeia_free_image(DesireeAIImage* img) {
    if (!img) return;
    delete[] img->data;
    img->data = nullptr;
    img->width = 0;
    img->height = 0;
    img->channels = 0;
}

DESIREEIA_API DesireeAIVisionCtx* desireeia_vision_create(const char* model_path,
                                                          desireeia_log_cb cb,
                                                          void* user) {
    if (!model_path) return nullptr;
    // Returns a pointer, not an error code: an exception becomes nullptr
    // (the documented failure value) after being logged.
    try {
        return desireeia::vision::vision_create_context(model_path, cb, user);
    } catch (const std::bad_alloc&) {
        report_exception(__func__, "out of memory", DESIREEIA_ERR_NO_MEM);
    } catch (const std::exception& e) {
        report_exception(__func__, e.what(), DESIREEIA_ERR_UNDEFINED);
    } catch (...) {
        report_exception(__func__, "unknown native exception", DESIREEIA_ERR_UNDEFINED);
    }
    return nullptr;
}

DESIREEIA_API void desireeia_vision_destroy(DesireeAIVisionCtx* ctx) {
    desireeia::vision::vision_destroy_context(ctx);
}

DESIREEIA_API desireeia_error desireeia_vision_get_config(const DesireeAIVisionCtx* ctx,
                                                          DesireeAIVisionConfig* out) {
    ABI_GUARD_BEGIN
    if (!ctx || !out) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::vision::vision_get_config(ctx, *out)) {
        return DESIREEIA_ERR_UNDEFINED;
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_vision_encode(DesireeAIVisionCtx* ctx,
                                                      const DesireeAIImage* image,
                                                      float* out_embd,
                                                      size_t out_capacity,
                                                      size_t* out_len,
                                                      uint32_t* out_dim) {
    ABI_GUARD_BEGIN
    if (!ctx || !image || !out_len || !out_dim) return DESIREEIA_ERR_INVALID_ARG;
    if (!image->data || image->width == 0 || image->height == 0) {
        return DESIREEIA_ERR_INVALID_ARG;
    }

    std::vector<float> embd;
    uint32_t dim = 0;
    if (!desireeia::vision::vision_encode(ctx, *image, embd, dim)) {
        return DESIREEIA_ERR_UNDEFINED;
    }

    *out_len = embd.size();
    *out_dim = dim;
    if (out_embd && out_capacity >= embd.size()) {
        std::memcpy(out_embd, embd.data(), embd.size() * sizeof(float));
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_vision_preprocess(const DesireeAIImage* input,
                                                          int32_t target_size,
                                                          float* out_pixels,
                                                          size_t out_capacity,
                                                          size_t* out_len) {
    ABI_GUARD_BEGIN
    if (!input || !out_len) return DESIREEIA_ERR_INVALID_ARG;
    if (!input->data || input->width == 0 || input->height == 0) {
        return DESIREEIA_ERR_INVALID_ARG;
    }

    std::vector<float> pixels;
    if (!desireeia::vision::vision_preprocess_image(*input, target_size, pixels)) {
        return DESIREEIA_ERR_UNDEFINED;
    }

    *out_len = pixels.size();
    if (out_pixels && out_capacity >= pixels.size()) {
        std::memcpy(out_pixels, pixels.data(), pixels.size() * sizeof(float));
    }
    return DESIREEIA_OK;
    ABI_GUARD_END
}

// ========================================================================
// Multimodal model support (ctx-based vision encoder)
// ========================================================================

DESIREEIA_API desireeia_error desireeia_has_vision(desireeia_ctx* ctx,
                                                   int32_t* out_has) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_has) return DESIREEIA_ERR_INVALID_ARG;
    *out_has = desireeia::engine_has_vision(ctx) ? 1 : 0;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_vision_token_count(desireeia_ctx* ctx,
                                                           int32_t* out_count) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_count) return DESIREEIA_ERR_INVALID_ARG;
    const int32_t n = desireeia::engine_vision_token_count(ctx);
    if (n <= 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    *out_count = n;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_vision_image_token(desireeia_ctx* ctx,
                                                           int32_t* out_id) {
    ABI_GUARD_BEGIN
    if (!ctx || !out_id) return DESIREEIA_ERR_INVALID_ARG;
    *out_id = desireeia::engine_vision_image_token(ctx);
    return *out_id >= 0 ? DESIREEIA_OK : DESIREEIA_ERR_NOT_SUPPORTED;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_vision_encode_ctx(desireeia_ctx* ctx,
                                                          const DesireeAIImage* image,
                                                          float* out_embd,
                                                          size_t out_capacity,
                                                          size_t* out_len,
                                                          uint32_t* out_dim) {
    ABI_GUARD_BEGIN
    if (!ctx || !image || !out_len || !out_dim) return DESIREEIA_ERR_INVALID_ARG;
    if (!image->data || image->width == 0 || image->height == 0) {
        return DESIREEIA_ERR_INVALID_ARG;
    }

    std::vector<float> embd;
    uint32_t dim = 0;
    if (!desireeia::engine_encode_image(ctx, *image, embd, dim)) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }

    *out_len = embd.size();
    *out_dim = dim;
    if (out_embd && out_capacity >= embd.size()) {
        std::memcpy(out_embd, embd.data(), embd.size() * sizeof(float));
    }
    if (embd.empty()) return DESIREEIA_ERR_UNDEFINED;
    return DESIREEIA_OK;
    ABI_GUARD_END
}

DESIREEIA_API desireeia_error desireeia_predict_image(desireeia_ctx* ctx,
                                                      const int32_t* tokens,
                                                      size_t n_tokens,
                                                      const float* embd,
                                                      size_t n_embd,
                                                      int32_t image_token,
                                                      int32_t* out_token) {
    ABI_GUARD_BEGIN
    if (!ctx || !tokens || !out_token || n_tokens == 0) return DESIREEIA_ERR_INVALID_ARG;
    if (!embd || n_embd == 0) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_has_vision(ctx)) return DESIREEIA_ERR_NOT_SUPPORTED;

    int32_t token = -1;
    if (!desireeia::engine_predict_vision(ctx, tokens, n_tokens, embd, n_embd,
                                          image_token, token)) {
        return DESIREEIA_ERR_UNDEFINED;
    }
    *out_token = token;
    return DESIREEIA_OK;
    ABI_GUARD_END
}
