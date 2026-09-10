#include "desireeia/abi.h"
#include "core/engine.h"
#include "core/profile.h"
#include "vision/vision_clip.h"
#include "vision/vision_image.h"

#include <cstring>

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
}

#define DESIREEIA_EMIT(level, msg) emit_impl(__FILE__, __LINE__, __func__, (level), (msg))

DESIREEIA_API const char* desireeia_version(void) {
    return "0.1.0";
}

DESIREEIA_API desireeia_error desireeia_set_logger(desireeia_log_cb cb, void* user) {
    g_log = cb;
    g_log_user = user;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_probe_hw(desireeia_hw_info* out) {
    if (!out) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    return desireeia::probe_hardware(*out);
}

DESIREEIA_API desireeia_error desireeia_make_plan(const desireeia_hw_info* hw,
                                         const char* model_path,
                                         const desireeia_plan* override,
                                         desireeia_plan* out) {
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
}

DESIREEIA_API desireeia_error desireeia_create(const char* model_path,
                                      const desireeia_plan* plan,
                                      desireeia_log_cb cb,
                                      void* user,
                                      desireeia_ctx** out) {
    if (!model_path || !plan || !out) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    if (cb) {
        g_log = cb;
        g_log_user = user;
    }
    desireeia_ctx* ctx = desireeia::engine_create(model_path, *plan, [](int32_t l, const char* m) {
        log_msg(l, m);
    });
    if (!ctx) {
        return DESIREEIA_ERR_IO;
    }
    *out = ctx;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_destroy(desireeia_ctx* ctx) {
    if (!ctx) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    desireeia::engine_destroy(ctx);
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_predict(desireeia_ctx* ctx,
                                       const int32_t* tokens,
                                       size_t n_tokens,
                                       int32_t* out_token) {
    if (!ctx || !tokens || !out_token) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    if (!desireeia::engine_predict(ctx, tokens, n_tokens, *out_token)) {
        return DESIREEIA_ERR_UNDEFINED;
    }
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_next_token(desireeia_ctx* ctx, int32_t* out_token) {
    if (!ctx || !out_token) {
        return DESIREEIA_ERR_INVALID_ARG;
    }
    int32_t t = 0;
    if (!desireeia::engine_next_token(ctx, t)) {
        return DESIREEIA_ERR_UNDEFINED;
    }
    *out_token = t;
    return DESIREEIA_OK;
}

DESIREEIA_API size_t desireeia_context_size(const desireeia_ctx* ctx) {
    if (!ctx) {
        return 0;
    }
    return desireeia::engine_context_size(ctx);
}

DESIREEIA_API desireeia_error desireeia_tokenize(desireeia_ctx* ctx,
                                        const char* text,
                                        int32_t add_bos,
                                        int32_t* out_ids,
                                        size_t max_ids,
                                        size_t* out_count) {
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
}

DESIREEIA_API desireeia_error desireeia_embed(desireeia_ctx* ctx,
                                     const int32_t* tokens,
                                     size_t n_tokens,
                                     float* out_embd,
                                     size_t out_capacity,
                                     size_t* out_len,
                                     uint32_t* out_embd_dim) {
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
}

DESIREEIA_API desireeia_error desireeia_token_piece(desireeia_ctx* ctx,
                                           int32_t id,
                                           char* out_buf,
                                           size_t buf_size) {
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
}

DESIREEIA_API desireeia_error desireeia_special_token_id(const desireeia_ctx* ctx,
                                                desireeia_special_token which,
                                                int32_t* out_id) {
    if (!ctx || !out_id) return DESIREEIA_ERR_INVALID_ARG;
    int32_t id = -1;
    if (!desireeia::engine_special_token_id(ctx, (int) which, id)) {
        return DESIREEIA_ERR_NOT_SUPPORTED;
    }
    *out_id = id;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_is_eog_token(const desireeia_ctx* ctx,
                                            int32_t id,
                                            int32_t* out_is_eog) {
    if (!ctx || !out_is_eog) return DESIREEIA_ERR_INVALID_ARG;
    *out_is_eog = desireeia::engine_is_eog_token(ctx, id) ? 1 : 0;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_set_sampling(desireeia_ctx* ctx,
                                            const desireeia_sampling* params) {
    if (!ctx || !params) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_set_sampling(ctx, *params)) return DESIREEIA_ERR_INVALID_ARG;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_get_sampling(const desireeia_ctx* ctx,
                                            desireeia_sampling* out) {
    if (!ctx || !out) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_get_sampling(ctx, *out)) return DESIREEIA_ERR_INVALID_ARG;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_load_lora_adapter(desireeia_ctx* ctx,
                                                            const char* lora_gguf_path,
                                                            float scale) {
    if (!ctx || !lora_gguf_path) return DESIREEIA_ERR_INVALID_ARG;
    std::string err;
    if (!desireeia::engine_load_lora(ctx, lora_gguf_path, scale, err)) {
        if (err.find("not a LoRA adapter") != std::string::npos) return DESIREEIA_ERR_PARSE;
        if (err.find("no generative forward engine") != std::string::npos) return DESIREEIA_ERR_NOT_SUPPORTED;
        return DESIREEIA_ERR_IO;
    }
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_clear_lora_adapters(desireeia_ctx* ctx) {
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_clear_lora(ctx)) return DESIREEIA_ERR_UNDEFINED;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_load_prerouter(desireeia_ctx* ctx, const char* path) {
    if (!ctx || !path) return DESIREEIA_ERR_INVALID_ARG;
    std::string err;
    if (!desireeia::engine_load_prerouter(ctx, path, err)) {
        if (err.find("no valid prerouter heads") != std::string::npos) return DESIREEIA_ERR_PARSE;
        if (err.find("no generative forward engine") != std::string::npos ||
            err.find("no MoE experts") != std::string::npos) return DESIREEIA_ERR_NOT_SUPPORTED;
        return DESIREEIA_ERR_IO;
    }
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_clear_prerouter(desireeia_ctx* ctx) {
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_clear_prerouter(ctx)) return DESIREEIA_ERR_UNDEFINED;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_set_prerouter_heuristic(desireeia_ctx* ctx, int32_t enabled) {
    if (!ctx) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::engine_set_prerouter_heuristic(ctx, enabled != 0)) return DESIREEIA_ERR_UNDEFINED;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_apply_chat_template(const desireeia_ctx* ctx,
                                                    const char** roles,
                                                    const char** contents,
                                                    size_t n_messages,
                                                    int add_assistant,
                                                    char* out_buf,
                                                    size_t buf_size,
                                                    size_t* out_len) {
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
    if (!path || !out) return DESIREEIA_ERR_INVALID_ARG;

    auto buf = desireeia::vision::load_image_from_file(path, expected_channels);
    if (buf.empty()) return DESIREEIA_ERR_IO;

    out->width = buf.width;
    out->height = buf.height;
    out->channels = buf.channels;
    out->data = new uint8_t[buf.data.size()];
    std::memcpy(out->data, buf.data.data(), buf.data.size());
    return DESIREEIA_OK;
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
    return desireeia::vision::vision_create_context(model_path, cb, user);
}

DESIREEIA_API void desireeia_vision_destroy(DesireeAIVisionCtx* ctx) {
    desireeia::vision::vision_destroy_context(ctx);
}

DESIREEIA_API desireeia_error desireeia_vision_get_config(const DesireeAIVisionCtx* ctx,
                                                          DesireeAIVisionConfig* out) {
    if (!ctx || !out) return DESIREEIA_ERR_INVALID_ARG;
    if (!desireeia::vision::vision_get_config(ctx, *out)) {
        return DESIREEIA_ERR_UNDEFINED;
    }
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_vision_encode(DesireeAIVisionCtx* ctx,
                                                      const DesireeAIImage* image,
                                                      float* out_embd,
                                                      size_t out_capacity,
                                                      size_t* out_len,
                                                      uint32_t* out_dim) {
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
}

DESIREEIA_API desireeia_error desireeia_vision_preprocess(const DesireeAIImage* input,
                                                          int32_t target_size,
                                                          float* out_pixels,
                                                          size_t out_capacity,
                                                          size_t* out_len) {
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
}

// ========================================================================
// Multimodal model support (ctx-based vision encoder)
// ========================================================================

DESIREEIA_API desireeia_error desireeia_has_vision(desireeia_ctx* ctx,
                                                   int32_t* out_has) {
    if (!ctx || !out_has) return DESIREEIA_ERR_INVALID_ARG;
    *out_has = desireeia::engine_has_vision(ctx) ? 1 : 0;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_vision_token_count(desireeia_ctx* ctx,
                                                           int32_t* out_count) {
    if (!ctx || !out_count) return DESIREEIA_ERR_INVALID_ARG;
    const int32_t n = desireeia::engine_vision_token_count(ctx);
    if (n <= 0) return DESIREEIA_ERR_NOT_SUPPORTED;
    *out_count = n;
    return DESIREEIA_OK;
}

DESIREEIA_API desireeia_error desireeia_vision_image_token(desireeia_ctx* ctx,
                                                           int32_t* out_id) {
    if (!ctx || !out_id) return DESIREEIA_ERR_INVALID_ARG;
    *out_id = desireeia::engine_vision_image_token(ctx);
    return *out_id >= 0 ? DESIREEIA_OK : DESIREEIA_ERR_NOT_SUPPORTED;
}

DESIREEIA_API desireeia_error desireeia_vision_encode_ctx(desireeia_ctx* ctx,
                                                          const DesireeAIImage* image,
                                                          float* out_embd,
                                                          size_t out_capacity,
                                                          size_t* out_len,
                                                          uint32_t* out_dim) {
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
}

DESIREEIA_API desireeia_error desireeia_predict_image(desireeia_ctx* ctx,
                                                      const int32_t* tokens,
                                                      size_t n_tokens,
                                                      const float* embd,
                                                      size_t n_embd,
                                                      int32_t image_token,
                                                      int32_t* out_token) {
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
}
