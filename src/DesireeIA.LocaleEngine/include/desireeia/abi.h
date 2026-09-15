// DesireeIA
// Copyright (c) Passaro Francesco Paolo. All rights reserved.
// Licensed under the DesireeIA License - see LICENSE and the "License"
// section of README.md for full terms: no modification, no unauthorized
// integration, no AI training/ingestion without explicit written consent
// from the author.

#ifndef DESIREEIA_ABI_H
#define DESIREEIA_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(DESIREEIA_BUILD)
        #define DESIREEIA_API __declspec(dllexport)
    #else
        #define DESIREEIA_API __declspec(dllimport)
    #endif
#else
    #define DESIREEIA_API __attribute__((visibility("default")))
#endif

#define DESIREEIA_MAX_PATH 1024

typedef enum desireeia_backend {
    DESIREEIA_BACKEND_CPU     = 1,
    DESIREEIA_BACKEND_CUDA    = 2,
    DESIREEIA_BACKEND_METAL   = 3,
    DESIREEIA_BACKEND_VULKAN  = 4,
    DESIREEIA_BACKEND_INTEL   = 5,
    DESIREEIA_BACKEND_AXELERA = 6
} desireeia_backend;

typedef enum desireeia_format {
    DESIREEIA_FORMAT_UNKNOWN    = 0,
    DESIREEIA_FORMAT_GGUF       = 1,
    DESIREEIA_FORMAT_SAFETENSORS = 2
} desireeia_format;

typedef enum desireeia_quant {
    DESIREEIA_QUANT_F32  = 0,
    DESIREEIA_QUANT_F16  = 1,
    DESIREEIA_QUANT_Q4_0 = 2,
    DESIREEIA_QUANT_Q4_1 = 3,
    DESIREEIA_QUANT_Q8_0 = 4,
    DESIREEIA_QUANT_Q4_K = 5,
    DESIREEIA_QUANT_Q5_K = 6,
    DESIREEIA_QUANT_Q6_K = 7
} desireeia_quant;

typedef struct desireeia_hw_info {
    int32_t cpu_threads;
    int32_t cpu_has_avx;
    int32_t cpu_has_avx2;
    int32_t cpu_has_avx512;
    int32_t cpu_has_neon;
    int32_t cuda_device_count;
    int32_t cuda_total_mb;
    uint64_t ram_total_mb;
    uint64_t ram_free_mb;
    int32_t has_metal;
    int32_t has_vulkan;
    int32_t intel_gpu_count;
    int32_t axelera_device_count;
} desireeia_hw_info;

/* How the SSD storage tier participates in serving model weights.
 *
 * OFF keeps every weight in the RAM/CPU path, which is the fastest option
 * whenever the model actually fits the RAM budget — the tier is not even
 * constructed, so nothing sits on the tensor read path.
 *
 * AUTO decides per model: if the weights fit the planned RAM budget the
 * behavior is identical to OFF; if they don't, the overflow is served from
 * the model file on SSD instead of failing or thrashing.
 *
 * ALWAYS forces reads through the SSD tier even when the model would fit
 * in RAM. This is slower by design and exists for two reasons: measuring
 * the tier honestly, and running a model far larger than RAM on a machine
 * where predictable streaming beats swapping. */
typedef enum desireeia_ssd_tier_mode {
    DESIREEIA_SSD_TIER_OFF    = 0,
    DESIREEIA_SSD_TIER_AUTO   = 1,
    DESIREEIA_SSD_TIER_ALWAYS = 2
} desireeia_ssd_tier_mode;

typedef struct desireeia_plan {
    desireeia_backend backend;
    desireeia_format format;
    desireeia_quant dense_quant;
    desireeia_quant expert_quant;
    int32_t n_threads;
    uint64_t ram_budget_mb;
    int32_t expert_cache_count;
    int32_t expert_prefetch_enabled;
    int32_t kv_compression_enabled;
    int32_t expert_pin_enabled;
    int32_t expert_prefetch_depth;
    int32_t batch_union_enabled;
    int32_t dual_ssd_enabled;
    /* desireeia_ssd_tier_mode. */
    int32_t ssd_tier_mode;
    /* RAM the SSD tier may use for its own hot-block cache, in MiB.
     * 0 means "derive it from ram_budget_mb". Ignored when the mode is OFF. */
    uint64_t ssd_tier_cache_mb;
} desireeia_plan;

typedef enum desireeia_error {
    DESIREEIA_OK                = 0,
    DESIREEIA_ERR_INVALID_ARG   = -1,
    DESIREEIA_ERR_NOT_SUPPORTED = -2,
    DESIREEIA_ERR_IO            = -3,
    DESIREEIA_ERR_PARSE         = -4,
    DESIREEIA_ERR_NO_MEM        = -5,
    DESIREEIA_ERR_UNDEFINED     = -6
} desireeia_error;

typedef struct desireeia_ctx desireeia_ctx;

typedef void (*desireeia_log_cb)(int32_t level, const char* msg, void* user);

DESIREEIA_API const char* desireeia_version(void);
DESIREEIA_API desireeia_error desireeia_set_logger(desireeia_log_cb cb, void* user);

DESIREEIA_API desireeia_error desireeia_probe_hw(desireeia_hw_info* out);
DESIREEIA_API desireeia_error desireeia_make_plan(const desireeia_hw_info* hw,
                                         const char* model_path,
                                         const desireeia_plan* override,
                                         desireeia_plan* out);

DESIREEIA_API desireeia_error desireeia_create(const char* model_path,
                                      const desireeia_plan* plan,
                                      desireeia_log_cb cb,
                                      void* user,
                                      desireeia_ctx** out);

DESIREEIA_API desireeia_error desireeia_destroy(desireeia_ctx* ctx);

DESIREEIA_API desireeia_error desireeia_predict(desireeia_ctx* ctx,
                                       const int32_t* tokens,
                                       size_t n_tokens,
                                       int32_t* out_token);

DESIREEIA_API desireeia_error desireeia_next_token(desireeia_ctx* ctx, int32_t* out_token);
DESIREEIA_API size_t desireeia_context_size(const desireeia_ctx* ctx);

/* Tokenizer (SentencePiece Unigram or byte-level BPE, auto-detected from
 * the model's metadata). Query-size convention: call with out_ids=NULL
 * (or max_ids=0) to get the number of tokens needed in out_count,
 * without writing anything; a second call with a buffer of that size
 * fills out_ids. Returns DESIREEIA_ERR_NOT_SUPPORTED if the model does
 * not have a recognized tokenizer. */
DESIREEIA_API desireeia_error desireeia_tokenize(desireeia_ctx* ctx,
                                        const char* text,
                                        int32_t add_bos,
                                        int32_t* out_ids,
                                        size_t max_ids,
                                        size_t* out_count);

/* Writes into out_buf (capacity buf_size, always NUL-terminated if buf_size>0)
 * the text of token `id`. Returns DESIREEIA_ERR_INVALID_ARG if the id is out
 * of range or the buffer is too small, DESIREEIA_ERR_NOT_SUPPORTED if the
 * model does not have a recognized tokenizer. */
DESIREEIA_API desireeia_error desireeia_token_piece(desireeia_ctx* ctx,
                                           int32_t id,
                                           char* out_buf,
                                           size_t buf_size);

/* Kind identifiers for special tokens, for desireeia_special_token_id
 * (Phase 0 "real inference engine": used by the generation loop to know
 * when to stop — EOS — or by chat template composition to know how to
 * open/close a turn). */
typedef enum desireeia_special_token {
    DESIREEIA_TOKEN_BOS = 0,
    DESIREEIA_TOKEN_EOS = 1,
    DESIREEIA_TOKEN_UNK = 2,
    DESIREEIA_TOKEN_PAD = 3
} desireeia_special_token;

/* Writes into *out_id the id of the requested special token, or -1 if the
 * model does not define it (GGUF metadata absent). DESIREEIA_ERR_NOT_SUPPORTED
 * if the model does not have a recognized vocabulary. */
DESIREEIA_API desireeia_error desireeia_special_token_id(const desireeia_ctx* ctx,
                                                desireeia_special_token which,
                                                int32_t* out_id);

/* Writes into *out_is_eog 1 if `id` is an END-OF-GENERATION token, 0
 * otherwise. Comparing against the EOS id alone is not enough: chat models
 * close the turn with dedicated tokens (gemma: <end_of_turn>), and stopping
 * only on EOS leaves the model repeating them forever. */
DESIREEIA_API desireeia_error desireeia_is_eog_token(const desireeia_ctx* ctx,
                                            int32_t id,
                                            int32_t* out_is_eog);

/* BERT (encoder-only): per-token embedding for the whole sequence, in a
 * single pass (no state across calls, unlike desireeia_predict/
 * desireeia_next_token — an embedding does not depend on previous calls).
 * Query-size convention as in desireeia_tokenize: call with out_embd=NULL
 * (or out_capacity=0) to get the number of floats needed in *out_len
 * (always n_tokens * *out_embd_dim) without writing anything; a second
 * call with a buffer of that size fills out_embd (token 0 in floats
 * [0, embd_dim), token 1 in [embd_dim, 2*embd_dim), ...). The embedding is
 * NOT aggregated/normalized by the engine (pooling/normalization are an
 * application-level choice). Returns DESIREEIA_ERR_NOT_SUPPORTED if the
 * loaded model is not a BERT encoder. */
DESIREEIA_API desireeia_error desireeia_embed(desireeia_ctx* ctx,
                                     const int32_t* tokens,
                                     size_t n_tokens,
                                     float* out_embd,
                                     size_t out_capacity,
                                     size_t* out_len,
                                     uint32_t* out_embd_dim);

/* Sampling parameters. Previously only greedy argmax existed: for the same
 * prompt the engine always answered identically, and on long generations it
 * would fall into repetitive loops it never escaped (measured: "France is a
 * country with a..." repeated until it degenerated into random tokens). The
 * repetition penalty exists specifically to address that.
 *
 * temperature <= 0 = greedy (previous behavior, still the default).
 * top_k <= 0 disables top-k; top_p >= 1 disables top-p;
 * penalty_repeat == 1 disables the penalty. seed == 0 = random seed. */
typedef struct desireeia_sampling {
    float    temperature;
    int32_t  top_k;
    float    top_p;
    float    penalty_repeat;
    float    penalty_freq;
    float    penalty_present;
    int32_t  penalty_last_n;
    uint32_t seed;
} desireeia_sampling;

/* Sets the context's sampling parameters. Can be called at any time; takes
 * effect starting from the next token. */
DESIREEIA_API desireeia_error desireeia_set_sampling(desireeia_ctx* ctx,
                                            const desireeia_sampling* params);

/* Fills *out with the current sampling parameters. */
DESIREEIA_API desireeia_error desireeia_get_sampling(const desireeia_ctx* ctx,
                                            desireeia_sampling* out);

/* Recover-LoRA: loads a LoRA adapter in GGUF format, desireeialmn
 * convention (tensors "<base_name>.lora_a" / "<base_name>.lora_b",
 * metadata "adapter.lora.alpha"). Applied at runtime without touching the
 * base weights: out = base_mm(x, W) + scale * B @ (A @ x). Can be called
 * multiple times to load several adapters at once (their contributions
 * add up); scale multiplies the adapter's alpha/rank factor (1.0 if the
 * adapter has no alpha). Covers attention, dense/MLA/shared-expert FFN;
 * does NOT cover routed MoE experts (different code path, see
 * docs/README). Returns DESIREEIA_ERR_NOT_SUPPORTED if the model does not
 * have a generative forward engine (e.g. a BERT encoder), DESIREEIA_ERR_IO
 * if the file cannot be opened, DESIREEIA_ERR_PARSE if it is not a valid
 * LoRA adapter. */
DESIREEIA_API desireeia_error desireeia_load_lora_adapter(desireeia_ctx* ctx,
                                                  const char* lora_gguf_path,
                                                  float scale);

/* Removes all LoRA adapters loaded on the context. */
DESIREEIA_API desireeia_error desireeia_clear_lora_adapters(desireeia_ctx* ctx);

/* Prerouter routing prediction: predicts, from layer L's data, which
 * experts layer L+1 will likely route to, so their weights can be
 * prefetched in the background one layer before the router's actual
 * computation (see src/models/prerouter.h for the math).
 * It never affects correctness: a wrong or missing prediction simply
 * leaves the existing read path to run when layer L+1 actually computes
 * its own router.
 *
 * `path`: a GGUF file using this engine's own naming convention (tensors
 * "prerouter.<N>.fc1.weight" / ".fc2.weight" / ".linear_init.weight" for
 * each "owner" layer N) — NOT byte-for-byte compatible with any external
 * reference format. Returns DESIREEIA_ERR_NOT_SUPPORTED if the model has
 * no MoE experts or no generative forward engine, DESIREEIA_ERR_PARSE if
 * the file contains no valid head in that format. */
DESIREEIA_API desireeia_error desireeia_load_prerouter(desireeia_ctx* ctx, const char* path);

/* Removes all loaded prerouter heads. */
DESIREEIA_API desireeia_error desireeia_clear_prerouter(desireeia_ctx* ctx);

/* Enables/disables the fallback heuristic ("layer L+1 routes to the same
 * experts layer L just used") for owner layers with no trained prerouter
 * head loaded. Off by default: no trained head is currently available for
 * any model, so this is the only way to get SOME prediction until a real
 * one exists — explicitly a placeholder, not an accurate prediction. */
DESIREEIA_API desireeia_error desireeia_set_prerouter_heuristic(desireeia_ctx* ctx, int32_t enabled);

/* Applies the chat format detected for the model (from the GGUF's
 * "tokenizer.chat_template", or a per-architecture default if absent) to a
 * sequence of messages, writing the resulting prompt into out_buf
 * (NUL-terminated if buf_size>0, truncated if the buffer is too small).
 * *out_len receives the required length (without the NUL): if greater than
 * buf_size, call again with a larger buffer. roles/contents are parallel
 * arrays of n_messages NUL-terminated UTF-8 strings (typical roles:
 * "system", "user", "assistant"). add_assistant=1 appends the assistant
 * turn's opening marker, so generation can start right after. */
DESIREEIA_API desireeia_error desireeia_apply_chat_template(const desireeia_ctx* ctx,
                                                    const char** roles,
                                                    const char** contents,
                                                    size_t n_messages,
                                                    int add_assistant,
                                                    char* out_buf,
                                                    size_t buf_size,
                                                    size_t* out_len);

/* Minimal always-on profiler (cumulative time per matmul kernel type:
 * activation quantization, Q4_0/Q4_K/Q6_K, float fallback).
 * desireeia_profile_dump writes a human-readable line into out_buf
 * (NUL-terminated if buf_size>0, truncated if the buffer is too small);
 * desireeia_profile_reset zeroes the counters (useful to isolate the
 * measurement to a window, e.g. just a benchmark's decode loop). The
 * counters are process-global, not per-model. */
DESIREEIA_API void desireeia_profile_dump(char* out_buf, size_t buf_size);
DESIREEIA_API void desireeia_profile_reset(void);

/* ========================================================================
 * DesireeAI Vision Module - Image processing and CLIP vision encoder
 * ======================================================================== */

/* Image representation: interleaved RGB/RGBA/grayscale pixel data.
 * Pixel layout: data[y * width * channels + x * channels + c].
 * Caller must free data pointer with desireeia_free_image(). */
typedef struct DesireeAIImage {
    uint32_t width;
    uint32_t height;
    uint32_t channels;  /* 1=grayscale, 3=RGB, 4=RGBA */
    uint8_t* data;
} DesireeAIImage;

/* Vision encoder configuration, read from model metadata or set explicitly. */
typedef struct DesireeAIVisionConfig {
    int32_t embedding_dim;      /* Vision embedding dimension (e.g. 768, 1024) */
    int32_t patch_size;         /* Patch size (e.g. 14, 16, 32) */
    int32_t image_size;         /* Input image size (e.g. 224, 384) */
    int32_t num_heads;          /* Number of attention heads */
    int32_t num_layers;         /* Number of transformer layers */
    int32_t projection_dim;     /* Output projection dimension */
    int32_t has_encoder;        /* 1 if vision encoder is loaded, 0 otherwise */
} DesireeAIVisionConfig;

/* Vision context: holds the loaded CLIP vision encoder weights and state.
 * Created by desireeia_vision_create(), destroyed by desireeia_vision_destroy(). */
typedef struct DesireeAIVisionCtx DesireeAIVisionCtx;

/* Load an image from disk. Supports PNG, JPEG, BMP via stb_image.
 * Returns DESIREEIA_OK on success, caller must free with desireeia_free_image(). */
DESIREEIA_API desireeia_error desireeia_load_image(const char* path,
                                                   int32_t expected_channels,
                                                   DesireeAIImage* out);

/* Free image data allocated by desireeia_load_image(). */
DESIREEIA_API void desireeia_free_image(DesireeAIImage* img);

/* Create a vision encoder context from a model file path.
 * The model must contain CLIP vision encoder weights.
 * Returns NULL on failure. */
DESIREEIA_API DesireeAIVisionCtx* desireeia_vision_create(const char* model_path,
                                                          desireeia_log_cb cb,
                                                          void* user);

/* Destroy a vision encoder context. */
DESIREEIA_API void desireeia_vision_destroy(DesireeAIVisionCtx* ctx);

/* Get the vision encoder configuration. */
DESIREEIA_API desireeia_error desireeia_vision_get_config(const DesireeAIVisionCtx* ctx,
                                                          DesireeAIVisionConfig* out);

/* Encode an image into embedding vectors.
 * Input: image (must match config image_size, will be resized/cropped).
 * Output: out_embd (caller-allocated buffer), out_len (number of floats),
 *         out_dim (embedding dimension per vector).
 * Returns DESIREEIA_OK on success. */
DESIREEIA_API desireeia_error desireeia_vision_encode(DesireeAIVisionCtx* ctx,
                                                      const DesireeAIImage* image,
                                                      float* out_embd,
                                                      size_t out_capacity,
                                                      size_t* out_len,
                                                      uint32_t* out_dim);

/* Preprocess image for vision encoder: resize to target_size x target_size,
 * normalize pixel values to [0, 1], convert to RGB if needed.
 * out_pixels is caller-allocated with out_capacity floats.
 * out_len receives the actual number of floats written.
 * Returns DESIREEIA_OK on success. */
DESIREEIA_API desireeia_error desireeia_vision_preprocess(const DesireeAIImage* input,
                                                          int32_t target_size,
                                                          float* out_pixels,
                                                          size_t out_capacity,
                                                          size_t* out_len);

/* ========================================================================
 * Multimodal model support (ctx-based, loaded from the model's own GGUF)
 * ========================================================================
 *
 * Variants of the vision API that operate on a *loaded model context*
 * (desireeia_ctx) instead of a standalone vision context. When the GGUF
 * file carries clip.vision.* metadata the vision encoder is auto-loaded at
 * desireeia_create() time and these functions work without extra setup. */

/* Writes 1 into *out_has if the loaded model has a vision encoder,
 * 0 otherwise. */
DESIREEIA_API desireeia_error desireeia_has_vision(desireeia_ctx* ctx,
                                                   int32_t* out_has);

/* Number of embedding vectors the vision encoder produces per image
 * (= number of occurrences of the placeholder token to insert in the
 * prompt). */
DESIREEIA_API desireeia_error desireeia_vision_token_count(desireeia_ctx* ctx,
                                                           int32_t* out_count);

/* Id of the image placeholder token resolved against the model's
 * vocabulary (e.g. <image>), or -1 if it cannot be resolved. */
DESIREEIA_API desireeia_error desireeia_vision_image_token(desireeia_ctx* ctx,
                                                           int32_t* out_id);

/* Encodes an image with the loaded model's vision encoder.
 * Same query-size convention as desireeia_embed: out_embd=NULL with
 * out_capacity=0 fills only *out_len. out_dim receives the width of
 * each vector (the text model's embedding dimension). */
DESIREEIA_API desireeia_error desireeia_vision_encode_ctx(desireeia_ctx* ctx,
                                                          const DesireeAIImage* image,
                                                          float* out_embd,
                                                          size_t out_capacity,
                                                          size_t* out_len,
                                                          uint32_t* out_dim);

/* Multimodal prefill on the loaded model: like desireeia_predict, but the
 * token stream contains N occurrences of the image placeholder token (N =
 * desireeia_vision_token_count) and `embd` contains the vision encoder's
 * vectors (N * out_dim floats, one per occurrence, in order). image_token
 * can be -1 to use the automatically resolved placeholder.
 * Returns DESIREEIA_ERR_NOT_SUPPORTED if the model has no vision or is not
 * a dense forward engine. */
DESIREEIA_API desireeia_error desireeia_predict_image(desireeia_ctx* ctx,
                                                      const int32_t* tokens,
                                                      size_t n_tokens,
                                                      const float* embd,
                                                      size_t n_embd,
                                                      int32_t image_token,
                                                      int32_t* out_token);

#ifdef __cplusplus
}
#endif

#endif
