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

/* Tokenizer (SentencePiece Unigram o BPE byte-level, auto-rilevato dai
 * metadati del modello). Convenzione query-size: chiamare con out_ids=NULL
 * (o max_ids=0) per ottenere in out_count il numero di token necessari,
 * senza scrivere nulla; una seconda chiamata con un buffer di quella
 * dimensione riempie out_ids. Ritorna DESIREEIA_ERR_NOT_SUPPORTED se il
 * modello non ha un tokenizer riconosciuto. */
DESIREEIA_API desireeia_error desireeia_tokenize(desireeia_ctx* ctx,
                                        const char* text,
                                        int32_t add_bos,
                                        int32_t* out_ids,
                                        size_t max_ids,
                                        size_t* out_count);

/* Scrive in out_buf (capacita' buf_size, sempre NUL-terminato se buf_size>0)
 * il testo del token `id`. Ritorna DESIREEIA_ERR_INVALID_ARG se l'id e' fuori
 * range o il buffer troppo piccolo, DESIREEIA_ERR_NOT_SUPPORTED se il modello
 * non ha un tokenizer riconosciuto. */
DESIREEIA_API desireeia_error desireeia_token_piece(desireeia_ctx* ctx,
                                           int32_t id,
                                           char* out_buf,
                                           size_t buf_size);

/* Identificatori dei kind di token speciale per desireeia_special_token_id
 * (Fase 0 "motore di inferenza reale": serve a chi genera per sapere
 * quando fermarsi — EOS — o a chi compone una chat template per sapere
 * come iniziare/finire un turno). */
typedef enum desireeia_special_token {
    DESIREEIA_TOKEN_BOS = 0,
    DESIREEIA_TOKEN_EOS = 1,
    DESIREEIA_TOKEN_UNK = 2,
    DESIREEIA_TOKEN_PAD = 3
} desireeia_special_token;

/* Scrive in *out_id l'id del token speciale richiesto, o -1 se il modello
 * non lo definisce (metadato GGUF assente). DESIREEIA_ERR_NOT_SUPPORTED se
 * il modello non ha un vocabolario riconosciuto. */
DESIREEIA_API desireeia_error desireeia_special_token_id(const desireeia_ctx* ctx,
                                                desireeia_special_token which,
                                                int32_t* out_id);

/* Scrive in *out_is_eog 1 se `id` e' un token di FINE GENERAZIONE, 0
 * altrimenti. Non basta confrontare con l'id EOS: i modelli chat chiudono
 * il turno con token dedicati (gemma: <end_of_turn>), e fermarsi solo su
 * EOS lascia il modello a ripeterli all'infinito. */
DESIREEIA_API desireeia_error desireeia_is_eog_token(const desireeia_ctx* ctx,
                                            int32_t id,
                                            int32_t* out_is_eog);

/* BERT (encoder-only): embedding per token dell'intera sequenza, in un
 * solo passo (nessuno stato fra chiamate, a differenza di desireeia_predict/
 * desireeia_next_token — un embedding non dipende da chiamate precedenti).
 * Convenzione query-size come desireeia_tokenize: chiamare con out_embd=NULL
 * (o out_capacity=0) per ottenere in *out_len il numero di float necessari
 * (sempre n_tokens * *out_embd_dim) senza scrivere nulla; una seconda
 * chiamata con un buffer di quella dimensione riempie out_embd (token 0 ai
 * float [0, embd_dim), token 1 a [embd_dim, 2*embd_dim), ...). L'embedding
 * NON e' aggregato/normalizzato dal motore (pooling/normalizzazione sono
 * una scelta dell'applicazione). Ritorna DESIREEIA_ERR_NOT_SUPPORTED se il
 * modello caricato non e' un encoder BERT. */
DESIREEIA_API desireeia_error desireeia_embed(desireeia_ctx* ctx,
                                     const int32_t* tokens,
                                     size_t n_tokens,
                                     float* out_embd,
                                     size_t out_capacity,
                                     size_t* out_len,
                                     uint32_t* out_embd_dim);

/* Parametri di campionamento. Prima esisteva solo l'argmax greedy: a
 * parita' di prompt il motore rispondeva sempre identico, e su generazioni
 * lunghe entrava in cicli ripetitivi da cui non usciva piu' (misurato:
 * "France is a country with a..." ripetuto fino a degenerare in token
 * casuali). La penalita' di ripetizione serve proprio a quello.
 *
 * temperature <= 0 = greedy (comportamento precedente, resta il default).
 * top_k <= 0 disattiva il top-k; top_p >= 1 disattiva il top-p;
 * penalty_repeat == 1 disattiva la penalita'. seed == 0 = seme casuale. */
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

/* Imposta i parametri di campionamento del contesto. Puo' essere chiamata
 * in qualsiasi momento; ha effetto dal token successivo. */
DESIREEIA_API desireeia_error desireeia_set_sampling(desireeia_ctx* ctx,
                                            const desireeia_sampling* params);

/* Riempie *out con i parametri di campionamento correnti. */
DESIREEIA_API desireeia_error desireeia_get_sampling(const desireeia_ctx* ctx,
                                            desireeia_sampling* out);

/* Applica il formato di chat rilevato per il modello (dal
 * "tokenizer.chat_template" del GGUF, o da un default per architettura se
 * assente) a una sequenza di messaggi, scrivendo il prompt risultante in
 * out_buf (NUL-terminato se buf_size>0, troncato se il buffer e' piccolo).
 * *out_len riceve la lunghezza necessaria (senza NUL): se maggiore di
 * buf_size, richiamare con un buffer piu' grande. roles/contents sono
 * array paralleli di n_messages stringhe UTF-8 NUL-terminate (ruoli tipici:
 * "system", "user", "assistant"). add_assistant=1 aggiunge il marcatore di
 * apertura del turno assistente, per far generare la risposta subito dopo. */
DESIREEIA_API desireeia_error desireeia_apply_chat_template(const desireeia_ctx* ctx,
                                                    const char** roles,
                                                    const char** contents,
                                                    size_t n_messages,
                                                    int add_assistant,
                                                    char* out_buf,
                                                    size_t buf_size,
                                                    size_t* out_len);

/* Profiler minimo sempre attivo (tempo cumulativo per tipo di kernel
 * matmul: quantizzazione attivazione, Q4_0/Q4_K/Q6_K, fallback float).
 * desireeia_profile_dump scrive una riga leggibile in out_buf (NUL-terminata
 * se buf_size>0, troncata se il buffer e' piccolo); desireeia_profile_reset
 * azzera i contatori (utile per isolare la misura a una finestra, es. solo
 * il ciclo di decode di un benchmark). I contatori sono globali di
 * processo, non per-modello. */
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

/* Scrive 1 in *out_has se il modello caricato ha un encoder visivo
 * (vision), 0 altrimenti. */
DESIREEIA_API desireeia_error desireeia_has_vision(desireeia_ctx* ctx,
                                                   int32_t* out_has);

/* Numero di vector di embedding che l'encoder visivo produce per immagine
 * (= numero di occorrenze del token placeholder da inserire nel prompt). */
DESIREEIA_API desireeia_error desireeia_vision_token_count(desireeia_ctx* ctx,
                                                           int32_t* out_count);

/* Id del token placeholder immagine risolto sul vocabolario del modello
 * (es. <image>), o -1 se non risolvibile. */
DESIREEIA_API desireeia_error desireeia_vision_image_token(desireeia_ctx* ctx,
                                                           int32_t* out_id);

/* Codifica un'immagine con l'encoder visivo del modello caricato.
 * Stessa convenzione query-size di desireeia_embed: out_embd=NULL con
 * out_capacity=0 riempie solo *out_len. out_dim riceve la larghezza di
 * ogni vector (dimensione embedding del modello di testo). */
DESIREEIA_API desireeia_error desireeia_vision_encode_ctx(desireeia_ctx* ctx,
                                                          const DesireeAIImage* image,
                                                          float* out_embd,
                                                          size_t out_capacity,
                                                          size_t* out_len,
                                                          uint32_t* out_dim);

/* Prefill multimodale sul modello caricato: like desireeia_predict, ma il
 * flusso token contiene N occorrenze del token placeholder immagine (N =
 * desireeia_vision_token_count) e `embd` contiene i vector dell'encoder
 * visivo (N * out_dim float, uno per occorrenza, in ordine). image_token
 * puo' essere -1 per usare il placeholder risolto automaticamente.
 * Ritorna DESIREEIA_ERR_NOT_SUPPORTED se il modello non ha vision o non e'
 * un motore forward denso. */
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
