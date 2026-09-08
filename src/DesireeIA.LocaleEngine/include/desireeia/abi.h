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

#ifdef __cplusplus
}
#endif

#endif
