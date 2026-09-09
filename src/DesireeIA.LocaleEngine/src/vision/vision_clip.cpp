#include "vision_clip.h"
#include "vision_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

namespace desireeia {
namespace vision {

// ============================================================
// Matrix-vector operations
// ============================================================

void matvec_f32(const float* input, const float* weight, float* output,
                int32_t in_dim, int32_t out_dim) {
    // output[out] = sum(input[in] * weight[out * in_dim + in])
    for (int32_t o = 0; o < out_dim; ++o) {
        float sum = 0.0f;
        const float* w_row = weight + static_cast<size_t>(o) * in_dim;
        for (int32_t i = 0; i < in_dim; ++i) {
            sum += input[i] * w_row[i];
        }
        output[o] = sum;
    }
}

// ============================================================
// Activation functions
// ============================================================

float gelu_approx(float x) {
    // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c1 = 0.7978845608f;   // sqrt(2/pi)
    const float c2 = 0.044715f;
    float x3 = x * x * x;
    float inner = c1 * (x + c2 * x3);
    return x * 0.5f * (1.0f + std::tanh(inner));
}

// ============================================================
// Normalization
// ============================================================

void apply_layer_norm(const float* input, float* output,
                      const float* gamma, const float* beta,
                      int32_t dim) {
    // Compute mean
    float mean = 0.0f;
    for (int32_t i = 0; i < dim; ++i) mean += input[i];
    mean /= dim;

    // Compute variance
    float var = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
    const float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int32_t i = 0; i < dim; ++i) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ============================================================
// Softmax
// ============================================================

void softmax(float* data, int32_t length) {
    if (length <= 0) return;
    float max_val = *std::max_element(data, data + length);
    float sum = 0.0f;
    for (int32_t i = 0; i < length; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    float inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < length; ++i) {
        data[i] *= inv_sum;
    }
}

// ============================================================
// Patch embedding computation
// ============================================================

void compute_patch_embeddings(DesireeAIVisionCtx* ctx,
                              const float* pixels,
                              int32_t num_patches,
                              std::vector<float>& out) {
    if (!ctx || !ctx->patch_embed_weight || ctx->patch_embed_weight->empty()) return;

    const int32_t embed_dim = ctx->config.embedding_dim;
    const int32_t patch_size = ctx->config.patch_size;
    const int32_t channels = 3;  // RGB

    out.resize(static_cast<size_t>(num_patches + 1) * embed_dim);

    // Copy class token to position 0
    std::memcpy(out.data(), ctx->class_token->ptr(), embed_dim * sizeof(float));

    // For each patch, compute dot product with patch embedding weights
    const float* w = ctx->patch_embed_weight->ptr();
    for (int32_t p = 0; p < num_patches; ++p) {
        // Simple linear projection: out[p] = pixels[p] . w^T
        // In full implementation, this would be a convolution
        float* out_row = out.data() + (p + 1) * embed_dim;
        for (int32_t d = 0; d < embed_dim; ++d) {
            float sum = 0.0f;
            for (int32_t c = 0; c < channels; ++c) {
                for (int32_t pp = 0; pp < patch_size * patch_size; ++pp) {
                    // Simplified: direct weight lookup
                    sum += pixels[p * channels * patch_size * patch_size +
                                   c * patch_size * patch_size + pp] *
                           w[d * channels * patch_size * patch_size +
                             c * patch_size * patch_size + pp];
                }
            }
            out_row[d] = sum;
        }
    }

    // Add position embeddings
    const float* pos_w = ctx->pos_embed_weight->ptr();
    for (int32_t i = 0; i < num_patches + 1; ++i) {
        float* out_row = out.data() + i * embed_dim;
        const float* pos_row = pos_w + i * embed_dim;
        for (int32_t d = 0; d < embed_dim; ++d) {
            out_row[d] += pos_row[d];
        }
    }
}

// ============================================================
// Transformer encoder
// ============================================================

void run_transformer_encoder(DesireeAIVisionCtx* ctx,
                             std::vector<float>& embeddings,
                             int32_t seq_len) {
    if (!ctx || ctx->layers.empty()) return;

    const int32_t embed_dim = ctx->config.embedding_dim;
    const int32_t num_heads = ctx->config.num_heads;
    const int32_t head_dim = embed_dim / num_heads;

    std::vector<float> norm_buf(embed_dim);
    std::vector<float> q_buf(embed_dim);
    std::vector<float> k_buf(embed_dim);
    std::vector<float> v_buf(embed_dim);
    std::vector<float> attn_out(embed_dim);
    std::vector<float> mlp_out(embed_dim * 2);

    for (auto& layer : ctx->layers) {
        // Layer norm 1
        for (int32_t i = 0; i < seq_len; ++i) {
            float* x = embeddings.data() + i * embed_dim;
            apply_layer_norm(x, norm_buf.data(),
                            layer.ln1_gamma->ptr(), layer.ln1_beta->ptr(),
                            embed_dim);
            std::memcpy(x, norm_buf.data(), embed_dim * sizeof(float));
        }

        // Self-attention (simplified: single head for now)
        for (int32_t i = 0; i < seq_len; ++i) {
            float* x = embeddings.data() + i * embed_dim;
            matvec_f32(x, layer.q_proj->ptr(), q_buf.data(), embed_dim, embed_dim);
            matvec_f32(x, layer.k_proj->ptr(), k_buf.data(), embed_dim, embed_dim);
            matvec_f32(x, layer.v_proj->ptr(), v_buf.data(), embed_dim, embed_dim);
        }

        // Attention scores (simplified)
        std::vector<float> scores(seq_len * seq_len);
        for (int32_t i = 0; i < seq_len; ++i) {
            for (int32_t j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int32_t d = 0; d < embed_dim; ++d) {
                    dot += embeddings[i * embed_dim + d] * embeddings[j * embed_dim + d];
                }
                scores[i * seq_len + j] = dot / std::sqrt(static_cast<float>(embed_dim));
            }
            softmax(scores.data() + i * seq_len, seq_len);
        }

        // Apply attention to values (simplified)
        for (int32_t i = 0; i < seq_len; ++i) {
            float* out = attn_out.data();
            std::memset(out, 0, embed_dim * sizeof(float));
            for (int32_t j = 0; j < seq_len; ++j) {
                float weight = scores[i * seq_len + j];
                for (int32_t d = 0; d < embed_dim; ++d) {
                    out[d] += weight * embeddings[j * embed_dim + d];
                }
            }
            // Output projection
            matvec_f32(out, layer.o_proj->ptr(),
                      embeddings.data() + i * embed_dim,
                      embed_dim, embed_dim);
        }

        // Residual connection
        // (already in embeddings)

        // Layer norm 2
        for (int32_t i = 0; i < seq_len; ++i) {
            float* x = embeddings.data() + i * embed_dim;
            apply_layer_norm(x, norm_buf.data(),
                            layer.ln2_gamma->ptr(), layer.ln2_beta->ptr(),
                            embed_dim);
            std::memcpy(x, norm_buf.data(), embed_dim * sizeof(float));
        }

        // MLP (simplified)
        for (int32_t i = 0; i < seq_len; ++i) {
            float* x = embeddings.data() + i * embed_dim;
            // fc1: embed_dim -> intermediate_size (2 * embed_dim for GELU)
            matvec_f32(x, layer.fc1->ptr(), mlp_out.data(),
                      embed_dim, embed_dim * 2);
            // GELU activation
            for (int32_t d = 0; d < embed_dim * 2; ++d) {
                mlp_out[d] = gelu_approx(mlp_out[d]);
            }
            // fc2: intermediate_size -> embed_dim
            matvec_f32(mlp_out.data(), layer.fc2->ptr(), x,
                      embed_dim * 2, embed_dim);
        }
    }
}

// ============================================================
// Image preprocessing
// ============================================================

bool vision_preprocess_image(const DesireeAIImage& input,
                             int32_t target_size,
                             std::vector<float>& out_pixels) {
    if (!input.data || input.width == 0 || input.height == 0) return false;

    // Convert to RGB if needed
    PixelBuffer rgb_buf;
    rgb_buf.width = input.width;
    rgb_buf.height = input.height;
    rgb_buf.channels = input.channels;
    rgb_buf.data.assign(input.data,
                       input.data + input.width * input.height * input.channels);

    if (input.channels == 1) {
        rgb_buf = grayscale_to_rgb(rgb_buf);
    } else if (input.channels == 4) {
        rgb_buf = rgba_to_rgb(rgb_buf);
    }

    // Center-crop and resize
    PixelBuffer resized = crop_and_resize(rgb_buf, target_size, target_size);
    if (resized.empty()) return false;

    // Convert to CHW format and normalize to [0, 1]
    const int32_t channels = 3;
    const size_t pixel_count = static_cast<size_t>(target_size) * target_size;
    out_pixels.resize(channels * pixel_count);

    for (int32_t c = 0; c < channels; ++c) {
        for (int32_t h = 0; h < target_size; ++h) {
            for (int32_t w = 0; w < target_size; ++w) {
                size_t src_idx = (h * target_size + w) * channels + c;
                size_t dst_idx = c * pixel_count + h * target_size + w;
                out_pixels[dst_idx] = resized.data[src_idx] / 255.0f;
            }
        }
    }

    return true;
}

// ============================================================
// Context management (stubs - real implementation loads from GGUF)
// ============================================================

DesireeAIVisionCtx* vision_create_context(const char* model_path,
                                          desireeia_log_cb log_cb,
                                          void* log_user) {
    (void)model_path;
    (void)log_cb;
    (void)log_user;

    // TODO: Load CLIP vision weights from GGUF file
    // For now, return a context with default config
    auto* ctx = new DesireeAIVisionCtx;
    ctx->config.embedding_dim = 768;
    ctx->config.patch_size = 14;
    ctx->config.image_size = 224;
    ctx->config.num_heads = 12;
    ctx->config.num_layers = 12;
    ctx->config.projection_dim = 512;
    ctx->config.has_encoder = 0;  // No weights loaded yet
    ctx->initialized = false;
    return ctx;
}

void vision_destroy_context(DesireeAIVisionCtx* ctx) {
    delete ctx;
}

bool vision_get_config(const DesireeAIVisionCtx* ctx, DesireeAIVisionConfig& out) {
    if (!ctx) return false;
    out = ctx->config;
    return true;
}

bool vision_encode(DesireeAIVisionCtx* ctx,
                   const DesireeAIImage& image,
                   std::vector<float>& out_embd,
                   uint32_t& out_dim) {
    if (!ctx || !ctx->initialized) return false;

    // Preprocess image
    std::vector<float> pixels;
    if (!vision_preprocess_image(image, ctx->config.image_size, pixels)) {
        return false;
    }

    // Compute patch embeddings
    const int32_t patch_count = (ctx->config.image_size / ctx->config.patch_size) *
                                (ctx->config.image_size / ctx->config.patch_size);
    std::vector<float> embeddings;
    compute_patch_embeddings(ctx, pixels.data(), patch_count, embeddings);

    // Run transformer encoder
    run_transformer_encoder(ctx, embeddings, patch_count + 1);

    // Extract class token embedding (position 0) as output
    out_dim = static_cast<uint32_t>(ctx->config.projection_dim);
    out_embd.resize(out_dim);

    // Apply projection if available
    if (ctx->visual_projection && !ctx->visual_projection->empty()) {
        matvec_f32(embeddings.data(), ctx->visual_projection->ptr(),
                  out_embd.data(), ctx->config.embedding_dim, out_dim);
    } else {
        // Just copy the class token
        std::memcpy(out_embd.data(), embeddings.data(),
                   std::min(out_embd.size() * sizeof(float),
                           embeddings.size() * sizeof(float)));
    }

    return true;
}

}  // namespace vision
}  // namespace desireeia
