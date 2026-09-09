#include "vision_gguf.h"
#include "../core/engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace desireeia {
namespace vision {

// Helper: get integer metadata with default fallback.
static uint32_t get_meta_u32(ModelReader& reader, const std::string& key, uint32_t def) {
    uint32_t v = def;
    reader.meta_u32(key, v);
    return v;
}

// Helper: get float metadata with default fallback.
static float get_meta_f32(ModelReader& reader, const std::string& key, float def) {
    float v = def;
    reader.meta_f32(key, v);
    return v;
}

// Helper: get string metadata.
static bool get_meta_str(ModelReader& reader, const std::string& key, std::string& out) {
    return reader.meta_str(key, out);
}

// Read a tensor from the model reader and store it in a Tensor2D.
static bool load_tensor_2d(ModelReader& reader, const std::string& name,
                           std::unique_ptr<Tensor2D>& tensor) {
    std::vector<float> data;
    if (!reader.read_tensor(name, data)) return false;
    if (data.empty()) return false;

    tensor = std::make_unique<Tensor2D>();
    tensor->data = std::move(data);

    // GGUF stores tensors as [dim0, dim1] in row-major order.
    // For vision weights, the tensor info in GGUF tells us the shape,
    // but read_tensor gives us a flat vector. We need to figure out dims.
    // The convention for GGUF vision tensors:
    //   - For 2D weight matrices: [out_features, in_features]
    //   - The flat array has out_features rows, each with in_features elements.
    // We store the total count and let the caller know the intended shape.
    tensor->rows = 0;
    tensor->cols = 0;
    return true;
}

// Read a tensor and assign known dimensions.
static bool load_tensor_2d(ModelReader& reader, const std::string& name,
                           std::unique_ptr<Tensor2D>& tensor,
                           int32_t rows, int32_t cols) {
    std::vector<float> data;
    if (!reader.read_tensor(name, data)) return false;
    if (static_cast<int32_t>(data.size()) != rows * cols) {
        // Size mismatch: try to continue with whatever we got
    }

    tensor = std::make_unique<Tensor2D>();
    tensor->data = std::move(data);
    tensor->rows = rows;
    tensor->cols = cols;
    return true;
}

// Read a 1D tensor (vector).
static bool load_tensor_1d(ModelReader& reader, const std::string& name,
                           std::unique_ptr<Tensor1D>& tensor, int32_t dim) {
    std::vector<float> data;
    if (!reader.read_tensor(name, data)) return false;
    if (data.empty()) return false;

    tensor = std::make_unique<Tensor1D>();
    tensor->data = std::move(data);
    tensor->dim = dim > 0 ? dim : static_cast<int32_t>(tensor->data.size());
    return true;
}

// Read a 4D tensor (patch embedding: [out_channels, in_channels, kH, kW]).
static bool load_tensor_4d(ModelReader& reader, const std::string& name,
                           std::unique_ptr<Tensor4D>& tensor,
                           int32_t d0, int32_t d1, int32_t d2, int32_t d3) {
    std::vector<float> data;
    if (!reader.read_tensor(name, data)) return false;
    if (data.empty()) return false;

    tensor = std::make_unique<Tensor4D>();
    tensor->data = std::move(data);
    tensor->d0 = d0;
    tensor->d1 = d1;
    tensor->d2 = d2;
    tensor->d3 = d3;
    return true;
}

// Read image normalization mean/std from GGUF metadata.
static void load_image_normalization(ModelReader& reader, VisionGGUFContext& ctx) {
    // clip.vision.image_mean is stored as an array of 3 floats.
    // GGUF arrays are type 9, but meta_f32 only reads scalar floats.
    // For now, use defaults. Real GGUF files store these as arrays.
    // TODO: Read array metadata properly.
    ctx.image_mean[0] = get_meta_f32(reader, "clip.vision.image_mean.0", 0.48145466f);
    ctx.image_mean[1] = get_meta_f32(reader, "clip.vision.image_mean.1", 0.4578275f);
    ctx.image_mean[2] = get_meta_f32(reader, "clip.vision.image_mean.2", 0.40821073f);
    ctx.image_std[0] = get_meta_f32(reader, "clip.vision.image_std.0", 0.26862954f);
    ctx.image_std[1] = get_meta_f32(reader, "clip.vision.image_std.1", 0.26130258f);
    ctx.image_std[2] = get_meta_f32(reader, "clip.vision.image_std.2", 0.27577711f);
}

// Detect and load multimodal projector weights.
// Tries multiple naming conventions used by different GGUF converters.
// Dimension conventions (all row-major, rows are the output dimension):
//   LLaVA 1.5 (llama.cpp convert_image_encoder_to_gguf.py):
//     mlp2x_gelu: mm.0.weight [H, D_img], mm.2.weight [D_txt, H], mm.1/mm.3 bias
//                 (H = projector intermediate, D_txt = text embedding dim)
//     linear:     mm.0.weight [D_txt, D_img], optional mm.1 bias
//   llava GGUF:   mm.image_projection.weight [D_txt, D_img]
//   MiniCPM-V:    v.mm.weight [D_txt, D_img], v.mm.bias [D_txt]
// The intermediate and output sizes are derived from the tensor byte counts
// vs the vision embedding dim — never assumed.
static bool load_mm_projector(ModelReader& reader, VisionGGUFContext& ctx) {
    // Try to detect projector type from metadata
    std::string proj_type;
    if (get_meta_str(reader, "llava.projector_type", proj_type)) {
        if (proj_type == "mlp2x_gelu") {
            ctx.mm_projector_type = 1;
        } else if (proj_type == "mlp_gelu") {
            ctx.mm_projector_type = 2;
        } else {
            ctx.mm_projector_type = 0;  // linear
        }
    }

    const size_t d_img = (size_t) std::max<int32_t>(ctx.config.embedding_dim, 1);

    // Convention 2: mm.0.weight / mm.2.weight (MLP2x with bias at mm.1/mm.3)
    {
        std::vector<float> w0;
        if (reader.read_tensor("mm.0.weight", w0) && !w0.empty() && w0.size() % d_img == 0) {
            std::vector<float> w1;
            bool has_w1 = reader.read_tensor("mm.2.weight", w1) && !w1.empty();

            if (!has_w1) {
                // Single linear: [D_txt, D_img]
                ctx.mm_hidden_size = static_cast<int32_t>(d_img);
                ctx.mm_output_size = static_cast<int32_t>(w0.size() / d_img);
                ctx.mm_proj_0 = std::make_unique<Tensor2D>();
                ctx.mm_proj_0->data = std::move(w0);
                ctx.mm_proj_0->rows = ctx.mm_output_size;
                ctx.mm_proj_0->cols = static_cast<int32_t>(d_img);

                std::vector<float> b0;
                if (reader.read_tensor("mm.1.bias", b0) && (size_t) b0.size() == ctx.mm_proj_0->rows) {
                    ctx.mm_proj_0_bias = std::make_unique<Tensor1D>();
                    ctx.mm_proj_0_bias->data = std::move(b0);
                    ctx.mm_proj_0_bias->dim = ctx.mm_proj_0->rows;
                }
                return true;
            }

            // MLP2x: w0 [H, D_img], w1 [D_txt, H]
            const int32_t h = static_cast<int32_t>(w0.size() / d_img);
            if (h <= 0 || w1.size() % (size_t) h != 0) return false;
            const int32_t out = static_cast<int32_t>(w1.size() / (size_t) h);

            ctx.mm_projector_type = 1;  // MLP2x
            ctx.mm_hidden_size = h;
            ctx.mm_output_size = out;
            ctx.mm_proj_0 = std::make_unique<Tensor2D>();
            ctx.mm_proj_0->data = std::move(w0);
            ctx.mm_proj_0->rows = h;
            ctx.mm_proj_0->cols = static_cast<int32_t>(d_img);

            ctx.mm_proj_1 = std::make_unique<Tensor2D>();
            ctx.mm_proj_1->data = std::move(w1);
            ctx.mm_proj_1->rows = out;
            ctx.mm_proj_1->cols = h;

            std::vector<float> b0;
            if (reader.read_tensor("mm.1.bias", b0) && (size_t) b0.size() == (size_t) h) {
                ctx.mm_proj_0_bias = std::make_unique<Tensor1D>();
                ctx.mm_proj_0_bias->data = std::move(b0);
                ctx.mm_proj_0_bias->dim = h;
            }
            std::vector<float> b1;
            if (reader.read_tensor("mm.3.bias", b1) && (size_t) b1.size() == (size_t) out) {
                ctx.mm_proj_1_bias = std::make_unique<Tensor1D>();
                ctx.mm_proj_1_bias->data = std::move(b1);
                ctx.mm_proj_1_bias->dim = out;
            }
            return true;
        }
    }

    // Convention 1: mm.image_projection.weight (LLaVA 1.5+ single linear)
    {
        std::vector<float> w;
        if (reader.read_tensor("mm.image_projection.weight", w) && !w.empty() && w.size() % d_img == 0) {
            ctx.mm_hidden_size = static_cast<int32_t>(d_img);
            ctx.mm_output_size = static_cast<int32_t>(w.size() / d_img);
            ctx.mm_proj_0 = std::make_unique<Tensor2D>();
            ctx.mm_proj_0->data = std::move(w);
            ctx.mm_proj_0->rows = ctx.mm_output_size;
            ctx.mm_proj_0->cols = static_cast<int32_t>(d_img);
            return true;
        }
    }

    // Convention 3: v.mm.weight / v.mm.bias (MiniCPM-V style)
    {
        std::vector<float> w;
        if (reader.read_tensor("v.mm.weight", w) && !w.empty() && w.size() % d_img == 0) {
            ctx.mm_hidden_size = static_cast<int32_t>(d_img);
            ctx.mm_output_size = static_cast<int32_t>(w.size() / d_img);
            ctx.mm_proj_0 = std::make_unique<Tensor2D>();
            ctx.mm_proj_0->data = std::move(w);
            ctx.mm_proj_0->rows = ctx.mm_output_size;
            ctx.mm_proj_0->cols = static_cast<int32_t>(d_img);

            std::vector<float> b;
            if (reader.read_tensor("v.mm.bias", b) && (size_t) b.size() == ctx.mm_proj_0->rows) {
                ctx.mm_proj_0_bias = std::make_unique<Tensor1D>();
                ctx.mm_proj_0_bias->data = std::move(b);
                ctx.mm_proj_0_bias->dim = ctx.mm_output_size;
            }
            return true;
        }
    }

    return false;
}

// Load a single CLIP transformer layer's weights from GGUF.
static bool load_clip_layer(ModelReader& reader, VisionGGUFContext& ctx, int32_t idx) {
    auto& layer = ctx.layers[idx];
    const int32_t embed_dim = ctx.config.embedding_dim;

    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "v.blk.%d.", idx);

    // Layer norm 1
    if (!load_tensor_1d(reader, std::string(prefix) + "ln1.weight", layer.ln1_gamma, embed_dim))
        return false;
    if (!load_tensor_1d(reader, std::string(prefix) + "ln1.bias", layer.ln1_beta, embed_dim))
        return false;

    // Self-attention Q/K/V/O projections
    if (!load_tensor_2d(reader, std::string(prefix) + "attn.q.weight",
                        layer.q_proj, embed_dim, embed_dim))
        return false;
    if (!load_tensor_2d(reader, std::string(prefix) + "attn.k.weight",
                        layer.k_proj, embed_dim, embed_dim))
        return false;
    if (!load_tensor_2d(reader, std::string(prefix) + "attn.v.weight",
                        layer.v_proj, embed_dim, embed_dim))
        return false;
    if (!load_tensor_2d(reader, std::string(prefix) + "attn.output.weight",
                        layer.o_proj, embed_dim, embed_dim))
        return false;

    // Layer norm 2
    if (!load_tensor_1d(reader, std::string(prefix) + "ln2.weight", layer.ln2_gamma, embed_dim))
        return false;
    if (!load_tensor_1d(reader, std::string(prefix) + "ln2.bias", layer.ln2_beta, embed_dim))
        return false;

    // MLP (feed-forward)
    const int32_t ff_dim = get_meta_u32(reader, "clip.vision.feed_forward_length", embed_dim * 4);
    if (!load_tensor_2d(reader, std::string(prefix) + "mlp.fc1.weight",
                        layer.fc1, ff_dim, embed_dim))
        return false;
    if (!load_tensor_2d(reader, std::string(prefix) + "mlp.fc2.weight",
                        layer.fc2, embed_dim, ff_dim))
        return false;

    return true;
}

// ============================================================
// Matrix-vector multiply: out = input * weight^T + bias
// weight is [out_dim, in_dim], row-major
// ============================================================
static void matvec_bias(const float* input, const float* weight, const float* bias,
                        float* output, int32_t in_dim, int32_t out_dim) {
    for (int32_t o = 0; o < out_dim; ++o) {
        float sum = bias ? bias[o] : 0.0f;
        const float* w_row = weight + static_cast<size_t>(o) * in_dim;
        for (int32_t i = 0; i < in_dim; ++i) {
            sum += input[i] * w_row[i];
        }
        output[o] = sum;
    }
}

// GELU approximation and softmax come from vision_clip.cpp (declared in
// vision_clip.h) — same math, no duplicate definitions here.

// Layer normalization
static void layer_norm(const float* input, float* output,
                       const float* gamma, const float* beta, int32_t dim) {
    float mean = 0.0f;
    for (int32_t i = 0; i < dim; ++i) mean += input[i];
    mean /= dim;

    float var = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    const float eps = 1e-5f;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int32_t i = 0; i < dim; ++i) {
        output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ============================================================
// Public API
// ============================================================

bool vision_load_from_gguf(VisionGGUFContext& ctx, ModelReader& reader) {
    // Check if this GGUF has vision encoder metadata
    uint32_t has_clip = 0;
    if (!reader.meta_u32("clip.vision.has_encoder", has_clip) && !has_clip) {
        // Also check by probing for vision tensors
        // (some GGUF files don't set has_encoder but have vision weights)
    }

    // Read vision encoder config from clip.vision.* metadata
    ctx.config.embedding_dim = static_cast<int32_t>(
        get_meta_u32(reader, "clip.vision.embedding_length", 768));
    ctx.config.patch_size = static_cast<int32_t>(
        get_meta_u32(reader, "clip.vision.patch_size", 14));
    ctx.config.image_size = static_cast<int32_t>(
        get_meta_u32(reader, "clip.vision.image_size", 224));
    ctx.config.num_heads = static_cast<int32_t>(
        get_meta_u32(reader, "clip.vision.attention.head_count", 12));
    ctx.config.num_layers = static_cast<int32_t>(
        get_meta_u32(reader, "clip.vision.block_count", 12));
    ctx.config.projection_dim = static_cast<int32_t>(
        get_meta_u32(reader, "clip.vision.projection_dim", 0));

    if (ctx.config.embedding_dim <= 0 || ctx.config.patch_size <= 0 ||
        ctx.config.image_size <= 0 || ctx.config.num_layers <= 0) {
        return false;  // Not a valid vision model
    }

    // Compute derived dimensions
    const int32_t embed_dim = ctx.config.embedding_dim;
    const int32_t num_patches = (ctx.config.image_size / ctx.config.patch_size) *
                                 (ctx.config.image_size / ctx.config.patch_size);
    const int32_t num_positions = num_patches + 1;  // +1 for class token

    // Load class token
    if (!load_tensor_1d(reader, "v.cls_emb.weight", ctx.class_token, embed_dim)) {
        // Some models use a different name
        if (!load_tensor_1d(reader, "v.class_token", ctx.class_token, embed_dim)) {
            return false;
        }
    }

    // Load position embeddings
    if (!load_tensor_2d(reader, "v.position_embd.weight", ctx.pos_embed_weight,
                        num_positions, embed_dim)) {
        return false;
    }

    // Load pre-norm layer norm
    if (!load_tensor_1d(reader, "v.pre_ln.weight", ctx.pre_ln_gamma, embed_dim)) {
        // Some models don't have pre-norm; initialize to ones/zeros
        ctx.pre_ln_gamma = std::make_unique<Tensor1D>();
        ctx.pre_ln_gamma->data.resize(embed_dim, 1.0f);
        ctx.pre_ln_gamma->dim = embed_dim;
    }
    if (!load_tensor_1d(reader, "v.pre_ln.bias", ctx.pre_ln_beta, embed_dim)) {
        ctx.pre_ln_beta = std::make_unique<Tensor1D>();
        ctx.pre_ln_beta->data.resize(embed_dim, 0.0f);
        ctx.pre_ln_beta->dim = embed_dim;
    }

    // Allocate transformer layers
    ctx.layers.resize(ctx.config.num_layers);

    // Load transformer layer weights
    for (int32_t i = 0; i < ctx.config.num_layers; ++i) {
        if (!load_clip_layer(reader, ctx, i)) {
            return false;
        }
    }

    // Load post-norm layer norm
    if (!load_tensor_1d(reader, "v.post_ln.weight", ctx.post_ln_gamma, embed_dim)) {
        ctx.post_ln_gamma = std::make_unique<Tensor1D>();
        ctx.post_ln_gamma->data.resize(embed_dim, 1.0f);
        ctx.post_ln_gamma->dim = embed_dim;
    }
    if (!load_tensor_1d(reader, "v.post_ln.bias", ctx.post_ln_beta, embed_dim)) {
        ctx.post_ln_beta = std::make_unique<Tensor1D>();
        ctx.post_ln_beta->data.resize(embed_dim, 0.0f);
        ctx.post_ln_beta->dim = embed_dim;
    }

    // Load image normalization
    load_image_normalization(reader, ctx);

    // Load multimodal projector
    if (!load_mm_projector(reader, ctx)) {
        // No projector found; vision encoder alone may still be useful
        // for embedding extraction (but not for chat integration)
    }

    // Find image token ID in vocabulary
    std::string img_token;
    if (get_meta_str(reader, "clip.vision.image_token", img_token)) {
        // Store for later token lookup
        // The actual token ID lookup needs the vocabulary, which is
        // managed by the caller (ctx.cpp).
    }

    ctx.has_vision = true;
    ctx.initialized = true;
    return true;
}

bool vision_preprocess_image_normalized(const VisionGGUFContext& ctx,
                                        const DesireeAIImage& input,
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

    // Convert to CHW format, normalize to [0,1], then apply mean/std normalization
    const int32_t channels = 3;
    const size_t pixel_count = static_cast<size_t>(target_size) * target_size;
    out_pixels.resize(channels * pixel_count);

    for (int32_t c = 0; c < channels; ++c) {
        for (int32_t h = 0; h < target_size; ++h) {
            for (int32_t w = 0; w < target_size; ++w) {
                size_t src_idx = (h * target_size + w) * channels + c;
                size_t dst_idx = c * pixel_count + h * target_size + w;
                float val = resized.data[src_idx] / 255.0f;
                // Normalize: (x - mean) / std
                val = (val - ctx.image_mean[c]) / ctx.image_std[c];
                out_pixels[dst_idx] = val;
            }
        }
    }

    return true;
}

bool vision_encode_image(const VisionGGUFContext& ctx,
                         const DesireeAIImage& image,
                         std::vector<float>& out_embd) {
    if (!ctx.initialized || !ctx.has_vision) return false;
    if (!image.data || image.width == 0 || image.height == 0) return false;

    const int32_t embed_dim = ctx.config.embedding_dim;
    const int32_t num_patches = (ctx.config.image_size / ctx.config.patch_size) *
                                 (ctx.config.image_size / ctx.config.patch_size);
    const int32_t seq_len = num_patches + 1;  // +1 for class token

    // Preprocess image
    std::vector<float> pixels;
    if (!vision_preprocess_image_normalized(ctx, image, ctx.config.image_size, pixels)) {
        return false;
    }

    // Step 1: Patch embedding
    std::vector<float> embeddings(seq_len * embed_dim);

    // Copy class token to position 0
    if (ctx.class_token) {
        std::memcpy(embeddings.data(), ctx.class_token->data.data(),
                   embed_dim * sizeof(float));
    }

    // Compute patch embeddings: linear projection of flattened patches
    if (ctx.patch_embed_weight) {
        const int32_t patch_size = ctx.config.patch_size;
        const int32_t channels = 3;
        const int32_t patch_dim = channels * patch_size * patch_size;
        const float* w = ctx.patch_embed_weight->data.data();

        for (int32_t p = 0; p < num_patches; ++p) {
            int32_t patch_y = (p / (ctx.config.image_size / patch_size)) * patch_size;
            int32_t patch_x = (p % (ctx.config.image_size / patch_size)) * patch_size;

            float* out_row = embeddings.data() + (p + 1) * embed_dim;
            for (int32_t d = 0; d < embed_dim; ++d) {
                float sum = 0.0f;
                const float* w_row = w + d * patch_dim;
                for (int32_t c = 0; c < channels; ++c) {
                    for (int32_t py = 0; py < patch_size; ++py) {
                        for (int32_t px = 0; px < patch_size; ++px) {
                            size_t pixel_idx = c * num_patches * patch_size * patch_size +
                                              (patch_y + py) * ctx.config.image_size * patch_size +
                                              (patch_x + px);
                            // CHW format: channels first
                            size_t src_idx = static_cast<size_t>(c) *
                                            (ctx.config.image_size / patch_size * patch_size) *
                                            (ctx.config.image_size / patch_size * patch_size) +
                                            (patch_y + py) * ctx.config.image_size + (patch_x + px);
                            // Simpler: just use the flat index from our preprocessed pixels
                            // pixels is CHW: [c * H * W + h * W + w]
                            size_t flat_idx = static_cast<size_t>(c) * ctx.config.image_size * ctx.config.image_size +
                                             (patch_y + py) * ctx.config.image_size + (patch_x + px);
                            if (flat_idx < pixels.size()) {
                                sum += pixels[flat_idx] * w_row[c * patch_size * patch_size + py * patch_size + px];
                            }
                        }
                    }
                }
                out_row[d] = sum;
            }
        }
    }

    // Step 2: Add position embeddings
    if (ctx.pos_embed_weight) {
        for (int32_t i = 0; i < seq_len; ++i) {
            float* out_row = embeddings.data() + i * embed_dim;
            const float* pos_row = ctx.pos_embed_weight->data.data() + i * embed_dim;
            for (int32_t d = 0; d < embed_dim; ++d) {
                out_row[d] += pos_row[d];
            }
        }
    }

    // Step 3: Pre-norm
    if (ctx.pre_ln_gamma && ctx.pre_ln_beta) {
        std::vector<float> norm_buf(embed_dim);
        for (int32_t i = 0; i < seq_len; ++i) {
            float* x = embeddings.data() + i * embed_dim;
            layer_norm(x, norm_buf.data(),
                      ctx.pre_ln_gamma->data.data(), ctx.pre_ln_beta->data.data(),
                      embed_dim);
            std::memcpy(x, norm_buf.data(), embed_dim * sizeof(float));
        }
    }

    // Step 4: Transformer layers (pre-LN residual blocks, multi-head)
    std::vector<float> norm_buf(embed_dim);
    std::vector<float> q_buf(embed_dim);
    std::vector<float> k_buf(embed_dim);
    std::vector<float> v_buf(embed_dim);
    std::vector<float> attn_buf(embed_dim);   // raw attention output before o_proj
    std::vector<float> o_out(embed_dim);      // after o_proj, added to residual
    std::vector<float> fc1_out(embed_dim * 4);

    const int32_t num_heads = std::max<int32_t>(ctx.config.num_heads, 1);
    const int32_t head_dim = embed_dim / num_heads;
    const float head_scale = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> scores(seq_len);

    for (const auto& layer : ctx.layers) {
        // Multi-head Q/K/V for every position, computed on ln1 output.
        std::vector<std::vector<float>> all_q(seq_len, std::vector<float>(embed_dim));
        std::vector<std::vector<float>> all_k(seq_len, std::vector<float>(embed_dim));
        std::vector<std::vector<float>> all_v(seq_len, std::vector<float>(embed_dim));

        for (int32_t i = 0; i < seq_len; ++i) {
            const float* x = embeddings.data() + i * embed_dim;
            layer_norm(x, norm_buf.data(),
                       layer.ln1_gamma->data.data(), layer.ln1_beta->data.data(),
                       embed_dim);
            matvec_bias(norm_buf.data(), layer.q_proj->data.data(), nullptr,
                       all_q[i].data(), embed_dim, embed_dim);
            matvec_bias(norm_buf.data(), layer.k_proj->data.data(), nullptr,
                       all_k[i].data(), embed_dim, embed_dim);
            matvec_bias(norm_buf.data(), layer.v_proj->data.data(), nullptr,
                       all_v[i].data(), embed_dim, embed_dim);
        }

        // Attention, head by head (scores scaled by sqrt(head_dim), as CLIP
        // does), then o_proj; add the result residually to the block input.
        for (int32_t i = 0; i < seq_len; ++i) {
            std::fill(attn_buf.begin(), attn_buf.end(), 0.0f);
            for (int32_t h = 0; h < num_heads; ++h) {
                const int32_t hd = h * head_dim;
                for (int32_t j = 0; j < seq_len; ++j) {
                    float dot = 0.0f;
                    for (int32_t d = 0; d < head_dim; ++d) {
                        dot += all_q[i][hd + d] * all_k[j][hd + d];
                    }
                    scores[j] = dot * head_scale;
                }
                softmax(scores.data(), seq_len);
                for (int32_t j = 0; j < seq_len; ++j) {
                    const float w = scores[j];
                    for (int32_t d = 0; d < head_dim; ++d) {
                        attn_buf[hd + d] += w * all_v[j][hd + d];
                    }
                }
            }
            matvec_bias(attn_buf.data(), layer.o_proj->data.data(), nullptr,
                       o_out.data(), embed_dim, embed_dim);
            float* x = embeddings.data() + i * embed_dim;
            for (int32_t d = 0; d < embed_dim; ++d) x[d] += o_out[d];
        }

        // MLP block: x = x + fc2(gelu(fc1(ln2(x))))
        const int32_t ff_dim = layer.fc1->rows;
        if ((size_t) (ff_dim * embed_dim) > fc1_out.size()) {
            fc1_out.resize((size_t) ff_dim * embed_dim);
        }
        std::vector<float> fc2_out(embed_dim);
        for (int32_t i = 0; i < seq_len; ++i) {
            float* x = embeddings.data() + i * embed_dim;
            layer_norm(x, norm_buf.data(),
                       layer.ln2_gamma->data.data(), layer.ln2_beta->data.data(),
                       embed_dim);
            matvec_bias(norm_buf.data(), layer.fc1->data.data(), nullptr,
                       fc1_out.data(), embed_dim, ff_dim);
            for (int32_t d = 0; d < ff_dim; ++d) fc1_out[d] = gelu_approx(fc1_out[d]);
            matvec_bias(fc1_out.data(), layer.fc2->data.data(), nullptr,
                       fc2_out.data(), ff_dim, embed_dim);
            for (int32_t d = 0; d < embed_dim; ++d) x[d] += fc2_out[d];
        }
    }

    // Step 5: Post-norm
    if (ctx.post_ln_gamma && ctx.post_ln_beta) {
        for (int32_t i = 0; i < seq_len; ++i) {
            float* x = embeddings.data() + i * embed_dim;
            layer_norm(x, norm_buf.data(),
                      ctx.post_ln_gamma->data.data(), ctx.post_ln_beta->data.data(),
                      embed_dim);
            std::memcpy(x, norm_buf.data(), embed_dim * sizeof(float));
        }
    }

    // Step 6: Multimodal projector
    const int32_t output_dim = vision_get_output_dim(ctx);

    if (ctx.mm_proj_0) {
        // Project all patch embeddings (excluding class token at position 0)
        out_embd.resize(num_patches * output_dim);
        for (int32_t p = 0; p < num_patches; ++p) {
            const float* patch_embd = embeddings.data() + (p + 1) * embed_dim;
            float* proj_out = out_embd.data() + p * output_dim;

            // First projection layer
            std::vector<float> hidden(ctx.mm_proj_0->rows);
            matvec_bias(patch_embd, ctx.mm_proj_0->data.data(),
                       ctx.mm_proj_0_bias ? ctx.mm_proj_0_bias->data.data() : nullptr,
                       hidden.data(), embed_dim, ctx.mm_proj_0->rows);

            if (ctx.mm_proj_1 && ctx.mm_projector_type == 1) {
                // MLP2x with GELU
                for (int32_t d = 0; d < ctx.mm_proj_0->rows; ++d) {
                    hidden[d] = gelu_approx(hidden[d]);
                }
                matvec_bias(hidden.data(), ctx.mm_proj_1->data.data(),
                           ctx.mm_proj_1_bias ? ctx.mm_proj_1_bias->data.data() : nullptr,
                           proj_out, ctx.mm_proj_0->rows, output_dim);
            } else {
                // Linear projection
                std::memcpy(proj_out, hidden.data(), output_dim * sizeof(float));
            }
        }
    } else {
        // No projector: return raw CLIP embeddings (class token)
        out_embd.resize(embed_dim);
        std::memcpy(out_embd.data(), embeddings.data(), embed_dim * sizeof(float));
    }

    return true;
}

int32_t vision_get_token_count(const VisionGGUFContext& ctx) {
    if (!ctx.initialized) return 0;
    const int32_t num_patches = (ctx.config.image_size / ctx.config.patch_size) *
                                 (ctx.config.image_size / ctx.config.patch_size);
    return num_patches;  // One token per patch (class token not counted as injection token)
}

int32_t vision_get_output_dim(const VisionGGUFContext& ctx) {
    if (!ctx.initialized) return 0;
    if (ctx.mm_proj_1) {
        return ctx.mm_proj_1->rows;  // MLP2x: final projection output
    }
    if (ctx.mm_proj_0) {
        return ctx.mm_proj_0->rows;  // Single linear: output rows
    }
    return ctx.config.embedding_dim;  // Raw CLIP embedding dimension
}

}  // namespace vision
}  // namespace desireeia
