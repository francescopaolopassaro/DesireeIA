#ifndef DESIREEIA_VISION_CLIP_H
#define DESIREEIA_VISION_CLIP_H

#include "desireeia/abi.h"
#include "vision_image.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace desireeia {
namespace vision {

// Internal tensor types for vision encoder weights. Defined here (not in the
// .cpp) because VisionGGUFContext / DesireeAIVisionCtx own them through
// std::unique_ptr and every TU that instantiates those structs (ctx.cpp,
// abi.cpp, ...) must see the complete type for unique_ptr's destructor.
struct Tensor1D {
    std::vector<float> data;
    int32_t dim = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct Tensor2D {
    std::vector<float> data;
    int32_t rows = 0;
    int32_t cols = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

struct Tensor4D {
    std::vector<float> data;
    int32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    bool empty() const { return data.empty(); }
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

}  // namespace vision
}  // namespace desireeia

// Vision encoder context: defined in global namespace to match abi.h forward decl.
struct DesireeAIVisionCtx {
    DesireeAIVisionConfig config;

    // Patch embedding weights: [embed_dim, channels, patch_size, patch_size]
    std::unique_ptr<desireeia::vision::Tensor4D> patch_embed_weight;
    // Class token: [embed_dim]
    std::unique_ptr<desireeia::vision::Tensor1D> class_token;
    // Position embeddings: [num_positions, embed_dim]
    std::unique_ptr<desireeia::vision::Tensor2D> pos_embed_weight;

    // Pre-norm layer norm: gamma [embed_dim], beta [embed_dim]
    std::unique_ptr<desireeia::vision::Tensor1D> pre_ln_gamma;
    std::unique_ptr<desireeia::vision::Tensor1D> pre_ln_beta;

    // Transformer layers: each has self-attention and MLP
    struct TransformerLayer {
        // Layer norm 1: gamma, beta
        std::unique_ptr<desireeia::vision::Tensor1D> ln1_gamma;
        std::unique_ptr<desireeia::vision::Tensor1D> ln1_beta;
        // Self-attention: Q, K, V projections + output projection
        std::unique_ptr<desireeia::vision::Tensor2D> q_proj;
        std::unique_ptr<desireeia::vision::Tensor2D> k_proj;
        std::unique_ptr<desireeia::vision::Tensor2D> v_proj;
        std::unique_ptr<desireeia::vision::Tensor2D> o_proj;
        // Layer norm 2: gamma, beta
        std::unique_ptr<desireeia::vision::Tensor1D> ln2_gamma;
        std::unique_ptr<desireeia::vision::Tensor1D> ln2_beta;
        // MLP: fc1, fc2
        std::unique_ptr<desireeia::vision::Tensor2D> fc1;
        std::unique_ptr<desireeia::vision::Tensor2D> fc2;
    };
    std::vector<TransformerLayer> layers;

    // Post-norm: gamma, beta
    std::unique_ptr<desireeia::vision::Tensor1D> post_ln_gamma;
    std::unique_ptr<desireeia::vision::Tensor1D> post_ln_beta;

    // Projection: [projection_dim, embed_dim]
    std::unique_ptr<desireeia::vision::Tensor2D> visual_projection;

    // Scratch buffers for intermediate computations
    std::vector<float> scratch_embed;
    std::vector<float> scratch_hidden;
    std::vector<float> scratch_output;

    bool initialized = false;
};

namespace desireeia {
namespace vision {

// Create a vision encoder context. Returns nullptr on failure.
// The model_path should point to a GGUF file containing CLIP vision weights.
::DesireeAIVisionCtx* vision_create_context(const char* model_path,
                                            desireeia_log_cb log_cb,
                                            void* log_user);

// Destroy a vision encoder context and free all resources.
void vision_destroy_context(::DesireeAIVisionCtx* ctx);

// Get the vision encoder configuration.
bool vision_get_config(const ::DesireeAIVisionCtx* ctx, DesireeAIVisionConfig& out);

// Encode an image into embedding vectors.
// Input: image (will be resized/preprocessed internally).
// Output: out_embd (flat array of floats), out_dim (embedding dimension).
bool vision_encode(::DesireeAIVisionCtx* ctx,
                   const DesireeAIImage& image,
                   std::vector<float>& out_embd,
                   uint32_t& out_dim);

// Preprocess image: resize, normalize, convert to RGB.
// Output: flat array of floats [0,1], shape [channels, height, width].
bool vision_preprocess_image(const DesireeAIImage& input,
                             int32_t target_size,
                             std::vector<float>& out_pixels);

// Internal: compute patch embeddings from preprocessed pixels.
void compute_patch_embeddings(::DesireeAIVisionCtx* ctx,
                              const float* pixels,
                              int32_t num_patches,
                              std::vector<float>& out);

// Internal: run transformer encoder on embeddings.
void run_transformer_encoder(::DesireeAIVisionCtx* ctx,
                             std::vector<float>& embeddings,
                             int32_t seq_len);

// Internal: apply layer normalization.
void apply_layer_norm(const float* input, float* output,
                      const float* gamma, const float* beta,
                      int32_t dim);

// Internal: matrix-vector multiply: out = input * weight^T + bias.
void matvec_f32(const float* input, const float* weight, float* output,
                int32_t in_dim, int32_t out_dim);

// Internal: GELU activation approximation.
float gelu_approx(float x);

// Internal: softmax over the last dimension.
void softmax(float* data, int32_t length);

}  // namespace vision
}  // namespace desireeia

#endif  // DESIREEIA_VISION_CLIP_H
