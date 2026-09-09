#ifndef DESIREEIA_VISION_GGUF_H
#define DESIREEIA_VISION_GGUF_H

#include "desireeia/abi.h"
#include "vision/vision_clip.h"
#include "vision/vision_image.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace desireeia {
class ModelReader;  // defined in core/engine.h; fwd-declared here at desireeia
                    // scope so unqualified `ModelReader` inside desireeia::vision
                    // resolves to the engine's reader, not a local shadow.

namespace vision {

// Vision encoder weights loaded from GGUF file.
// Stores dequantized float weights for CLIP/SigLIP vision encoder
// and the multimodal projector (MLP that maps vision embeddings
// into the text model's embedding space).
struct VisionGGUFContext {
    // Vision encoder config from clip.vision.* GGUF metadata
    DesireeAIVisionConfig config;

    // Multimodal projector config
    int32_t mm_hidden_size = 0;       // vision embedding dimension (hidden size of projector input)
    int32_t mm_output_size = 0;       // text model embedding dimension (projector output)
    int32_t mm_projector_type = 0;    // 0=linear, 1=mlp2x_gelu, 2=mlp_gelu

    // CLIP vision encoder weights
    std::unique_ptr<Tensor4D> patch_embed_weight;  // [embed_dim, channels, patch_size, patch_size]
    std::unique_ptr<Tensor1D> class_token;          // [embed_dim]
    std::unique_ptr<Tensor2D> pos_embed_weight;     // [num_positions, embed_dim]
    std::unique_ptr<Tensor1D> pre_ln_gamma;         // [embed_dim]
    std::unique_ptr<Tensor1D> pre_ln_beta;          // [embed_dim]

    struct TransformerLayer {
        std::unique_ptr<Tensor1D> ln1_gamma;
        std::unique_ptr<Tensor1D> ln1_beta;
        std::unique_ptr<Tensor2D> q_proj;
        std::unique_ptr<Tensor2D> k_proj;
        std::unique_ptr<Tensor2D> v_proj;
        std::unique_ptr<Tensor2D> o_proj;
        std::unique_ptr<Tensor1D> ln2_gamma;
        std::unique_ptr<Tensor1D> ln2_beta;
        std::unique_ptr<Tensor2D> fc1;
        std::unique_ptr<Tensor2D> fc2;
    };
    std::vector<TransformerLayer> layers;

    std::unique_ptr<Tensor1D> post_ln_gamma;
    std::unique_ptr<Tensor1D> post_ln_beta;

    // Multimodal projector weights (2-layer MLP with GELU, or single linear)
    std::unique_ptr<Tensor2D> mm_proj_0;  // first layer: [mm_hidden_size, mm_output_size]
    std::unique_ptr<Tensor2D> mm_proj_1;  // second layer (MLP2x): [mm_output_size, mm_hidden_size]
    std::unique_ptr<Tensor1D> mm_proj_0_bias;  // optional bias
    std::unique_ptr<Tensor1D> mm_proj_1_bias;  // optional bias

    // Image normalization constants (from clip.vision.image_mean/std, default [0.48145466, 0.4578275, 0.40821073])
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    // Special token for image embedding injection
    int32_t image_token_id = -1;  // token used as placeholder for image embeddings

    bool has_vision = false;
    bool initialized = false;

    // Scratch buffers
    std::vector<float> scratch_embed;
    std::vector<float> scratch_hidden;
    std::vector<float> scratch_output;
};

// Forward declaration of ModelReader lives at desireeia scope (above), so
// this function's `ModelReader&` parameter is the engine's reader.

// Load vision encoder weights from a GGUF file.
// The GGUF file must contain clip.vision.* metadata keys and
// vision encoder tensor data (v.blk.*, v.patch_embd.*, etc.).
// Returns true on success.
bool vision_load_from_gguf(VisionGGUFContext& ctx, ModelReader& reader);

// Encode an image using the loaded vision encoder.
// Input: raw RGB pixel data, dimensions, and preprocessing target size.
// Output: projected embedding vector suitable for text model injection.
// Returns true on success.
bool vision_encode_image(const VisionGGUFContext& ctx,
                         const DesireeAIImage& image,
                         std::vector<float>& out_embd);

// Preprocess image: resize, normalize, convert to RGB.
// Uses the vision encoder's mean/std for normalization.
bool vision_preprocess_image_normalized(const VisionGGUFContext& ctx,
                                        const DesireeAIImage& input,
                                        int32_t target_size,
                                        std::vector<float>& out_pixels);

// Get the number of embedding tokens produced by the vision encoder
// for a given image size. This is (image_size / patch_size)^2 + 1 (class token).
int32_t vision_get_token_count(const VisionGGUFContext& ctx);

// Get the embedding dimension of the projected output.
int32_t vision_get_output_dim(const VisionGGUFContext& ctx);

}  // namespace vision
}  // namespace desireeia

#endif  // DESIREEIA_VISION_GGUF_H
