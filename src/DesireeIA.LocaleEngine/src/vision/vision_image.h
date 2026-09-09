#ifndef DESIREEIA_VISION_IMAGE_H
#define DESIREEIA_VISION_IMAGE_H

#include "desireeia/abi.h"
#include <cstdint>
#include <string>
#include <vector>

namespace desireeia {
namespace vision {

// Pixel data container for internal image operations.
// Stores interleaved RGB/RGBA/grayscale data in uint8 format.
struct PixelBuffer {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    std::vector<uint8_t> data;

    bool empty() const { return data.empty(); }
    size_t pixel_count() const { return static_cast<size_t>(width) * height; }
    size_t byte_count() const { return pixel_count() * channels; }
};

// Supported image file formats for encoding output.
enum class ImageFormat {
    PNG,
    JPEG,
    BMP,
    UNKNOWN
};

// Detect image format from file extension.
ImageFormat detect_format_from_path(const std::string& path);

// Load image from file path. Returns empty PixelBuffer on failure.
PixelBuffer load_image_from_file(const char* path, int expected_channels = 3);

// Load image from memory buffer. Returns empty PixelBuffer on failure.
PixelBuffer load_image_from_memory(const uint8_t* bytes, size_t len, int expected_channels = 3);

// Save image to file. Returns true on success.
bool save_image_to_file(const std::string& path, const PixelBuffer& img, int quality = 90);

// Convert PixelBuffer to DesireeAIImage (caller owns the data pointer).
DesireeAIImage pixel_buffer_to_image(const PixelBuffer& buf);

// Convert DesireeAIImage to PixelBuffer (copies data).
PixelBuffer image_to_pixel_buffer(const DesireeAIImage& img);

// Resize image using bilinear interpolation.
PixelBuffer resize_image(const PixelBuffer& src, int new_width, int new_height);

// Center-crop image to target aspect ratio, then resize.
PixelBuffer crop_and_resize(const PixelBuffer& src, int target_width, int target_height);

// Convert grayscale to RGB by replicating channels.
PixelBuffer grayscale_to_rgb(const PixelBuffer& src);

// Convert RGBA to RGB by dropping alpha channel.
PixelBuffer rgba_to_rgb(const PixelBuffer& src);

}  // namespace vision
}  // namespace desireeia

#endif  // DESIREEIA_VISION_IMAGE_H
