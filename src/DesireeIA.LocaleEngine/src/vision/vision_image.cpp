#include "vision_image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace desireeia {
namespace vision {

namespace {

// Extract file extension and convert to lowercase.
std::string get_extension_lower(const std::string& path) {
    auto pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    std::string ext = path.substr(pos);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

}  // namespace

ImageFormat detect_format_from_path(const std::string& path) {
    std::string ext = get_extension_lower(path);
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".jpe") return ImageFormat::JPEG;
    if (ext == ".png") return ImageFormat::PNG;
    if (ext == ".bmp") return ImageFormat::BMP;
    return ImageFormat::UNKNOWN;
}

PixelBuffer load_image_from_file(const char* path, int expected_channels) {
    PixelBuffer result;
    if (!path || path[0] == '\0') return result;

    int w = 0, h = 0, c = 0;
    uint8_t* pixels = stbi_load(path, &w, &h, &c, expected_channels);
    if (!pixels || w <= 0 || h <= 0) {
        if (pixels) stbi_image_free(pixels);
        return result;
    }

    result.width = static_cast<uint32_t>(w);
    result.height = static_cast<uint32_t>(h);
    result.channels = static_cast<uint32_t>(expected_channels);
    size_t byte_count = static_cast<size_t>(w) * h * expected_channels;
    result.data.assign(pixels, pixels + byte_count);
    stbi_image_free(pixels);
    return result;
}

PixelBuffer load_image_from_memory(const uint8_t* bytes, size_t len, int expected_channels) {
    PixelBuffer result;
    if (!bytes || len == 0) return result;

    int w = 0, h = 0, c = 0;
    uint8_t* pixels = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc*>(bytes),
        static_cast<int>(len), &w, &h, &c, expected_channels);
    if (!pixels || w <= 0 || h <= 0) {
        if (pixels) stbi_image_free(pixels);
        return result;
    }

    result.width = static_cast<uint32_t>(w);
    result.height = static_cast<uint32_t>(h);
    result.channels = static_cast<uint32_t>(expected_channels);
    size_t byte_count = static_cast<size_t>(w) * h * expected_channels;
    result.data.assign(pixels, pixels + byte_count);
    stbi_image_free(pixels);
    return result;
}

bool save_image_to_file(const std::string& path, const PixelBuffer& img, int quality) {
    if (img.empty() || img.width == 0 || img.height == 0) return false;

    int w = static_cast<int>(img.width);
    int h = static_cast<int>(img.height);
    int ch = static_cast<int>(img.channels);

    ImageFormat fmt = detect_format_from_path(path);
    switch (fmt) {
        case ImageFormat::JPEG:
            return stbi_write_jpg(path.c_str(), w, h, ch,
                                  img.data.data(), quality) != 0;
        case ImageFormat::PNG:
            return stbi_write_png(path.c_str(), w, h, ch,
                                  img.data.data(), 0) != 0;
        case ImageFormat::BMP:
            return stbi_write_bmp(path.c_str(), w, h, ch,
                                  img.data.data()) != 0;
        default:
            return false;
    }
}

DesireeAIImage pixel_buffer_to_image(const PixelBuffer& buf) {
    DesireeAIImage img;
    img.width = buf.width;
    img.height = buf.height;
    img.channels = buf.channels;
    if (!buf.data.empty()) {
        img.data = new uint8_t[buf.data.size()];
        std::memcpy(img.data, buf.data.data(), buf.data.size());
    } else {
        img.data = nullptr;
    }
    return img;
}

PixelBuffer image_to_pixel_buffer(const DesireeAIImage& img) {
    PixelBuffer buf;
    buf.width = img.width;
    buf.height = img.height;
    buf.channels = img.channels;
    if (img.data && img.width > 0 && img.height > 0) {
        size_t byte_count = static_cast<size_t>(img.width) * img.height * img.channels;
        buf.data.assign(img.data, img.data + byte_count);
    }
    return buf;
}

PixelBuffer resize_image(const PixelBuffer& src, int new_width, int new_height) {
    PixelBuffer result;
    if (src.empty() || new_width <= 0 || new_height <= 0) return result;

    result.width = static_cast<uint32_t>(new_width);
    result.height = static_cast<uint32_t>(new_height);
    result.channels = src.channels;
    result.data.resize(static_cast<size_t>(new_width) * new_height * src.channels);

    stbir_resize(
        src.data.data(),
        static_cast<int>(src.width), static_cast<int>(src.height), 0,
        result.data.data(),
        new_width, new_height, 0,
        STBIR_TYPE_UINT8,
        static_cast<int>(src.channels),
        STBIR_ALPHA_CHANNEL_NONE, 0,
        STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
        STBIR_FILTER_BOX, STBIR_FILTER_BOX,
        STBIR_COLORSPACE_SRGB, nullptr);

    return result;
}

PixelBuffer crop_and_resize(const PixelBuffer& src, int target_width, int target_height) {
    if (src.empty() || target_width <= 0 || target_height <= 0) {
        return PixelBuffer();
    }

    float dst_aspect = static_cast<float>(target_width) / target_height;
    float src_aspect = static_cast<float>(src.width) / src.height;

    int crop_x = 0, crop_y = 0;
    int crop_w = static_cast<int>(src.width);
    int crop_h = static_cast<int>(src.height);

    if (src_aspect > dst_aspect) {
        crop_w = static_cast<int>(src.height * dst_aspect);
        crop_x = (static_cast<int>(src.width) - crop_w) / 2;
    } else if (src_aspect < dst_aspect) {
        crop_h = static_cast<int>(src.width / dst_aspect);
        crop_y = (static_cast<int>(src.height) - crop_h) / 2;
    }

    // Center-crop the source image.
    PixelBuffer cropped;
    cropped.width = static_cast<uint32_t>(crop_w);
    cropped.height = static_cast<uint32_t>(crop_h);
    cropped.channels = src.channels;
    cropped.data.resize(static_cast<size_t>(crop_w) * crop_h * src.channels);

    for (int row = 0; row < crop_h; ++row) {
        const uint8_t* src_row = src.data.data() +
            static_cast<size_t>((crop_y + row) * src.width + crop_x) * src.channels;
        uint8_t* dst_row = cropped.data.data() + static_cast<size_t>(row * crop_w) * src.channels;
        std::memcpy(dst_row, src_row, static_cast<size_t>(crop_w) * src.channels);
    }

    // Resize to target dimensions.
    return resize_image(cropped, target_width, target_height);
}

PixelBuffer grayscale_to_rgb(const PixelBuffer& src) {
    PixelBuffer result;
    if (src.empty() || src.channels != 1) return src;

    result.width = src.width;
    result.height = src.height;
    result.channels = 3;
    result.data.resize(static_cast<size_t>(src.width) * src.height * 3);

    for (size_t i = 0; i < src.pixel_count(); ++i) {
        result.data[i * 3 + 0] = src.data[i];
        result.data[i * 3 + 1] = src.data[i];
        result.data[i * 3 + 2] = src.data[i];
    }
    return result;
}

PixelBuffer rgba_to_rgb(const PixelBuffer& src) {
    PixelBuffer result;
    if (src.empty() || src.channels != 4) return src;

    result.width = src.width;
    result.height = src.height;
    result.channels = 3;
    result.data.resize(static_cast<size_t>(src.width) * src.height * 3);

    for (size_t i = 0; i < src.pixel_count(); ++i) {
        result.data[i * 3 + 0] = src.data[i * 4 + 0];
        result.data[i * 3 + 1] = src.data[i * 4 + 1];
        result.data[i * 3 + 2] = src.data[i * 4 + 2];
    }
    return result;
}

}  // namespace vision
}  // namespace desireeia
