#include "sensor/SuperResolution.h"

#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace sensor {
namespace sr {

// FSR CAS dictates weights applied to color based on local Luma contrast
inline float computeLuma(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

// Helper to safely get the normalized RGB pixel using clamping for border replication
inline void getPixel(const std::vector<uint8_t>& img, int x, int y, int w, int h, float out[3], float& out_luma) {
    x = std::clamp(x, 0, w - 1);
    y = std::clamp(y, 0, h - 1);
    int idx = (y * w + x) * 3;
    out[0] = img[idx + 0] / 255.0f; // R
    out[1] = img[idx + 1] / 255.0f; // G
    out[2] = img[idx + 2] / 255.0f; // B
    out_luma = computeLuma(out[0], out[1], out[2]);
}

void applyCAS_CPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness) {
    if (rgb.empty() || width <= 0 || height <= 0) return;

    std::vector<uint8_t> scratch_buffer(rgb.size());

    // Map sharpness [0.0, 1.0] to FSR peak [-1/8, -1/5]
    float t = std::clamp(sharpness, 0.0f, 1.0f);
    float peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);

    // Parallelize processing by row. schedule(static) reduces locking overhead since workload is uniform.
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            
            // FSR CAS operates on a precise '+' cross pattern:
            float a[3], b[3], c[3], d[3], e[3];
            float l_a, l_b, l_c, l_d, l_e;
            
            getPixel(rgb, x, y - 1, width, height, a, l_a); // Top
            getPixel(rgb, x - 1, y, width, height, b, l_b); // Left
            getPixel(rgb, x, y, width, height, c, l_c);     // Center (Original)
            getPixel(rgb, x + 1, y, width, height, d, l_d); // Right
            getPixel(rgb, x, y + 1, width, height, e, l_e); // Bottom
            
            // Find local neighborhood Luma min and max (Global intensity anchor)
            float min_luma = std::min({l_a, l_b, l_c, l_d, l_e});
            float max_luma = std::max({l_a, l_b, l_c, l_d, l_e});
            
            // Contrast adaptive weighting
            float d_min = min_luma;
            float d_max = 1.0f - max_luma;
            float max_val_safe = max_luma + 1e-6f;
            
            // Singular weight applied uniformly to R, G, B to prevent chromatic bleeding
            float w = std::sqrt(std::min(d_min, d_max) / max_val_safe) * peak;
            float weight_sum = 1.0f + 4.0f * w;
            
            for (int ch = 0; ch < 3; ++ch) {
                // Apply the uniformly calculated Luma weight to the individual color channels
                float enhanced = (c[ch] + w * (a[ch] + b[ch] + d[ch] + e[ch])) / weight_sum;
                scratch_buffer[(y * width + x) * 3 + ch] = static_cast<uint8_t>(std::clamp(enhanced, 0.0f, 1.0f) * 255.0f);
            }
        }
    }
    
    // Copy computed scratch output back into the original vector (preserves original capacity and pointers)
    std::memcpy(rgb.data(), scratch_buffer.data(), rgb.size());
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
