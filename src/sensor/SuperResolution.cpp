#include "sensor/SuperResolution.h"

#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace sensor {
namespace sr {

// Helper to safely get the normalized RGB pixel using clamping for border replication
inline void getPixel(const std::vector<uint8_t>& img, int x, int y, int w, int h, float out[3]) {
    x = std::clamp(x, 0, w - 1);
    y = std::clamp(y, 0, h - 1);
    int idx = (y * w + x) * 3;
    out[0] = img[idx + 0] / 255.0f; // R
    out[1] = img[idx + 1] / 255.0f; // G
    out[2] = img[idx + 2] / 255.0f; // B
}

void applyCAS(std::vector<uint8_t>& rgb, int width, int height, float sharpness) {
    if (rgb.empty() || width <= 0 || height <= 0) return;

    // We must use a scratch buffer to prevent feedback loops during the convolution
    std::vector<uint8_t> out_buffer(rgb.size());

    // Map sharpness [0.0, 1.0] to FSR peak [-1/8, -1/5]
    // std::lerp is C++20, so using standard manual lerp for C++17 compatibility
    float t = std::clamp(sharpness, 0.0f, 1.0f);
    float peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);

    // Parallelize processing by row to maximize cache coherency
    #pragma omp parallel for schedule(dynamic, 8)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            
            // FSR CAS operates on a precise '+' cross pattern:
            //   a
            // b c d
            //   e
            float a[3], b[3], c[3], d[3], e[3];
            getPixel(rgb, x, y - 1, width, height, a); // Top
            getPixel(rgb, x - 1, y, width, height, b); // Left
            getPixel(rgb, x, y, width, height, c);     // Center
            getPixel(rgb, x + 1, y, width, height, d); // Right
            getPixel(rgb, x, y + 1, width, height, e); // Bottom
            
            for (int ch = 0; ch < 3; ++ch) {
                // Find local neighborhood min and max
                float min_val = std::min({a[ch], b[ch], c[ch], d[ch], e[ch]});
                float max_val = std::max({a[ch], b[ch], c[ch], d[ch], e[ch]});
                
                // Contrast adaptive weighting
                float d_min = min_val;
                float d_max = 1.0f - max_val;
                // Add a small epsilon to avoid division by zero
                float max_val_safe = max_val + 1e-6f;
                float w = std::sqrt(std::min(d_min, d_max) / max_val_safe) * peak;
                
                // Accumulate the 5-tap filter
                float weight_sum = 1.0f + 4.0f * w;
                float enhanced = (c[ch] + w * (a[ch] + b[ch] + d[ch] + e[ch])) / weight_sum;
                
                // Clamp and encode back to 8-bit
                out_buffer[(y * width + x) * 3 + ch] = static_cast<uint8_t>(std::clamp(enhanced, 0.0f, 1.0f) * 255.0f);
            }
        }
    }
    
    // Efficiently move the enhanced buffer back into rgb
    rgb = std::move(out_buffer);
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
