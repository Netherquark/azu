#pragma once

#include <vector>
#include <cstdint>

namespace kfusion {
namespace sensor {
namespace sr {

/**
 * @brief Applies CPU-based Contrast Adaptive Sharpening (AMD FSR 1.0 CAS math) in-place to an RGB image.
 */
void applyCAS_CPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness = 0.85f);

#ifdef CUDA_ENABLED
/**
 * @brief Applies GPU-accelerated Contrast Adaptive Sharpening (AMD FSR 1.0 CAS math) in-place.
 */
void applyCAS_GPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness = 0.85f);
#endif

/**
 * @brief Master wrapper. Dispatches to GPU if CUDA is enabled and available, otherwise CPU.
 * 
 * @param rgb Flat RGB byte array of size (width * height * 3)
 * @param width Image width
 * @param height Image height
 * @param sharpness Range [0.0, 1.0]. 0.0 is softest, 1.0 is sharpest. Default is 0.85.
 */
inline void applyCAS(std::vector<uint8_t>& rgb, int width, int height, float sharpness = 0.85f) {
#ifdef CUDA_ENABLED
    applyCAS_GPU(rgb, width, height, sharpness);
#else
    applyCAS_CPU(rgb, width, height, sharpness);
#endif
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
