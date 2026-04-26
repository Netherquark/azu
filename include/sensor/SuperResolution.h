#pragma once

#include <vector>
#include <cstdint>

namespace kfusion {
namespace sensor {
namespace sr {

/**
 * @brief Applies Contrast Adaptive Sharpening (AMD FSR 1.0 CAS math) in-place to an RGB image.
 * 
 * @param rgb Flat RGB byte array of size (width * height * 3)
 * @param width Image width
 * @param height Image height
 * @param sharpness Range [0.0, 1.0]. 0.0 is softest, 1.0 is sharpest. Default is 0.8.
 */
void applyCAS(std::vector<uint8_t>& rgb, int width, int height, float sharpness = 0.8f);

} // namespace sr
} // namespace sensor
} // namespace kfusion
