#include "sensor/SignalConditioner.h"

#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "sensor/SuperResolution.h"

#include <algorithm>
#include <array>
#include <cmath>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace kfusion {
namespace sensor {

namespace {

constexpr int kClaheTileSize = 8;
constexpr int kHoleFillRadius = 2;
constexpr int kGuidedRadius = 9;
constexpr int kRgbBilateralRadius = 2;
constexpr int kDepthMedianRadius = 1;
constexpr float kEmaJumpResetMeters = 0.05f;
constexpr float kGuidedSigmaLuma = 0.10f;
constexpr float kGuidedSigmaDepth = 0.04f;
constexpr float kRgbSigmaSpatial = 2.0f;
constexpr float kRgbSigmaRange = 28.0f;

inline uint8_t clampToByte(float v) {
    return static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
}

inline float rgbLuma(const uint8_t* rgb) {
    return 0.299f * static_cast<float>(rgb[0]) +
           0.587f * static_cast<float>(rgb[1]) +
           0.114f * static_cast<float>(rgb[2]);
}

inline uint16_t metersToRawDepth(float depth_m) {
    if (depth_m <= 0.0f) return 0;
    const float raw = (1.0f / depth_m - 3.3309495161f) / -0.0030711016f;
    // Use explicit long cast to avoid signed/unsigned narrowing on MSVC where
    // long is 32-bit: std::lround returns long, clamp literals must match.
    const long clamped = std::clamp(static_cast<long>(std::lround(raw)), 1L, 2046L);
    return static_cast<uint16_t>(clamped);
}

inline bool isValidDepthMeters(float d, float min_depth_m, float max_depth_m) {
    return d >= min_depth_m && d <= max_depth_m;
}

void medianBlur3x3(std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> scratch(rgb.size(), 0);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            for (int c = 0; c < 3; ++c) {
                uint8_t window[9];
                int count = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int sy = std::clamp(y + dy, 0, FRAME_H - 1);
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = std::clamp(x + dx, 0, FRAME_W - 1);
                        window[count++] = rgb[(sy * FRAME_W + sx) * 3 + c];
                    }
                }
                std::nth_element(window, window + 4, window + count);
                scratch[(y * FRAME_W + x) * 3 + c] = window[4];
            }
        }
    }

    rgb.swap(scratch);
}

void bilateralDenoiseRgb(const std::vector<uint8_t>& src, std::vector<uint8_t>& dst) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int center_idx = (y * FRAME_W + x) * 3;
            const float center_rgb[3] = {
                static_cast<float>(src[center_idx + 0]),
                static_cast<float>(src[center_idx + 1]),
                static_cast<float>(src[center_idx + 2])
            };

            float accum[3] = {0.0f, 0.0f, 0.0f};
            float weight_sum = 0.0f;

            for (int dy = -kRgbBilateralRadius; dy <= kRgbBilateralRadius; ++dy) {
                const int sy = std::clamp(y + dy, 0, FRAME_H - 1);
                for (int dx = -kRgbBilateralRadius; dx <= kRgbBilateralRadius; ++dx) {
                    const int sx = std::clamp(x + dx, 0, FRAME_W - 1);
                    const int sample_idx = (sy * FRAME_W + sx) * 3;
                    const float spatial_dist_sq = static_cast<float>(dx * dx + dy * dy);

                    float color_dist_sq = 0.0f;
                    for (int c = 0; c < 3; ++c) {
                        const float delta = static_cast<float>(src[sample_idx + c]) - center_rgb[c];
                        color_dist_sq += delta * delta;
                    }

                    const float spatial_weight = std::exp(-spatial_dist_sq / (2.0f * kRgbSigmaSpatial * kRgbSigmaSpatial));
                    const float range_weight = std::exp(-color_dist_sq / (2.0f * kRgbSigmaRange * kRgbSigmaRange));
                    const float weight = spatial_weight * range_weight;

                    for (int c = 0; c < 3; ++c) {
                        accum[c] += weight * static_cast<float>(src[sample_idx + c]);
                    }
                    weight_sum += weight;
                }
            }

            for (int c = 0; c < 3; ++c) {
                dst[center_idx + c] = (weight_sum > 1e-6f)
                    ? clampToByte(accum[c] / weight_sum)
                    : src[center_idx + c];
            }
        }
    }
}

} // namespace

SignalConditioner::SignalConditioner()
    : ema_buf_m_(FRAME_W * FRAME_H, 0.0f),
      sr_rgb_(FRAME_W * FRAME_H * 3, 0),
      rgb_scratch_(FRAME_W * FRAME_H * 3, 0),
      guidance_luma_(FRAME_W * FRAME_H, 0.0f),
      depth_scratch_(FRAME_W * FRAME_H, 0) {}

void SignalConditioner::reset() {
    resetEMA();
    std::fill(sr_rgb_.begin(), sr_rgb_.end(), 0);
    std::fill(rgb_scratch_.begin(), rgb_scratch_.end(), 0);
    std::fill(guidance_luma_.begin(), guidance_luma_.end(), 0.0f);
    std::fill(depth_scratch_.begin(), depth_scratch_.end(), 0);

#ifdef CUDA_ENABLED
    if (d_rgb_in_) cudaMemset(d_rgb_in_.get(), 0, sr_rgb_.size());
    if (d_rgb_out_) cudaMemset(d_rgb_out_.get(), 0, sr_rgb_.size());
    if (d_guidance_luma_) cudaMemset(d_guidance_luma_.get(), 0, guidance_luma_.size() * sizeof(float));
    if (d_depth_in_) cudaMemset(d_depth_in_.get(), 0, depth_scratch_.size() * sizeof(uint16_t));
    if (d_depth_out_) cudaMemset(d_depth_out_.get(), 0, depth_scratch_.size() * sizeof(uint16_t));
#endif
}

void SignalConditioner::resetEMA() {
    std::fill(ema_buf_m_.begin(), ema_buf_m_.end(), 0.0f);
#ifdef CUDA_ENABLED
    if (d_ema_buf_m_) cudaMemset(d_ema_buf_m_.get(), 0, ema_buf_m_.size() * sizeof(float));
#endif
}

void SignalConditioner::process(RawFrame& raw, cudaStream_t cuda_stream, float min_depth_m, float max_depth_m) {
    (void)cuda_stream;

    if (raw.rgb.size() != sr_rgb_.size() || raw.depth.size() != depth_scratch_.size()) {
        return;
    }

#ifdef CUDA_ENABLED
    if (cuda_stream && processCuda(raw, cuda_stream, min_depth_m, max_depth_m)) {
        return;
    }
#endif

    processCpu(raw, min_depth_m, max_depth_m);
}

void SignalConditioner::processCpu(RawFrame& raw, float min_depth_m, float max_depth_m) {
    preprocessRgb(raw.rgb);
    buildSuperResolutionGuidance(raw.rgb);
    denoiseDepthSpatial(raw.depth, min_depth_m, max_depth_m);
    applyDepthEma(raw.depth, min_depth_m, max_depth_m);
    fillDepthHoles(raw.depth, min_depth_m, max_depth_m);
    guidedDepthFilter(raw.depth, min_depth_m, max_depth_m);
}

void SignalConditioner::preprocessRgb(std::vector<uint8_t>& rgb) {
    bilateralDenoiseRgb(rgb, rgb_scratch_);
    rgb.swap(rgb_scratch_);

    // Block-based CLAHE removed because independent 8x8 tiled histogram equalization
    // without bilinear interpolation causes false grid boundary artifacts that corrupt
    // the downstream guided depth filter. We rely on the `medianBlur3x3` for noise 
    // reduction and `applyCAS` (SuperResolution) later for contrast enhancement.

    medianBlur3x3(rgb);
}

void SignalConditioner::buildSuperResolutionGuidance(const std::vector<uint8_t>& rgb) {
    sr_rgb_ = rgb;

    // CAS is the in-tree fallback guidance sharpener until a DNN SR backend is added.
    sr::applyCAS(sr_rgb_, FRAME_W, FRAME_H, 0.85f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < FRAME_W * FRAME_H; ++i) {
        guidance_luma_[i] = rgbLuma(&sr_rgb_[i * 3]) / 255.0f;
    }
}

void SignalConditioner::denoiseDepthSpatial(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    depth_scratch_ = depth;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int idx = y * FRAME_W + x;
            const float center_depth_m = rawDepthToMeters(depth[idx]);
            if (!isValidDepthMeters(center_depth_m, min_depth_m, max_depth_m)) {
                continue;
            }

            float window[9];
            int count = 0;
            for (int dy = -kDepthMedianRadius; dy <= kDepthMedianRadius; ++dy) {
                const int sy = std::clamp(y + dy, 0, FRAME_H - 1);
                for (int dx = -kDepthMedianRadius; dx <= kDepthMedianRadius; ++dx) {
                    const int sx = std::clamp(x + dx, 0, FRAME_W - 1);
                    const float sample_depth_m = rawDepthToMeters(depth[sy * FRAME_W + sx]);
                    if (!isValidDepthMeters(sample_depth_m, min_depth_m, max_depth_m)) {
                        continue;
                    }
                    window[count++] = sample_depth_m;
                }
            }

            if (count >= 3) {
                std::nth_element(window, window + count / 2, window + count);
                depth_scratch_[idx] = metersToRawDepth(window[count / 2]);
            }
        }
    }

    depth.swap(depth_scratch_);
}

void SignalConditioner::applyDepthEma(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < FRAME_W * FRAME_H; ++i) {
        const float depth_m = rawDepthToMeters(depth[i]);
        if (!isValidDepthMeters(depth_m, min_depth_m, max_depth_m)) {
            ema_buf_m_[i] = 0.0f; // Prevent ghosting by resetting stale EMA values
            continue;
        }

        const float previous = ema_buf_m_[i];
        const float delta = std::abs(depth_m - previous);
        const float filtered = (previous <= 0.0f || delta > kEmaJumpResetMeters)
            ? depth_m
            : (0.7f * depth_m + 0.3f * previous);

        ema_buf_m_[i] = filtered;
        depth[i] = metersToRawDepth(filtered);
    }
}

void SignalConditioner::fillDepthHoles(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    // NOTE: depth_scratch_ is a member variable used as a write-scratch buffer.
    // This function must only be called from a single thread per SignalConditioner
    // instance. PipelineController guarantees this via the single trackingLoop thread.
    depth_scratch_ = depth;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int idx = y * FRAME_W + x;
            if (depth[idx] != 0) {
                continue;
            }

            float best_depth_m = 0.0f;
            int best_dist_sq = (kHoleFillRadius + 1) * (kHoleFillRadius + 1) * 2;

            for (int dy = -kHoleFillRadius; dy <= kHoleFillRadius; ++dy) {
                const int sy = y + dy;
                if (sy < 0 || sy >= FRAME_H) {
                    continue;
                }
                for (int dx = -kHoleFillRadius; dx <= kHoleFillRadius; ++dx) {
                    const int sx = x + dx;
                    if (sx < 0 || sx >= FRAME_W || (dx == 0 && dy == 0)) {
                        continue;
                    }

                    const float candidate_m = rawDepthToMeters(depth[sy * FRAME_W + sx]);
                    if (!isValidDepthMeters(candidate_m, min_depth_m, max_depth_m)) {
                        continue;
                    }

                    const int dist_sq = dx * dx + dy * dy;
                    if (dist_sq < best_dist_sq) {
                        best_dist_sq = dist_sq;
                        best_depth_m = candidate_m;
                    }
                }
            }

            if (best_depth_m > 0.0f) {
                depth_scratch_[idx] = metersToRawDepth(best_depth_m);
            }
        }
    }

    depth.swap(depth_scratch_);
}

void SignalConditioner::guidedDepthFilter(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m) {
    // PERFORMANCE NOTE: kGuidedRadius=9 produces a 19x19 kernel (~361 samples/pixel).
    // At 640x480 this is ~110M multiply-adds per frame on the CPU path. If real-time
    // performance is required, consider reducing kGuidedRadius to 4-5, or replacing
    // with a separable bilateral approximation.
    depth_scratch_ = depth;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int idx = y * FRAME_W + x;
            const float center_depth = rawDepthToMeters(depth[idx]);
            
            // Prevent guided filter from treating empty space as a massive hole-filler 
            // which smears sharp object geometry boundaries into the void.
            if (!isValidDepthMeters(center_depth, min_depth_m, max_depth_m)) {
                continue;
            }

            const float center_luma = guidance_luma_[idx];

            float sum_weights = 0.0f;
            float sum_depth = 0.0f;

            for (int dy = -kGuidedRadius; dy <= kGuidedRadius; ++dy) {
                const int sy = y + dy;
                if (sy < 0 || sy >= FRAME_H) {
                    continue;
                }
                for (int dx = -kGuidedRadius; dx <= kGuidedRadius; ++dx) {
                    const int sx = x + dx;
                    if (sx < 0 || sx >= FRAME_W) {
                        continue;
                    }

                    const int nidx = sy * FRAME_W + sx;
                    const float neighbor_depth = rawDepthToMeters(depth[nidx]);
                    if (!isValidDepthMeters(neighbor_depth, min_depth_m, max_depth_m)) {
                        continue;
                    }

                    const float spatial_dist_sq = static_cast<float>(dx * dx + dy * dy);
                    const float guidance_delta = guidance_luma_[nidx] - center_luma;
                    const float depth_delta = (center_depth > 0.0f) ? (neighbor_depth - center_depth) : 0.0f;

                    const float spatial_weight = std::exp(-spatial_dist_sq / (2.0f * kGuidedRadius * kGuidedRadius));
                    const float luma_weight = std::exp(-(guidance_delta * guidance_delta) / (2.0f * kGuidedSigmaLuma * kGuidedSigmaLuma + 0.01f));
                    const float depth_weight = std::exp(-(depth_delta * depth_delta) / (2.0f * kGuidedSigmaDepth * kGuidedSigmaDepth + 0.01f));
                    const float weight = spatial_weight * luma_weight * depth_weight;

                    sum_weights += weight;
                    sum_depth += weight * neighbor_depth;
                }
            }

            if (sum_weights > 1e-6f) {
                depth_scratch_[idx] = metersToRawDepth(sum_depth / sum_weights);
            }
        }
    }

    depth.swap(depth_scratch_);
}

} // namespace sensor
} // namespace kfusion
