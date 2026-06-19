#ifdef HIP_ENABLED

#include "sensor/SignalConditioner.h"
#include "sensor/KinectSensor.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

#ifndef HIP_CHECK_LAST
#define HIP_CHECK_LAST() \
    do { \
        hipError_t err = hipGetLastError(); \
        if (err != hipSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        } \
    } while(0)
#endif

namespace kfusion {
namespace sensor {

// Constants matching SignalConditioner_omp.cpp
namespace {
    constexpr int kHoleFillRadius = 2;
    constexpr int kGuidedRadius = 9;
    constexpr int kRgbBilateralRadius = 2;
    constexpr int kDepthMedianRadius = 1;
    constexpr float kEmaJumpResetMeters = 0.05f;
    constexpr float kGuidedSigmaLuma = 0.10f;
    constexpr float kGuidedSigmaDepth = 0.04f;
    constexpr float kRgbSigmaSpatial = 2.0f;
    constexpr float kRgbSigmaRange = 28.0f;

    __device__ inline float rgbLumaGPU(uint8_t r, uint8_t g, uint8_t b) {
        return 0.299f * (float)r + 0.587f * (float)g + 0.114f * (float)b;
    }

    __device__ inline float rawDepthToMetersGPU(uint16_t depth) {
        if (depth == 0 || depth == 2047) return 0.0f;
        return 1.0f / (depth * -0.0030711016f + 3.3309495161f);
    }

    __device__ inline uint16_t metersToRawDepthGPU(float depth_m) {
        if (depth_m <= 0.0f) return 0;
        float raw = (1.0f / depth_m - 3.3309495161f) / -0.0030711016f;
        return (uint16_t)max(1.0f, min(2046.0f, floorf(raw + 0.5f)));
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ inline float computeLumaGPU(float r, float g, float b) {
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

__device__ inline void getPixelGPU(const uint8_t* img, int x, int y, int w, int h, float out[3], float& out_luma) {
    x = max(0, min(x, w - 1));
    y = max(0, min(y, h - 1));
    int idx = (y * w + x) * 3;
    out[0] = img[idx + 0] / 255.0f;
    out[1] = img[idx + 1] / 255.0f;
    out[2] = img[idx + 2] / 255.0f;
    out_luma = computeLumaGPU(out[0], out[1], out[2]);
}

__global__ void applyCASKernel(const uint8_t* d_in, uint8_t* d_out, int width, int height, float peak) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float a[3], b[3], c[3], d[3], e[3];
    float l_a, l_b, l_c, l_d, l_e;
    
    getPixelGPU(d_in, x, y - 1, width, height, a, l_a);
    getPixelGPU(d_in, x - 1, y, width, height, b, l_b);
    getPixelGPU(d_in, x, y, width, height, c, l_c);
    getPixelGPU(d_in, x + 1, y, width, height, d, l_d);
    getPixelGPU(d_in, x, y + 1, width, height, e, l_e);

    float min_luma = min(l_a, min(l_b, min(l_c, min(l_d, l_e))));
    float max_luma = max(l_a, max(l_b, max(l_c, max(l_d, l_e))));
    
    float d_min = min_luma;
    float d_max = 1.0f - max_luma;
    float max_val_safe = max_luma + 1e-6f;
    
    float w = sqrt(min(d_min, d_max) / max_val_safe) * peak;
    float weight_sum = 1.0f + 4.0f * w;
    
    int out_idx = (y * width + x) * 3;
    for (int ch = 0; ch < 3; ++ch) {
        float enhanced = (c[ch] + w * (a[ch] + b[ch] + d[ch] + e[ch])) / weight_sum;
        d_out[out_idx + ch] = (uint8_t)(max(0.0f, min(1.0f, enhanced)) * 255.0f);
    }
}

__global__ void bilateralDenoiseRgbKernel(const uint8_t* src, uint8_t* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int center_idx = (y * w + x) * 3;
    float center_r = src[center_idx + 0];
    float center_g = src[center_idx + 1];
    float center_b = src[center_idx + 2];

    float accum_r = 0, accum_g = 0, accum_b = 0, weight_sum = 0;

    for (int dy = -kRgbBilateralRadius; dy <= kRgbBilateralRadius; ++dy) {
        int sy = max(0, min(y + dy, h - 1));
        for (int dx = -kRgbBilateralRadius; dx <= kRgbBilateralRadius; ++dx) {
            int sx = max(0, min(x + dx, w - 1));
            int sidx = (sy * w + sx) * 3;

            float sr = src[sidx + 0];
            float sg = src[sidx + 1];
            float sb = src[sidx + 2];

            float spatial_dist_sq = (float)(dx * dx + dy * dy);
            float color_dist_sq = (sr - center_r) * (sr - center_r) +
                                  (sg - center_g) * (sg - center_g) +
                                  (sb - center_b) * (sb - center_b);

            float weight = expf(-spatial_dist_sq / (2.0f * kRgbSigmaSpatial * kRgbSigmaSpatial)) *
                           expf(-color_dist_sq / (2.0f * kRgbSigmaRange * kRgbSigmaRange));

            accum_r += weight * sr;
            accum_g += weight * sg;
            accum_b += weight * sb;
            weight_sum += weight;
        }
    }

    dst[center_idx + 0] = (uint8_t)(accum_r / weight_sum);
    dst[center_idx + 1] = (uint8_t)(accum_g / weight_sum);
    dst[center_idx + 2] = (uint8_t)(accum_b / weight_sum);
}

__global__ void medianBlur3x3Kernel(const uint8_t* src, uint8_t* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    for (int c = 0; c < 3; ++c) {
        uint8_t window[9];
        int count = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            int sy = max(0, min(y + dy, h - 1));
            for (int dx = -1; dx <= 1; ++dx) {
                int sx = max(0, min(x + dx, w - 1));
                window[count++] = src[(sy * w + sx) * 3 + c];
            }
        }
        // Simple bubble sort for median of 9
        for (int i = 0; i < 5; ++i) {
            for (int j = i + 1; j < 9; ++j) {
                if (window[i] > window[j]) {
                    uint8_t tmp = window[i];
                    window[i] = window[j];
                    window[j] = tmp;
                }
            }
        }
        dst[(y * w + x) * 3 + c] = window[4];
    }
}

__global__ void computeLumaKernel(const uint8_t* rgb, float* luma, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    luma[idx] = rgbLumaGPU(rgb[idx * 3 + 0], rgb[idx * 3 + 1], rgb[idx * 3 + 2]) / 255.0f;
}

__global__ void denoiseDepthSpatialKernel(const uint16_t* src, uint16_t* dst, int w, int h, float min_d, float max_d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    float center_d = rawDepthToMetersGPU(src[idx]);
    if (center_d < min_d || center_d > max_d) {
        dst[idx] = src[idx];
        return;
    }

    float window[9];
    int count = 0;
    for (int dy = -kDepthMedianRadius; dy <= kDepthMedianRadius; ++dy) {
        int sy = max(0, min(y + dy, h - 1));
        for (int dx = -kDepthMedianRadius; dx <= kDepthMedianRadius; ++dx) {
            int sx = max(0, min(x + dx, w - 1));
            float d = rawDepthToMetersGPU(src[sy * w + sx]);
            if (d >= min_d && d <= max_d) {
                window[count++] = d;
            }
        }
    }

    if (count >= 3) {
        for (int i = 0; i <= count / 2; ++i) {
            for (int j = i + 1; j < count; ++j) {
                if (window[i] > window[j]) {
                    float tmp = window[i];
                    window[i] = window[j];
                    window[j] = tmp;
                }
            }
        }
        dst[idx] = metersToRawDepthGPU(window[count / 2]);
    } else {
        dst[idx] = src[idx];
    }
}

__global__ void applyDepthEmaKernel(uint16_t* depth, float* ema_buf, int w, int h, float min_d, float max_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    float d = rawDepthToMetersGPU(depth[idx]);
    if (d < min_d || d > max_d || isnan(d) || isinf(d)) {
        ema_buf[idx] = 0.0f;
        return;
    }

    float previous = ema_buf[idx];
    float delta = fabsf(d - previous);
    float filtered = (isnan(previous) || isinf(previous) || previous <= 0.0f || delta > kEmaJumpResetMeters)
        ? d
        : (0.7f * d + 0.3f * previous);

    ema_buf[idx] = filtered;
    depth[idx] = metersToRawDepthGPU(filtered);
}

__global__ void fillDepthHolesKernel(const uint16_t* src, uint16_t* dst, int w, int h, float min_d, float max_d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    if (src[idx] != 0 && src[idx] != 2047) {
        dst[idx] = src[idx];
        return;
    }

    float best_d = 0.0f;
    int best_dist_sq = (kHoleFillRadius + 1) * (kHoleFillRadius + 1) * 2;

    for (int dy = -kHoleFillRadius; dy <= kHoleFillRadius; ++dy) {
        int sy = y + dy;
        if (sy < 0 || sy >= h) continue;
        for (int dx = -kHoleFillRadius; dx <= kHoleFillRadius; ++dx) {
            int sx = x + dx;
            if (sx < 0 || sx >= w || (dx == 0 && dy == 0)) continue;

            float candidate = rawDepthToMetersGPU(src[sy * w + sx]);
            if (candidate >= min_d && candidate <= max_d) {
                int dist_sq = dx * dx + dy * dy;
                if (dist_sq < best_dist_sq) {
                    best_dist_sq = dist_sq;
                    best_d = candidate;
                }
            }
        }
    }

    if (best_d > 0.0f) {
        dst[idx] = metersToRawDepthGPU(best_d);
    } else {
        dst[idx] = 0;
    }
}

__global__ void guidedDepthFilterKernel(const uint16_t* depth, uint16_t* dst, const float* luma, int w, int h, float min_d, float max_d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    float center_d = rawDepthToMetersGPU(depth[idx]);
    if (center_d <= 0.01f) {
        dst[idx] = 0;
        return;
    }
    if (center_d < min_d || center_d > max_d) {
        dst[idx] = depth[idx];
        return;
    }

    float center_l = luma[idx];
    float sum_w = 0.0f;
    float sum_d = 0.0f;

    for (int dy = -kGuidedRadius; dy <= kGuidedRadius; ++dy) {
        int sy = y + dy;
        if (sy < 0 || sy >= h) continue;
        for (int dx = -kGuidedRadius; dx <= kGuidedRadius; ++dx) {
            int sx = x + dx;
            if (sx < 0 || sx >= w) continue;

            int nidx = sy * w + sx;
            float nd = rawDepthToMetersGPU(depth[nidx]);
            if (nd < min_d || nd > max_d) continue;

            float spatial_dist_sq = (float)(dx * dx + dy * dy);
            float l_delta = luma[nidx] - center_l;
            float d_delta = nd - center_d;

            float weight = expf(-spatial_dist_sq / (2.0f * kGuidedRadius * kGuidedRadius)) *
                           expf(-(l_delta * l_delta) / (2.0f * kGuidedSigmaLuma * kGuidedSigmaLuma + 0.01f)) *
                           expf(-(d_delta * d_delta) / (2.0f * kGuidedSigmaDepth * kGuidedSigmaDepth + 0.01f));

            sum_w += weight;
            sum_d += weight * nd;
        }
    }

    if (sum_w > 1e-6f) {
        dst[idx] = metersToRawDepthGPU(sum_d / sum_w);
    } else {
        dst[idx] = depth[idx];
    }
}

__global__ void rawToMetersKernel(const uint16_t* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = rawDepthToMetersGPU(src[idx]);
}

// ---------------------------------------------------------------------------
// SignalConditioner::processCuda
// ---------------------------------------------------------------------------

bool SignalConditioner::processCuda(RawFrame& raw,
                                    hipStream_t stream,
                                    float min_depth_m,
                                    float max_depth_m) {
    int w = RGB_WIDTH;
    int h = RGB_HEIGHT;
    size_t n = w * h;

    // Allocate resources on first use
    if (!d_rgb_in_) {
        d_rgb_in_ = utils::make_hip_unique<uint8_t>(n * 3);
        d_rgb_out_ = utils::make_hip_unique<uint8_t>(n * 3);
        d_depth_in_ = utils::make_hip_unique<uint16_t>(n);
        d_depth_out_ = utils::make_hip_unique<uint16_t>(n);
        d_depth_meters_ = utils::make_hip_unique<float>(n);
        d_ema_buf_m_ = utils::make_hip_unique<float>(n);
        d_guidance_luma_ = utils::make_hip_unique<float>(n);
        hipMemsetAsync(d_ema_buf_m_.get(), 0, n * sizeof(float), stream);
    }

    // Upload
    hipMemcpyAsync(d_rgb_in_.get(), raw.rgb.data(), n * 3, hipMemcpyHostToDevice, stream);
    hipMemcpyAsync(d_depth_in_.get(), raw.depth.data(), n * sizeof(uint16_t), hipMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    // 1. Preprocess RGB: Bilateral + Median
    bilateralDenoiseRgbKernel<<<grid, block, 0, stream>>>(d_rgb_in_.get(), d_rgb_out_.get(), w, h);
    HIP_CHECK_LAST();
    medianBlur3x3Kernel<<<grid, block, 0, stream>>>(d_rgb_out_.get(), d_rgb_in_.get(), w, h);
    HIP_CHECK_LAST();

    // 2. Super Resolution Guidance (CAS)
    float sharpness = 0.85f;
    float t = std::max(0.0f, std::min(sharpness, 1.0f));
    float peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);
    applyCASKernel<<<grid, block, 0, stream>>>(d_rgb_in_.get(), d_rgb_out_.get(), w, h, peak);
    HIP_CHECK_LAST();
    
    // Compute Luma for guided filter
    computeLumaKernel<<<grid, block, 0, stream>>>(d_rgb_out_.get(), d_guidance_luma_.get(), w, h);
    HIP_CHECK_LAST();

    // 3. Depth pipeline
    denoiseDepthSpatialKernel<<<grid, block, 0, stream>>>(d_depth_in_.get(), d_depth_out_.get(), w, h, min_depth_m, max_depth_m);
    HIP_CHECK_LAST();
    
    fillDepthHolesKernel<<<grid, block, 0, stream>>>(d_depth_out_.get(), d_depth_in_.get(), w, h, min_depth_m, max_depth_m);
    HIP_CHECK_LAST();
    
    // 4. Guided Depth Filter
    guidedDepthFilterKernel<<<grid, block, 0, stream>>>(d_depth_in_.get(), d_depth_out_.get(), d_guidance_luma_.get(), w, h, min_depth_m, max_depth_m);
    HIP_CHECK_LAST();

    // 5. Temporal Stabilization (EMA)
    int total_threads = n;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    applyDepthEmaKernel<<<grid_size, block_size, 0, stream>>>(d_depth_out_.get(), d_ema_buf_m_.get(), w, h, min_depth_m, max_depth_m);
    HIP_CHECK_LAST();

    // 5. Final meters conversion for TSDF integration (Zero-copy GPU path)
    rawToMetersKernel<<<grid_size, block_size, 0, stream>>>(d_depth_out_.get(), d_depth_meters_.get(), n);
    HIP_CHECK_LAST();

    // Download (Only for UI preview or CPU-mode compatibility)
    hipMemcpyAsync(raw.rgb.data(), d_rgb_out_.get(), n * 3, hipMemcpyDeviceToHost, stream);
    hipMemcpyAsync(raw.depth.data(), d_depth_out_.get(), n * sizeof(uint16_t), hipMemcpyDeviceToHost, stream);

    hipStreamSynchronize(stream);
    return true;
}

} // namespace sensor
} // namespace kfusion

#endif
