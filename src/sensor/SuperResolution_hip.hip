#include "sensor/SuperResolution.h"

#ifdef HIP_ENABLED
#include <hip/hip_runtime.h>
#include <iostream>

namespace kfusion {
namespace sensor {
namespace sr {

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

    // Luma min/max
    float min_luma = min(l_a, min(l_b, min(l_c, min(l_d, l_e))));
    float max_luma = max(l_a, max(l_b, max(l_c, max(l_d, l_e))));
    
    // Contrast adaptive weighting computation
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

struct CudaBufferRAII {
    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    size_t size = 0;
    ~CudaBufferRAII() {
        if (d_in) (void)hipFree(d_in);
        if (d_out) (void)hipFree(d_out);
    }
    void resize(size_t req_size) {
        if (size != req_size) {
            if (d_in) (void)hipFree(d_in);
            if (d_out) (void)hipFree(d_out);
            (void)hipMalloc(&d_in, req_size);
            (void)hipMalloc(&d_out, req_size);
            size = req_size;
        }
    }
};

void applyCAS_GPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness) {
    if (rgb.empty() || width <= 0 || height <= 0) return;
    
    size_t req_size = width * height * 3;
    static thread_local CudaBufferRAII buffers;
    buffers.resize(req_size);

    float t = std::max(0.0f, std::min(sharpness, 1.0f));
    float peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);

    // Async block
    (void)hipMemcpy(buffers.d_in, rgb.data(), req_size, hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    applyCASKernel<<<grid, block>>>(buffers.d_in, buffers.d_out, width, height, peak);
    (void)hipDeviceSynchronize();

    (void)hipMemcpy(rgb.data(), buffers.d_out, req_size, hipMemcpyDeviceToHost);
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
#endif // HIP_ENABLED
