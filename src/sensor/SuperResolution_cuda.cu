#include "sensor/SuperResolution.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <mutex>
#include <cstdio>

namespace kfusion {
namespace sensor {
namespace sr {

// ---------------------------------------------------------------------------
// FSR 1.0 RCAS — CUDA Kernel
// ---------------------------------------------------------------------------

// Reflect boundary handling to eliminate vertical banding
__device__ static inline int reflectCoordCAS(int x, int max_val) {
    if (x < 0) return -x - 1;
    if (x >= max_val) return 2 * max_val - x - 1;
    return x;
}

__device__ static inline void rcasGetPixel(const uint8_t* img,
                                            int x, int y, int w, int h,
                                            float out[3]) {
    x = reflectCoordCAS(x, w);
    y = reflectCoordCAS(y, h);
    const int idx = (y * w + x) * 3;
    out[0] = (float)img[idx + 0] * (1.0f / 255.0f);
    out[1] = (float)img[idx + 1] * (1.0f / 255.0f);
    out[2] = (float)img[idx + 2] * (1.0f / 255.0f);
}

__global__ void rcasKernel(const uint8_t* __restrict__ d_in,
                                 uint8_t* __restrict__ d_out,
                                 int width, int height, float peak) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float a[3], b[3], c[3], d[3], e[3];
    rcasGetPixel(d_in, x,     y - 1, width, height, a);
    rcasGetPixel(d_in, x - 1, y,     width, height, b);
    rcasGetPixel(d_in, x,     y,     width, height, c);
    rcasGetPixel(d_in, x + 1, y,     width, height, d);
    rcasGetPixel(d_in, x,     y + 1, width, height, e);

    float amp = 1.0f;
    for (int ch = 0; ch < 3; ++ch) {
        float mn     = fminf(a[ch], fminf(b[ch], fminf(c[ch], fminf(d[ch], e[ch]))));
        float mx     = fmaxf(a[ch], fmaxf(b[ch], fmaxf(c[ch], fmaxf(d[ch], e[ch]))));
        float mx_s   = fmaxf(mx, 1e-6f);
        float amp_ch = fminf(mn, 1.0f - mx) / mx_s;
        amp = fminf(amp, amp_ch);
    }

    const float w          = amp * peak;
    const float weight_sum = 1.0f + 4.0f * w;
    const int   out_idx    = (y * width + x) * 3;

    for (int ch = 0; ch < 3; ++ch) {
        float val = (c[ch] + w * (a[ch] + b[ch] + d[ch] + e[ch])) / weight_sum;
        d_out[out_idx + ch] = (uint8_t)(fmaxf(0.0f, fminf(1.0f, val)) * 255.0f);
    }
}

struct CudaBufferRAII {
    uint8_t* d_in  = nullptr;
    uint8_t* d_out = nullptr;
    size_t   size  = 0;
    std::mutex mtx;

    void resize(size_t req) {
        if (size == req) return;
        if (d_in)  cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        d_in = d_out = nullptr; size = 0;
        cudaError_t r1 = cudaMalloc(&d_in,  req);
        cudaError_t r2 = cudaMalloc(&d_out, req);
        if (r1 != cudaSuccess || r2 != cudaSuccess) {
            fprintf(stderr, "[RCAS-CUDA] cudaMalloc failed: %s\n",
                    cudaGetErrorString(r1 != cudaSuccess ? r1 : r2));
            if (d_in)  { cudaFree(d_in);  d_in  = nullptr; }
            if (d_out) { cudaFree(d_out); d_out = nullptr; }
            return;
        }
        size = req;
    }

    ~CudaBufferRAII() {
        if (d_in)  cudaFree(d_in);
        if (d_out) cudaFree(d_out);
    }
};

static CudaBufferRAII g_rcas_buffers;

void applyCAS_GPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness) {
    if (rgb.empty() || width <= 0 || height <= 0) return;

    const size_t req  = (size_t)width * height * 3;
    const float  t    = std::max(0.0f, std::min(sharpness, 1.0f));
    const float  peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);

    std::lock_guard<std::mutex> lk(g_rcas_buffers.mtx);
    g_rcas_buffers.resize(req);
    if (!g_rcas_buffers.d_in || !g_rcas_buffers.d_out) return;

    cudaMemcpy(g_rcas_buffers.d_in, rgb.data(), req, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rcasKernel<<<grid, block>>>(g_rcas_buffers.d_in, g_rcas_buffers.d_out,
                                      width, height, peak);
    cudaDeviceSynchronize();
    cudaMemcpy(rgb.data(), g_rcas_buffers.d_out, req, cudaMemcpyDeviceToHost);
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
#endif // CUDA_ENABLED
