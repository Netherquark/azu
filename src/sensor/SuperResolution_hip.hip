#include "sensor/SuperResolution.h"

#ifdef HIP_ENABLED
#include <hip/hip_runtime.h>
#include <mutex>
#include <cstdio>

namespace kfusion {
namespace sensor {
namespace sr {

// ---------------------------------------------------------------------------
// FSR 1.0 RCAS — HIP Kernel
//
// Implements AMD FidelityFX FSR 1.0 RCAS (Robust Contrast Adaptive Sharpening)
// correctly with per-channel contrast-adaptive amplitude limits.
// ---------------------------------------------------------------------------

// Reads a clamped-border pixel, returns normalised [0,1] floats per channel.
__device__ static inline void rcasGetPixel(const uint8_t* img,
                                            int x, int y, int w, int h,
                                            float out[3]) {
    x = max(0, min(x, w - 1));
    y = max(0, min(y, h - 1));
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

    // Cross-pattern neighbours
    float a[3], b[3], c[3], d[3], e[3];
    rcasGetPixel(d_in, x,     y - 1, width, height, a); // Top
    rcasGetPixel(d_in, x - 1, y,     width, height, b); // Left
    rcasGetPixel(d_in, x,     y,     width, height, c); // Centre
    rcasGetPixel(d_in, x + 1, y,     width, height, d); // Right
    rcasGetPixel(d_in, x,     y + 1, width, height, e); // Bottom

    // Per-channel contrast-adaptive amplitude (FSR 1.0 RCAS exact algorithm)
    float amp = 1.0f;
    for (int ch = 0; ch < 3; ++ch) {
        float mn    = fminf(a[ch], fminf(b[ch], fminf(c[ch], fminf(d[ch], e[ch]))));
        float mx    = fmaxf(a[ch], fmaxf(b[ch], fmaxf(c[ch], fmaxf(d[ch], e[ch]))));
        float mx_s  = fmaxf(mx, 1e-6f);
        float amp_ch = fminf(mn, 1.0f - mx) / mx_s;
        amp = fminf(amp, amp_ch);
    }

    const float w          = amp * peak;
    const float weight_sum = 1.0f + 4.0f * w;

    const int out_idx = (y * width + x) * 3;
    for (int ch = 0; ch < 3; ++ch) {
        float val = (c[ch] + w * (a[ch] + b[ch] + d[ch] + e[ch])) / weight_sum;
        d_out[out_idx + ch] = (uint8_t)(fmaxf(0.0f, fminf(1.0f, val)) * 255.0f);
    }
}

// ---------------------------------------------------------------------------
// Buffer management — plain static with mutex; thread_local is wrong here
// because the AMD ROCm driver context is per-process not per-thread.
// ---------------------------------------------------------------------------
struct HipBufferRAII {
    uint8_t* d_in  = nullptr;
    uint8_t* d_out = nullptr;
    size_t   size  = 0;
    std::mutex mtx;

    void resize(size_t req) {
        if (size == req) return;
        if (d_in)  (void)hipFree(d_in);
        if (d_out) (void)hipFree(d_out);
        d_in = d_out = nullptr; size = 0;
        hipError_t r1 = hipMalloc(&d_in,  req);
        hipError_t r2 = hipMalloc(&d_out, req);
        if (r1 != hipSuccess || r2 != hipSuccess) {
            fprintf(stderr, "[RCAS-HIP] hipMalloc failed: %s\n",
                    hipGetErrorString(r1 != hipSuccess ? r1 : r2));
            if (d_in)  { hipFree(d_in);  d_in  = nullptr; }
            if (d_out) { hipFree(d_out); d_out = nullptr; }
            return;
        }
        size = req;
    }

    ~HipBufferRAII() {
        if (d_in)  (void)hipFree(d_in);
        if (d_out) (void)hipFree(d_out);
    }
};

static HipBufferRAII g_rcas_buffers; // process-lifetime singleton

void applyCAS_GPU(std::vector<uint8_t>& rgb, int width, int height, float sharpness) {
    if (rgb.empty() || width <= 0 || height <= 0) return;

    const size_t req  = (size_t)width * height * 3;
    const float  t    = std::max(0.0f, std::min(sharpness, 1.0f));
    const float  peak = -1.0f / ((1.0f - t) * 8.0f + t * 5.0f);

    std::lock_guard<std::mutex> lk(g_rcas_buffers.mtx);
    g_rcas_buffers.resize(req);
    if (!g_rcas_buffers.d_in || !g_rcas_buffers.d_out) return; // alloc failed

    // Upload → kernel → download (all on default stream, synchronous)
    hipError_t err;
    err = hipMemcpy(g_rcas_buffers.d_in, rgb.data(), req, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "[RCAS-HIP] Upload failed: %s\n", hipGetErrorString(err));
        return;
    }

    dim3 block(16, 16);
    dim3 grid((width  + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    rcasKernel<<<grid, block>>>(g_rcas_buffers.d_in, g_rcas_buffers.d_out,
                                      width, height, peak);

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "[RCAS-HIP] Kernel error: %s\n", hipGetErrorString(err));
        return;
    }

    err = hipMemcpy(rgb.data(), g_rcas_buffers.d_out, req, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        fprintf(stderr, "[RCAS-HIP] Download failed: %s\n", hipGetErrorString(err));
    }
}

} // namespace sr
} // namespace sensor
} // namespace kfusion
#endif // HIP_ENABLED
