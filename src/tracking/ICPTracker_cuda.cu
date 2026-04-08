#ifdef CUDA_ENABLED

#include "tracking/ICPTracker.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// ---------------------------------------------------------------------------
// GPU-accelerated normal map downsampling kernel
// Used to build pyramid levels on GPU before ICP (optional acceleration path)
// ---------------------------------------------------------------------------

__global__ void downsampleKernel(
    const float* __restrict__ src_depth,
    float*                    dst_depth,
    const float* __restrict__ src_vx,
    const float* __restrict__ src_vy,
    const float* __restrict__ src_vz,
    float*                    dst_vx,
    float*                    dst_vy,
    float*                    dst_vz,
    int src_w, int src_h,
    int dst_w, int dst_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int sx = x * 2;
    int sy = y * 2;

    float sum_d = 0.0f, sum_vx = 0.0f, sum_vy = 0.0f, sum_vz = 0.0f;
    int count = 0;

    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int sidx = (sy + dy) * src_w + (sx + dx);
            float d = src_depth[sidx];
            if (d > 0.0f) {
                sum_d  += d;
                sum_vx += src_vx[sidx];
                sum_vy += src_vy[sidx];
                sum_vz += src_vz[sidx];
                ++count;
            }
        }
    }

    int didx = y * dst_w + x;
    if (count > 0) {
        float inv = 1.0f / count;
        dst_depth[didx] = sum_d  * inv;
        dst_vx[didx]    = sum_vx * inv;
        dst_vy[didx]    = sum_vy * inv;
        dst_vz[didx]    = sum_vz * inv;
    } else {
        dst_depth[didx] = 0.0f;
        dst_vx[didx] = dst_vy[didx] = dst_vz[didx] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Normal computation kernel (cross-product of neighbors)
// ---------------------------------------------------------------------------

__global__ void computeNormalsKernel(
    const float* vx, const float* vy, const float* vz,
    const float* depth,
    float* nx, float* ny, float* nz,
    int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x <= 0 || x >= W-1 || y <= 0 || y >= H-1) return;

    int c = y*W + x;
    int r = y*W + (x+1);
    int l = y*W + (x-1);
    int u = (y-1)*W + x;
    int d = (y+1)*W + x;

    if (depth[c] <= 0.0f || depth[r] <= 0.0f || depth[l] <= 0.0f ||
        depth[u] <= 0.0f || depth[d] <= 0.0f) {
        nx[c] = ny[c] = nz[c] = 0.0f;
        return;
    }

    float dxx = vx[r] - vx[l];
    float dxy = vy[r] - vy[l];
    float dxz = vz[r] - vz[l];

    float dyx = vx[d] - vx[u];
    float dyy = vy[d] - vy[u];
    float dyz = vz[d] - vz[u];

    // cross product dx × dy
    float ncx = dxy*dyz - dxz*dyy;
    float ncy = dxz*dyx - dxx*dyz;
    float ncz = dxx*dyy - dxy*dyx;

    float len = sqrtf(ncx*ncx + ncy*ncy + ncz*ncz);
    if (len > 1e-6f) {
        nx[c] = ncx / len;
        ny[c] = ncy / len;
        nz[c] = ncz / len;
    } else {
        nx[c] = ny[c] = nz[c] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Host-side wrapper: GPU normal computation
// Called optionally from FrameData processing when GPU is available
// ---------------------------------------------------------------------------

namespace kfusion {
namespace tracking {

void computeNormalsGPU(
    const float* d_vx, const float* d_vy, const float* d_vz,
    const float* d_depth,
    float* d_nx, float* d_ny, float* d_nz,
    int W, int H)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    computeNormalsKernel<<<grid, block>>>(
        d_vx, d_vy, d_vz, d_depth,
        d_nx, d_ny, d_nz, W, H);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "[ICP CUDA] Normal kernel error: " << cudaGetErrorString(err) << "\n";
}

} // namespace tracking
} // namespace kfusion

#endif // CUDA_ENABLED
