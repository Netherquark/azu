#ifdef CUDA_ENABLED

#include "tsdf/TSDFVolume.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>

namespace kfusion {
namespace tsdf {

// -------------------------------------------------------------------
// Device structs
// -------------------------------------------------------------------
struct VoxelGPU {
    float   tsdf;
    float   weight;
    uint8_t r, g, b;
    uint8_t pad[3]; // alignment
};

// -------------------------------------------------------------------
// Integration kernel
// -------------------------------------------------------------------
__global__ void integrationKernel(
    VoxelGPU*      voxels,
    int            resolution,
    float          voxel_size,
    float3         origin,
    float          truncation,
    float          max_weight,
    const float*   depth,
    const uint8_t* rgb,
    int            width,
    int            height,
    float          fx, float fy,
    float          cx, float cy,
    // World-to-camera (row-major 4x4, but we pass 3x3 R + 3 t)
    float r00, float r01, float r02,
    float r10, float r11, float r12,
    float r20, float r21, float r22,
    float tx,  float ty,  float tz)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z;

    if (xi >= resolution || yi >= resolution || zi >= resolution) return;

    // World pos of voxel center
    float wx = origin.x + xi * voxel_size + voxel_size * 0.5f;
    float wy = origin.y + yi * voxel_size + voxel_size * 0.5f;
    float wz = origin.z + zi * voxel_size + voxel_size * 0.5f;

    // Camera pos
    float cx_ = r00*wx + r01*wy + r02*wz + tx;
    float cy_ = r10*wx + r11*wy + r12*wz + ty;
    float cz_ = r20*wx + r21*wy + r22*wz + tz;

    if (cz_ <= 0.0f) return;

    float px = fx * cx_ / cz_ + cx;
    float py = fy * cy_ / cz_ + cy;

    int ix = (int)(px + 0.5f);
    int iy = (int)(py + 0.5f);
    if (ix < 0 || ix >= width || iy < 0 || iy >= height) return;

    float d_meas = depth[iy * width + ix];
    if (d_meas <= 0.0f) return;

    float sdf = d_meas - cz_;
    if (sdf < -truncation) return;

    float tsdf_new = fminf(1.0f, sdf / truncation);

    int linear_idx = zi * resolution * resolution + yi * resolution + xi;
    VoxelGPU& vox = voxels[linear_idx];

    float w_old = vox.weight;
    float w_new = 1.0f;
    float w_sum = fminf(w_old + w_new, max_weight);

    vox.tsdf   = (vox.tsdf * w_old + tsdf_new * w_new) / (w_old + w_new + 1e-10f);
    vox.weight = w_sum;

    if (rgb) {
        int pidx = iy * width + ix;
        vox.r = (uint8_t)((vox.r * w_old + rgb[pidx*3+0] * w_new) / (w_old + w_new + 1e-6f));
        vox.g = (uint8_t)((vox.g * w_old + rgb[pidx*3+1] * w_new) / (w_old + w_new + 1e-6f));
        vox.b = (uint8_t)((vox.b * w_old + rgb[pidx*3+2] * w_new) / (w_old + w_new + 1e-6f));
    }
}

// -------------------------------------------------------------------
// Host-side GPU management
// -------------------------------------------------------------------
void TSDFVolume::initGPU() {
    size_t n = static_cast<size_t>(params_.resolution) * params_.resolution * params_.resolution;
    size_t bytes = n * sizeof(VoxelGPU);

    cudaError_t err = cudaMalloc(&d_voxels_, bytes);
    if (err != cudaSuccess) {
        std::cerr << "[TSDF GPU] cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        d_voxels_  = nullptr;
        gpu_valid_ = false;
        return;
    }

    gpu_valid_ = true;
    syncToGPU();
}

void TSDFVolume::freeGPU() {
    if (d_voxels_) {
        cudaFree(d_voxels_);
        d_voxels_  = nullptr;
        gpu_valid_ = false;
    }
}

void TSDFVolume::syncToGPU() {
    if (!gpu_valid_ || !d_voxels_) return;

    size_t n     = voxels_.size();
    size_t bytes = n * sizeof(VoxelGPU);

    // Build GPU-layout copy
    std::vector<VoxelGPU> gpu_data(n);
    for (size_t i = 0; i < n; ++i) {
        gpu_data[i].tsdf   = voxels_[i].tsdf;
        gpu_data[i].weight = voxels_[i].weight;
        gpu_data[i].r      = voxels_[i].r;
        gpu_data[i].g      = voxels_[i].g;
        gpu_data[i].b      = voxels_[i].b;
    }

    cudaMemcpy(d_voxels_, gpu_data.data(), bytes, cudaMemcpyHostToDevice);
}

void TSDFVolume::syncFromGPU() {
    if (!gpu_valid_ || !d_voxels_) return;

    size_t n = voxels_.size();
    std::vector<VoxelGPU> gpu_data(n);
    cudaMemcpy(gpu_data.data(), d_voxels_,
               n * sizeof(VoxelGPU), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < n; ++i) {
        voxels_[i].tsdf   = gpu_data[i].tsdf;
        voxels_[i].weight = gpu_data[i].weight;
        voxels_[i].r      = gpu_data[i].r;
        voxels_[i].g      = gpu_data[i].g;
        voxels_[i].b      = gpu_data[i].b;
    }
}

void TSDFVolume::integrateGPU(const float*           depth_meters,
                               const uint8_t*         rgb,
                               const Eigen::Matrix4f& pose,
                               float fx, float fy,
                               float cx, float cy,
                               int   width, int height)
{
    if (!gpu_valid_ || !d_voxels_) {
        // Fallback to CPU
        integrateCPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height);
        return;
    }

    // Upload depth and rgb to GPU
    float*   d_depth = nullptr;
    uint8_t* d_rgb   = nullptr;

    size_t depth_bytes = static_cast<size_t>(width * height) * sizeof(float);
    size_t rgb_bytes   = static_cast<size_t>(width * height) * 3;

    cudaMalloc(&d_depth, depth_bytes);
    cudaMemcpy(d_depth, depth_meters, depth_bytes, cudaMemcpyHostToDevice);

    if (rgb) {
        cudaMalloc(&d_rgb, rgb_bytes);
        cudaMemcpy(d_rgb, rgb, rgb_bytes, cudaMemcpyHostToDevice);
    }

    // World-to-camera matrix
    Eigen::Matrix4f pose_inv = pose.inverse();
    Eigen::Matrix3f R_wc = pose_inv.block<3,3>(0,0);
    Eigen::Vector3f t_wc = pose_inv.block<3,1>(0,3);

    float3 origin;
    origin.x = params_.origin.x();
    origin.y = params_.origin.y();
    origin.z = params_.origin.z();

    int RES = params_.resolution;
    dim3 block(8, 8, 8);
    dim3 grid(
        (RES + block.x - 1) / block.x,
        (RES + block.y - 1) / block.y,
        (RES + block.z - 1) / block.z
    );

    integrationKernel<<<grid, block>>>(
        reinterpret_cast<VoxelGPU*>(d_voxels_),
        RES,
        params_.voxel_size,
        origin,
        params_.truncation,
        params_.max_weight,
        d_depth, d_rgb,
        width, height,
        fx, fy, cx, cy,
        R_wc(0,0), R_wc(0,1), R_wc(0,2),
        R_wc(1,0), R_wc(1,1), R_wc(1,2),
        R_wc(2,0), R_wc(2,1), R_wc(2,2),
        t_wc.x(), t_wc.y(), t_wc.z()
    );

    cudaDeviceSynchronize();

    cudaFree(d_depth);
    if (d_rgb) cudaFree(d_rgb);

    // Sync back to CPU (needed for marching cubes on CPU)
    syncFromGPU();
}

} // namespace tsdf
} // namespace kfusion

#endif // CUDA_ENABLED
