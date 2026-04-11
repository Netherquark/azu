#ifdef CUDA_ENABLED

#include "tsdf/TSDFVolume.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstring>

namespace kfusion {
namespace tsdf {

#include "tsdf/VoxelGPU.h"

// -------------------------------------------------------------------
// Integration kernel: Image-Centric (Pixel Parallel)
// -------------------------------------------------------------------
__global__ void integrationKernel_PixelParallel(
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
    // Camera-to-world (for ray direction)
    float rcw00, float rcw01, float rcw02,
    float rcw10, float rcw11, float rcw12,
    float rcw20, float rcw21, float rcw22,
    float tcw_x, float tcw_y, float tcw_z,
    // World-to-camera (for voxel projection)
    float rwc00, float rwc01, float rwc02,
    float rwc10, float rwc11, float rwc12,
    float rwc20, float rwc21, float rwc22,
    float twc_x, float twc_y, float twc_z)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    float d_meas = depth[py * width + px];
    if (d_meas <= 0.0f) return;

    // Ray direction in camera space
    float rx_c = (px - cx) / fx;
    float ry_c = (py - cy) / fy;
    float rz_c = 1.0f;

    // Normalize ray in world space
    float rx_w = rcw00*rx_c + rcw01*ry_c + rcw02*rz_c;
    float ry_w = rcw10*rx_c + rcw11*ry_c + rcw12*rz_c;
    float rz_w = rcw20*rx_c + rcw21*ry_c + rcw22*rz_c;
    float inv_len = 1.0f / sqrtf(rx_w*rx_w + ry_w*ry_w + rz_w*rz_w);
    rx_w *= inv_len; ry_w *= inv_len; rz_w *= inv_len;

    float t_min = fmaxf(0.1f, d_meas - truncation);
    float t_max = d_meas + truncation;
    float step  = voxel_size * 0.75f;

    for (float t = t_min; t <= t_max; t += step) {
        float wx = tcw_x + rx_w * t;
        float wy = tcw_y + ry_w * t;
        float wz = tcw_z + rz_w * t;

        int vx = (int)((wx - origin.x) / voxel_size);
        int vy = (int)((wy - origin.y) / voxel_size);
        int vz = (int)((wz - origin.z) / voxel_size);

        if (vx < 0 || vx >= resolution || vy < 0 || vy >= resolution || vz < 0 || vz >= resolution)
            continue;

        float cz_ = rwc20*wx + rwc21*wy + rwc22*wz + twc_z;
        float sdf = d_meas - cz_;
        if (sdf < -truncation) continue;

        float tsdf_new = fminf(1.0f, sdf / truncation);
        int lidx = vz * resolution * resolution + vy * resolution + vx;
        VoxelGPU& vox = voxels[lidx];

        // Atomic integration with CAS clamp
        union TSDFPack {
            struct { float tsdf; float weight; } fields;
            unsigned long long int as_ull;
        };

        unsigned long long int* addr = (unsigned long long int*)&vox.tsdf;
        unsigned long long int old_val = *addr, assumed, new_val;
        float w_old, w_new;
        do {
            assumed = old_val;
            TSDFPack pack; pack.as_ull = assumed;
            w_old = pack.fields.weight;
            w_new = fminf(w_old + 1.0f, max_weight);
            float t_new = (pack.fields.tsdf * w_old + tsdf_new) / (w_new + 1e-6f);
            
            TSDFPack new_pack; 
            new_pack.fields.tsdf = t_new; 
            new_pack.fields.weight = w_new;
            new_val = new_pack.as_ull;
            old_val = atomicCAS(addr, assumed, new_val);
        } while (assumed != old_val);
        
        if (rgb) {
            int pidx = py * width + px;
            float r_meas = (float)rgb[pidx*3+0];
            float g_meas = (float)rgb[pidx*3+1];
            float b_meas = (float)rgb[pidx*3+2];
            
            auto atomicUpdateColor = [&](float* c_addr, float m) {
                int* c_addr_i = (int*)c_addr;
                int c_old_val = *c_addr_i, c_assumed, c_new_val;
                do {
                    c_assumed = c_old_val;
                    float c_old = __int_as_float(c_assumed);
                    float c_new = (c_old * w_old + m) / (w_new + 1e-6f);
                    c_new_val = __float_as_int(c_new);
                    c_old_val = atomicCAS(c_addr_i, c_assumed, c_new_val);
                } while (c_assumed != c_old_val);
            };
            
            atomicUpdateColor(&vox.r, r_meas);
            atomicUpdateColor(&vox.g, g_meas);
            atomicUpdateColor(&vox.b, b_meas);
        }
    }
}

// -------------------------------------------------------------------
// Host-side GPU management
// -------------------------------------------------------------------
void TSDFVolume::freeGPU() {
    d_voxels_.reset();
    d_depth_.reset();
    d_rgb_.reset();
    gpu_valid_ = false;
}

void TSDFVolume::initGPU() {
    const int res = params_.resolution;
    size_t n = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);
    d_voxels_ = utils::make_cuda_unique<void>(n * sizeof(VoxelGPU));
    d_depth_  = utils::make_cuda_unique<float>(640 * 480);
    d_rgb_    = utils::make_cuda_unique<uint8_t>(640 * 480 * 3);
    gpu_valid_ = true;
    syncToGPU();
}

void TSDFVolume::syncToGPU() {
    if (!gpu_valid_) return;
    size_t n = voxels_.size();
    std::vector<VoxelGPU> gpu_data(n);
    for (size_t i = 0; i < n; ++i) {
        gpu_data[i].tsdf   = voxels_[i].tsdf;
        gpu_data[i].weight = voxels_[i].weight;
        gpu_data[i].r      = voxels_[i].r;
        gpu_data[i].g      = voxels_[i].g;
        gpu_data[i].b      = voxels_[i].b;
    }
    cudaMemcpy(d_voxels_.get(), gpu_data.data(), n * sizeof(VoxelGPU), cudaMemcpyHostToDevice);
}

void TSDFVolume::syncFromGPU() {
    if (!gpu_valid_) return;
    size_t n = voxels_.size();
    std::vector<VoxelGPU> gpu_data(n);
    cudaMemcpy(gpu_data.data(), d_voxels_.get(), n * sizeof(VoxelGPU), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i) {
        float w = gpu_data[i].weight;
        voxels_[i].weight = w;
        if (w > 0.001f) {
            voxels_[i].tsdf = gpu_data[i].tsdf;
            voxels_[i].r = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].r));
            voxels_[i].g = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].g));
            voxels_[i].b = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].b));
        } else {
            voxels_[i].tsdf = 1.0f;
        }
    }
}

void TSDFVolume::integrateGPU(const float*           depth_meters,
                               const uint8_t*         rgb,
                               const Eigen::Matrix4f& pose,
                               float fx, float fy,
                               float cx, float cy,
                               int   width, int height)
{
    if (!gpu_valid_) return;
    cudaMemcpy(d_depth_.get(), depth_meters, width * height * sizeof(float), cudaMemcpyHostToDevice);
    if (rgb) cudaMemcpy(d_rgb_.get(), rgb, width * height * 3, cudaMemcpyHostToDevice);

    Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    Eigen::Vector3f t_cw = pose.block<3,1>(0,3);
    Eigen::Matrix4f inv  = pose.inverse();
    Eigen::Matrix3f R_wc = inv.block<3,3>(0,0);
    Eigen::Vector3f t_wc = inv.block<3,1>(0,3);

    float3 origin = {params_.origin.x(), params_.origin.y(), params_.origin.z()};
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    integrationKernel_PixelParallel<<<grid, block>>>(
        (VoxelGPU*)d_voxels_.get(), params_.resolution, params_.voxel_size, origin, params_.truncation, params_.max_weight,
        d_depth_.get(), d_rgb_.get(), width, height, fx, fy, cx, cy,
        R_cw(0,0), R_cw(0,1), R_cw(0,2), R_cw(1,0), R_cw(1,1), R_cw(1,2), R_cw(2,0), R_cw(2,1), R_cw(2,2),
        t_cw.x(), t_cw.y(), t_cw.z(),
        R_wc(0,0), R_wc(0,1), R_wc(0,2), R_wc(1,0), R_wc(1,1), R_wc(1,2), R_wc(2,0), R_wc(2,1), R_wc(2,2),
        t_wc.x(), t_wc.y(), t_wc.z()
    );
    cudaDeviceSynchronize();
}

__device__ float3 computeNormalGPU(void* voxels_void, int resolution, int x, int y, int z) {
    VoxelGPU* voxels = (VoxelGPU*)voxels_void;

    auto get_tsdf = [&](int xi, int yi, int zi) {
        if (xi < 0 || xi >= resolution || yi < 0 || yi >= resolution || zi < 0 || zi >= resolution) return 1.0f;
        VoxelGPU& v = voxels[zi * resolution * resolution + yi * resolution + xi];
        return (v.weight <= 0.001f) ? 1.0f : v.tsdf;
    };

    float3 n;
    n.x = get_tsdf(x + 1, y, z) - get_tsdf(x - 1, y, z);
    n.y = get_tsdf(x, y + 1, z) - get_tsdf(x, y - 1, z);
    n.z = get_tsdf(x, y, z + 1) - get_tsdf(x, y, z - 1);

    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-6f) {
        n.x /= len; n.y /= len; n.z /= len;
    } else {
        n = make_float3(0, 0, -1);
    }
    return n;
}

__global__ void raycastKernel(
    void* voxels_void, int resolution, float voxel_size, float3 origin,
    float fx, float fy, float cx, float cy,
    float r00, float r01, float r02, float tx,
    float r10, float r11, float r12, float ty,
    float r20, float r21, float r22, float tz,
    int width, int height,
    float3* out_v, float3* out_n)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    VoxelGPU* voxels = (VoxelGPU*)voxels_void;

    auto get_tsdf = [&](int xi, int yi, int zi) {
        VoxelGPU& v = voxels[zi * resolution * resolution + yi * resolution + xi];
        return (v.weight <= 0.001f) ? 1.0f : v.tsdf;
    };

    // Ray in camera space
    float3 ray_c = make_float3((px - cx) / fx, (py - cy) / fy, 1.0f);
    float rlen = sqrtf(ray_c.x*ray_c.x + ray_c.y*ray_c.y + 1.0f);
    ray_c.x /= rlen; ray_c.y /= rlen; ray_c.z /= rlen;

    // Ray in world space
    float3 ray_w;
    ray_w.x = r00 * ray_c.x + r01 * ray_c.y + r02 * ray_c.z;
    ray_w.y = r10 * ray_c.x + r11 * ray_c.y + r12 * ray_c.z;
    ray_w.z = r20 * ray_c.x + r21 * ray_c.y + r22 * ray_c.z;

    float3 pos_w = make_float3(tx, ty, tz);
    float t = 0.4f; // min depth
    float prev_val = 1.0f;

    for (int i = 0; i < 400; ++i) { // max steps
        float3 p = make_float3(pos_w.x + ray_w.x * t, pos_w.y + ray_w.y * t, pos_w.z + ray_w.z * t);
        int vx = (int)((p.x - origin.x) / voxel_size);
        int vy = (int)((p.y - origin.y) / voxel_size);
        int vz = (int)((p.z - origin.z) / voxel_size);

        if (vx < 0 || vx >= resolution || vy < 0 || vy >= resolution || vz < 0 || vz >= resolution) {
            t += voxel_size;
            continue;
        }

        float val = get_tsdf(vx, vy, vz);
        if (prev_val > 0 && val < 0) {
            // Surface found
            float t_fine = t - voxel_size * (val / (val - prev_val));
            float3 pf = make_float3(pos_w.x + ray_w.x * t_fine, pos_w.y + ray_w.y * t_fine, pos_w.z + ray_w.z * t_fine);
            
            // Transform back to camera space for the output vertex map
            // cam = R^T (world - t)
            float3 pc;
            pc.x = r00 * (pf.x - tx) + r10 * (pf.y - ty) + r20 * (pf.z - tz);
            pc.y = r01 * (pf.x - tx) + r11 * (pf.y - ty) + r21 * (pf.z - tz);
            pc.z = r02 * (pf.x - tx) + r12 * (pf.y - ty) + r22 * (pf.z - tz);
            
            out_v[py * width + px] = pc;
            
            // Compute normal in world space then rotate to camera space
            float3 n_w = computeNormalGPU(voxels_void, resolution, vx, vy, vz);
            float3 n_c;
            n_c.x = r00 * n_w.x + r10 * n_w.y + r20 * n_w.z;
            n_c.y = r01 * n_w.x + r11 * n_w.y + r21 * n_w.z;
            n_c.z = r02 * n_w.x + r12 * n_w.y + r22 * n_w.z;
            out_n[py * width + px] = n_c;
            return;
        }
        prev_val = val;
        t += fmaxf(voxel_size, val * 0.1f); // skipping
        if (t > 5.0f) break;
    }
    out_v[py * width + px] = make_float3(0,0,0);
    out_n[py * width + px] = make_float3(0,0,0);
}

void TSDFVolume::raycastGPU(const Eigen::Matrix4f& pose,
                             float fx, float fy, float cx, float cy,
                             int width, int height,
                             float3* d_vertices, float3* d_normals)
{
    if (!gpu_valid_) return;

    Eigen::Matrix3f R = pose.block<3,3>(0,0);
    Eigen::Vector3f t = pose.block<3,1>(0,3);

    float3 f_origin = {params_.origin.x(), params_.origin.y(), params_.origin.z()};
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    raycastKernel<<<grid, block>>>(
        d_voxels_.get(), params_.resolution, params_.voxel_size, f_origin,
        fx, fy, cx, cy,
        R(0,0), R(0,1), R(0,2), t.x(),
        R(1,0), R(1,1), R(1,2), t.y(),
        R(2,0), R(2,1), R(2,2), t.z(),
        width, height, d_vertices, d_normals
    );
    cudaDeviceSynchronize();
}

} // namespace tsdf
} // namespace kfusion

#endif // CUDA_ENABLED
