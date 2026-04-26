#ifdef CUDA_ENABLED

#include "tsdf/TSDFVolume.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <iostream>
#include <cstring>
#include "tsdf/VoxelGPU.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)

namespace kfusion {
namespace tsdf {

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
    float ray_dist_scale = sqrtf(rx_c*rx_c + ry_c*ry_c + rz_c*rz_c);

    // Normalize ray in world space
    float rx_w = rcw00*rx_c + rcw01*ry_c + rcw02*rz_c;
    float ry_w = rcw10*rx_c + rcw11*ry_c + rcw12*rz_c;
    float rz_w = rcw20*rx_c + rcw21*ry_c + rcw22*rz_c;
    float inv_len = 1.0f / sqrtf(rx_w*rx_w + ry_w*ry_w + rz_w*rz_w);
    rx_w *= inv_len; ry_w *= inv_len; rz_w *= inv_len;

    float t_meas = d_meas * ray_dist_scale;
    float t_min = fmaxf(0.1f, t_meas - truncation);
    float t_max = t_meas + truncation;
    float step  = voxel_size * 0.75f;

    // Hoist the Z-projection scale to avoid matrix-vector dot in the loop
    // cz_ = t * (ray_direction_in_camera_space.z) = t / ray_dist_scale
    float z_scale = 1.0f / ray_dist_scale;

    for (float t = t_min; t <= t_max; t += step) {
        float wx = tcw_x + rx_w * t;
        float wy = tcw_y + ry_w * t;
        float wz = tcw_z + rz_w * t;

        int vx = (int)((wx - origin.x) / voxel_size);
        int vy = (int)((wy - origin.y) / voxel_size);
        int vz = (int)((wz - origin.z) / voxel_size);

        if (vx < 0 || vx >= resolution || vy < 0 || vy >= resolution || vz < 0 || vz >= resolution)
            continue;

        float cz_ = t * z_scale;
        float sdf = d_meas - cz_;
        if (sdf < -truncation) continue;

        float tsdf_new = fminf(1.0f, sdf / truncation);
        int lidx = vz * resolution * resolution + vy * resolution + vx;
        VoxelGPU& vox = voxels[lidx];

        // Simultaneous 64-bit atomic update for weight and TSDF
        unsigned long long* addr_tsdf_weight = (unsigned long long*)&vox.tsdf;
        unsigned long long assumed_tw, old_tw = *addr_tsdf_weight;
        float w_new;
        do {
            assumed_tw = old_tw;
            
            // Unpack 64-bit int into two 32-bit ints (Little-Endian: tsdf is lower 32-bits)
            unsigned int t_old_int = (unsigned int)(assumed_tw & 0xFFFFFFFF);
            unsigned int w_old_int = (unsigned int)(assumed_tw >> 32);
            
            float t_old = __int_as_float(t_old_int);
            float w_old = __int_as_float(w_old_int);
            
            w_new = fminf(w_old + 1.0f, max_weight);
            float t_new = (t_old * w_old + tsdf_new) / (w_old + 1.0f);
            
            unsigned int t_new_int = __float_as_int(t_new);
            unsigned int w_new_int = __float_as_int(w_new);
            
            unsigned long long new_tw = ((unsigned long long)w_new_int << 32) | (unsigned long long)t_new_int;
            
            old_tw = atomicCAS(addr_tsdf_weight, assumed_tw, new_tw);
        } while (assumed_tw != old_tw);
        
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
                    float c_new = (c_old * w_old + m) / (w_old + 1.0f);
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
    d_pc_is_valid_.reset();
    d_pc_offsets_.reset();
    gpu_valid_ = false;
}

void TSDFVolume::initGPU() {
    const int res = params_.resolution;
    size_t n = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);
    d_voxels_ = utils::make_cuda_unique<VoxelGPU>(n);
    d_depth_  = utils::make_cuda_unique<float>(640 * 480);
    d_rgb_    = utils::make_cuda_unique<uint8_t>(640 * 480 * 3);
    gpu_valid_ = true;
    syncToGPU();
}

void TSDFVolume::syncToGPU() {
    if (!gpu_valid_) return;
    size_t n = voxels_.size();
    const size_t CHUNK_SIZE = 1024 * 1024;
    std::vector<VoxelGPU> gpu_data(std::min(n, CHUNK_SIZE));
    for (size_t offset = 0; offset < n; offset += CHUNK_SIZE) {
        size_t current_chunk = std::min(CHUNK_SIZE, n - offset);
        for (size_t i = 0; i < current_chunk; ++i) {
            size_t idx = offset + i;
            gpu_data[i].tsdf   = voxels_[idx].tsdf;
            gpu_data[i].weight = voxels_[idx].weight;
            gpu_data[i].r      = voxels_[idx].r;
            gpu_data[i].g      = voxels_[idx].g;
            gpu_data[i].b      = voxels_[idx].b;
        }
        CUDA_CHECK(cudaMemcpy(d_voxels_.get() + offset, gpu_data.data(), current_chunk * sizeof(VoxelGPU), cudaMemcpyHostToDevice));
    }
}

void TSDFVolume::syncFromGPU() {
    if (!gpu_valid_) return;
    size_t n = voxels_.size();
    const size_t CHUNK_SIZE = 1024 * 1024;
    std::vector<VoxelGPU> gpu_data(std::min(n, CHUNK_SIZE));
    for (size_t offset = 0; offset < n; offset += CHUNK_SIZE) {
        size_t current_chunk = std::min(CHUNK_SIZE, n - offset);
        CUDA_CHECK(cudaMemcpy(gpu_data.data(), d_voxels_.get() + offset, current_chunk * sizeof(VoxelGPU), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < current_chunk; ++i) {
            size_t idx = offset + i;
            float w = gpu_data[i].weight;
            voxels_[idx].weight = w;
            if (w > 0.001f) {
                voxels_[idx].tsdf = gpu_data[i].tsdf;
                voxels_[idx].r = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].r));
                voxels_[idx].g = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].g));
                voxels_[idx].b = (uint8_t)fminf(255.0f, fmaxf(0.0f, gpu_data[i].b));
            } else {
                voxels_[idx].tsdf = 1.0f;
            }
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
    CUDA_CHECK(cudaMemcpy(d_depth_.get(), depth_meters, width * height * sizeof(float), cudaMemcpyHostToDevice));
    if (rgb) CUDA_CHECK(cudaMemcpy(d_rgb_.get(), rgb, width * height * 3, cudaMemcpyHostToDevice));

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
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}

__device__ float3 computeNormalGPU(void* voxels_void, int resolution, int x, int y, int z) {
    VoxelGPU* voxels = (VoxelGPU*)voxels_void;

    auto get_tsdf_and_valid = [&](int xi, int yi, int zi, float& tsdf_val, bool& valid) {
        if (xi < 0 || xi >= resolution || yi < 0 || yi >= resolution || zi < 0 || zi >= resolution) {
            valid = false;
            tsdf_val = 1.0f;
            return;
        }
        VoxelGPU& v = voxels[zi * resolution * resolution + yi * resolution + xi];
        if (v.weight <= 0.001f) {
            valid = false;
            tsdf_val = 1.0f;
        } else {
            valid = true;
            tsdf_val = v.tsdf;
        }
    };

    float tsdf_x_p, tsdf_x_n, tsdf_y_p, tsdf_y_n, tsdf_z_p, tsdf_z_n, tsdf_c;
    bool v_x_p, v_x_n, v_y_p, v_y_n, v_z_p, v_z_n, v_c;
    
    get_tsdf_and_valid(x, y, z, tsdf_c, v_c);
    get_tsdf_and_valid(x + 1, y, z, tsdf_x_p, v_x_p);
    get_tsdf_and_valid(x - 1, y, z, tsdf_x_n, v_x_n);
    get_tsdf_and_valid(x, y + 1, z, tsdf_y_p, v_y_p);
    get_tsdf_and_valid(x, y - 1, z, tsdf_y_n, v_y_n);
    get_tsdf_and_valid(x, y, z + 1, tsdf_z_p, v_z_p);
    get_tsdf_and_valid(x, y, z - 1, tsdf_z_n, v_z_n);

    float3 n;
    n.x = (v_x_p && v_x_n) ? (tsdf_x_p - tsdf_x_n) : (v_x_p ? (tsdf_x_p - tsdf_c) : (v_x_n ? (tsdf_c - tsdf_x_n) : 0.0f));
    n.y = (v_y_p && v_y_n) ? (tsdf_y_p - tsdf_y_n) : (v_y_p ? (tsdf_y_p - tsdf_c) : (v_y_n ? (tsdf_c - tsdf_y_n) : 0.0f));
    n.z = (v_z_p && v_z_n) ? (tsdf_z_p - tsdf_z_n) : (v_z_p ? (tsdf_z_p - tsdf_c) : (v_z_n ? (tsdf_c - tsdf_z_n) : 0.0f));

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
    float3* out_v, float3* out_n, uchar3* out_c)
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
    bool inside_surface = false;

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
        
        if (i == 0 && val < 0) {
            inside_surface = true;
        }
        if (inside_surface && val > 0) {
            inside_surface = false;
            prev_val = val;
        }

        if (!inside_surface && prev_val > 0 && val < 0) {
            // Surface found
            float t_fine = t - voxel_size * (val / (val - prev_val));
            float3 pf = make_float3(pos_w.x + ray_w.x * t_fine, pos_w.y + ray_w.y * t_fine, pos_w.z + ray_w.z * t_fine);
            
            // Output WORLD SPACE vertex (consistent with CPU)
            out_v[py * width + px] = pf;
            
            // Compute normal in world space
            float3 n_w = computeNormalGPU(voxels_void, resolution, vx, vy, vz);
            out_n[py * width + px] = n_w;

            if (out_c) {
                VoxelGPU& v = voxels[vz * resolution * resolution + vy * resolution + vx];
                out_c[py * width + px] = make_uchar3((uint8_t)v.r, (uint8_t)v.g, (uint8_t)v.b);
            }
            return;
        }
        prev_val = val;
        t += fmaxf(voxel_size, fabsf(val) * 0.1f); // skipping
        if (t > 5.0f) break;
    }
    out_v[py * width + px] = make_float3(0,0,0);
    out_n[py * width + px] = make_float3(0,0,0);
    if (out_c) out_c[py * width + px] = make_uchar3(0,0,0);
}

void TSDFVolume::raycastGPU(const Eigen::Matrix4f& pose,
                             float fx, float fy, float cx, float cy,
                             int width, int height,
                             float3* d_vertices, float3* d_normals,
                             uchar3* d_colors)
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
        width, height, d_vertices, d_normals, d_colors
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -------------------------------------------------------------------
// Global Point Cloud Extraction (GPU-accelerated)
// -------------------------------------------------------------------

__global__ void classifyVoxelKernel(
    const VoxelGPU* voxels,
    int             resolution,
    uint32_t*       is_valid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution || y >= resolution || z >= resolution) return;

    int idx = z * resolution * resolution + y * resolution + x;
    const VoxelGPU& v = voxels[idx];

    // Same criteria as CPU extractGlobalPointCloud: weight > 1.0 and surface proximity
    if (v.weight > 1.0f && fabsf(v.tsdf) < 0.2f) {
        is_valid[idx] = 1;
    } else {
        is_valid[idx] = 0;
    }
}

__global__ void compactPointsKernel(
    const VoxelGPU* voxels,
    const uint32_t* is_valid,
    const uint32_t* offsets,
    int             resolution,
    float           voxel_size,
    float3          origin,
    float3*         out_points,
    uchar3*         out_colors)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= resolution || y >= resolution || z >= resolution) return;

    int idx = z * resolution * resolution + y * resolution + x;
    if (is_valid[idx]) {
        uint32_t out_idx = offsets[idx];
        out_points[out_idx] = make_float3(
            origin.x + x * voxel_size,
            origin.y + y * voxel_size,
            origin.z + z * voxel_size
        );
        const VoxelGPU& v = voxels[idx];
        out_colors[out_idx] = make_uchar3(
            (uint8_t)fmaxf(0, fminf(255, v.r)),
            (uint8_t)fmaxf(0, fminf(255, v.g)),
            (uint8_t)fmaxf(0, fminf(255, v.b))
        );
    }
}

void TSDFVolume::extractGlobalPointCloudGPU(std::vector<Eigen::Vector3f>& points_out,
                                            std::vector<uint8_t>&         colors_out) const
{
    int res = params_.resolution;
    int total_voxels = res * res * res;

    if (!d_pc_is_valid_) d_pc_is_valid_ = utils::make_cuda_unique<uint32_t>(total_voxels);
    if (!d_pc_offsets_) d_pc_offsets_ = utils::make_cuda_unique<uint32_t>(total_voxels);
    if (!d_pc_out_points_) d_pc_out_points_ = utils::make_cuda_unique<float3>(total_voxels);
    if (!d_pc_out_colors_) d_pc_out_colors_ = utils::make_cuda_unique<uchar3>(total_voxels);

    dim3 block(8, 8, 8);
    dim3 grid((res + block.x - 1) / block.x, (res + block.y - 1) / block.y, (res + block.z - 1) / block.z);

    classifyVoxelKernel<<<grid, block>>>(d_voxels_.get(), res, d_pc_is_valid_.get());
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Use Thrust for exclusive scan to get compaction offsets
    thrust::device_ptr<uint32_t> d_ptr_valid(d_pc_is_valid_.get());
    thrust::device_ptr<uint32_t> d_ptr_offsets(d_pc_offsets_.get());
    thrust::exclusive_scan(d_ptr_valid, d_ptr_valid + total_voxels, d_ptr_offsets);
    
    uint32_t total_points = thrust::reduce(d_ptr_valid, d_ptr_valid + total_voxels);

    if (total_points == 0) return;

    compactPointsKernel<<<grid, block>>>(
        d_voxels_.get(), d_pc_is_valid_.get(), d_pc_offsets_.get(),
        res, params_.voxel_size, 
        make_float3(params_.origin.x(), params_.origin.y(), params_.origin.z()),
        d_pc_out_points_.get(), d_pc_out_colors_.get()
    );
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    points_out.resize(total_points);
    colors_out.resize(total_points * 3);
    
    // Direct copy from device into the correctly sized Eigen/byte vectors
    CUDA_CHECK(cudaMemcpy(points_out.data(), d_pc_out_points_.get(), total_points * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colors_out.data(), d_pc_out_colors_.get(), total_points * sizeof(uchar3), cudaMemcpyDeviceToHost));
}

} // namespace tsdf
} // namespace kfusion

#endif // CUDA_ENABLED
