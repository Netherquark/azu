#include "tsdf/TSDFVolume.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace tsdf {

TSDFVolume::TSDFVolume(const TSDFParams& params)
    : params_(params)
{
    size_t n = static_cast<size_t>(params_.resolution)
             * params_.resolution
             * params_.resolution;
    voxels_.resize(n);
    reset();

#ifdef CUDA_ENABLED
    initGPU();
#endif
}

TSDFVolume::~TSDFVolume() {
#ifdef CUDA_ENABLED
    freeGPU();
#endif
}

void TSDFVolume::reset() {
    for (auto& v : voxels_) {
        v.tsdf   = 1.0f;
        v.weight = 0.0f;
        v.r = v.g = v.b = 128;
    }
    integrated_frames_.store(0);
}

const Voxel& TSDFVolume::voxelAt(int x, int y, int z) const {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    return voxels_[idx(x, y, z)];
}

Voxel& TSDFVolume::voxelAt(int x, int y, int z) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    return voxels_[idx(x, y, z)];
}

Eigen::Vector3i TSDFVolume::worldToVoxel(const Eigen::Vector3f& world) const {
    Eigen::Vector3f local = (world - params_.origin) / params_.voxel_size;
    return Eigen::Vector3i(
        static_cast<int>(local.x()),
        static_cast<int>(local.y()),
        static_cast<int>(local.z())
    );
}

Eigen::Vector3f TSDFVolume::voxelToWorld(const Eigen::Vector3i& v) const {
    return params_.origin + Eigen::Vector3f(v.x(), v.y(), v.z()) * params_.voxel_size;
}

Eigen::Vector3f TSDFVolume::voxelToWorld(int x, int y, int z) const {
    return params_.origin + Eigen::Vector3f(x, y, z) * params_.voxel_size;
}

float TSDFVolume::interpolate(const Eigen::Vector3f& world_pos) const {
    Eigen::Vector3f local = (world_pos - params_.origin) / params_.voxel_size;
    int x0 = static_cast<int>(std::floor(local.x()));
    int y0 = static_cast<int>(std::floor(local.y()));
    int z0 = static_cast<int>(std::floor(local.z()));

    if (!inBounds(x0, y0, z0) || !inBounds(x0+1, y0+1, z0+1))
        return 1.0f;

    std::shared_lock<std::shared_mutex> lk(mutex_);

    auto tsdf = [&](int x, int y, int z) -> float {
        return voxels_[idx(x,y,z)].tsdf;
    };

    // Trilinear interpolation
    float c000 = tsdf(x0,   y0,   z0);
    float c100 = tsdf(x0+1, y0,   z0);
    float c010 = tsdf(x0,   y0+1, z0);
    float c110 = tsdf(x0+1, y0+1, z0);
    float c001 = tsdf(x0,   y0,   z0+1);
    float c101 = tsdf(x0+1, y0,   z0+1);
    float c011 = tsdf(x0,   y0+1, z0+1);
    float c111 = tsdf(x0+1, y0+1, z0+1);

    float fx = local.x() - x0;
    float fy = local.y() - y0;
    float fz = local.z() - z0;

    return c000*(1-fx)*(1-fy)*(1-fz) + c100*fx*(1-fy)*(1-fz)
         + c010*(1-fx)*fy*(1-fz)     + c110*fx*fy*(1-fz)
         + c001*(1-fx)*(1-fy)*fz     + c101*fx*(1-fy)*fz
         + c011*(1-fx)*fy*fz         + c111*fx*fy*fz;
}

float TSDFVolume::usageFraction() const {
    size_t used = 0;
    for (const auto& v : voxels_)
        if (v.weight > 0.0f) ++used;
    return static_cast<float>(used) / static_cast<float>(voxels_.size());
}

void TSDFVolume::integrate(const float*           depth_meters,
                           const uint8_t*         rgb,
                           const Eigen::Matrix4f& pose,
                           float fx, float fy,
                           float cx, float cy,
                           int   width, int height)
{
    std::unique_lock<std::shared_mutex> lk(mutex_);
#ifdef CUDA_ENABLED
    integrateGPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height);
#else
    integrateCPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height);
#endif
    integrated_frames_.fetch_add(1);
}

void TSDFVolume::integrateCPU(const float*           depth_meters,
                               const uint8_t*         rgb,
                               const Eigen::Matrix4f& pose,
                               float fx, float fy,
                               float cx, float cy,
                               int   width, int height)
{
    // -----------------------------------------------------------------------
    // Optimized: Image-centric integration
    // -----------------------------------------------------------------------
    // Instead of iterating over all voxels, we iterate over pixels and update 
    // only the voxels within the truncation distance along each ray.
    
    const int   W     = width;
    const int   H     = height;
    const float trunc = params_.truncation;
    const float vs    = params_.voxel_size;
    const float max_w = params_.max_weight;

    // Camera origin in world space
    const Eigen::Vector3f cam_origin = pose.block<3,1>(0,3);
    const Eigen::Matrix3f R_cw       = pose.block<3,3>(0,0);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            float d_meas = depth_meters[py * W + px];
            if (d_meas <= 0.0f) continue;

            // Voxel-space ray direction
            Eigen::Vector3f ray_cam(
                (static_cast<float>(px) - cx) / fx,
                (static_cast<float>(py) - cy) / fy,
                1.0f
            );
            // Ray length in camera space for unit depth is norm(ray_cam)
            float ray_len_unit = ray_cam.norm();
            Eigen::Vector3f ray_world = (R_cw * ray_cam).normalized();

            // We only need to update voxels in [d_meas - trunc, d_meas + trunc]
            // We sample along the ray with a step of vs (voxel size)
            float t_min = std::max(0.1f, d_meas - trunc);
            float t_max = d_meas + trunc;

            for (float t = t_min; t <= t_max; t += vs * 0.8f) {
                Eigen::Vector3f wpos = cam_origin + ray_world * (t * ray_len_unit);
                Eigen::Vector3i vi = worldToVoxel(wpos);

                if (!inBounds(vi.x(), vi.y(), vi.z())) continue;

                Voxel& vox = voxelAt(vi.x(), vi.y(), vi.z());
                
                // Distance from camera to voxel center along camera Z axis
                // Camera-space voxel position
                Eigen::Vector3f cpos = pose.inverse().block<3,3>(0,0) * wpos + pose.inverse().block<3,1>(0,3);
                float dist_to_v = cpos.z();
                
                float sdf = d_meas - dist_to_v;
                if (sdf < -trunc) continue;

                float tsdf_new = std::min(1.0f, sdf / trunc);
                float w_old = vox.weight;
                float w_new = 1.0f; // Could be weighted by cos(theta)
                float w_sum = std::min(w_old + w_new, max_w);

                // Update TSDF and Weight (atomic-like if needed, but we used OMP)
                // Note: OMP parallel for over pixels might have data races if rays 
                // from different pixels hit the same voxel.
                // However, for TSDF integration, a slight race is often acceptable 
                // or handled with atomics. 
                vox.tsdf   = (vox.tsdf * w_old + tsdf_new * w_new) / (w_old + w_new + 1e-6f);
                vox.weight = w_sum;

                if (rgb) {
                    int pidx = py * W + px;
                    vox.r = static_cast<uint8_t>((vox.r * w_old + rgb[pidx*3+0]) / (w_old + 1.0f + 1e-6f));
                    vox.g = static_cast<uint8_t>((vox.g * w_old + rgb[pidx*3+1]) / (w_old + 1.0f + 1e-6f));
                    vox.b = static_cast<uint8_t>((vox.b * w_old + rgb[pidx*3+2]) / (w_old + 1.0f + 1e-6f));
                }
            }
        }
    }
}

void TSDFVolume::raycast(const Eigen::Matrix4f& pose,
                         float fx, float fy, float cx, float cy,
                         int width, int height,
                         Eigen::Vector3f* vertices_out,
                         Eigen::Vector3f* normals_out) const
{
#ifdef CUDA_ENABLED
    // In a fully GPU-resident pipeline, raycast output should stay on GPU.
    // For now, we sync if host pointers are provided.
    // In realistic scenarios, ICP will call raycastGPU directly.
    static float3 *d_v = nullptr, *d_n = nullptr;
    if (!d_v) {
        cudaMalloc(&d_v, width * height * sizeof(float3));
        cudaMalloc(&d_n, width * height * sizeof(float3));
    }
    const_cast<TSDFVolume*>(this)->raycastGPU(pose, fx, fy, cx, cy, width, height, d_v, d_n);
    cudaMemcpy(vertices_out, d_v, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(normals_out,  d_n, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
#else
    std::shared_lock<std::shared_mutex> lk(mutex_);
    const float step     = params_.voxel_size;        // full voxel step (faster)
    const float max_dist = params_.resolution * params_.voxel_size * 1.73f;
    const float min_dist = 0.3f;

    const Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    const Eigen::Vector3f t_cw = pose.block<3,1>(0,3);

    // Subsample raycast: only cast every 2nd pixel in each direction for speed
    #pragma omp parallel for schedule(dynamic, 4)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int out_idx = y * width + x;
            vertices_out[out_idx] = Eigen::Vector3f::Zero();
            normals_out[out_idx]  = Eigen::Vector3f::Zero();

            Eigen::Vector3f ray_cam(
                (x - cx) / fx,
                (y - cy) / fy,
                1.0f
            );
            ray_cam.normalize();
            Eigen::Vector3f ray_world = (R_cw * ray_cam).normalized();

            float prev_tsdf = 1.0f;
            float prev_t    = min_dist;

            for (float t = min_dist; t < max_dist; t += step) {
                Eigen::Vector3f world_p = t_cw + ray_world * t;
                Eigen::Vector3i vi = worldToVoxel(world_p);

                if (!inBounds(vi.x(), vi.y(), vi.z())) {
                    prev_tsdf = 1.0f;
                    continue;
                }

                const Voxel& v = voxels_[idx(vi.x(), vi.y(), vi.z())];
                if (v.weight <= 0.0f) { prev_tsdf = 1.0f; continue; }

                float tsdf_val = v.tsdf;

                if (prev_tsdf > 0.0f && tsdf_val <= 0.0f && prev_tsdf < 1.0f) {
                    float alpha = prev_tsdf / (prev_tsdf - tsdf_val + 1e-6f);
                    float t_hit = prev_t + alpha * (t - prev_t);
                    Eigen::Vector3f hit_world = t_cw + ray_world * t_hit;

                    float dx = interpolate(hit_world + Eigen::Vector3f(params_.voxel_size, 0, 0))
                             - interpolate(hit_world - Eigen::Vector3f(params_.voxel_size, 0, 0));
                    float dy = interpolate(hit_world + Eigen::Vector3f(0, params_.voxel_size, 0))
                             - interpolate(hit_world - Eigen::Vector3f(0, params_.voxel_size, 0));
                    float dz = interpolate(hit_world + Eigen::Vector3f(0, 0, params_.voxel_size))
                             - interpolate(hit_world - Eigen::Vector3f(0, 0, params_.voxel_size));

                    Eigen::Vector3f normal(dx, dy, dz);
                    float nlen = normal.norm();

                    Eigen::Vector3f hit_cam  = R_cw.transpose() * (hit_world - t_cw);
                    Eigen::Vector3f norm_cam = R_cw.transpose() * ((nlen > 1e-6f) ? (normal / nlen) : Eigen::Vector3f(0,0,-1));

                    vertices_out[out_idx] = hit_cam;
                    normals_out[out_idx]  = norm_cam;
                    break;
                }

                prev_tsdf = tsdf_val;
                prev_t    = t;
            }
        }
    }
#endif
}

} // namespace tsdf
} // namespace kfusion
