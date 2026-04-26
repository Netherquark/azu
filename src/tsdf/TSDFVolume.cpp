#include "tsdf/TSDFVolume.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <mutex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace tsdf {

TSDFVolume::TSDFVolume(const TSDFParams& params)
    : params_(params)
{
    size_t total = static_cast<size_t>(params.resolution) * params.resolution * params.resolution;
    voxels_.resize(total);
    reset();
}

TSDFVolume::~TSDFVolume() {
#ifdef CUDA_ENABLED
    freeGPU();
#endif
}

void TSDFVolume::reset() {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    std::fill(voxels_.begin(), voxels_.end(), Voxel{0.0f, 0.0f, 128, 128, 128});
    integrated_frames_.store(0);
}

void TSDFVolume::setParams(const TSDFParams& p) {
    std::unique_lock<std::shared_mutex> lk(mutex_);
    bool resized = (p.resolution != params_.resolution);
    params_ = p;
    if (resized) {
        size_t total = static_cast<size_t>(params_.resolution) * params_.resolution * params_.resolution;
        voxels_.resize(total);
        reset();
    }
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
    if (gpu_enabled_) {
        integrateGPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height);
    } else {
        integrateCPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height);
    }
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
    const int   W     = width;
    const int   H     = height;
    const float trunc = params_.truncation;
    const float max_w = params_.max_weight;

    const Eigen::Matrix4f world_to_cam = pose.inverse();
    const Eigen::Matrix3f R_rc = world_to_cam.block<3,3>(0,0);
    const Eigen::Vector3f t_rc = world_to_cam.block<3,1>(0,3);

    const int res = params_.resolution;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < res; ++y) {
        for (int z = 0; z < res; ++z) {
            for (int x = 0; x < res; ++x) {
                const int vidx = idx(x, y, z);
                Voxel& vox = voxels_[vidx];

                Eigen::Vector3f wpos = voxelToWorld(Eigen::Vector3i(x, y, z));
                Eigen::Vector3f cpos = R_rc * wpos + t_rc;
                float v_depth = cpos.z();

                if (v_depth <= 0.1f) continue;

                float inv_z = 1.0f / v_depth;
                int px = static_cast<int>(fx * cpos.x() * inv_z + cx + 0.5f);
                int py = static_cast<int>(fy * cpos.y() * inv_z + cy + 0.5f);

                if (px < 0 || px >= W || py < 0 || py >= H) continue;

                float d_meas = depth_meters[py * W + px];
                if (d_meas <= 0.0f) continue;

                float sdf = d_meas - v_depth;
                if (sdf < -trunc) continue;

                float tsdf_new = std::min(1.0f, sdf / trunc);
                float w_old = vox.weight;
                float w_new = 1.0f;
                float w_sum = std::min(w_old + w_new, max_w);

                vox.tsdf   = (vox.tsdf * w_old + tsdf_new * w_new) / (w_old + w_new + 1e-6f);
                vox.weight = w_sum;

                if (rgb && sdf > -trunc * 0.2f) { // Strict check to prevent backside smearing
                    int pidx = (py * W + px) * 3;
                    vox.r = static_cast<uint8_t>((vox.r * w_old + rgb[pidx+0]) / (w_old + 1.0f + 1e-6f));
                    vox.g = static_cast<uint8_t>((vox.g * w_old + rgb[pidx+1]) / (w_old + 1.0f + 1e-6f));
                    vox.b = static_cast<uint8_t>((vox.b * w_old + rgb[pidx+2]) / (w_old + 1.0f + 1e-6f));
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
    std::shared_lock<std::shared_mutex> lk(mutex_);
#ifdef CUDA_ENABLED
    if (gpu_enabled_) {
        // raycastGPU should have a similar signature or be updated
        // For now, we fall back to CPU if we need both vertices and normals out 
        // unless raycastGPU is ready for it.
    }
#endif

    const float vs    = params_.voxel_size;
    const Eigen::Vector3f cam_origin = pose.block<3,1>(0,3);
    const Eigen::Matrix3f R_cw       = pose.block<3,3>(0,0);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int out_idx = y * width + x;
            vertices_out[out_idx] = Eigen::Vector3f::Zero();
            normals_out[out_idx]  = Eigen::Vector3f::Zero();

            Eigen::Vector3f ray_cam((x - cx) / fx, (y - cy) / fy, 1.0f);
            Eigen::Vector3f ray_world = (R_cw * ray_cam).normalized();

            float t = 0.1f;
            float prev_tsdf = 1.0f;
            bool hit = false;

            while (t < 5.0f) {
                Eigen::Vector3f p = cam_origin + ray_world * t;
                Eigen::Vector3i vi = worldToVoxel(p);

                if (inBounds(vi.x(), vi.y(), vi.z())) {
                    const Voxel& vox = voxels_[idx(vi.x(), vi.y(), vi.z())];
                    if (vox.weight > 0.0f) {
                        float tsdf = vox.tsdf;
                        if (prev_tsdf > 0.0f && tsdf <= 0.0f) {
                            float t_hit = t - (0.5f * vs) * tsdf / (tsdf - prev_tsdf + 1e-6f);
                            Eigen::Vector3f hit_world = cam_origin + ray_world * t_hit;
                            vertices_out[out_idx] = hit_world;
                            normals_out[out_idx]  = computeNormal(hit_world);
                            hit = true;
                            break;
                        }
                        prev_tsdf = tsdf;
                    } else {
                        prev_tsdf = 1.0f;
                    }
                } else {
                    prev_tsdf = 1.0f;
                }
                t += vs * 0.5f;
            }
        }
    }
}

Eigen::Vector3f TSDFVolume::computeNormal(const Eigen::Vector3f& world_pos) const {
    const float vs = params_.voxel_size;
    Eigen::Vector3f n;
    n.x() = (getTSDF(world_pos + Eigen::Vector3f(vs, 0, 0)) - getTSDF(world_pos - Eigen::Vector3f(vs, 0, 0)));
    n.y() = (getTSDF(world_pos + Eigen::Vector3f(0, vs, 0)) - getTSDF(world_pos - Eigen::Vector3f(0, vs, 0)));
    n.z() = (getTSDF(world_pos + Eigen::Vector3f(0, 0, vs)) - getTSDF(world_pos - Eigen::Vector3f(0, 0, vs)));
    float len = n.norm();
    if (len > 1e-6f) return n / len;
    return Eigen::Vector3f::Zero();
}

float TSDFVolume::getTSDF(const Eigen::Vector3f& world_pos) const {
    Eigen::Vector3i vi = worldToVoxel(world_pos);
    if (!inBounds(vi.x(), vi.y(), vi.z())) return 1.0f;
    const Voxel& v = voxels_[idx(vi.x(), vi.y(), vi.z())];
    return (v.weight > 0.0f) ? v.tsdf : 1.0f;
}

const Voxel& TSDFVolume::voxelAt(int x, int y, int z) const {
    return voxels_[idx(x, y, z)];
}

Voxel& TSDFVolume::voxelAt(int x, int y, int z) {
    return voxels_[idx(x, y, z)];
}

Eigen::Vector3i TSDFVolume::worldToVoxel(const Eigen::Vector3f& world) const {
    Eigen::Vector3f v = (world - params_.origin) / params_.voxel_size;
    return Eigen::Vector3i(static_cast<int>(v.x()), static_cast<int>(v.y()), static_cast<int>(v.z()));
}

Eigen::Vector3f TSDFVolume::voxelToWorld(const Eigen::Vector3i& v) const {
    return voxelToWorld(v.x(), v.y(), v.z());
}

Eigen::Vector3f TSDFVolume::voxelToWorld(int x, int y, int z) const {
    return params_.origin + Eigen::Vector3f(x, y, z) * params_.voxel_size;
}

float TSDFVolume::usageFraction() const {
    size_t count = 0;
    for (const auto& v : voxels_) {
        if (v.weight > 0.0f) count++;
    }
    return static_cast<float>(count) / voxels_.size();
}

} // namespace tsdf
} // namespace kfusion
