#include "tsdf/TSDFVolume.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

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
    return voxels_[idx(x, y, z)];
}

Voxel& TSDFVolume::voxelAt(int x, int y, int z) {
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

    float fx = local.x() - x0;
    float fy = local.y() - y0;
    float fz = local.z() - z0;

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
    const int   RES   = params_.resolution;
    const float trunc = params_.truncation;
    const float max_w = params_.max_weight;
    const float vs    = params_.voxel_size;

    // World-to-camera transform
    const Eigen::Matrix4f pose_inv = pose.inverse();
    const Eigen::Matrix3f R_wc = pose_inv.block<3,3>(0,0);
    const Eigen::Vector3f t_wc = pose_inv.block<3,1>(0,3);

    // Iterate over image pixels — project each voxel column only once
    // Use slice-parallel approach: parallelize over Z slices
    #pragma omp parallel for schedule(dynamic, 2)
    for (int z = 0; z < RES; ++z) {
        for (int y = 0; y < RES; ++y) {
            for (int x = 0; x < RES; ++x) {
                // World position of voxel center
                Eigen::Vector3f wpos = voxelToWorld(x, y, z);
                wpos += Eigen::Vector3f(vs*0.5f, vs*0.5f, vs*0.5f);

                // Camera space
                Eigen::Vector3f cpos = R_wc * wpos + t_wc;
                if (cpos.z() <= 0.1f) continue;

                // Project
                float px = fx * cpos.x() / cpos.z() + cx;
                float py = fy * cpos.y() / cpos.z() + cy;
                int ix = static_cast<int>(px + 0.5f);
                int iy = static_cast<int>(py + 0.5f);
                if (ix < 0 || ix >= width || iy < 0 || iy >= height) continue;

                float d_meas = depth_meters[iy * width + ix];
                if (d_meas <= 0.0f) continue;

                float sdf = d_meas - cpos.z();
                if (sdf < -trunc) continue;

                float tsdf_new = std::min(1.0f, sdf / trunc);

                Voxel& vox = voxelAt(x, y, z);
                float  w_old = vox.weight;
                float  w_new = 1.0f;
                float  w_sum = std::min(w_old + w_new, max_w);

                vox.tsdf   = (vox.tsdf * w_old + tsdf_new * w_new) / (w_old + w_new + 1e-6f);
                vox.weight = w_sum;

                if (rgb) {
                    int pidx = iy * width + ix;
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
}

} // namespace tsdf
} // namespace kfusion
