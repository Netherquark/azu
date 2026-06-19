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
#elif defined(HIP_ENABLED)
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
        if (!d_depth_integ_) {
            d_depth_integ_ = utils::make_cuda_unique<float>(width * height);
            if (rgb) d_rgb_integ_ = utils::make_cuda_unique<uint8_t>(width * height * 3);
        }
        cudaMemcpy(d_depth_integ_.get(), depth_meters, width * height * sizeof(float), cudaMemcpyHostToDevice);
        if (rgb) {
            cudaMemcpy(d_rgb_integ_.get(), rgb, width * height * 3, cudaMemcpyHostToDevice);
        }
        integrateGPU(d_depth_integ_.get(), rgb ? d_rgb_integ_.get() : nullptr, pose, fx, fy, cx, cy, width, height);
    } else {
        integrateCPU(depth_meters, rgb, pose, fx, fy, cx, cy, width, height);
    }
#elif defined(HIP_ENABLED)
    if (gpu_enabled_) {
        if (!d_depth_integ_) {
            d_depth_integ_ = utils::make_hip_unique<float>(width * height);
            if (rgb) d_rgb_integ_ = utils::make_hip_unique<uint8_t>(width * height * 3);
        }
        (void)hipMemcpy(d_depth_integ_.get(), depth_meters, width * height * sizeof(float), hipMemcpyHostToDevice);
        if (rgb) {
            (void)hipMemcpy(d_rgb_integ_.get(), rgb, width * height * 3, hipMemcpyHostToDevice);
        }
        integrateGPU(d_depth_integ_.get(), rgb ? d_rgb_integ_.get() : nullptr, pose, fx, fy, cx, cy, width, height);
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
    const float trunc = params_.truncation;
    const float max_w = params_.max_weight;
    const float vs    = params_.voxel_size;
    const int   res   = params_.resolution;
    const Eigen::Vector3f origin = params_.origin;

    const Eigen::Matrix4f world_to_cam = pose.inverse();
    const Eigen::Matrix3f R_wc = world_to_cam.block<3,3>(0,0);
    const Eigen::Vector3f t_wc = world_to_cam.block<3,1>(0,3);

    const Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    const Eigen::Vector3f t_cw = pose.block<3,1>(0,3);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float d_meas = depth_meters[y * width + x];
            if (d_meas <= 0.1f) continue;

            // Ray direction in camera space
            Eigen::Vector3f ray_cam((x - cx) / fx, (y - cy) / fy, 1.0f);
            float ray_dist_scale = ray_cam.norm(); // Euclidean / Z ratio
            ray_cam.normalize();

            // Convert Z-depth to Euclidean distance along the ray
            float t_meas = d_meas * ray_dist_scale;
            float t_min  = std::max(0.1f, t_meas - trunc);
            float t_max  = t_meas + trunc;

            // Convert to world space
            Eigen::Vector3f ray_w = R_cw * ray_cam;
            Eigen::Vector3f cam_pos_w = t_cw;

            // March through the truncation segment
            for (float t = t_min; t <= t_max; t += vs * 0.75f) {
                Eigen::Vector3f wpos = cam_pos_w + ray_w * t;
                
                // World to voxel index
                Eigen::Vector3f vpos = (wpos - origin) / vs;
                int vx = static_cast<int>(std::floor(vpos.x()));
                int vy = static_cast<int>(std::floor(vpos.y()));
                int vz = static_cast<int>(std::floor(vpos.z()));

                if (vx < 0 || vx >= res || vy < 0 || vy >= res || vz < 0 || vz >= res)
                    continue;

                // Project voxel back to camera plane to get precise Z-depth difference
                Eigen::Vector3f cpos = R_wc * wpos + t_wc;
                float sdf = d_meas - cpos.z();
                
                if (sdf < -trunc) continue;

                float tsdf_new = std::min(1.0f, sdf / trunc);
                if (std::isnan(tsdf_new) || std::isinf(tsdf_new)) continue;
                int vidx = idx(vx, vy, vz);
                Voxel& vox = voxels_[vidx];

                float w_old = vox.weight;
                float w_new = 1.0f;
                float w_sum = std::min(w_old + w_new, max_w);

                float next_tsdf = (vox.tsdf * w_old + tsdf_new * w_new) / (w_old + w_new + 1e-6f);
                if (std::isnan(next_tsdf) || std::isinf(next_tsdf)) continue;

                vox.tsdf   = next_tsdf;
                vox.weight = w_sum;

                if (rgb && sdf > -trunc * 0.5f) {
                    int pidx = (y * width + x) * 3;
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
                         Eigen::Vector3f* normals_out,
                         uint8_t* colors_out) const
{
    std::shared_lock<std::shared_mutex> lk(mutex_);

    const float vs    = params_.voxel_size;
    const Eigen::Vector3f cam_origin = pose.block<3,1>(0,3);
    const Eigen::Matrix3f R_cw       = pose.block<3,3>(0,0);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int out_idx = y * width + x;
            vertices_out[out_idx] = Eigen::Vector3f::Zero();
            normals_out[out_idx]  = Eigen::Vector3f::Zero();
            if (colors_out) {
                colors_out[out_idx*3+0] = 0;
                colors_out[out_idx*3+1] = 0;
                colors_out[out_idx*3+2] = 0;
            }

            Eigen::Vector3f ray_cam((x - cx) / fx, (y - cy) / fy, 1.0f);
            Eigen::Vector3f ray_world = (R_cw * ray_cam).normalized();

            float t = 0.3f; // min depth for Kinect v1
            float prev_tsdf = 1.0f;
            const float trunc = params_.truncation;

            while (t < 5.0f) {
                Eigen::Vector3f p = cam_origin + ray_world * t;
                float tsdf = getTSDF(p);

                if (tsdf < 1.0f) { // Probable geometry region
                    if (prev_tsdf > 0.0f && tsdf <= 0.0f) {
                        // Surface zero-crossing found
                        float t_hit = t - vs * tsdf / (tsdf - prev_tsdf + 1e-6f);
                        Eigen::Vector3f hit_world = cam_origin + ray_world * t_hit;
                        
                        vertices_out[out_idx] = hit_world;
                        normals_out[out_idx]  = computeNormal(hit_world);
                        
                        if (colors_out) {
                            Eigen::Vector3i vi = worldToVoxel(hit_world);
                            if (inBounds(vi.x(), vi.y(), vi.z())) {
                                const Voxel& vox = voxels_[idx(vi.x(), vi.y(), vi.z())];
                                colors_out[out_idx*3+0] = vox.r;
                                colors_out[out_idx*3+1] = vox.g;
                                colors_out[out_idx*3+2] = vox.b;
                            }
                        }
                        break;
                    }
                    prev_tsdf = tsdf;
                    t += vs * 0.5f; // Small steps near surface
                } else {
                    // Unknown or empty space
                    t += vs; 
                    prev_tsdf = 1.0f;
                }
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
    const float vs = params_.voxel_size;
    const int res = params_.resolution;
    
    Eigen::Vector3f v = (world_pos - params_.origin) / vs;
    
    // Trilinear interpolation
    int x0 = static_cast<int>(std::floor(v.x()));
    int y0 = static_cast<int>(std::floor(v.y()));
    int z0 = static_cast<int>(std::floor(v.z()));
    
    if (x0 < 0 || x0 >= res - 1 || y0 < 0 || y0 >= res - 1 || z0 < 0 || z0 >= res - 1) {
        Eigen::Vector3i vi = worldToVoxel(world_pos);
        if (!inBounds(vi.x(), vi.y(), vi.z())) return 1.0f;
        const Voxel& vox = voxels_[idx(vi.x(), vi.y(), vi.z())];
        return (vox.weight > 0.0f) ? vox.tsdf : 1.0f;
    }

    float tx = v.x() - x0;
    float ty = v.y() - y0;
    float tz = v.z() - z0;

    auto getV = [&](int x, int y, int z) {
        const Voxel& vox = voxels_[idx(x, y, z)];
        return (vox.weight > 0.0f) ? vox.tsdf : 1.0f;
    };

    float v000 = getV(x0, y0, z0);
    float v100 = getV(x0+1, y0, z0);
    float v010 = getV(x0, y0+1, z0);
    float v110 = getV(x0+1, y0+1, z0);
    float v001 = getV(x0, y0, z0+1);
    float v101 = getV(x0+1, y0, z0+1);
    float v011 = getV(x0, y0+1, z0+1);
    float v111 = getV(x0+1, y0+1, z0+1);

    float v00 = v000 * (1 - tx) + v100 * tx;
    float v01 = v001 * (1 - tx) + v101 * tx;
    float v10 = v010 * (1 - tx) + v110 * tx;
    float v11 = v011 * (1 - tx) + v111 * tx;

    float v0 = v00 * (1 - ty) + v10 * ty;
    float v1 = v01 * (1 - ty) + v11 * ty;

    return v0 * (1 - tz) + v1 * tz;
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

void TSDFVolume::extractGlobalPointCloud(std::vector<Eigen::Vector3f>& points_out,
                                         std::vector<uint8_t>&         colors_out) const
{
#ifdef CUDA_ENABLED
    if (gpu_enabled_) {
        extractGlobalPointCloudGPU(points_out, colors_out);
        return;
    }
#elif defined(HIP_ENABLED)
    if (gpu_enabled_) {
        extractGlobalPointCloudGPU(points_out, colors_out);
        return;
    }
#endif
    std::shared_lock<std::shared_mutex> lk(mutex_);
    const int res = params_.resolution;
    
    points_out.clear();
    colors_out.clear();

    // Estimate capacity to avoid excessive reallocations (~5% of volume)
    points_out.reserve(voxels_.size() / 20);
    colors_out.reserve(voxels_.size() / 20 * 3);

    #pragma omp parallel
    {
        std::vector<Eigen::Vector3f> local_points;
        std::vector<uint8_t>         local_colors;

        #pragma omp for nowait
        for (int z = 0; z < res; ++z) {
            for (int y = 0; y < res; ++y) {
                for (int x = 0; x < res; ++x) {
                    const Voxel& vox = voxels_[idx(x, y, z)];
                    // Only extract voxels near the surface (low absolute TSDF) with significant weight
                    if (vox.weight > 1.0f && std::abs(vox.tsdf) < 0.2f) {
                        local_points.push_back(voxelToWorld(x, y, z));
                        local_colors.push_back(vox.r);
                        local_colors.push_back(vox.g);
                        local_colors.push_back(vox.b);
                    }
                }
            }
        }

        #pragma omp critical
        {
            points_out.insert(points_out.end(), local_points.begin(), local_points.end());
            colors_out.insert(colors_out.end(), local_colors.begin(), local_colors.end());
        }
    }
}

} // namespace tsdf
} // namespace kfusion
