#include "tsdf/TSDFVolume.hpp"
#include "utils/Math.hpp"
#include "utils/Logger.hpp"
#include "meshing/MarchingCubes.hpp"
#include <omp.h>
#include <algorithm>

namespace kf {

TSDFVolume::TSDFVolume(const Config& config) : config_(config) {
    data_.resize(TOTAL_VOXELS);
    std::fill(data_.begin(), data_.end(), TSDFVoxel());

    // Volume is centered at origin, extends in both positive and negative directions
    float total_size = RESOLUTION * config_.voxel_size;
    volume_center_ = Vector3f::Zero();
    volume_extent_ = total_size / 2.0f;

    KF_LOG_INFO("Initialized TSDF Volume: ", RESOLUTION, "^3 voxels, ",
               config_.voxel_size, "m per voxel, ", total_size, "m^3 total");
}

TSDFVolume::~TSDFVolume() = default;

Vector3i TSDFVolume::world_to_voxel(const Vector3f& world_pos) const {
    Vector3i idx;

    idx.x() =
        static_cast<int>((world_pos.x() - (volume_center_.x() - volume_extent_)) /
                        config_.voxel_size);
    idx.y() =
        static_cast<int>((world_pos.y() - (volume_center_.y() - volume_extent_)) /
                        config_.voxel_size);
    idx.z() =
        static_cast<int>((world_pos.z() - (volume_center_.z() - volume_extent_)) /
                        config_.voxel_size);

    return idx;
}

Vector3f TSDFVolume::voxel_to_world(int x, int y, int z) const {
    Vector3f world;
    world.x() = (volume_center_.x() - volume_extent_) + x * config_.voxel_size;
    world.y() = (volume_center_.y() - volume_extent_) + y * config_.voxel_size;
    world.z() = (volume_center_.z() - volume_extent_) + z * config_.voxel_size;
    return world;
}

Vector3f TSDFVolume::voxel_to_world(const Vector3i& idx) const {
    return voxel_to_world(idx.x(), idx.y(), idx.z());
}

bool TSDFVolume::is_valid_index(int x, int y, int z) const {
    return x >= 0 && x < RESOLUTION && y >= 0 && y < RESOLUTION && z >= 0 &&
           z < RESOLUTION;
}

bool TSDFVolume::is_valid_index(const Vector3i& idx) const {
    return is_valid_index(idx.x(), idx.y(), idx.z());
}

TSDFVoxel& TSDFVolume::voxel(int x, int y, int z) {
    return data_[get_linear_index(x, y, z)];
}

const TSDFVoxel& TSDFVolume::voxel(int x, int y, int z) const {
    return data_[get_linear_index(x, y, z)];
}

TSDFVoxel& TSDFVolume::voxel(const Vector3i& idx) {
    return voxel(idx.x(), idx.y(), idx.z());
}

const TSDFVoxel& TSDFVolume::voxel(const Vector3i& idx) const {
    return voxel(idx.x(), idx.y(), idx.z());
}

void TSDFVolume::reset() {
    std::fill(data_.begin(), data_.end(), TSDFVoxel());
    KF_LOG_INFO("TSDF Volume reset");
}

float TSDFVolume::get_voxel_grid_usage() const {
    int active = 0;
#pragma omp parallel for reduction(+ : active)
    for (size_t i = 0; i < data_.size(); ++i) {
        if (data_[i].weight > 0.1f) active++;
    }
    return 100.0f * active / TOTAL_VOXELS;
}

bool TSDFVolume::interpolate_trilinear(const Vector3f& world_pos,
                                     float& tsdf_val, float& weight) {
    Vector3f pos_voxel =
        (world_pos - (volume_center_ - Vector3f::Ones() * volume_extent_)) /
        config_.voxel_size;

    int x0 = static_cast<int>(pos_voxel.x());
    int y0 = static_cast<int>(pos_voxel.y());
    int z0 = static_cast<int>(pos_voxel.z());

    if (!is_valid_index(x0, y0, z0) || !is_valid_index(x0 + 1, y0 + 1, z0 + 1)) {
        return false;
    }

    float fx = pos_voxel.x() - x0;
    float fy = pos_voxel.y() - y0;
    float fz = pos_voxel.z() - z0;

    // Trilinear interpolation
    float v000 = voxel(x0, y0, z0).tsdf;
    float v001 = voxel(x0, y0, z0 + 1).tsdf;
    float v010 = voxel(x0, y0 + 1, z0).tsdf;
    float v011 = voxel(x0, y0 + 1, z0 + 1).tsdf;
    float v100 = voxel(x0 + 1, y0, z0).tsdf;
    float v101 = voxel(x0 + 1, y0, z0 + 1).tsdf;
    float v110 = voxel(x0 + 1, y0 + 1, z0).tsdf;
    float v111 = voxel(x0 + 1, y0 + 1, z0 + 1).tsdf;

    tsdf_val = (1 - fx) * (1 - fy) * (1 - fz) * v000 +
              (1 - fx) * (1 - fy) * fz * v001 +
              (1 - fx) * fy * (1 - fz) * v010 +
              (1 - fx) * fy * fz * v011 + fx * (1 - fy) * (1 - fz) * v100 +
              fx * (1 - fy) * fz * v101 + fx * fy * (1 - fz) * v110 +
              fx * fy * fz * v111;

    float w000 = voxel(x0, y0, z0).weight;
    float w111 = voxel(x0 + 1, y0 + 1, z0 + 1).weight;
    weight =
        (1 - fx) * (1 - fy) * (1 - fz) * w000 + fx * fy * fz * w111;

    return true;
}

void TSDFVolume::integrate(const DepthFrame& depth, const CameraPose& pose,
                          const ColorFrame* color) {
#ifdef USE_CUDA
    integrate_frame_cuda(depth, pose, color);
#else
    integrate_frame(depth, pose, color);
#endif
}

void TSDFVolume::integrate_frame(const DepthFrame& depth, const CameraPose& pose,
                                const ColorFrame* color) {
    // Transform from camera to world
    CameraPose pose_inv = pose.inverse();

    // Iterate over voxels in world space
    // For efficiency, only process voxels potentially visible in the depth frame

#pragma omp parallel for collapse(3)
    for (int z = 0; z < RESOLUTION; ++z) {
        for (int y = 0; y < RESOLUTION; ++y) {
            for (int x = 0; x < RESOLUTION; ++x) {
                // World position of voxel center
                Vector3f world_pos = voxel_to_world(x, y, z);

                // Transform to camera coordinates
                Vector3f cam_pos = pose_inv.transform_point(world_pos);

                // Check if behind camera
                if (cam_pos.z() < 0.1f) continue;

                // Project to depth image
                Vector2f proj = math::project_3d_to_2d(cam_pos, depth.fx,
                                                      depth.fy, depth.cx,
                                                      depth.cy);

                int u = static_cast<int>(proj.x());
                int v = static_cast<int>(proj.y());

                // Check bounds
                if (u < 0 || u >= DepthFrame::WIDTH || v < 0 ||
                   v >= DepthFrame::HEIGHT) {
                    continue;
                }

                // Get depth from frame
                float measured_depth = depth.get_depth_m(u, v);

                if (measured_depth <= 0.0f) continue;

                // Compute signed distance
                float dist = measured_depth - cam_pos.z();

                // Truncate
                if (dist < -config_.truncation_distance) continue;

                float tsdf =
                    std::min(1.0f, dist / config_.truncation_distance);

                // Update voxel (weighted average)
                TSDFVoxel& voxel_ref = voxel(x, y, z);
                float w = voxel_ref.weight;
                float w_new = std::min(w + 1.0f, config_.max_weight);

                voxel_ref.tsdf = (voxel_ref.tsdf * w + tsdf) / w_new;
                voxel_ref.weight = w_new;

                // Update color if provided
                if (color) {
                    uint8_t r, g, b;
                    color->get_pixel(u, v, r, g, b);
                    voxel_ref.r = static_cast<uint8_t>((voxel_ref.r * w + r) / w_new);
                    voxel_ref.g = static_cast<uint8_t>((voxel_ref.g * w + g) / w_new);
                    voxel_ref.b = static_cast<uint8_t>((voxel_ref.b * w + b) / w_new);
                }
            }
        }
    }
}

void TSDFVolume::integrate_frame_cuda(const DepthFrame& depth,
                                     const CameraPose& pose,
                                     const ColorFrame* color) {
    // CPU fallback for now - CUDA kernel would optimize this
    integrate_frame(depth, pose, color);
}

void TSDFVolume::raycast(std::vector<Vector3f>& vertices,
                        std::vector<Vector3f>& normals,
                        std::vector<uint8_t>& colors) {
    vertices.clear();
    normals.clear();
    colors.clear();

    // Raycast from each pixel
    // For each ray, find the zero-crossing of the TSDF (surface)
    // This is a simplified raycaster - step along each ray

#pragma omp parallel for collapse(2)
    for (int vy = 0; vy < DepthFrame::HEIGHT; ++vy) {
        for (int vx = 0; vx < DepthFrame::WIDTH; ++vx) {
            // Unproject pixel to camera ray
            Vector3f ray_dir = math::safe_normalize(Vector3f(
                (vx - DepthFrame::WIDTH / 2.0f) / (DepthFrame::WIDTH / 2.0f),
                (vy - DepthFrame::HEIGHT / 2.0f) / (DepthFrame::HEIGHT / 2.0f),
                1.0f));

            // Raycast along direction
            Vector3f ray_pos = Vector3f::Zero();
            float step_size = config_.voxel_size * 2.0f;
            float max_dist = 5.0f;  // Max 5m

            float last_tsdf = 1.0f;
            float last_weight = 0.0f;

            for (float t = 0.0f; t < max_dist; t += step_size) {
                Vector3f world_pos = ray_pos + ray_dir * t;
                float tsdf_val, weight;

                if (!interpolate_trilinear(world_pos, tsdf_val, weight)) {
                    continue;
                }

                // Zero crossing detection
                if (last_tsdf > 0 && tsdf_val < 0) {
                    // Found surface, interpolate position
                    float alpha = last_tsdf / (last_tsdf - tsdf_val);
                    Vector3f p1 = ray_pos + ray_dir * (t - step_size);
                    Vector3f p2 = ray_pos + ray_dir * t;
                    // Linear interpolation: p = (1 - alpha) * p1 + alpha * p2
                    Vector3f surface_pos = (1.0f - alpha) * p1 + alpha * p2;

                    // Compute normal via finite differences
                    float eps = config_.voxel_size / 2.0f;
                    float tsdf_dx, w_dx;
                    float tsdf_dy, w_dy;
                    float tsdf_dz, w_dz;

                    interpolate_trilinear(
                        Vector3f(surface_pos.x() + eps, surface_pos.y(),
                                surface_pos.z()),
                        tsdf_dx, w_dx);
                    interpolate_trilinear(
                        Vector3f(surface_pos.x(), surface_pos.y() + eps,
                                surface_pos.z()),
                        tsdf_dy, w_dy);
                    interpolate_trilinear(
                        Vector3f(surface_pos.x(), surface_pos.y(),
                                surface_pos.z() + eps),
                        tsdf_dz, w_dz);

                    interpolate_trilinear(
                        Vector3f(surface_pos.x() - eps, surface_pos.y(),
                                surface_pos.z()),
                        tsdf_dx, w_dx);
                    interpolate_trilinear(
                        Vector3f(surface_pos.x(), surface_pos.y() - eps,
                                surface_pos.z()),
                        tsdf_dy, w_dy);
                    interpolate_trilinear(
                        Vector3f(surface_pos.x(), surface_pos.y(),
                                surface_pos.z() - eps),
                        tsdf_dz, w_dz);

                    Vector3f normal =
                        math::safe_normalize(Vector3f(tsdf_dx, tsdf_dy, tsdf_dz));

#pragma omp critical
                    {
                        vertices.push_back(surface_pos);
                        normals.push_back(normal);
                        colors.push_back(255);
                        colors.push_back(255);
                        colors.push_back(255);
                    }

                    break;  // Only one surface per ray
                }

                last_tsdf = tsdf_val;
                last_weight = weight;
            }
        }
    }
}

MeshPtr TSDFVolume::extract_mesh() {
    // Use Marching Cubes to extract mesh from TSDF
    MarchingCubes mc(*this);
    return mc.extract();
}

}  // namespace kf
