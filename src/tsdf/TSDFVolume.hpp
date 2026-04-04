#pragma once

#include "utils/Types.hpp"
#include <vector>
#include <memory>

namespace kf {

// ============================================================================
// TSDF Volume - 512^3 Truncated Signed Distance Function
// ============================================================================

class TSDFVolume {
public:
    static constexpr int RESOLUTION = 512;
    static constexpr int TOTAL_VOXELS = RESOLUTION * RESOLUTION * RESOLUTION;

    struct Config {
        float voxel_size = 0.005f;           // Size of each voxel in meters
        float truncation_distance = 0.01f;   // Truncation distance
        float max_weight = 255.0f;           // Maximum integration weight
    };

    explicit TSDFVolume(const Config& config = Config());
    ~TSDFVolume();

    // Integration: add depth frame to volume
    void integrate(const DepthFrame& depth, const CameraPose& pose,
                   const ColorFrame* color = nullptr);

    // Raycasting: extract surface points
    void raycast(std::vector<Vector3f>& vertices,
                std::vector<Vector3f>& normals,
                std::vector<uint8_t>& colors);

    // Mesh extraction via marching cubes
    MeshPtr extract_mesh();

    // Reset volume
    void reset();

    // Statistics
    float get_voxel_grid_usage() const;
    size_t get_voxel_count() const { return TOTAL_VOXELS; }

    // World to voxel conversion
    Vector3i world_to_voxel(const Vector3f& world_pos) const;
    Vector3f voxel_to_world(int x, int y, int z) const;
    Vector3f voxel_to_world(const Vector3i& idx) const;

    // Voxel access
    TSDFVoxel& voxel(int x, int y, int z);
    const TSDFVoxel& voxel(int x, int y, int z) const;
    TSDFVoxel& voxel(const Vector3i& idx);
    const TSDFVoxel& voxel(const Vector3i& idx) const;

    bool is_valid_index(int x, int y, int z) const;
    bool is_valid_index(const Vector3i& idx) const;

    // Interpolation
    bool interpolate_trilinear(const Vector3f& world_pos,
                              float& tsdf_val, float& weight);

private:
    Config config_;
    std::vector<TSDFVoxel> data_;
    Vector3f volume_center_;
    float volume_extent_;

    // Voxel index calculation
    size_t get_linear_index(int x, int y, int z) const {
        return z * RESOLUTION * RESOLUTION + y * RESOLUTION + x;
    }

    // Integration helpers
    void integrate_frame(const DepthFrame& depth, const CameraPose& pose,
                        const ColorFrame* color);

    // CUDA integration (if available)
    void integrate_frame_cuda(const DepthFrame& depth,
                             const CameraPose& pose,
                             const ColorFrame* color);
};

}  // namespace kf