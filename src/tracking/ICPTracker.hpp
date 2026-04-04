#pragma once

#include "utils/Types.hpp"
#include "utils/CameraModel.hpp"
#include "tracking/PointToPlaneICP.hpp"
#include <memory>
#include <vector>

namespace kf {

// ============================================================================
// ICP Tracker - Frame-to-Model Tracking
// ============================================================================

class ICPTracker {
public:
    struct Config {
        CameraModel camera;
        int pyramid_levels = 3;
        PointToPlaneICP::Config icp_config;
        bool use_color_weighting = false;
    };

    explicit ICPTracker(const Config& config);

    // Convert depth frame to vertex map and compute normals
    VertexMapPtr depth_to_vertex_map(const DepthFrame& depth);

    // Multi-resolution tracking
    AlignmentResult track(const VertexMapPtr& frame_vertices,
                         const std::vector<Vector3f>& model_vertices,
                         const std::vector<Vector3f>& model_normals,
                         const CameraPose& initial_pose);

    // Single level tracking (for debugging)
    AlignmentResult track_single_level(const VertexMapPtr& frame_vertices,
                                      const std::vector<Vector3f>& model_vertices,
                                      const std::vector<Vector3f>& model_normals,
                                      const CameraPose& initial_pose);

    // Get current pose
    const CameraPose& get_pose() const { return current_pose_; }

    // Reset pose
    void reset_pose() { current_pose_ = CameraPose(); }

    // Statistics
    struct Stats {
        float total_error;
        float inlier_ratio;
        int iterations;
        bool success;
    };

    const Stats& get_last_stats() const { return last_stats_; }

private:
    Config config_;
    CameraPose current_pose_;
    Stats last_stats_;

    // Downsample vertex map for pyramid
    VertexMapPtr downsample_vertex_map(const VertexMapPtr& src);

    // Compute normals for vertex map
    void compute_normals(VertexMapPtr& vmap);

    // Extract valid vertices and normals
    std::vector<Vector3f> extract_valid_vertices(const VertexMapPtr& vmap);
    std::vector<Vector3f> extract_valid_normals(const VertexMapPtr& vmap);

    // ICP solver instances for each pyramid level
    std::vector<std::unique_ptr<PointToPlaneICP>> icp_solvers_;
};

}  // namespace kf