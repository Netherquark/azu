#pragma once

#include "utils/Types.hpp"
#include "utils/CameraModel.hpp"
#include <vector>

namespace kf {

// ============================================================================
// Point-to-Plane ICP Solver
// ============================================================================

class PointToPlaneICP {
public:
    struct Config {
        float distance_threshold = 0.05f;     // Max distance for correspondence
        float normal_angle_threshold = 45.0f; // Max angle (degrees)
        int max_iterations = 50;
        float convergence_threshold = 0.0001f;
        int min_correspondences = 100;
    };

    explicit PointToPlaneICP(const Config& config = Config());

    // Core alignment function
    //   source_vertices: Points from current frame (camera coordinates)
    //   source_normals:  Normals from current frame
    //   model_vertices:  Vertices from TSDF raycasting (world coordinates)
    //   model_normals:   Normals from model
    //   initial_pose:    Initial guess for pose (camera pose in world)
    //   camera_intrinsic: Camera model for projection
    AlignmentResult align(const std::vector<Vector3f>& source_vertices,
                         const std::vector<Vector3f>& source_normals,
                         const std::vector<Vector3f>& model_vertices,
                         const std::vector<Vector3f>& model_normals,
                         const CameraPose& initial_pose,
                         const CameraModel& camera_intrinsic);

private:
    Config config_;

    // Find correspondences between source and model
    struct Correspondence {
        int source_idx;
        int model_idx;
        Vector3f normal;
        float distance;
    };

    std::vector<Correspondence> find_correspondences(
        const std::vector<Vector3f>& source_verts,
        const std::vector<Vector3f>& source_norms,
        const std::vector<Vector3f>& model_verts,
        const std::vector<Vector3f>& model_norms,
        const CameraPose& pose);

    // Build and solve linear system
    bool solve_linear_system(const std::vector<Correspondence>& correspondences,
                            const std::vector<Vector3f>& source_verts,
                            const std::vector<Vector3f>& model_verts,
                            const CameraPose& current_pose,
                            Vector3f& delta_rotation,  // Output
                            Vector3f& delta_translation);

    // Update pose from delta
    CameraPose update_pose(const CameraPose& current_pose,
                          const Vector3f& delta_rotation,
                          const Vector3f& delta_translation);

    // Compute error metrics
    float compute_error(const std::vector<Correspondence>& correspondences);
    float compute_inlier_ratio(const std::vector<Correspondence>& correspondences);
};

}  // namespace kf
