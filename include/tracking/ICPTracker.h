#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "sensor/FrameData.h"

namespace kfusion {
namespace tracking {

struct ICPParams {
    int    max_iterations[sensor::FramePyramid::LEVELS] = {10, 5, 4};
    float  dist_threshold    = 0.1f;  // meters
    float  angle_threshold   = 30.0f; // degrees
    float  min_depth         = 0.3f;
    float  max_depth         = 5.0f;
};

struct ICPResult {
    Eigen::Matrix4f pose;        // Updated camera pose (world-from-camera)
    float           error       = 0.0f;
    int             inliers     = 0;
    bool            converged   = false;
    bool            tracking_ok = false;
};

// Raycasted model points and normals used as reference for ICP
struct ModelFrame {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    int width  = sensor::FRAME_W;
    int height = sensor::FRAME_H;

    ModelFrame() {
        vertices.assign(width * height, Eigen::Vector3f::Zero());
        normals.assign(width * height, Eigen::Vector3f::Zero());
    }
};

class ICPTracker {
public:
    explicit ICPTracker(const ICPParams& params = ICPParams{});

    // Track: given live pyramid + previous model frame + previous pose → new pose
    ICPResult track(const sensor::FramePyramid& live,
                    const ModelFrame&           model,
                    const Eigen::Matrix4f&      prev_pose);

    const ICPParams& params() const { return params_; }

private:
    ICPParams params_;

    // Per-level ICP
    ICPResult trackLevel(const sensor::FrameData& live_level,
                         const ModelFrame&        model,
                         const Eigen::Matrix4f&   pose_estimate,
                         const Eigen::Matrix4f&   ref_pose,
                         int                      level,
                         int                      max_iter);

    // Point-to-plane linear system build
    bool buildLinearSystem(const sensor::FrameData& live,
                           const ModelFrame&        model,
                           const Eigen::Matrix4f&   pose,
                           const Eigen::Matrix4f&   ref_pose,
                           Eigen::Matrix<float,6,6>& A,
                           Eigen::Matrix<float,6,1>& b,
                           float& residual,
                           int&   inlier_count);

    // Project model into current frame space
    void projectModel(const ModelFrame&      model,
                      const Eigen::Matrix4f& pose,
                      int                    level,
                      std::vector<Eigen::Vector3f>& proj_vertices,
                      std::vector<Eigen::Vector3f>& proj_normals);
};

} // namespace tracking
} // namespace kfusion
