#include "tracking/ICPTracker.h"
#include "sensor/KinectSensor.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace tracking {

// Kinect intrinsics at each pyramid level
static float getFx(int level) { return static_cast<float>(sensor::FX) / (1 << level); }
static float getFy(int level) { return static_cast<float>(sensor::FY) / (1 << level); }
static float getCx(int level) { return static_cast<float>(sensor::CX) / (1 << level); }
static float getCy(int level) { return static_cast<float>(sensor::CY) / (1 << level); }

ICPTracker::ICPTracker(const ICPParams& params)
    : params_(params)
{}

ICPResult ICPTracker::track(const sensor::FramePyramid& live,
                            const ModelFrame&           model,
                            const Eigen::Matrix4f&      prev_pose)
{
    ICPResult result;
    result.pose = prev_pose;

    // Coarse to fine: level 2 → 1 → 0
    for (int level = sensor::FramePyramid::LEVELS - 1; level >= 0; --level) {
        result = trackLevel(live.levels[level], model, result.pose, level,
                            params_.max_iterations[level]);
    }

    result.tracking_ok = result.converged && result.inliers > 100;
    return result;
}

ICPResult ICPTracker::trackLevel(const sensor::FrameData& live_level,
                                 const ModelFrame&        model,
                                 const Eigen::Matrix4f&   pose_estimate,
                                 int                      level,
                                 int                      max_iter)
{
    ICPResult result;
    result.pose = pose_estimate;

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
        float residual    = 0.0f;
        int   inlier_count = 0;

        if (!buildLinearSystem(live_level, model, result.pose, A, b, residual, inlier_count))
            break;

        if (inlier_count < 10) break;

        // Solve: Ax = b using Cholesky decomposition
        Eigen::Matrix<float, 6, 1> x = A.ldlt().solve(b);

        // Update pose: twist to SE3
        // x = [tx, ty, tz, rx, ry, rz]
        float tx = x(0), ty = x(1), tz = x(2);
        float rx = x(3), ry = x(4), rz = x(5);

        // Rodrigues: small angle → rotation matrix
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        float angle = std::sqrt(rx*rx + ry*ry + rz*rz);
        if (angle > 1e-6f) {
            Eigen::Vector3f axis(rx, ry, rz);
            axis.normalize();
            R = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
        } else {
            // Small angle approximation
            R(0,1) = -rz; R(0,2) =  ry;
            R(1,0) =  rz; R(1,2) = -rx;
            R(2,0) = -ry; R(2,1) =  rx;
        }

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = R;
        delta(0,3) = tx; delta(1,3) = ty; delta(2,3) = tz;

        result.pose     = delta * result.pose;
        result.error    = residual / static_cast<float>(std::max(inlier_count, 1));
        result.inliers  = inlier_count;

        if (x.norm() < 1e-5f) {
            result.converged = true;
            break;
        }
    }

    return result;
}

bool ICPTracker::buildLinearSystem(const sensor::FrameData& live,
                                   const ModelFrame&        model,
                                   const Eigen::Matrix4f&   pose,
                                   Eigen::Matrix<float,6,6>& A,
                                   Eigen::Matrix<float,6,1>& b,
                                   float& residual,
                                   int&   inlier_count)
{
    const int W = live.width;
    const int H = live.height;

    // Compute level from width
    int level = 0;
    if (W == sensor::FRAME_W / 2) level = 1;
    else if (W == sensor::FRAME_W / 4) level = 2;

    const float fx = getFx(level);
    const float fy = getFy(level);
    const float cx = getCx(level);
    const float cy = getCy(level);

    const float angle_thresh_cos = std::cos(params_.angle_threshold * M_PI / 180.0f);

    // Camera-to-world rotation and translation from pose
    const Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    const Eigen::Vector3f t_cw = pose.block<3,1>(0,3);
    // World-to-camera
    const Eigen::Matrix3f R_wc = R_cw.transpose();
    const Eigen::Vector3f t_wc = -R_wc * t_cw;

    // Thread-local accumulators
    struct LocalAcc {
        Eigen::Matrix<float,6,6> A;
        Eigen::Matrix<float,6,1> b;
        float residual;
        int   count;
        LocalAcc() : A(Eigen::Matrix<float,6,6>::Zero()),
                     b(Eigen::Matrix<float,6,1>::Zero()),
                     residual(0.0f), count(0) {}
    };

    const int num_threads = 1;
    std::vector<LocalAcc> local(num_threads);

    #pragma omp parallel for schedule(dynamic, 8) num_threads(num_threads)
    for (int y = 1; y < H - 1; ++y) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& acc = local[tid];

        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;

            // Skip invalid live points
            if (live.depth_meters[idx] <= 0.0f) continue;

            const Eigen::Vector3f& live_v = live.vertices[idx];
            if (live_v.norm() < 1e-6f) continue;

            // Transform live vertex to world space
            Eigen::Vector3f live_v_world = R_cw * live_v + t_cw;

            // Project into model image coords (model is in world space, camera at identity)
            // For simplicity: project live_v_world back into camera space = live_v (since model frame is raycasted from same camera)
            // More precisely: find corresponding model point by projecting live vertex into model frame
            float model_x = fx * live_v[0] / live_v[2] + cx;
            float model_y = fy * live_v[1] / live_v[2] + cy;

            int mx = static_cast<int>(std::round(model_x));
            int my = static_cast<int>(std::round(model_y));

            // Use full-res model dimensions
            if (mx < 0 || mx >= sensor::FRAME_W || my < 0 || my >= sensor::FRAME_H) continue;
            int midx = my * sensor::FRAME_W + mx;

            const Eigen::Vector3f& model_v = model.vertices[midx];
            const Eigen::Vector3f& model_n = model.normals[midx];
            if (model_v.norm() < 1e-6f || model_n.norm() < 1e-6f) continue;

            // Distance check
            float dist = (live_v - model_v).norm();
            if (dist > params_.dist_threshold) continue;

            // Normal angle check
            const Eigen::Vector3f& live_n = live.normals[idx];
            if (live_n.norm() < 1e-6f) continue;
            float dot = std::abs(live_n.dot(model_n));
            if (dot < angle_thresh_cos) continue;

            // Point-to-plane error: n^T * (live - model)
            float err = model_n.dot(live_v - model_v);

            // Jacobian for point-to-plane ICP
            // J = [n; cross(live_v, n)]
            Eigen::Vector3f cross = live_v.cross(model_n);
            Eigen::Matrix<float,6,1> J;
            J(0) = model_n(0); J(1) = model_n(1); J(2) = model_n(2);
            J(3) = cross(0);   J(4) = cross(1);   J(5) = cross(2);

            acc.A += J * J.transpose();
            acc.b -= J * err;
            acc.residual += err * err;
            acc.count++;
        }
    }

    // Merge thread-local results
    for (auto& acc : local) {
        A += acc.A;
        b += acc.b;
        residual    += acc.residual;
        inlier_count += acc.count;
    }

    return inlier_count > 0;
}

} // namespace tracking
} // namespace kfusion
