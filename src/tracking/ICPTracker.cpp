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
        result = trackLevel(live.levels[level], model, result.pose, prev_pose, level,
                            params_.max_iterations[level]);
    }

    result.tracking_ok = result.converged && result.inliers > 100;
    return result;
}

ICPResult ICPTracker::trackLevel(const sensor::FrameData& live_level,
                                 const ModelFrame&        model,
                                 const Eigen::Matrix4f&   pose_estimate,
                                 const Eigen::Matrix4f&   ref_pose,
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

        if (!buildLinearSystem(live_level, model, result.pose, ref_pose, A, b, residual, inlier_count,
                               result.valid_live_points, result.valid_model_points, result.projected_points,
                               result.dist_filtered, result.angle_filtered))
            break;

        if (inlier_count < 10) break;

        // Regularize the Hessian (Tikhonov / Levenberg-Marquardt style)
        // Helps avoid singular matrices in flat areas
        A += Eigen::Matrix<float, 6, 6>::Identity() * 0.1f;

        // Solve: Ax = b using Cholesky decomposition
        Eigen::Matrix<float, 6, 1> x = A.ldlt().solve(b);

        // Update pose: twist to SE3
        // x = [tx, ty, tz, rx, ry, rz]
        float tx = x(0), ty = x(1), tz = x(2);
        float rx = x(3), ry = x(4), rz = x(5);

        // Rodrigues: small angle → rotation matrix
        Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        float angle = std::sqrt(rx*rx + ry*ry + rz*rz);
        if (angle > 1e-4f) {
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

        if (x.norm() < 5e-5f) {
            result.converged = true;
            break;
        }
    }

    return result;
}

bool ICPTracker::buildLinearSystem(const sensor::FrameData& live,
                                   const ModelFrame&        model,
                                   const Eigen::Matrix4f&   pose,
                                   const Eigen::Matrix4f&   ref_pose,
                                   Eigen::Matrix<float,6,6>& A,
                                   Eigen::Matrix<float,6,1>& b,
                                   float& residual,
                                   int&   inlier_count,
                                   int&   valid_live,
                                   int&   valid_model,
                                   int&   projected,
                                   int&   dist_filtered,
                                   int&   angle_filtered)
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

    // Camera-to-world rotation and translation from current estimated pose
    const Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    const Eigen::Vector3f t_cw = pose.block<3,1>(0,3);

    // World-to-camera for the REFERENCE pose (where the model frame was raycasted)
    const Eigen::Matrix4f ref_inv = ref_pose.inverse();
    const Eigen::Matrix3f R_rc = ref_inv.block<3,3>(0,0);
    const Eigen::Vector3f t_rc = ref_inv.block<3,1>(0,3);

    // Combined relative transform: ref_cam <- world <- live_cam
    const Eigen::Matrix3f R_rel = R_rc * R_cw;
    const Eigen::Vector3f t_rel = R_rc * t_cw + t_rc;

    // Use user-configured thread count or auto-detect
    int num_threads = num_threads_.load();
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }
    
    // Thread-local accumulators
    struct LocalAcc {
        Eigen::Matrix<float,6,6> A;
        Eigen::Matrix<float,6,1> b;
        float residual;
        int   count;
        int   valid_live;
        int   valid_model;
        int   projected;
        int   dist_filtered;
        int   angle_filtered;

        LocalAcc() : A(Eigen::Matrix<float,6,6>::Zero()),
                     b(Eigen::Matrix<float,6,1>::Zero()),
                     residual(0.0f), count(0),
                     valid_live(0), valid_model(0), projected(0),
                     dist_filtered(0), angle_filtered(0) {}
    };
    std::vector<LocalAcc> local(num_threads);

    #pragma omp parallel for schedule(dynamic, 8) num_threads(num_threads)
    for (int y = 1; y < H - 1; ++y) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid >= num_threads) tid = 0; // Safety
        auto& acc = local[tid];

        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;

            // -----------------------------------------------------------------------
            // Fix: Transform live vertex at current estimate to reference camera
            // -----------------------------------------------------------------------
            const Eigen::Vector3f& live_v = live.vertices[idx];
            if (live_v.norm() < 1e-6f) continue;

            // Map live_v (current cam space) to ref_cam space
            Eigen::Vector3f v_ref = R_rel * live_v + t_rel;
            
            if (v_ref.z() <= 0.001f) continue;
            acc.valid_live++;

            // Project v_ref into the model frame (always 640x480)
            // Use full-res intrinsics here, NOT level-adjusted ones!
            float model_x = static_cast<float>(sensor::FX) * v_ref.x() / v_ref.z() + static_cast<float>(sensor::CX);
            float model_y = static_cast<float>(sensor::FY) * v_ref.y() / v_ref.z() + static_cast<float>(sensor::CY);

            int mx = static_cast<int>(std::round(model_x));
            int my = static_cast<int>(std::round(model_y));

            // Use full-res model dimensions
            if (mx < 0 || mx >= sensor::FRAME_W || my < 0 || my >= sensor::FRAME_H) continue;
            acc.projected++;
            
            int midx = my * sensor::FRAME_W + mx;

            const Eigen::Vector3f& model_v = model.vertices[midx];
            const Eigen::Vector3f& model_n = model.normals[midx];
            if (model_v.norm() < 1e-6f || model_n.norm() < 1e-6f) continue;
            acc.valid_model++;

            // Correspondence check (Distance)
            // Fix: Both must be in the same space for a valid distance check
            float dist = (v_ref - model_v).norm();
            if (dist > params_.dist_threshold) {
                acc.dist_filtered++;
                continue;
            }

            // Normal angle check
            const Eigen::Vector3f& live_n = live.normals[idx];
            if (live_n.norm() < 1e-6f) continue;
            
            // Map live normal to reference space
            Eigen::Vector3f live_n_ref = R_rel * live_n;
            float dot = std::abs(live_n_ref.dot(model_n));
            if (dot < angle_thresh_cos) {
                acc.angle_filtered++;
                continue;
            }

            // Point-to-plane error: n^T * (live_in_ref - model)
            float err = model_n.dot(v_ref - model_v);

            // Jacobian for point-to-plane ICP
            // We use the live vertex in CURRENT camera space for the Jacobian
            // J = [model_n_in_live; cross(live_v, model_n_in_live)]
            // But usually we work in current camera space.
            // Let's transform everything to current camera space: 
            // model_v_in_live = R_rel^T * (model_v - t_rel)
            // model_n_in_live = R_rel^T * model_n
            
            Eigen::Vector3f model_n_live = R_rel.transpose() * model_n;
            Eigen::Vector3f model_v_live = R_rel.transpose() * (model_v - t_rel);

            Eigen::Vector3f cross = live_v.cross(model_n_live);
            Eigen::Matrix<float,6,1> J;
            J(0) = model_n_live(0); J(1) = model_n_live(1); J(2) = model_n_live(2);
            J(3) = cross(0);        J(4) = cross(1);        J(5) = cross(2);

            acc.A += J * J.transpose();
            acc.b -= J * err;
            acc.residual += err * err;
            acc.count++;
        }
    }

    // Merge thread-local results
    valid_live = 0;
    valid_model = 0;
    projected = 0;
    dist_filtered = 0;
    angle_filtered = 0;

    for (auto& acc : local) {
        A += acc.A;
        b += acc.b;
        residual    += acc.residual;
        inlier_count += acc.count;
        valid_live     += acc.valid_live;
        valid_model    += acc.valid_model;
        projected      += acc.projected;
        dist_filtered  += acc.dist_filtered;
        angle_filtered += acc.angle_filtered;
    }

    return inlier_count > 0;
}

#ifdef CUDA_ENABLED
ICPResult ICPTracker::trackGPU(const sensor::FramePyramid& live,
                               const ModelFrame&           model,
                               const Eigen::Matrix4f&      pose_estimate)
{
    ICPResult result;
    result.pose = pose_estimate;
    result.tracking_ok = true;

    // 1. Upload live pyramid to GPU buffers (Reuse persistent buffers)
    for (int level = 0; level < sensor::FramePyramid::LEVELS; ++level) {
        const auto& ld = live.levels[level];
        size_t sz = ld.width * ld.height * sizeof(float3);
        
        if (!d_pyramid_v[level]) {
            d_pyramid_v[level] = utils::make_cuda_unique<float3>(ld.width * ld.height);
            d_pyramid_n[level] = utils::make_cuda_unique<float3>(ld.width * ld.height);
        }
        
        cudaMemcpy(d_pyramid_v[level].get(), ld.vertices.data(), sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_pyramid_n[level].get(), ld.normals.data(),  sz, cudaMemcpyHostToDevice);
    }

    // 2. Coarse-to-fine iterations
    for (int level = sensor::FramePyramid::LEVELS - 1; level >= 0; --level) {
        result = trackLevelGPU(
            d_pyramid_v[level].get(), 
            d_pyramid_n[level].get(),
            live.levels[level].width,
            live.levels[level].height,
            model, 
            result.pose, 
            pose_estimate, 
            level, 
            params_.max_iterations[level]
        );
        
        if (result.inliers < 100) {
            result.tracking_ok = false;
            break;
        }
    }

    return result;
}
#endif

} // namespace tracking
} // namespace kfusion
