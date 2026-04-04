#include "tracking/PointToPlaneICP.hpp"
#include "utils/Math.hpp"
#include "utils/Logger.hpp"
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>

namespace kf {

PointToPlaneICP::PointToPlaneICP(const Config& config) : config_(config) {}

AlignmentResult PointToPlaneICP::align(
    const std::vector<Vector3f>& source_vertices,
    const std::vector<Vector3f>& source_normals,
    const std::vector<Vector3f>& model_vertices,
    const std::vector<Vector3f>& model_normals,
    const CameraPose& initial_pose, const CameraModel& camera_intrinsic) {

    AlignmentResult result;
    result.pose = initial_pose;
    result.iterations = 0;

    if (source_vertices.empty() || model_vertices.empty()) {
        KF_LOG_WARN("ICP: Empty input vertices");
        return result;
    }

    // Iterative refinement
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        result.iterations = iter + 1;

        // Find correspondences
        auto correspondences =
            find_correspondences(source_vertices, source_normals,
                               model_vertices, model_normals, result.pose);

        if (correspondences.size() < config_.min_correspondences) {
            KF_LOG_WARN("ICP: Too few correspondences: ", correspondences.size());
            result.success = false;
            return result;
        }

        // Compute error before update
        float prev_error = compute_error(correspondences);

        // Solve linear system
        Vector3f delta_rot, delta_trans;
        if (!solve_linear_system(correspondences, source_vertices, delta_rot,
                                delta_trans)) {
            result.success = false;
            return result;
        }

        // Update pose
        result.pose = update_pose(result.pose, delta_rot, delta_trans);

        // Compute new error
        result.error = prev_error;
        result.inlier_ratio = compute_inlier_ratio(correspondences);

        // Check convergence
        float delta_norm = delta_rot.norm() + delta_trans.norm();
        if (delta_norm < config_.convergence_threshold) {
            result.success = true;
            KF_LOG_DEBUG("ICP converged at iteration ", iter);
            return result;
        }
    }

    result.success = true;
    return result;
}

std::vector<PointToPlaneICP::Correspondence>
PointToPlaneICP::find_correspondences(
    const std::vector<Vector3f>& source_verts,
    const std::vector<Vector3f>& source_norms,
    const std::vector<Vector3f>& model_verts,
    const std::vector<Vector3f>& model_norms, const CameraPose& pose) {

    std::vector<Correspondence> correspondences;
    correspondences.reserve(source_verts.size() / 2);

    // Transform source points to world coordinates
    for (size_t i = 0; i < source_verts.size(); ++i) {
        const Vector3f& src_point = source_verts[i];
        if (!math::is_valid_3d_point(src_point)) continue;

        const Vector3f& src_normal = source_norms[i];
        if (src_normal.norm() < 0.5f) continue;  // Invalid normal

        // Transform to world
        Vector3f world_point = pose.transform_point(src_point);
        Vector3f world_normal = pose.transform_normal(src_normal);

        // Find nearest model point (brute force for now)
        float min_dist = config_.distance_threshold;
        int best_model_idx = -1;

        for (size_t j = 0; j < model_verts.size(); ++j) {
            const Vector3f& model_point = model_verts[j];
            if (!math::is_valid_3d_point(model_point)) continue;

            Vector3f diff = model_point - world_point;
            float dist = diff.norm();

            if (dist > min_dist) continue;

            // Check normal alignment
            const Vector3f& model_normal = model_norms[j];
            if (model_normal.norm() < 0.5f) continue;

            float angle =
                std::acos(std::max(-1.0f, std::min(1.0f,
                    std::abs(world_normal.normalized().dot(
                        model_normal.normalized())))));
            float angle_deg = angle * math::RAD_TO_DEG;

            if (angle_deg > config_.normal_angle_threshold) continue;

            min_dist = dist;
            best_model_idx = j;
        }

        if (best_model_idx >= 0) {
            Correspondence c;
            c.source_idx = i;
            c.model_idx = best_model_idx;
            c.normal = model_norms[best_model_idx];
            c.distance = min_dist;
            correspondences.push_back(c);
        }
    }

    return correspondences;
}

bool PointToPlaneICP::solve_linear_system(
    const std::vector<Correspondence>& correspondences,
    const std::vector<Vector3f>& source_verts, Vector3f& delta_rotation,
    Vector3f& delta_translation) {

    // Point-to-plane ICP: min ||n_i^T (R*p_i - q_i)||^2
    // Where n_i is model normal, p_i is source point, q_i is model point

    int n = correspondences.size();
    if (n == 0) return false;

    Eigen::MatrixXf A(n, 6);
    Eigen::VectorXf b(n);

    for (int i = 0; i < n; ++i) {
        const Correspondence& c = correspondences[i];
        const Vector3f& p = source_verts[c.source_idx];
        const Vector3f& n = c.normal.normalized();

        // Skew symmetric matrix for cross product
        Vector3f p_cross_n = p.cross(n);

        A.row(i) << p_cross_n.x(), p_cross_n.y(), p_cross_n.z(), n.x(), n.y(),
            n.z();

        b(i) = 0.0f;  // Point-to-plane residual is always 0 at optimum
    }

    // Solve using SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU |
                                          Eigen::ComputeThinV);
    Eigen::VectorXf x = svd.solve(b);

    delta_rotation = Vector3f(x(0), x(1), x(2));
    delta_translation = Vector3f(x(3), x(4), x(5));

    return true;
}

CameraPose PointToPlaneICP::update_pose(const CameraPose& current_pose,
                                       const Vector3f& delta_rotation,
                                       const Vector3f& delta_translation) {
    // Small angle approximation: R_delta ≈ I + [delta_rot]_x
    float angle = delta_rotation.norm();
    if (angle > 1e-6f) {
        Vector3f axis = delta_rotation / angle;
        Matrix3f R_delta = current_pose.R;
        R_delta += math::skew_symmetric(delta_rotation);  // First-order approx
        R_delta = R_delta.householderQr().householderQ();  // Re-orthogonalize
        return CameraPose(R_delta, current_pose.t + delta_translation);
    }

    return CameraPose(current_pose.R,
                     current_pose.t + delta_translation);
}

float PointToPlaneICP::compute_error(
    const std::vector<Correspondence>& correspondences) {
    if (correspondences.empty()) return std::numeric_limits<float>::max();

    float error_sum = 0.0f;
    for (const auto& c : correspondences) {
        error_sum += c.distance * c.distance;
    }

    return error_sum / correspondences.size();
}

float PointToPlaneICP::compute_inlier_ratio(
    const std::vector<Correspondence>& correspondences) {
    if (correspondences.empty()) return 0.0f;

    int inliers = 0;
    for (const auto& c : correspondences) {
        if (c.distance < config_.distance_threshold) {
            inliers++;
        }
    }

    return static_cast<float>(inliers) / correspondences.size();
}

}  // namespace kf
