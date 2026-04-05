#include "tracking/ICPTracker.hpp"
#include "utils/Math.hpp"
#include "utils/Logger.hpp"
#include <omp.h>

namespace kf {

ICPTracker::ICPTracker(const Config& config) : config_(config) {
    // Initialize ICP solvers for each pyramid level
    for (int level = 0; level < config_.pyramid_levels; ++level) {
        auto icp_config = config_.icp_config;
        // Adjust thresholds for coarser levels (more lenient)
        icp_config.distance_threshold *= (1 << level);
        icp_config.max_iterations = 20 - level * 2;  // Fewer iterations at coarse levels
        icp_solvers_.push_back(
            std::make_unique<PointToPlaneICP>(icp_config));
    }
}

VertexMapPtr ICPTracker::depth_to_vertex_map(const DepthFrame& depth) {
    auto vmap = std::make_shared<VertexMap>(DepthFrame::WIDTH,
                                           DepthFrame::HEIGHT);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < DepthFrame::HEIGHT; ++y) {
        for (int x = 0; x < DepthFrame::WIDTH; ++x) {
            float depth_m = depth.get_depth_m(x, y);

            if (depth_m > 0.0f && depth_m < 10.0f) {
                Vector3f p =
                    math::depth_to_3d(x, y, depth_m, depth.fx, depth.fy,
                                     depth.cx, depth.cy);
                vmap->vertex(x, y) = Vector4f(p.x(), p.y(), p.z(), 1.0f);
            } else {
                vmap->vertex(x, y) = Vector4f::Zero();
            }
        }
    }

    compute_normals(vmap);
    return vmap;
}

void ICPTracker::compute_normals(VertexMapPtr& vmap) {
#pragma omp parallel for collapse(2)
    for (int y = 1; y < vmap->height - 1; ++y) {
        for (int x = 1; x < vmap->width - 1; ++x) {
            vmap->normal(x, y) =
                math::compute_vertex_normal(*vmap, x, y);
        }
    }
}

std::vector<Vector3f> ICPTracker::extract_valid_vertices(
    const VertexMapPtr& vmap) {
    std::vector<Vector3f> vertices;
    vertices.reserve(vmap->width * vmap->height / 2);

    for (int y = 0; y < vmap->height; ++y) {
        for (int x = 0; x < vmap->width; ++x) {
            Vector4f v = vmap->vertex(x, y);
            if (v.w() > 0.5f && math::is_valid_3d_point(v.head<3>())) {
                vertices.push_back(v.head<3>());
            }
        }
    }

    return vertices;
}

std::vector<Vector3f> ICPTracker::extract_valid_normals(
    const VertexMapPtr& vmap) {
    std::vector<Vector3f> normals;
    normals.reserve(vmap->width * vmap->height / 2);

    for (int y = 0; y < vmap->height; ++y) {
        for (int x = 0; x < vmap->width; ++x) {
            Vector4f v = vmap->vertex(x, y);
            if (v.w() > 0.5f && math::is_valid_3d_point(v.head<3>())) {
                Vector3f n = vmap->normal(x, y);
                if (n.norm() > 0.5f) {
                    normals.push_back(n.normalized());
                } else {
                    normals.push_back(Vector3f(0, 0, 1));  // Default normal
                }
            }
        }
    }

    return normals;
}

VertexMapPtr ICPTracker::downsample_vertex_map(const VertexMapPtr& src) {
    int new_width = src->width / 2;
    int new_height = src->height / 2;
    auto dst =
        std::make_shared<VertexMap>(new_width, new_height);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            // Take center of 2x2 block
            Vector4f v = src->vertex(x * 2 + 1, y * 2 + 1);
            dst->vertex(x, y) = v;
        }
    }

    compute_normals(dst);
    return dst;
}

AlignmentResult ICPTracker::track_single_level(
    const VertexMapPtr& frame_vertices,
    const std::vector<Vector3f>& model_vertices,
    const std::vector<Vector3f>& model_normals,
    const CameraPose& initial_pose) {

    auto frame_verts = extract_valid_vertices(frame_vertices);
    auto frame_norms = extract_valid_normals(frame_vertices);

    if (frame_verts.size() < 100) {
        KF_LOG_WARN("ICPTracker: Too few frame vertices: ", frame_verts.size());
        AlignmentResult result;
        result.success = false;
        return result;
    }

    // Use finest-level solver (index 0 is finest resolution)
    AlignmentResult result =
        icp_solvers_[0]->align(frame_verts, frame_norms, model_vertices,
                              model_normals, initial_pose, config_.camera);

    last_stats_ = {result.error, result.inlier_ratio, result.iterations,
                  result.success};
    current_pose_ = result.pose;

    return result;
}

AlignmentResult ICPTracker::track(
    const VertexMapPtr& frame_vertices,
    const std::vector<Vector3f>& model_vertices,
    const std::vector<Vector3f>& model_normals,
    const CameraPose& initial_pose) {

    CameraPose pose = initial_pose;

    // Coarse-to-fine pyramid
    std::vector<VertexMapPtr> pyramid;
    pyramid.push_back(frame_vertices);

    for (int level = 1; level < config_.pyramid_levels; ++level) {
        pyramid.push_back(downsample_vertex_map(pyramid.back()));
    }

    // Track from coarsest to finest
    for (int level = config_.pyramid_levels - 1; level >= 0; --level) {
        auto frame_verts = extract_valid_vertices(pyramid[level]);
        auto frame_norms = extract_valid_normals(pyramid[level]);

        if (frame_verts.size() < 50) {
            KF_LOG_WARN("ICPTracker level ", level, ": Too few vertices");
            continue;
        }

        AlignmentResult result =
            icp_solvers_[level]->align(frame_verts, frame_norms,
                                      model_vertices, model_normals, pose,
                                      config_.camera);

        if (result.success) {
            pose = result.pose;
        } else {
            KF_LOG_WARN("ICP level ", level, " failed");
        }

        last_stats_ = {result.error, result.inlier_ratio, result.iterations,
                      result.success};
    }

    current_pose_ = pose;
    return AlignmentResult(pose, last_stats_.total_error,
                          last_stats_.inlier_ratio, last_stats_.iterations);
}

}  // namespace kf
