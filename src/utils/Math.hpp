#pragma once

#include "Types.hpp"
#include <cmath>
#include <algorithm>

namespace kf::math {

// ============================================================================
// Constants
// ============================================================================

static constexpr float PI = 3.14159265359f;
static constexpr float DEG_TO_RAD = PI / 180.0f;
static constexpr float RAD_TO_DEG = 180.0f / PI;

// ============================================================================
// Vector Operations
// ============================================================================

inline float safe_length(const Vector3f& v, float epsilon = 1e-6f) {
    float len = v.norm();
    return len > epsilon ? len : epsilon;
}

inline Vector3f safe_normalize(const Vector3f& v, float epsilon = 1e-6f) {
    float len = safe_length(v, epsilon);
    return v / len;
}

inline float clamp(float x, float min_val, float max_val) {
    return std::max(min_val, std::min(x, max_val));
}

inline bool is_valid_3d_point(const Vector3f& p, float min_depth = 0.1f,
                               float max_depth = 10.0f) {
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
        return false;
    float z = p.z();
    return z >= min_depth && z <= max_depth;
}

// ============================================================================
// Depth Conversion
// ============================================================================

inline Vector3f depth_to_3d(int x, int y, float depth_m, float fx, float fy,
                            float cx, float cy) {
    if (depth_m <= 0.0f) return Vector3f::Zero();
    float X = (x - cx) * depth_m / fx;
    float Y = (y - cy) * depth_m / fy;
    float Z = depth_m;
    return Vector3f(X, Y, Z);
}

inline Vector2f project_3d_to_2d(const Vector3f& p, float fx, float fy,
                                 float cx, float cy) {
    if (p.z() <= 0.0f) return Vector2f(-1, -1);
    float u = fx * p.x() / p.z() + cx;
    float v = fy * p.y() / p.z() + cy;
    return Vector2f(u, v);
}

// ============================================================================
// Normal Computation
// ============================================================================

inline Vector3f compute_normal_cross(const Vector3f& p0, const Vector3f& p1,
                                     const Vector3f& p2) {
    Vector3f v1 = p1 - p0;
    Vector3f v2 = p2 - p0;
    Vector3f n = v1.cross(v2);
    return safe_normalize(n);
}

// Compute normal from vertex map using central differences
inline Vector3f compute_vertex_normal(const VertexMap& vmap, int x, int y) {
    const Vector4f& p_center = vmap.vertex(x, y);
    if (p_center.w() < 0.5f) return Vector3f::Zero();

    int dx = 1, dy = 1;
    const Vector4f& p_right = 
        (x + dx < vmap.width) ? vmap.vertex(x + dx, y) : p_center;
    const Vector4f& p_down = 
        (y + dy < vmap.height) ? vmap.vertex(x, y + dy) : p_center;

    if (p_right.w() < 0.5f || p_down.w() < 0.5f) return Vector3f::Zero();

    Vector3f v1 = p_right.head<3>() - p_center.head<3>();
    Vector3f v2 = p_down.head<3>() - p_center.head<3>();
    Vector3f normal = v1.cross(v2);
    return safe_normalize(normal);
}

// ============================================================================
// Coordinate System Conversion
// ============================================================================

// Kinect depth frame to metric space
inline Vector3f kinect_depth_to_metric(int x, int y, uint16_t raw_depth,
                                       const DepthFrame& frame) {
    float depth_m = raw_depth * frame.depth_scale;
    return depth_to_3d(x, y, depth_m, frame.fx, frame.fy, frame.cx, frame.cy);
}

// Metric space to Unity space (convert handedness, scale)
// Unity: right-handed, Y-up
// Kinect: right-handed, Y-down
inline Vector3f metric_to_unity(const Vector3f& p) {
    // Kinect: X right, Y down, Z forward
    // Unity: X right, Y up, Z forward
    // Simple conversion: flip Y
    return Vector3f(p.x(), -p.y(), p.z());
}

inline Vector3f unity_to_metric(const Vector3f& p) {
    return Vector3f(p.x(), -p.y(), p.z());
}

// ============================================================================
// Quaternion Utils
// ============================================================================

inline Quaternionf angle_axis_to_quaternion(const Vector3f& axis, float angle) {
    Vector3f normalized_axis = safe_normalize(axis);
    return Quaternionf(Eigen::AngleAxisf(angle, normalized_axis));
}

inline void quaternion_to_angle_axis(const Quaternionf& q, Vector3f& axis,
                                     float& angle) {
    Eigen::AngleAxisf aa(q);
    axis = aa.axis();
    angle = aa.angle();
}

// ============================================================================
// Distance Metrics
// ============================================================================

inline float point_to_plane_distance(const Vector3f& point,
                                     const Vector3f& plane_point,
                                     const Vector3f& plane_normal) {
    Vector3f v = point - plane_point;
    return std::abs(v.dot(plane_normal));
}

inline float point_to_plane_signed_distance(const Vector3f& point,
                                           const Vector3f& plane_point,
                                           const Vector3f& plane_normal) {
    Vector3f v = point - plane_point;
    return v.dot(plane_normal);
}

// ============================================================================
// Matrix Operations
// ============================================================================

inline bool invert_4x4(const Matrix4f& in, Matrix4f& out) {
    out = in.inverse();
    return true;
}

inline Matrix3f skew_symmetric(const Vector3f& v) {
    Matrix3f S;
    S << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return S;
}

// ============================================================================
// Pyramid Level Computation
// ============================================================================

inline int compute_pyramid_level(int target_width, int base_width = 640) {
    int level = 0;
    while (base_width > target_width && base_width > 8) {
        base_width /= 2;
        level++;
    }
    return level;
}

inline int pyramid_width(int level, int base_width = 640) {
    return base_width >> level;
}

inline int pyramid_height(int level, int base_height = 480) {
    return base_height >> level;
}

}  // namespace kf::math
