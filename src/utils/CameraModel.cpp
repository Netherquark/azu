#include "utils/CameraModel.hpp"

namespace kf {

CameraModel::CameraModel(float fx, float fy, float cx, float cy, int width,
                         int height)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), width_(width), height_(height) {}

Matrix3f CameraModel::K() const {
    if (K_dirty_) {
        K_cached_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        K_inv_cached_ = K_cached_.inverse();
        K_dirty_ = false;
    }
    return K_cached_;
}

Matrix3f CameraModel::K_inv() const {
    if (K_dirty_) {
        (void)K();  // Trigger cache computation
    }
    return K_inv_cached_;
}

Vector3f CameraModel::unproject(float x, float y, float depth_m) const {
    if (depth_m <= 0.0f) return Vector3f::Zero();
    Vector3f p_norm(x, y, 1.0f);
    Vector3f p_cam = K_inv() * p_norm;
    return depth_m * p_cam;
}

Vector2f CameraModel::project(const Vector3f& point) const {
    if (point.z() <= 0.0f) return Vector2f::Zero();
    Vector3f p_cam = K() * point;
    return p_cam.head<2>() / p_cam.z();
}

Vector2f CameraModel::project_with_check(const Vector3f& point) const {
    Vector2f p = project(point);
    // Check if in bounds
    if (p.x() < 0 || p.x() >= width_ || p.y() < 0 || p.y() >= height_) {
        return Vector2f(-1, -1);
    }
    return p;
}

CameraModel CameraModel::get_pyramid_level(int level) const {
    int scale = 1 << level;  // 2^level
    return CameraModel(fx_ / scale, fy_ / scale, cx_ / scale, cy_ / scale,
                      width_ / scale, height_ / scale);
}

void CameraModel::invalidate_cache() {
    K_dirty_ = true;
}

}  // namespace kf
