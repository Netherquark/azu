#include "rendering/Camera.h"
#include <cmath>
#include <algorithm>

namespace kfusion {
namespace rendering {

OrbitCamera::OrbitCamera() = default;

void OrbitCamera::orbit(float dx, float dy) {
    azimuth_   += dx * 0.01f;
    elevation_ += dy * 0.01f;
    elevation_  = std::clamp(elevation_, -1.5f, 1.5f);
}

void OrbitCamera::zoom(float delta) {
    distance_ -= delta * 0.3f;
    distance_  = std::clamp(distance_, 0.1f, 100.0f);
}

void OrbitCamera::pan(float dx, float dy) {
    // Build right and up from current view
    float cos_az  = std::cos(azimuth_);
    float sin_az  = std::sin(azimuth_);
    float cos_el  = std::cos(elevation_);

    Eigen::Vector3f right( cos_az, 0.0f, -sin_az);
    Eigen::Vector3f up   (-sin_az * std::sin(elevation_), cos_el, -cos_az * std::sin(elevation_));

    target_ += right * (-dx * 0.005f * distance_)
             + up    * ( dy * 0.005f * distance_);
}

void OrbitCamera::reset() {
    azimuth_   = 0.0f;
    elevation_ = 0.3f;
    distance_  = 3.0f;
    target_    = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
}

Eigen::Matrix4f OrbitCamera::viewMatrix() const {
    float cos_az  = std::cos(azimuth_);
    float sin_az  = std::sin(azimuth_);
    float cos_el  = std::cos(elevation_);
    float sin_el  = std::sin(elevation_);

    // Eye position in spherical coords around target
    Eigen::Vector3f offset(
        distance_ * cos_el * sin_az,
        distance_ * sin_el,
        distance_ * cos_el * cos_az
    );
    Eigen::Vector3f eye = target_ + offset;
    Eigen::Vector3f up(0.0f, 1.0f, 0.0f);

    // Look-at
    Eigen::Vector3f f = (target_ - eye).normalized();
    Eigen::Vector3f r = f.cross(up).normalized();
    Eigen::Vector3f u = r.cross(f);

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V(0,0) =  r.x(); V(0,1) =  r.y(); V(0,2) =  r.z(); V(0,3) = -r.dot(eye);
    V(1,0) =  u.x(); V(1,1) =  u.y(); V(1,2) =  u.z(); V(1,3) = -u.dot(eye);
    V(2,0) = -f.x(); V(2,1) = -f.y(); V(2,2) = -f.z(); V(2,3) =  f.dot(eye);
    return V;
}

Eigen::Matrix4f OrbitCamera::projectionMatrix(float aspect, float near_z, float far_z) const {
    float fov_rad  = fov_degrees * M_PI / 180.0f;
    float tan_half = std::tan(fov_rad * 0.5f);

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0,0) = 1.0f / (aspect * tan_half);
    P(1,1) = 1.0f / tan_half;
    P(2,2) = -(far_z + near_z) / (far_z - near_z);
    P(2,3) = -(2.0f * far_z * near_z) / (far_z - near_z);
    P(3,2) = -1.0f;
    return P;
}

} // namespace rendering
} // namespace kfusion
