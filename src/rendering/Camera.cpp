#include "rendering/Camera.h"
#include <cmath>
#include <algorithm>

namespace kfusion {
namespace rendering {

Camera::Camera() {
    reset();
}

void Camera::rotate(float dx, float dy) {
    float sensitivity = (mode_ == Mode::Orbit) ? 0.01f : 0.005f;
    yaw_   -= dx * sensitivity;
    pitch_ -= dy * sensitivity;
    pitch_  = std::clamp(pitch_, -1.5f, 1.5f);
}

void Camera::zoom(float delta) {
    if (mode_ == Mode::Orbit) {
        distance_ -= delta * 0.3f;
        distance_  = std::clamp(distance_, 0.1f, 100.0f);
    } else {
        move(Eigen::Vector3f(0.0f, 0.0f, -1.0f), delta * 0.3f);
    }
}

void Camera::pan(float dx, float dy) {
    if (mode_ == Mode::Orbit) {
        float cos_y = std::cos(yaw_);
        float sin_y = std::sin(yaw_);
        float cos_p = std::cos(pitch_);

        Eigen::Vector3f right(cos_y, 0.0f, -sin_y);
        Eigen::Vector3f up(-sin_y * std::sin(pitch_), cos_p, -cos_y * std::sin(pitch_));

        target_ += right * (-dx * 0.005f * distance_)
                 + up    * (-dy * 0.005f * distance_);
    } else {
        move(Eigen::Vector3f(1.0f, 0.0f, 0.0f), -dx * 0.005f);
        move(Eigen::Vector3f(0.0f, 1.0f, 0.0f),  dy * 0.005f);
    }
}

void Camera::move(const Eigen::Vector3f& dir, float amount) {
    float cos_y = std::cos(yaw_);
    float sin_y = std::sin(yaw_);
    float cos_p = std::cos(pitch_);
    float sin_p = std::sin(pitch_);

    Eigen::Vector3f forward(-sin_y * cos_p, sin_p, -cos_y * cos_p);
    Eigen::Vector3f right(cos_y, 0.0f, -sin_y);
    Eigen::Vector3f up = right.cross(forward).normalized();

    if (mode_ == Mode::Free) {
        position_ += right * (dir.x() * amount) +
                     up * (dir.y() * amount) +
                     forward * (dir.z() * amount);
    } else {
        // In Orbit mode, move() translates the target
        target_ += right * (dir.x() * amount) +
                   up * (dir.y() * amount) +
                   forward * (dir.z() * amount);
    }
}

void Camera::setMode(Mode mode) {
    if (mode_ == mode) return;
    
    if (mode == Mode::Free) {
        updateFreeFromOrbit();
    } else {
        updateOrbitFromFree();
    }
    mode_ = mode;
}

void Camera::updateFreeFromOrbit() {
    float cos_y = std::cos(yaw_);
    float sin_y = std::sin(yaw_);
    float cos_p = std::cos(pitch_);
    float sin_p = std::sin(pitch_);

    Eigen::Vector3f offset(
        distance_ * cos_p * sin_y,
        distance_ * sin_p,
        distance_ * cos_p * cos_y
    );
    position_ = target_ + offset;
}

void Camera::updateOrbitFromFree() {
    float cos_y = std::cos(yaw_);
    float sin_y = std::sin(yaw_);
    float cos_p = std::cos(pitch_);
    float sin_p = std::sin(pitch_);

    Eigen::Vector3f forward(-sin_y * cos_p, sin_p, -cos_y * cos_p);
    target_ = position_ + forward * distance_;
}

void Camera::reset() {
    mode_ = Mode::Orbit;
    yaw_       = 0.0f;
    pitch_     = 0.3f;
    roll_      = 0.0f;
    distance_  = 3.0f;
    target_    = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    position_  = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
}

Eigen::Matrix4f Camera::viewMatrix() const {
    float cos_y = std::cos(yaw_);
    float sin_y = std::sin(yaw_);
    float cos_p = std::cos(pitch_);
    float sin_p = std::sin(pitch_);

    Eigen::Vector3f forward(-sin_y * cos_p, sin_p, -cos_y * cos_p);
    Eigen::Vector3f right(cos_y, 0.0f, -sin_y);
    Eigen::Vector3f up = right.cross(forward).normalized();

    Eigen::Vector3f eye;
    if (mode_ == Mode::Orbit) {
        Eigen::Vector3f offset(
            distance_ * cos_p * sin_y,
            distance_ * sin_p,
            distance_ * cos_p * cos_y
        );
        eye = target_ + offset;
    } else {
        eye = position_;
    }

    // Apply roll
    Eigen::AngleAxisf roll_rot(roll_, forward);
    up = roll_rot * up;
    right = forward.cross(up).normalized();

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V(0,0) =  right.x(); V(0,1) =  right.y(); V(0,2) =  right.z(); V(0,3) = -right.dot(eye);
    V(1,0) =  up.x();    V(1,1) =  up.y();    V(1,2) =  up.z();    V(1,3) = -up.dot(eye);
    V(2,0) = -forward.x(); V(2,1) = -forward.y(); V(2,2) = -forward.z(); V(2,3) =  forward.dot(eye);
    return V;
}

Eigen::Matrix4f Camera::projectionMatrix(float aspect, float near_z, float far_z) const {
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
