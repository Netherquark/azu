#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kfusion {
namespace rendering {

class OrbitCamera {
public:
    OrbitCamera();

    void orbit(float dx, float dy);
    void zoom(float delta);
    void pan(float dx, float dy);
    void reset();

    Eigen::Matrix4f viewMatrix() const;
    Eigen::Matrix4f projectionMatrix(float aspect, float near_z = 0.01f, float far_z = 100.0f) const;

    float fov_degrees = 60.0f;

private:
    float azimuth_  = 0.0f;    // radians
    float elevation_ = 0.3f;   // radians
    float distance_  = 3.0f;   // meters
    Eigen::Vector3f target_{0.0f, 0.0f, 1.0f};
};

} // namespace rendering
} // namespace kfusion
