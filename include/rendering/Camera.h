#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kfusion {
namespace rendering {

class Camera {
public:
    enum class Mode {
        Orbit,
        Free
    };

    Camera();

    void rotate(float dx, float dy);
    void zoom(float delta);
    void pan(float dx, float dy);
    void move(const Eigen::Vector3f& dir, float amount);
    void reset();

    void setMode(Mode mode);
    Mode mode() const { return mode_; }

    void setAzimuth(float az) { azimuth_ = az; }
    void setElevation(float el) { elevation_ = el; }
    void setRoll(float r) { roll_ = r; }

    float azimuth() const { return azimuth_; }
    float elevation() const { return elevation_; }
    float roll() const { return roll_; }

    void setTarget(const Eigen::Vector3f& t) { target_ = t; }
    Eigen::Vector3f target() const { return target_; }

    Eigen::Matrix4f viewMatrix() const;
    Eigen::Matrix4f projectionMatrix(float aspect, float near_z = 0.01f, float far_z = 100.0f) const;

    float fov_degrees = 60.0f;

private:
    Mode mode_ = Mode::Orbit;

    // Rotation (radians)
    float yaw_   = 0.0f;
    float pitch_ = 0.3f;
    float roll_  = 0.0f;

    // Orbit-specific
    float distance_  = 3.0f;
    Eigen::Vector3f target_{0.0f, 0.0f, 1.0f};

    // Free-specific
    Eigen::Vector3f position_{0.0f, 0.0f, 0.0f};

    void updateFreeFromOrbit();
    void updateOrbitFromFree();
};

} // namespace rendering
} // namespace kfusion
