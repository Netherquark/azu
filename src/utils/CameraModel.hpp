#pragma once

#include "Types.hpp"

namespace kf {

class CameraModel {
public:
    // Kinect v1 standard intrinsics
    static constexpr float KINECT_V1_FX = 525.0f;
    static constexpr float KINECT_V1_FY = 525.0f;
    static constexpr float KINECT_V1_CX = 319.5f;
    static constexpr float KINECT_V1_CY = 239.5f;

    CameraModel(float fx = KINECT_V1_FX, float fy = KINECT_V1_FY,
                float cx = KINECT_V1_CX, float cy = KINECT_V1_CY,
                int width = 640, int height = 480);

    // Getters
    float fx() const { return fx_; }
    float fy() const { return fy_; }
    float cx() const { return cx_; }
    float cy() const { return cy_; }
    int width() const { return width_; }
    int height() const { return height_; }

    // Camera matrix
    Matrix3f K() const;
    Matrix3f K_inv() const;

    // Operations
    Vector3f unproject(float x, float y, float depth_m) const;
    Vector2f project(const Vector3f& point) const;
    Vector2f project_with_check(const Vector3f& point) const;

    // Pyramid levels for coarse-to-fine tracking
    CameraModel get_pyramid_level(int level) const;

private:
    float fx_, fy_, cx_, cy_;
    int width_, height_;
    mutable Matrix3f K_cached_;
    mutable Matrix3f K_inv_cached_;
    mutable bool K_dirty_ = true;

    void invalidate_cache();
};

}  // namespace kf
