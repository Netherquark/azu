#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <memory>
#include <cstdint>

namespace kf {

// ============================================================================
// Basic Math Types
// ============================================================================

using Vector2f = Eigen::Vector2f;
using Vector3f = Eigen::Vector3f;
using Vector4f = Eigen::Vector4f;
using Matrix3f = Eigen::Matrix3f;
using Matrix4f = Eigen::Matrix4f;
using Quaternionf = Eigen::Quaternionf;

using Vector2d = Eigen::Vector2d;
using Vector3d = Eigen::Vector3d;
using Vector4d = Eigen::Vector4d;
using Matrix3d = Eigen::Matrix3d;
using Matrix4d = Eigen::Matrix4d;
using Quaterniond = Eigen::Quaterniond;

// ============================================================================
// Depth Frame Data
// ============================================================================

struct DepthFrame {
    static constexpr int WIDTH = 640;
    static constexpr int HEIGHT = 480;
    static constexpr size_t TOTAL_PIXELS = WIDTH * HEIGHT;

    std::vector<uint16_t> data;        // Raw depth (11-bit)
    int64_t timestamp_us;
    float fx, fy, cx, cy;              // Camera intrinsics
    float depth_scale;                 // Scale factor for depth conversion

    DepthFrame() 
        : data(TOTAL_PIXELS, 0), timestamp_us(0),
          fx(525.0f), fy(525.0f), cx(319.5f), cy(239.5f),
          depth_scale(0.001f) {}

    float get_depth_m(int x, int y) const {
        if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return 0.0f;
        return static_cast<float>(data[y * WIDTH + x]) * depth_scale;
    }

    uint16_t get_raw_depth(int x, int y) const {
        if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return 0;
        return data[y * WIDTH + x];
    }
};

// ============================================================================
// Color Frame Data
// ============================================================================

struct ColorFrame {
    static constexpr int WIDTH = 640;
    static constexpr int HEIGHT = 480;
    static constexpr size_t TOTAL_PIXELS = WIDTH * HEIGHT;
    static constexpr size_t CHANNELS = 3;

    std::vector<uint8_t> data;  // RGB data (3 channels)
    int64_t timestamp_us;

    ColorFrame() 
        : data(TOTAL_PIXELS * CHANNELS, 0), timestamp_us(0) {}

    void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) return;
        size_t idx = (y * WIDTH + x) * CHANNELS;
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
    }

    void get_pixel(int x, int y, uint8_t& r, uint8_t& g, uint8_t& b) const {
        if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
            r = g = b = 0;
            return;
        }
        size_t idx = (y * WIDTH + x) * CHANNELS;
        r = data[idx];
        g = data[idx + 1];
        b = data[idx + 2];
    }
};

// ============================================================================
// Vertex Map (depth converted to 3D)
// ============================================================================

struct VertexMap {
    std::vector<Vector4f> vertices;  // XYZ + confidence
    std::vector<Vector3f> normals;
    int width, height;

    VertexMap(int w = 640, int h = 480)
        : width(w), height(h) {
        vertices.resize(w * h);
        normals.resize(w * h);
    }

    size_t get_index(int x, int y) const { return y * width + x; }

    Vector4f& vertex(int x, int y) { return vertices[get_index(x, y)]; }
    const Vector4f& vertex(int x, int y) const { return vertices[get_index(x, y)]; }

    Vector3f& normal(int x, int y) { return normals[get_index(x, y)]; }
    const Vector3f& normal(int x, int y) const { return normals[get_index(x, y)]; }
};

// ============================================================================
// TSDF Voxel
// ============================================================================

struct TSDFVoxel {
    float tsdf;      // Truncated Signed Distance Function
    float weight;    // Integration weight
    uint8_t r, g, b; // Color

    TSDFVoxel() : tsdf(0.0f), weight(0.0f), r(0), g(0), b(0) {}
};

// ============================================================================
// Triangle Mesh
// ============================================================================

struct Mesh {
    std::vector<Vector3f> positions;
    std::vector<Vector3f> normals;
    std::vector<uint8_t> colors;  // RGB per vertex
    std::vector<uint32_t> indices; // Triangle indices

    size_t vertex_count() const { return positions.size(); }
    size_t triangle_count() const { return indices.size() / 3; }
    bool is_empty() const { return positions.empty(); }

    void clear() {
        positions.clear();
        normals.clear();
        colors.clear();
        indices.clear();
    }

    void add_vertex(const Vector3f& pos, const Vector3f& normal,
                    uint8_t r, uint8_t g, uint8_t b) {
        positions.push_back(pos);
        normals.push_back(normal.normalized());
        colors.push_back(r);
        colors.push_back(g);
        colors.push_back(b);
    }

    void add_triangle(uint32_t i0, uint32_t i1, uint32_t i2) {
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }
};

// ============================================================================
// Camera Pose (SE(3))
// ============================================================================

struct CameraPose {
    Matrix3f R;      // Rotation
    Vector3f t;      // Translation

    CameraPose() : R(Matrix3f::Identity()), t(Vector3f::Zero()) {}

    explicit CameraPose(const Matrix4f& T) {
        R = T.block<3, 3>(0, 0);
        t = T.block<3, 1>(0, 3);
    }

    Matrix4f to_matrix() const {
        Matrix4f T = Matrix4f::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = t;
        return T;
    }

    Vector3f transform_point(const Vector3f& p) const {
        return R * p + t;
    }

    Vector3f transform_normal(const Vector3f& n) const {
        return R * n;
    }

    CameraPose inverse() const {
        CameraPose inv;
        inv.R = R.transpose();
        inv.t = -inv.R * t;
        return inv;
    }

    CameraPose operator*(const CameraPose& other) const {
        CameraPose result;
        result.R = R * other.R;
        result.t = R * other.t + t;
        return result;
    }
};

// ============================================================================
// ICP Alignment Result
// ============================================================================

struct AlignmentResult {
    bool success;
    CameraPose pose;
    float error;           // Mean squared error
    float inlier_ratio;    // Percentage of inlier correspondences
    int iterations;
    Matrix4f covariance;   // 6x6 reduced to 4x4

    AlignmentResult()
        : success(false), error(std::numeric_limits<float>::max()),
          inlier_ratio(0.0f), iterations(0),
          covariance(Matrix4f::Zero()) {}

    AlignmentResult(const CameraPose& p, float e, float ir, int it)
        : success(true), pose(p), error(e), inlier_ratio(ir),
          iterations(it), covariance(Matrix4f::Zero()) {}
};

// ============================================================================
// Export Result
// ============================================================================

enum class ExportFormat {
    PLY,
    GLB
};

struct ExportResult {
    bool success;
    std::string output_path;
    std::string error_message;
    size_t file_size_bytes;
    float processing_time_ms;

    ExportResult()
        : success(false), file_size_bytes(0), processing_time_ms(0.0f) {}
};

// ============================================================================
// Statistics
// ============================================================================

struct ReconstructionStats {
    int frame_count = 0;
    int integrated_frames = 0;
    int total_triangles = 0;
    float total_voxels_used = 0.0f;
    float voxel_grid_usage_percent = 0.0f;
    float current_icp_error = 0.0f;
    bool tracking_ok = true;
    float fps_capture = 0.0f;
    float fps_tracking = 0.0f;
    float gpu_memory_mb = 0.0f;
    float cpu_memory_mb = 0.0f;
};

// ============================================================================
// Shared Pointers
// ============================================================================

using DepthFramePtr = std::shared_ptr<DepthFrame>;
using ColorFramePtr = std::shared_ptr<ColorFrame>;
using VertexMapPtr = std::shared_ptr<VertexMap>;
using MeshPtr = std::shared_ptr<Mesh>;

}  // namespace kf
