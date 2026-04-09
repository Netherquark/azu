#pragma once

#include <vector>
#include <Eigen/Core>
#include <mutex>
#include <memory>

namespace kfusion {
namespace meshing {

struct Triangle {
    Eigen::Vector3f v[3];
    Eigen::Vector3f n[3];
    uint8_t         c[3][3]; // RGB per vertex
};

struct MeshData {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> normals;
    std::vector<uint8_t>         colors;    // RGB per vertex (positions.size() * 3)
    std::vector<uint32_t>        indices;   // triangle list

    void clear() {
        positions.clear();
        normals.clear();
        colors.clear();
        indices.clear();
    }

    bool empty() const { return positions.empty(); }
    size_t triangleCount() const { return indices.size() / 3; }

    void reserve(size_t tri_count) {
        positions.reserve(tri_count * 3);
        normals.reserve(tri_count * 3);
        colors.reserve(tri_count * 3 * 3);
        indices.reserve(tri_count * 3);
    }

    void addTriangle(const Triangle& tri) {
        for (int i = 0; i < 3; ++i) {
            uint32_t vidx = static_cast<uint32_t>(positions.size());
            positions.push_back(tri.v[i]);
            normals.push_back(tri.n[i]);
            colors.push_back(tri.c[i][0]);
            colors.push_back(tri.c[i][1]);
            colors.push_back(tri.c[i][2]);
            indices.push_back(vidx);
        }
    }
};

// Thread-safe mesh container used between extraction thread and render thread
class SharedMesh {
public:
    void update(std::shared_ptr<MeshData> mesh) {
        std::lock_guard<std::mutex> lk(mutex_);
        mesh_ = std::move(mesh);
        version_++;
    }

    // Returns shared pointer for zero-copy rendering access
    std::shared_ptr<MeshData> snapshot(uint64_t& version_out) const {
        std::lock_guard<std::mutex> lk(mutex_);
        version_out = version_;
        return mesh_;
    }

    uint64_t version() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return version_;
    }

private:
    mutable std::mutex        mutex_;
    std::shared_ptr<MeshData> mesh_;
    uint64_t           version_ = 0;
};

} // namespace meshing
} // namespace kfusion
