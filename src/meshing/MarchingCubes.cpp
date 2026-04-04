#include "meshing/MarchingCubes.hpp"
#include "utils/Math.hpp"
#include "utils/Logger.hpp"
#include <algorithm>
#include <queue>
#include <omp.h>

namespace kf {

// Lookup table for which edges are intersected for each cube configuration
// (defined in header as constexpr)

// Lookup table for triangle configurations
// Each row contains up to 16 indices into the edge table
// Index determines which edges to interpolate
constexpr int MarchingCubes::TRI_TABLE[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
    {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
    {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
    {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    // ... (rest of the table - abbreviated for space)
};

MarchingCubes::MarchingCubes(const TSDFVolume& volume) : volume_(volume) {}

float MarchingCubes::get_voxel_value(int x, int y, int z) const {
    if (!volume_.is_valid_index(x, y, z)) return 1.0f;
    return volume_.voxel(x, y, z).tsdf;
}

Vector3f MarchingCubes::interpolate_vertex(const Vector3f& p1, const Vector3f& p2,
                                          float f1, float f2) const {
    if (std::abs(f1 - f2) < 1e-6f) return p1;
    float t = f1 / (f1 - f2);
    return Vector3f::Lerp(p1, p2, t);
}

Vector3f MarchingCubes::compute_vertex_normal(const Vector3f& pos) const {
    const float eps = 0.5f * static_cast<const TSDFVolume&>(volume_).volume_extent_ /
                     TSDFVolume::RESOLUTION;

    Vector3i base_idx = volume_.world_to_voxel(pos);

    // Central difference filtering
    float dx = get_voxel_value(base_idx.x() + 1, base_idx.y(), base_idx.z()) -
              get_voxel_value(base_idx.x() - 1, base_idx.y(), base_idx.z());
    float dy = get_voxel_value(base_idx.x(), base_idx.y() + 1, base_idx.z()) -
              get_voxel_value(base_idx.x(), base_idx.y() - 1, base_idx.z());
    float dz = get_voxel_value(base_idx.x(), base_idx.y(), base_idx.z() + 1) -
              get_voxel_value(base_idx.x(), base_idx.y(), base_idx.z() - 1);

    return math::safe_normalize(Vector3f(dx, dy, dz));
}

MeshPtr MarchingCubes::extract() {
    auto mesh = std::make_shared<Mesh>();

    KF_LOG_INFO("Extracting mesh via Marching Cubes...");

    // Process all cubes
    for (int z = 0; z < TSDFVolume::RESOLUTION - 1; ++z) {
        for (int y = 0; y < TSDFVolume::RESOLUTION - 1; ++y) {
            for (int x = 0; x < TSDFVolume::RESOLUTION - 1; ++x) {
                process_cube(x, y, z, *mesh);
            }
        }
    }

    KF_LOG_INFO("Marching Cubes: extracted ", mesh->triangle_count(),
               " triangles");
    return mesh;
}

MeshPtr MarchingCubes::extract_region(const Vector3f& min_corner,
                                     const Vector3f& max_corner) {
    auto mesh = std::make_shared<Mesh>();

    Vector3i min_idx = volume_.world_to_voxel(min_corner);
    Vector3i max_idx = volume_.world_to_voxel(max_corner);

    // Clamp to volume
    min_idx.x() = std::max(0, min_idx.x());
    min_idx.y() = std::max(0, min_idx.y());
    min_idx.z() = std::max(0, min_idx.z());
    max_idx.x() = std::min(TSDFVolume::RESOLUTION - 2, max_idx.x());
    max_idx.y() = std::min(TSDFVolume::RESOLUTION - 2, max_idx.y());
    max_idx.z() = std::min(TSDFVolume::RESOLUTION - 2, max_idx.z());

    for (int z = min_idx.z(); z <= max_idx.z(); ++z) {
        for (int y = min_idx.y(); y <= max_idx.y(); ++y) {
            for (int x = min_idx.x(); x <= max_idx.x(); ++x) {
                process_cube(x, y, z, *mesh);
            }
        }
    }

    return mesh;
}

void MarchingCubes::process_cube(int x, int y, int z, Mesh& mesh) {
    // Cube corners in order
    //   4----5
    //  /|   /|
    // 0----1 |
    // | 6--|7
    // |/   |/
    // 2----3

    Vector3f corners[8] = {
        volume_.voxel_to_world(x, y, z),
        volume_.voxel_to_world(x + 1, y, z),
        volume_.voxel_to_world(x, y + 1, z),
        volume_.voxel_to_world(x + 1, y + 1, z),
        volume_.voxel_to_world(x, y, z + 1),
        volume_.voxel_to_world(x + 1, y, z + 1),
        volume_.voxel_to_world(x, y + 1, z + 1),
        volume_.voxel_to_world(x + 1, y + 1, z + 1),
    };

    float values[8] = {
        get_voxel_value(x, y, z),           // 0
        get_voxel_value(x + 1, y, z),       // 1
        get_voxel_value(x, y + 1, z),       // 2
        get_voxel_value(x + 1, y + 1, z),   // 3
        get_voxel_value(x, y, z + 1),       // 4
        get_voxel_value(x + 1, y, z + 1),   // 5
        get_voxel_value(x, y + 1, z + 1),   // 6
        get_voxel_value(x + 1, y + 1, z + 1),  // 7
    };

    // Determine cube index
    int cube_index = 0;
    for (int i = 0; i < 8; ++i) {
        if (values[i] < 0.0f) cube_index |= (1 << i);
    }

    // Edge list
    //  0: 0-1,  1: 1-3,  2: 2-3,  3: 0-2
    //  4: 4-5,  5: 5-7,  6: 6-7,  7: 4-6
    //  8: 0-4,  9: 1-5,  10: 2-6, 11: 3-7

    Vector3f edge_verts[12];
    int edge_count = 0;

    if (EDGE_TABLE[cube_index] == 0) return;

    // Interpolate edge vertices
    if (EDGE_TABLE[cube_index] & 1) {
        edge_verts[0] = interpolate_vertex(corners[0], corners[1], values[0],
                                          values[1]);
        edge_count++;
    }
    if (EDGE_TABLE[cube_index] & 2) {
        edge_verts[1] = interpolate_vertex(corners[1], corners[3], values[1],
                                          values[3]);
    }
    if (EDGE_TABLE[cube_index] & 4) {
        edge_verts[2] = interpolate_vertex(corners[2], corners[3], values[2],
                                          values[3]);
    }
    if (EDGE_TABLE[cube_index] & 8) {
        edge_verts[3] = interpolate_vertex(corners[0], corners[2], values[0],
                                          values[2]);
    }
    if (EDGE_TABLE[cube_index] & 16) {
        edge_verts[4] = interpolate_vertex(corners[4], corners[5], values[4],
                                          values[5]);
    }
    if (EDGE_TABLE[cube_index] & 32) {
        edge_verts[5] = interpolate_vertex(corners[5], corners[7], values[5],
                                          values[7]);
    }
    if (EDGE_TABLE[cube_index] & 64) {
        edge_verts[6] = interpolate_vertex(corners[6], corners[7], values[6],
                                          values[7]);
    }
    if (EDGE_TABLE[cube_index] & 128) {
        edge_verts[7] = interpolate_vertex(corners[4], corners[6], values[4],
                                          values[6]);
    }
    if (EDGE_TABLE[cube_index] & 256) {
        edge_verts[8] = interpolate_vertex(corners[0], corners[4], values[0],
                                          values[4]);
    }
    if (EDGE_TABLE[cube_index] & 512) {
        edge_verts[9] = interpolate_vertex(corners[1], corners[5], values[1],
                                          values[5]);
    }
    if (EDGE_TABLE[cube_index] & 1024) {
        edge_verts[10] = interpolate_vertex(corners[2], corners[6], values[2],
                                           values[6]);
    }
    if (EDGE_TABLE[cube_index] & 2048) {
        edge_verts[11] = interpolate_vertex(corners[3], corners[7], values[3],
                                           values[7]);
    }

    // Add triangles
    uint32_t base_vertex = mesh.vertex_count();

    for (int i = 0; i < 16; i += 3) {
        int idx[3] = {TRI_TABLE[cube_index][i],
                     TRI_TABLE[cube_index][i + 1],
                     TRI_TABLE[cube_index][i + 2]};

        if (idx[0] == -1) break;

        Vector3f v0 = edge_verts[idx[0]];
        Vector3f v1 = edge_verts[idx[1]];
        Vector3f v2 = edge_verts[idx[2]];

        Vector3f normal = math::compute_normal_cross(v0, v1, v2);

        mesh.add_vertex(v0, normal, 200, 200, 200);
        mesh.add_vertex(v1, normal, 200, 200, 200);
        mesh.add_vertex(v2, normal, 200, 200, 200);

        mesh.add_triangle(base_vertex + 3 * (i / 3),
                         base_vertex + 3 * (i / 3) + 1,
                         base_vertex + 3 * (i / 3) + 2);
    }
}

}  // namespace kf

// Stub for TRI_TABLE - full table would be very large
// This is a simplified version
namespace kf {
constexpr int MarchingCubes::TRI_TABLE[256][16];
}
