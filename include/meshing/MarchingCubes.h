#pragma once

#include "meshing/MeshData.h"
#include "tsdf/TSDFVolume.h"
#include <functional>

namespace kfusion {
namespace meshing {

using ProgressCallback = std::function<void(float)>; // 0..1

class MarchingCubes {
public:
    // Extract mesh from TSDF volume
    // If progress_cb is set, called with fraction 0..1 during extraction
    static MeshData extract(const tsdf::TSDFVolume& volume,
                            ProgressCallback        progress_cb = nullptr);

    // Marching cubes edge table (256 entries)
    static const int edge_table[256];
    // Triangle table (256 x 16 entries)
    static const int tri_table[256][16];

private:
    static Eigen::Vector3f interpolateEdge(
        const Eigen::Vector3f& p1, float v1,
        const Eigen::Vector3f& p2, float v2);

    static Eigen::Vector3f computeNormal(
        const tsdf::TSDFVolume& vol, int x, int y, int z);
};

} // namespace meshing
} // namespace kfusion
