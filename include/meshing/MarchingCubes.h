#pragma once

#include "meshing/MeshData.h"
#include "tsdf/TSDFVolume.h"
#include <functional>

namespace kfusion {
namespace meshing {

using ProgressCallback = std::function<void(float)>; // 0..1

class MarchingCubes {
public:
    // CPU extraction path
    static MeshData extract(const tsdf::TSDFVolume& volume,
                            ProgressCallback        cb = nullptr);

    // GPU extraction path
    static MeshData extractGPU(const tsdf::TSDFVolume& volume);

    // Look-up tables for Marching Cubes
    static const int edge_table[256];
    static const int tri_table[256][16];

#ifdef CUDA_ENABLED
private:
    // Persistent GPU buffers
    static uint32_t* d_voxel_tri_counts_;
    static uint32_t* d_voxel_offsets_;
    static float3*   d_mesh_vertices_;
    static float3*   d_mesh_normals_;
    static uint8_t*  d_mesh_colors_;
    
    static void initGPU(int resolution);
    static void freeGPU();
#endif

private:
    static Eigen::Vector3f interpolateEdge(
        const Eigen::Vector3f& p1, float v1,
        const Eigen::Vector3f& p2, float v2);

    static Eigen::Vector3f computeNormal(
        const tsdf::TSDFVolume& vol, int x, int y, int z);
};

} // namespace meshing
} // namespace kfusion
