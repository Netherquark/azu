#pragma once

#include "meshing/MeshData.h"
#include "tsdf/TSDFVolume.h"
#include <functional>

namespace kfusion {
namespace meshing {

using ProgressCallback = std::function<void(float)>; // 0..1

class MarchingCubes {
public:
    MarchingCubes();
    ~MarchingCubes();

    // CPU extraction path
    std::shared_ptr<MeshData> extract(const tsdf::TSDFVolume& volume,
                                      ProgressCallback        cb = nullptr);

    // GPU extraction path
    std::shared_ptr<MeshData> extractGPU(const tsdf::TSDFVolume& volume);

    // Look-up tables for Marching Cubes
    static const int edge_table[256];
    static const int tri_table[256][16];

private:
#ifdef CUDA_ENABLED
    // Persistent GPU buffers owned by this instance
    uint32_t* d_voxel_tri_counts_ = nullptr;
    uint32_t* d_voxel_offsets_    = nullptr;
    float3*   d_mesh_vertices_    = nullptr;
    float3*   d_mesh_normals_     = nullptr;
    uint8_t*  d_mesh_colors_      = nullptr;
    
    size_t    max_triangles_      = 2000000;
    int       last_resolution_    = 0;

    void initGPU(int resolution);
    void freeGPU();
#endif

    static Eigen::Vector3f interpolateEdge(
        const Eigen::Vector3f& p1, float v1,
        const Eigen::Vector3f& p2, float v2);

    static Eigen::Vector3f computeNormal(
        const tsdf::TSDFVolume& vol, int x, int y, int z);
};

} // namespace meshing
} // namespace kfusion
