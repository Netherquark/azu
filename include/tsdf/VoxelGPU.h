#pragma once
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ALIGN32 __align__(32)
#else
#define ALIGN32 alignas(32)
#endif

namespace kfusion {
namespace tsdf {

// Unified GPU Voxel Structure
struct ALIGN32 VoxelGPU {
    float tsdf;     // Pre-divided tsdf average
    float weight;   // Capped integration weight
    float r;        // Pre-divided r average
    float g;        // Pre-divided g average
    float b;        // Pre-divided b average
    float padding[3];
};

} // namespace tsdf
} // namespace kfusion
