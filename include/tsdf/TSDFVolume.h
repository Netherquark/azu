#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cstdint>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include "utils/CudaUniquePtr.h"
#include "tsdf/VoxelGPU.h"

namespace kfusion {
namespace tsdf {

struct TSDFParams {
    int   resolution     = 256;          // 256³ voxels — much faster than 512³
    float voxel_size     = 0.010f;       // meters per voxel (256 * 0.010 = 2.56m cube)
    float truncation     = 0.030f;       // meters (3 voxels)
    float max_weight     = 128.0f;
    Eigen::Vector3f origin = {-1.28f, -1.28f, 0.0f};
};

struct Voxel {
    float    tsdf   = 1.0f;  // normalized [-1,1]
    float    weight = 0.0f;
    uint8_t  r = 128, g = 128, b = 128;
};

class TSDFVolume {
public:
    explicit TSDFVolume(const TSDFParams& params = TSDFParams{});
    ~TSDFVolume();

    // Non-copyable (large memory block)
    TSDFVolume(const TSDFVolume&) = delete;
    TSDFVolume& operator=(const TSDFVolume&) = delete;

    const TSDFParams& params() const { return params_; }

    /** Replace parameters. If resolution changes, voxel storage is reallocated and the volume is cleared. */
    void setParams(const TSDFParams& p);

    // Integrate a depth frame into the volume
    // pose: world-from-camera 4x4 matrix
    // depth: meters per pixel (FRAME_W x FRAME_H)
    // rgb: optional (FRAME_W x FRAME_H * 3)
    void integrate(const float*            depth_meters,
                   const uint8_t*          rgb,
                   const Eigen::Matrix4f&  pose,
                   float                   fx, float fy,
                   float                   cx, float cy,
                   int                     width, int height);

    // Raycast the TSDF volume to produce a model frame
    void raycast(const Eigen::Matrix4f&  pose,
                 float fx, float fy, float cx, float cy,
                 int width, int height,
                 Eigen::Vector3f*        vertices_out,
                 Eigen::Vector3f*        normals_out,
                 uint8_t*                colors_out = nullptr) const;

    /** Extract all occupied voxels as a point cloud (for full model view). */
    void extractGlobalPointCloud(std::vector<Eigen::Vector3f>& points_out,
                                 std::vector<uint8_t>&         colors_out) const;

    // Reset all voxels to initial state
    void reset();

    // Get voxel at integer coordinates (bounds checked)
    const Voxel& voxelAt(int x, int y, int z) const;
    Voxel&       voxelAt(int x, int y, int z);

    // Volume usage: fraction of voxels with weight > 0
    float usageFraction() const;

    // integrated frame count
    int integratedFrames() const { return integrated_frames_.load(); }

    // Voxel data for mesh extraction (read-only)
    const std::vector<Voxel>& voxelData() const { return voxels_; }

    // Convert world position to voxel index
    Eigen::Vector3i worldToVoxel(const Eigen::Vector3f& world) const;
    Eigen::Vector3f voxelToWorld(const Eigen::Vector3i& v) const;
    Eigen::Vector3f voxelToWorld(int x, int y, int z) const;

#ifdef CUDA_ENABLED
    void initGPU();
    void freeGPU();
    void syncToGPU();
    void syncFromGPU();
    void integrateGPU(const float* depth_meters,
                      const uint8_t* rgb,
                      const Eigen::Matrix4f& pose,
                      float fx, float fy, 
                      float cx, float cy,
                      int width, int height);
    void raycastGPU(const Eigen::Matrix4f& pose,
                    float fx, float fy, float cx, float cy,
                    int width, int height,
                    float3* d_vertices, float3* d_normals,
                    uchar3* d_colors = nullptr);

    void extractGlobalPointCloudGPU(std::vector<Eigen::Vector3f>& points_out,
                                    std::vector<uint8_t>&         colors_out) const;

    void* getGPUVoxels() const { return (void*)d_voxels_.get(); }
#endif

    void setGPUEnabled(bool enabled) { gpu_enabled_ = enabled; }
    bool isGPUEnabled() const { return gpu_enabled_; }

private:
    TSDFParams           params_;
    std::vector<Voxel>   voxels_;
    std::atomic<int>     integrated_frames_{0};
    mutable std::shared_mutex mutex_;

    inline int idx(int x, int y, int z) const {
        return z * params_.resolution * params_.resolution
             + y * params_.resolution
             + x;
    }

    bool inBounds(int x, int y, int z) const {
        return x >= 0 && x < params_.resolution &&
               y >= 0 && y < params_.resolution &&
               z >= 0 && z < params_.resolution;
    }

    // CPU integration kernel
    void integrateCPU(const float* depth_meters,
                      const uint8_t* rgb,
                      const Eigen::Matrix4f& pose,
                      float fx, float fy, float cx, float cy,
                      int width, int height);

#ifdef CUDA_ENABLED
    // GPU state
    utils::CudaUniquePtr<VoxelGPU> d_voxels_; 
    utils::CudaUniquePtr<float> d_depth_;
    utils::CudaUniquePtr<uint8_t> d_rgb_;
    bool    gpu_valid_ = false;
#endif
    bool    gpu_enabled_ = false;

    // Internal helpers
    float getTSDF(const Eigen::Vector3f& world_pos) const;
    Eigen::Vector3f computeNormal(const Eigen::Vector3f& world_pos) const;
};

} // namespace tsdf
} // namespace kfusion
