#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <string>

#include "sensor/KinectSensor.h"
#include "sensor/FrameData.h"
#include "tracking/ICPTracker.h"
#include "tsdf/TSDFVolume.h"
#include "meshing/MarchingCubes.h"
#include "meshing/MeshData.h"
#include <Eigen/Core>

namespace kfusion {
namespace app {

enum class PipelineState {
    Idle,
    Running,
    TrackingLost,
    Error,
    Stopped
};

struct PipelineMetrics {
    float    capture_fps        = 0.0f;
    float    tracking_fps       = 0.0f;
    int      frame_count        = 0;
    int      integrated_frames  = 0;
    float    icp_error          = 0.0f;
    bool     tracking_ok        = true;
    float    volume_usage_pct   = 0.0f;
    size_t   mesh_triangles     = 0;
    float    mesh_extract_pct   = 0.0f;
    float    export_pct         = 0.0f;
    PipelineState state         = PipelineState::Idle;
};

using MetricsCallback    = std::function<void(const PipelineMetrics&)>;
using FrameReadyCallback = std::function<void(const sensor::FrameData&)>;
using MeshReadyCallback  = std::function<void()>; // mesh updated in shared mesh

class PipelineController {
public:
    PipelineController();
    ~PipelineController();

    bool start();
    void stop();
    void reset();

    bool exportPLY(const std::string& path);
    bool exportGLB(const std::string& path);

    void setMetricsCallback(MetricsCallback cb)    { metrics_cb_ = std::move(cb); }
    void setFrameReadyCallback(FrameReadyCallback cb) { frame_ready_cb_ = std::move(cb); }
    void setMeshReadyCallback(MeshReadyCallback cb)   { mesh_ready_cb_ = std::move(cb); }

    meshing::SharedMesh& sharedMesh() { return shared_mesh_; }
    const PipelineMetrics& metrics() const { return metrics_; }
    PipelineState state() const { return state_.load(); }

private:
    // Components
    std::unique_ptr<sensor::KinectSensor> sensor_;
    std::unique_ptr<tracking::ICPTracker> tracker_;
    std::unique_ptr<tsdf::TSDFVolume>     tsdf_;
    meshing::SharedMesh                   shared_mesh_;

    // State
    std::atomic<PipelineState> state_{PipelineState::Idle};
    Eigen::Matrix4f            current_pose_;
    std::mutex                 pose_mutex_;

    // Metrics
    PipelineMetrics            metrics_;
    mutable std::mutex         metrics_mutex_;
    MetricsCallback            metrics_cb_;
    FrameReadyCallback         frame_ready_cb_;
    MeshReadyCallback          mesh_ready_cb_;

    // Threads
    std::thread                tracking_thread_;
    std::thread                integration_thread_;
    std::thread                meshing_thread_;
    std::atomic<bool>          running_{false};

    // Raw frame queue (sensor callback → tracking thread)
    std::queue<std::shared_ptr<sensor::RawFrame>>  raw_queue_;
    std::mutex                                     tracking_queue_mutex_;
    std::condition_variable                        tracking_queue_cv_;

    std::queue<std::shared_ptr<sensor::FrameData>> integration_queue_;
    std::mutex                                     integration_queue_mutex_;
    std::condition_variable                        integration_queue_cv_;

    // Timing
    std::chrono::steady_clock::time_point last_capture_time_;
    std::chrono::steady_clock::time_point last_tracking_time_;
    int                                   frame_count_    = 0;
    bool                                  first_frame_    = true;

    // Tracking model frame (raycasted)
    tracking::ModelFrame                  model_frame_;
    std::mutex                            model_frame_mutex_;

    // Mesh extraction trigger
    std::atomic<bool>                     mesh_extraction_requested_{false};
    std::atomic<float>                    mesh_extract_progress_{0.0f};
    std::atomic<float>                    export_progress_{0.0f};

    void onRawFrame(const sensor::RawFrame& raw);
    void trackingLoop();
    void integrationLoop();
    void meshingLoop();

    void updateMetrics();
    void postMetrics();
};

} // namespace app
} // namespace kfusion
