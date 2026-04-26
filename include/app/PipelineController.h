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
#include "sensor/SignalConditioner.h"
#include "tracking/ICPTracker.h"
#include "tsdf/TSDFVolume.h"
#include "meshing/MarchingCubes.h"
#include "meshing/MeshData.h"
#include "app/FusionHyperparams.h"
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

    /** True while capture + pipeline worker threads are active. */
    bool isRunning() const { return running_.load(); }

    bool exportPLY(const std::string& path);
    bool exportGLB(const std::string& path);

    /** Optional subscriber; not invoked automatically — poll metricsSnapshot() from a timer instead. */
    void setMetricsCallback(MetricsCallback cb)    { metrics_cb_ = std::move(cb); }
    void setFrameReadyCallback(FrameReadyCallback cb) { frame_ready_cb_ = std::move(cb); }
    void setMeshReadyCallback(MeshReadyCallback cb)   { mesh_ready_cb_ = std::move(cb); }

    meshing::SharedMesh& sharedMesh() { return shared_mesh_; }
    /** Thread-safe copy for UI / diagnostics (locks internal metrics mutex). */
    PipelineMetrics metricsSnapshot() const;
    PipelineState state() const { return state_.load(); }

    FusionHyperparams hyperparamsSnapshot() const;
    void              setHyperparams(const FusionHyperparams& h);

private:
    FusionHyperparams          hyperparams_{FusionHyperparams::defaults()};
    mutable std::mutex         hyper_mutex_;
    sensor::SignalConditioner  signal_conditioner_;

    // Components
    std::unique_ptr<sensor::KinectSensor>    sensor_;
    std::unique_ptr<tracking::ICPTracker>    tracker_;
    std::unique_ptr<tsdf::TSDFVolume>        tsdf_;
    std::unique_ptr<meshing::MarchingCubes>  cubes_;
    meshing::SharedMesh                      shared_mesh_;

    // State
    std::atomic<PipelineState> state_{PipelineState::Idle};
    Eigen::Matrix4f            current_pose_;
    Eigen::Matrix4f            last_pose_{Eigen::Matrix4f::Identity()};
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
    std::atomic<int>           num_threads_{0}; // 0 = Auto

    // Frame recycling pool (FrameData) - Self-recycling via custom deleter
    static constexpr size_t    DATA_POOL_SIZE = 6;
    struct DataPool {
        std::vector<std::shared_ptr<sensor::FrameData>> data_pool;
        std::queue<sensor::FrameData*>                  free_data_queue;
        std::mutex                                      mutex;
    };
    std::shared_ptr<DataPool> data_pool_state_;

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

    // Per-instance log throttle counters — replaces static locals in worker threads
    // to avoid UB data races when threads are stopped and restarted.
    int                                   ui_skip_counter_      = 0;
    int                                   lost_log_counter_     = 0;
    int                                   success_log_counter_  = 0;

    // Tracking model frame (double-buffered PingPong)
    struct PingPongModel {
        std::shared_ptr<tracking::ModelFrame> buffers[2];
        std::atomic<int>     front_idx{0}; // Read by tracking
        std::atomic<int>     back_idx{1};  // Written by integration
        std::mutex           mtx;

        void swap() {
            int f = front_idx.load();
            front_idx.store(back_idx.load());
            back_idx.store(f);
        }
    } model_pingpong_;

    // Mesh extraction trigger
    std::atomic<bool>                     mesh_extraction_requested_{false};
    std::atomic<bool>                     is_meshing_{false};
    std::atomic<float>                    mesh_extract_progress_{0.0f};
    std::atomic<float>                    export_progress_{0.0f};
    std::atomic<bool>                     use_gpu_{false};
    sensor::cudaStream_t                  cuda_stream_ = nullptr;
    mutable std::mutex                    control_mutex_;

    void onRawFrame(std::shared_ptr<sensor::RawFrame> raw);
    void trackingLoop();
    void integrationLoop();
    void meshingLoop();

    // Pool helpers
    std::shared_ptr<sensor::FrameData> acquireFreeData();
    void releaseData(std::shared_ptr<sensor::FrameData> data);
    
public:
    void setNumThreads(int n) {
        num_threads_.store(n);
        if (tracker_) tracker_->setNumThreads(n);
    }
};

} // namespace app
} // namespace kfusion
