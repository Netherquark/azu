#include "app/PipelineController.h"
#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "export/PLYExporter.h"
#include "export/GLBExporter.h"
#include <QCoreApplication>
#include <iostream>
#include <chrono>
#include <cmath>

namespace kfusion {
namespace app {

using namespace std::chrono;

PipelineController::PipelineController()
    : current_pose_(Eigen::Matrix4f::Identity())
{
    sensor_  = std::make_unique<sensor::KinectSensor>();
    tracker_ = std::make_unique<tracking::ICPTracker>();
    tsdf_    = std::make_unique<tsdf::TSDFVolume>();
}

PipelineController::~PipelineController() {
    stop();
}

bool PipelineController::start() {
    if (running_.load()) return true;

    if (!sensor_->init()) {
        std::cerr << "[Pipeline] Sensor init failed\n";
        state_.store(PipelineState::Error);
        return false;
    }

    running_.store(true);
    state_.store(PipelineState::Running);
    first_frame_   = true;
    frame_count_   = 0;
    current_pose_  = Eigen::Matrix4f::Identity();

    // Set frame callback before starting capture
    sensor_->setFrameCallback([this](const sensor::RawFrame& raw) {
        onRawFrame(raw);
    });

    if (!sensor_->start()) {
        std::cerr << "[Pipeline] Sensor start failed\n";
        running_.store(false);
        state_.store(PipelineState::Error);
        return false;
    }

    // Launch pipeline threads
    tracking_thread_    = std::thread(&PipelineController::trackingLoop, this);
    integration_thread_ = std::thread(&PipelineController::integrationLoop, this);
    meshing_thread_     = std::thread(&PipelineController::meshingLoop, this);

    last_capture_time_  = steady_clock::now();
    last_tracking_time_ = steady_clock::now();

    return true;
}

void PipelineController::stop() {
    if (!running_.load()) return;
    running_.store(false);
    state_.store(PipelineState::Stopped);

    sensor_->stop();

    // Wake up threads blocked on condition variables
    tracking_queue_cv_.notify_all();
    integration_queue_cv_.notify_all();

    if (tracking_thread_.joinable())    tracking_thread_.join();
    if (integration_thread_.joinable()) integration_thread_.join();
    if (meshing_thread_.joinable())     meshing_thread_.join();
}

void PipelineController::reset() {
    stop();
    tsdf_->reset();
    {
        std::lock_guard<std::mutex> lk(pose_mutex_);
        current_pose_ = Eigen::Matrix4f::Identity();
    }
    first_frame_  = true;
    frame_count_  = 0;
    metrics_      = PipelineMetrics{};
    shared_mesh_.update(meshing::MeshData{});
    state_.store(PipelineState::Idle);
}

// Called from sensor capture thread — must be lightweight, just copy raw data
void PipelineController::onRawFrame(const sensor::RawFrame& raw) {
    if (!running_.load()) return;

    // Copy raw frame onto heap and enqueue for tracking thread to process
    auto raw_copy = std::make_shared<sensor::RawFrame>(raw);

    {
        std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
        if (raw_queue_.size() < 3) {
            // Store raw frame — processing happens in tracking thread
            raw_queue_.push(raw_copy);
        }
    }
    tracking_queue_cv_.notify_one();

    // Update capture FPS
    auto now = steady_clock::now();
    float dt = duration<float>(now - last_capture_time_).count();
    last_capture_time_ = now;

    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.capture_fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
        metrics_.frame_count = ++frame_count_;
    }
}

void PipelineController::trackingLoop() {
    while (running_.load()) {
        std::shared_ptr<sensor::RawFrame> raw;
        {
            std::unique_lock<std::mutex> lk(tracking_queue_mutex_);
            tracking_queue_cv_.wait(lk, [&] {
                return !running_.load() || !raw_queue_.empty();
            });
            if (!running_.load()) break;
            if (raw_queue_.empty()) continue;
            raw = raw_queue_.front();
            raw_queue_.pop();
        }

        // Build processed frame here (on tracking thread, safe for OpenMP)
        auto frame = std::make_shared<sensor::FrameData>();
        frame->frame_id = raw->frame_id;
        sensor::buildFrameData(raw->depth.data(), raw->rgb.data(), *frame);

        // Notify UI (throttled, safe shared_ptr copy)
        if (frame_ready_cb_) {
            static int ui_skip = 0;
            if (++ui_skip % 3 == 0) {
                auto frame_ui = frame; // shared_ptr copy, cheap
                QMetaObject::invokeMethod(
                    qApp, [this, frame_ui]() {
                        if (frame_ready_cb_) frame_ready_cb_(*frame_ui);
                    }, Qt::QueuedConnection);
            }
        }

        // First frame: initialize pose; skip tracking
        if (first_frame_) {
            first_frame_ = false;
            // Enqueue for integration
            {
                std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                if (integration_queue_.size() < 3)
                    integration_queue_.push(frame);
            }
            integration_queue_cv_.notify_one();

            {
                std::lock_guard<std::mutex> lk(metrics_mutex_);
                metrics_.tracking_ok = true;
                metrics_.icp_error   = 0.0f;
            }
            continue;
        }

        // Build pyramid for ICP
        sensor::FramePyramid pyramid;
        sensor::buildFramePyramid(*frame, pyramid);

        // Get model frame (raycasted from TSDF)
        tracking::ModelFrame model;
        {
            std::lock_guard<std::mutex> lk(model_frame_mutex_);
            model = model_frame_;
        }

        // Run ICP
        Eigen::Matrix4f prev_pose;
        {
            std::lock_guard<std::mutex> lk(pose_mutex_);
            prev_pose = current_pose_;
        }

        auto icp_result = tracker_->track(pyramid, model, prev_pose);

        // Update tracking fps
        auto now = steady_clock::now();
        float dt = duration<float>(now - last_tracking_time_).count();
        last_tracking_time_ = now;

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.tracking_fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
            metrics_.icp_error    = icp_result.error;
            metrics_.tracking_ok  = icp_result.tracking_ok;
        }

        if (icp_result.tracking_ok) {
            {
                std::lock_guard<std::mutex> lk(pose_mutex_);
                current_pose_ = icp_result.pose;
            }
            state_.store(PipelineState::Running);

            // Enqueue for integration
            {
                std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                if (integration_queue_.size() < 3)
                    integration_queue_.push(frame);
            }
            integration_queue_cv_.notify_one();
        } else {
            state_.store(PipelineState::TrackingLost);
        }
    }
}

void PipelineController::integrationLoop() {
    static constexpr int MESH_TRIGGER_FRAMES = 10; // trigger mesh update every N integrated frames
    int frames_since_mesh = 0;

    while (running_.load()) {
        std::shared_ptr<sensor::FrameData> frame;
        {
            std::unique_lock<std::mutex> lk(integration_queue_mutex_);
            integration_queue_cv_.wait(lk, [&] {
                return !running_.load() || !integration_queue_.empty();
            });
            if (!running_.load()) break;
            if (integration_queue_.empty()) continue;
            frame = integration_queue_.front();
            integration_queue_.pop();
        }

        Eigen::Matrix4f pose;
        {
            std::lock_guard<std::mutex> lk(pose_mutex_);
            pose = current_pose_;
        }

        // Integrate into TSDF
        tsdf_->integrate(frame->depth_meters.data(),
                         frame->rgb.data(),
                         pose,
                         static_cast<float>(sensor::FX),
                         static_cast<float>(sensor::FY),
                         static_cast<float>(sensor::CX),
                         static_cast<float>(sensor::CY),
                         frame->width, frame->height);

        // Raycast updated model (every 5 frames to save time)
        static int raycast_skip = 0;
        if (++raycast_skip % 5 == 0) {
            std::lock_guard<std::mutex> lk(model_frame_mutex_);
            tsdf_->raycast(pose,
                           static_cast<float>(sensor::FX),
                           static_cast<float>(sensor::FY),
                           static_cast<float>(sensor::CX),
                           static_cast<float>(sensor::CY),
                           sensor::FRAME_W, sensor::FRAME_H,
                           model_frame_.vertices.data(),
                           model_frame_.normals.data());
        }

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.integrated_frames = tsdf_->integratedFrames();
            metrics_.volume_usage_pct  = tsdf_->usageFraction() * 100.0f;
        }

        ++frames_since_mesh;
        if (frames_since_mesh >= MESH_TRIGGER_FRAMES) {
            frames_since_mesh = 0;
            mesh_extraction_requested_.store(true);
        }
    }
}

void PipelineController::meshingLoop() {
    while (running_.load()) {
        // Poll for mesh extraction request
        if (!mesh_extraction_requested_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        mesh_extraction_requested_.store(false);

        mesh_extract_progress_.store(0.0f);

        auto mesh = meshing::MarchingCubes::extract(*tsdf_, [this](float p) {
            mesh_extract_progress_.store(p);
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.mesh_extract_pct = p * 100.0f;
        });

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.mesh_triangles   = mesh.triangleCount();
            metrics_.mesh_extract_pct = 100.0f;
        }

        shared_mesh_.update(std::move(mesh));

        if (mesh_ready_cb_) mesh_ready_cb_();
    }
}

bool PipelineController::exportPLY(const std::string& path) {
    // Force a mesh extraction if we don't have one yet
    uint64_t ver;
    meshing::MeshData mesh = shared_mesh_.snapshot(ver);
    if (mesh.empty()) {
        std::cout << "[Pipeline] No mesh yet, extracting now...\n";
        mesh = meshing::MarchingCubes::extract(*tsdf_, nullptr);
        if (!mesh.empty()) shared_mesh_.update(meshing::MeshData(mesh));
    }
    if (mesh.empty()) {
        std::cerr << "[Pipeline] No mesh to export — scan more frames first.\n";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = 50.0f;
    }
    bool ok = export_io::PLYExporter::writeBinary(mesh, path);
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = ok ? 100.0f : 0.0f;
    }
    return ok;
}

bool PipelineController::exportGLB(const std::string& path) {
    uint64_t ver;
    meshing::MeshData mesh = shared_mesh_.snapshot(ver);
    if (mesh.empty()) {
        std::cout << "[Pipeline] No mesh yet, extracting now...\n";
        mesh = meshing::MarchingCubes::extract(*tsdf_, nullptr);
        if (!mesh.empty()) shared_mesh_.update(meshing::MeshData(mesh));
    }
    if (mesh.empty()) {
        std::cerr << "[Pipeline] No mesh to export — scan more frames first.\n";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = 50.0f;
    }
    bool ok = export_io::GLBExporter::write(mesh, path);
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = ok ? 100.0f : 0.0f;
    }
    return ok;
}

} // namespace app
} // namespace kfusion
