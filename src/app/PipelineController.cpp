#include "app/PipelineController.h"
#include "tsdf/TSDFVolume.h"
#include "meshing/MeshData.h"
#include "meshing/MarchingCubes.h"
#include "export/GLBExporter.h"
#include "export/PLYExporter.h"
#include <QCoreApplication>
#include <iostream>
#include <chrono>
#include <cmath>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace kfusion {
namespace app {

using namespace std::chrono;

PipelineController::PipelineController()
    : current_pose_(Eigen::Matrix4f::Identity())
{
    sensor_  = std::make_unique<sensor::KinectSensor>();
    tracker_ = std::make_unique<tracking::ICPTracker>();
    tsdf_    = std::make_unique<tsdf::TSDFVolume>();
    cubes_   = std::make_unique<meshing::MarchingCubes>();
    
    // Initialize data pool for automated recycling
    for (size_t i = 0; i < DATA_POOL_SIZE; ++i) {
        data_pool_.push_back(std::make_shared<sensor::FrameData>());
        free_data_queue_.push(data_pool_.back().get());
    }

    // Initialize ping-pong buffers
    for (int i = 0; i < 2; ++i) {
        model_pingpong_.buffers[i] = std::make_shared<tracking::ModelFrame>();
    }
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
    sensor_->setFrameCallback([this](std::shared_ptr<sensor::RawFrame> raw) {
        onRawFrame(std::move(raw));
    });

    if (!sensor_->start()) {
        std::cerr << "[Pipeline] Sensor start failed\n";
        running_.store(false);
        state_.store(PipelineState::Error);
        return false;
    }

#ifdef CUDA_ENABLED
    tsdf_->initGPU();
    tracker_->initGPU();
    // ModelFrame buffers are managed via RAII in the tracking::ModelFrame struct
    for (int i = 0; i < 2; ++i) {
        model_pingpong_.buffers[i]->d_vertices = utils::make_cuda_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
        model_pingpong_.buffers[i]->d_normals  = utils::make_cuda_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
    }
#endif

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

#ifdef CUDA_ENABLED
    tsdf_->freeGPU();
    tracker_->freeGPU();
    for (int i = 0; i < 2; ++i) {
        model_pingpong_.buffers[i].freeGPU();
    }
#endif
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

// Called from sensor capture thread — must be lightweight
void PipelineController::onRawFrame(std::shared_ptr<sensor::RawFrame> raw) {
    if (!running_.load()) return;

    {
        std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
        if (raw_queue_.size() < 3) {
            // Store raw frame — processing happens in tracking thread
            raw_queue_.push(std::move(raw));
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

        // Build processed frame here (using pool)
        auto frame = acquireFreeData();
        if (!frame) {
            sensor_->releaseFrame(std::move(raw)); // Don't forget to recycle raw
            continue;
        }
        
        frame->frame_id = raw->frame_id;
        sensor::buildFrameData(raw->depth.data(), raw->rgb.data(), *frame);
        
        // We can release 'raw' immediately after buildFrameData copies it
        sensor_->releaseFrame(std::move(raw));

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
            frame->pose = Eigen::Matrix4f::Identity();
            // Enqueue for integration
            {
                std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                if (integration_queue_.size() < 3)
                    integration_queue_.push(frame);
                else
                    releaseData(std::move(frame));
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

        // Get model frame safely from PingPong storage (ZERO-COPY)
        std::shared_ptr<tracking::ModelFrame> model_ref;
        {
            std::lock_guard<std::mutex> lk(model_pingpong_.mtx);
            model_ref = model_pingpong_.buffers[model_pingpong_.front_idx.load()];
        }

        // Run ICP
        Eigen::Matrix4f prev_pose;
        {
            std::lock_guard<std::mutex> lk(pose_mutex_);
            prev_pose = current_pose_;
        }

#ifdef CUDA_ENABLED
        auto icp_result = tracker_->trackGPU(pyramid, *model_ref, prev_pose);
#else
        auto icp_result = tracker_->track(pyramid, *model_ref, prev_pose);
#endif

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
            frame->pose = icp_result.pose; // Couple pose with frame!
            state_.store(PipelineState::Running);

            // Enqueue for integration
            {
                std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                if (integration_queue_.size() < 3)
                    integration_queue_.push(frame);
                else
                    releaseData(std::move(frame)); // Drop if queue full
            }
            integration_queue_cv_.notify_one();
        } else {
            state_.store(PipelineState::TrackingLost);
            releaseData(std::move(frame));
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

        // Integrate into TSDF using THE POSE AT CAPTURE
        tsdf_->integrate(frame->depth_meters.data(),
                         frame->rgb.data(),
                         frame->pose,
                         static_cast<float>(sensor::FX),
                         static_cast<float>(sensor::FY),
                         static_cast<float>(sensor::CX),
                         static_cast<float>(sensor::CY),
                         frame->width, frame->height);

        // 2. Generate model frame for tracking (Ping-Pong back buffer)
        int integrated_count = tsdf_->integratedFrames();
        if (integrated_count % 1 == 0) { // Raycast every frame for better tracking
            int back = model_pingpong_.back_idx.load();
            auto& model_back = *(model_pingpong_.buffers[back]);

#ifdef CUDA_ENABLED
            tsdf_->raycastGPU(frame->pose,
                              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
                              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
                              sensor::DEPTH_WIDTH, sensor::DEPTH_HEIGHT,
                              model_back.d_vertices.get(), model_back.d_normals.get());
#else
            tsdf_->raycast(frame->pose,
                           static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
                           static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
                           sensor::DEPTH_WIDTH, sensor::DEPTH_HEIGHT,
                           model_back.vertices.data(), model_back.normals.data());
#endif
            {
                std::lock_guard<std::mutex> lk(model_pingpong_.mtx);
                model_pingpong_.swap();
            }
        }

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.integrated_frames = tsdf_->integratedFrames();
            metrics_.volume_usage_pct  = tsdf_->usageFraction() * 100.0f;
        }

        // Return frame to pool!
        releaseData(std::move(frame));

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

#ifdef CUDA_ENABLED
        auto mesh = cubes_->extractGPU(*tsdf_);
#else
        auto mesh = cubes_->extract(*tsdf_, [this](float p) {
            mesh_extract_progress_.store(p);
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.mesh_extract_pct = p * 100.0f;
        });
#endif

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.mesh_triangles   = mesh ? mesh->triangleCount() : 0;
            metrics_.mesh_extract_pct = 100.0f;
        }

        if (mesh) {
            shared_mesh_.update(std::move(mesh));
            if (mesh_ready_cb_) mesh_ready_cb_();
        }
    }
}

bool PipelineController::exportPLY(const std::string& path) {
    uint64_t ver;
    auto mesh = shared_mesh_.snapshot(ver);
    if (!mesh || mesh->empty()) {
        std::cout << "[Pipeline] No mesh yet, extracting via meshingLoop...\n";
        mesh_extraction_requested_.store(true);
        int waits = 0;
        while (waits < 100) {
            auto m = shared_mesh_.snapshot(ver);
            if (m && !m->empty()) {
                mesh = m;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            waits++;
        }
    }
    if (!mesh || mesh->empty()) {
        std::cerr << "[Pipeline] No mesh to export — scan more frames first.\n";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = 50.0f;
    }
    bool ok = export_io::PLYExporter::writeBinary(*mesh, path);
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = ok ? 100.0f : 0.0f;
    }
    return ok;
}

bool PipelineController::exportGLB(const std::string& path) {
    uint64_t ver;
    auto mesh = shared_mesh_.snapshot(ver);
    if (!mesh || mesh->empty()) {
        std::cout << "[Pipeline] No mesh yet, extracting via meshingLoop...\n";
        mesh_extraction_requested_.store(true);
        int waits = 0;
        while (waits < 100) {
            auto m = shared_mesh_.snapshot(ver);
            if (m && !m->empty()) {
                mesh = m;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            waits++;
        }
    }
    if (!mesh || mesh->empty()) {
        std::cerr << "[Pipeline] No mesh to export — scan more frames first.\n";
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = 50.0f;
    }
    bool ok = export_io::GLBExporter::write(*mesh, path);
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.export_pct = ok ? 100.0f : 0.0f;
    }
    return ok;
}

std::shared_ptr<sensor::FrameData> PipelineController::acquireFreeData() {
    std::lock_guard<std::mutex> lk(data_pool_mutex_);
    if (free_data_queue_.empty()) return nullptr;
    
    auto* raw_ptr = free_data_queue_.front();
    free_data_queue_.pop();

    // Zero-out or reset frame data if needed
    // raw_ptr->reset(); 

    return std::shared_ptr<sensor::FrameData>(raw_ptr, [this, raw_ptr](sensor::FrameData*) {
        std::lock_guard<std::mutex> lk_inner(this->data_pool_mutex_);
        this->free_data_queue_.push(raw_ptr);
    });
}

void PipelineController::releaseData(std::shared_ptr<sensor::FrameData>) {
    // Managed by custom deleter
}

} // namespace app
} // namespace kfusion
