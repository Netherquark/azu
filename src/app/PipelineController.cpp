#include "app/PipelineController.h"
#include "tsdf/TSDFVolume.h"
#include "meshing/MeshData.h"
#include "meshing/MarchingCubes.h"
#include "export/GLBExporter.h"
#include "export/PLYExporter.h"
#include "utils/Logger.h"
#include <QCoreApplication>
#include <iostream>
#include <sstream>
#include <algorithm>
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
    syncIcpDepthFromRange(hyperparams_);
    sensor_  = std::make_unique<sensor::KinectSensor>();
    tracker_ = std::make_unique<tracking::ICPTracker>(hyperparams_.icp);
    tsdf_    = std::make_unique<tsdf::TSDFVolume>(hyperparams_.tsdf);
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
    std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
    if (running_.load()) return true;

    KFLOG_INFO("Pipeline", "Pipeline starting...");
    
    if (!sensor_->init()) {
        KFLOG_ERROR("Pipeline", "Sensor init failed");
        state_.store(PipelineState::Error);
        return false;
    }

    // Clear queues
    {
        std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
        while (!raw_queue_.empty()) raw_queue_.pop();
    }
    {
        std::lock_guard<std::mutex> lk(integration_queue_mutex_);
        while (!integration_queue_.empty()) integration_queue_.pop();
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
        KFLOG_ERROR("Pipeline", "Sensor start failed");
        running_.store(false);
        state_.store(PipelineState::Error);
        return false;
    }

#ifdef CUDA_ENABLED
    try {
        tsdf_->initGPU();
        tracker_->initGPU();
        // ModelFrame buffers are managed via RAII in the tracking::ModelFrame struct
        for (int i = 0; i < 2; ++i) {
            model_pingpong_.buffers[i]->d_vertices = utils::make_cuda_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
            model_pingpong_.buffers[i]->d_normals = utils::make_cuda_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
        }
        use_gpu_ = true;
        tsdf_->setGPUEnabled(true);
        KFLOG_INFO("Pipeline", "CUDA initialization successful — using GPU acceleration.");
    } catch (const std::exception& e) {
        use_gpu_ = false;
        tsdf_->setGPUEnabled(false);
        KFLOG_WARN("Pipeline", "CUDA initialization failed: " + std::string(e.what()) + " — falling back to CPU mode.");
    }
#else
    use_gpu_ = false;
    tsdf_->setGPUEnabled(false);
#endif

    // Launch pipeline threads
    tracking_thread_    = std::thread(&PipelineController::trackingLoop, this);
    integration_thread_ = std::thread(&PipelineController::integrationLoop, this);
    meshing_thread_     = std::thread(&PipelineController::meshingLoop, this);

    last_capture_time_  = steady_clock::now();
    last_tracking_time_ = steady_clock::now();

    KFLOG_INFO("Pipeline", "Pipeline started (tracking + integration + meshing threads running)");
    return true;
}

PipelineMetrics PipelineController::metricsSnapshot() const {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    PipelineMetrics m = metrics_;
    m.state = state_.load();
    return m;
}

FusionHyperparams PipelineController::hyperparamsSnapshot() const {
    std::lock_guard<std::mutex> lk(hyper_mutex_);
    return hyperparams_;
}

void PipelineController::setHyperparams(const FusionHyperparams& h) {
    FusionHyperparams hp = h;
    syncIcpDepthFromRange(hp);
    {
        std::lock_guard<std::mutex> lk(hyper_mutex_);
        hyperparams_ = hp;
    }
    tracker_->setParams(hp.icp);
    tsdf_->setParams(hp.tsdf);
    KFLOG_INFO("Pipeline",
               "Hyperparameters applied (depth clip, TSDF grid, ICP). Changing resolution clears the volume.");
}

void PipelineController::stop() {
    std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
    if (!running_.load()) return;
    running_.store(false);
    state_.store(PipelineState::Stopped);

    KFLOG_INFO("Pipeline", "Pipeline stopping (joining worker threads)...");
    sensor_->stop();
    sensor_->setFrameCallback(nullptr); // Unset to avoid late callbacks

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
        model_pingpong_.buffers[i]->d_vertices.reset();
        model_pingpong_.buffers[i]->d_normals.reset();
    }
#endif
    KFLOG_INFO("Pipeline", "Pipeline stopped");
}

void PipelineController::reset() {
    // Note: stop() already has control_mutex_, so we don't lock here yet
    // to avoid deadlock, but we ensure reset is sequential.
    stop();
    std::lock_guard<std::mutex> ctrl_lk(control_mutex_);

    // Clear queues
    {
        std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
        while (!raw_queue_.empty()) raw_queue_.pop();
    }
    {
        std::lock_guard<std::mutex> lk(integration_queue_mutex_);
        while (!integration_queue_.empty()) integration_queue_.pop();
    }

    mesh_extraction_requested_.store(false);

    // Clear Ping-Pong buffers under lock
    {
        std::lock_guard<std::mutex> lk(model_pingpong_.mtx);
        for (int i = 0; i < 2; ++i) {
            if (!model_pingpong_.buffers[i]) continue;
            auto& mf = *model_pingpong_.buffers[i];
            std::fill(mf.vertices.begin(), mf.vertices.end(), Eigen::Vector3f::Zero());
            std::fill(mf.normals.begin(), mf.normals.end(), Eigen::Vector3f::Zero());
        }
        model_pingpong_.front_idx.store(0);
        model_pingpong_.back_idx.store(1);
    }

    tsdf_->reset();
    {
        std::lock_guard<std::mutex> lk(pose_mutex_);
        current_pose_ = Eigen::Matrix4f::Identity();
    }
    first_frame_  = true;
    frame_count_  = 0;
    metrics_      = PipelineMetrics{};
    shared_mesh_.update(std::make_shared<meshing::MeshData>());
    state_.store(PipelineState::Idle);
    KFLOG_INFO("Pipeline", "Volume and pose reset; queues drained; mesh cleared");
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

    int fc = 0;
    float inst_fps = 0.0f;
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.capture_fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
        metrics_.frame_count = ++frame_count_;
        fc = metrics_.frame_count;
        inst_fps = metrics_.capture_fps;
    }
    if (fc % 60 == 0) {
        std::ostringstream oss;
        oss << "sensor→pipeline: paired frames received=" << fc
            << " instantaneous_capture_fps=" << inst_fps;
        KFLOG_DEBUG("Pipeline", oss.str());
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
            KFLOG_WARN("Pipeline", "FrameData pool exhausted — dropping raw frame (increase pool or slow capture)");
            sensor_->releaseFrame(std::move(raw)); // Don't forget to recycle raw
            continue;
        }
        
        frame->frame_id = raw->frame_id;
        float d_min = 0.3f, d_max = 5.0f;
        {
            std::lock_guard<std::mutex> lk(hyper_mutex_);
            d_min = hyperparams_.min_depth;
            d_max = hyperparams_.max_depth;
        }
        sensor::buildFrameData(raw->depth.data(), raw->rgb.data(), *frame, d_min, d_max);
        
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
            KFLOG_INFO("Pipeline", "First depth frame: pose=identity, queued for TSDF integration (ICP starts next frame)");
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
        Eigen::Matrix4f predicted_pose;
        {
            std::lock_guard<std::mutex> lk(pose_mutex_);
            prev_pose = current_pose_;
            
            // Motion Model: predicted = current * (last_delta)
            // last_delta = last_pose.inv * current_pose
            static Eigen::Matrix4f last_pose = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f delta = last_pose.inverse() * current_pose_;
            predicted_pose = current_pose_ * delta;
            last_pose = current_pose_;
        }

        tracking::ICPResult icp_result;
        bool is_lost = (state_.load() == PipelineState::TrackingLost);

        if (use_gpu_.load()) {
#ifdef CUDA_ENABLED
            icp_result = tracker_->trackGPU(pyramid, *model_ref, predicted_pose);
#endif
        } else {
            if (is_lost) {
                // RELOCALIZATION MODE: Try harder to catch the pose
                tracking::ICPParams recovery_params = hyperparams_.icp;
                recovery_params.dist_threshold *= 3.0f; // 30cm instead of 10cm
                recovery_params.angle_threshold = 60.0f;
                
                // Backup original params
                tracking::ICPParams original_params = tracker_->params();
                tracker_->setParams(recovery_params);
                
                // Try relocalizing at the coarsest level first with more iterations
                icp_result = tracker_->track(pyramid, *model_ref, prev_pose);
                
                // Restore params
                tracker_->setParams(original_params);
            } else {
                icp_result = tracker_->track(pyramid, *model_ref, predicted_pose);
                
                // If predicted pose fails, retry with stationary pose (safety fallback)
                if (!icp_result.tracking_ok) {
                    icp_result = tracker_->track(pyramid, *model_ref, prev_pose);
                }
            }
        }

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

        if (!icp_result.tracking_ok) {
            static int lost_log = 0;
            if (++lost_log % 45 == 1) {
                std::ostringstream oss;
                oss << (is_lost ? "[RELOCALIZING...] " : "[TRACKING LOST] ")
                    << "ICP failed — inliers=" << icp_result.inliers
                    << " err=" << icp_result.error
                    << " dist_fail=" << icp_result.dist_filtered;
                KFLOG_WARN("Pipeline", oss.str());
            }
        }

        if (icp_result.tracking_ok) {
            {
                std::lock_guard<std::mutex> lk(pose_mutex_);
                current_pose_ = icp_result.pose;
            }
            frame->pose = icp_result.pose; 
            state_.store(PipelineState::Running);

            // Enqueue for integration
            {
                std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                if (integration_queue_.size() < 3)
                    integration_queue_.push(frame);
                else
                    releaseData(std::move(frame)); 
            }
            integration_queue_cv_.notify_one();
        } else {
            // RELOCALIZATION: If tracking lost, don't update state or pose, but don't stop.
            // We just don't integrate the frame. This keeps the model "clean".
            state_.store(PipelineState::TrackingLost);
            releaseData(std::move(frame));
            
            static int recover_log = 0;
            if (++recover_log % 30 == 0) {
                KFLOG_WARN("Pipeline", "Tracking Lost — Integration suspended. Attempting relocalization...");
            }
        }
    }
}

void PipelineController::integrationLoop() {
    static constexpr int MESH_TRIGGER_FRAMES = 5; // trigger mesh update every N integrated frames
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
            metrics_.integrated_frames = integrated_count;
            metrics_.volume_usage_pct  = tsdf_->usageFraction() * 100.0f;
        }

        if (integrated_count % 20 == 0 && integrated_count > 0) {
            std::ostringstream oss;
            oss << "TSDF: integrated_frames=" << integrated_count
                << " volume_fill=" << (tsdf_->usageFraction() * 100.0f) << "%";
            KFLOG_INFO("Pipeline", oss.str());
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
        KFLOG_INFO("Pipeline", "Marching cubes: mesh extraction started (CPU/GPU path)");

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

        if (mesh && !mesh->empty()) {
            std::ostringstream oss;
            oss << "Mesh ready: triangles=" << mesh->triangleCount()
                << " vertices=" << mesh->positions.size();
            KFLOG_INFO("Pipeline", oss.str());
            shared_mesh_.update(std::move(mesh));
            if (mesh_ready_cb_) mesh_ready_cb_();
        } else {
            KFLOG_WARN("Pipeline", "Mesh extraction produced empty mesh (scan more surface or check TSDF fill)");
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
