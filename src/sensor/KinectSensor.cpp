#include "sensor/KinectSensor.h"
#include <sys/time.h>
#include <cstring>
#include <chrono>
#include <iostream>

namespace kfusion {
namespace sensor {

KinectSensor::KinectSensor() {
    // Initialize pool with custom deleters to automatically recycle frames
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        auto raw = new RawFrame();
        std::shared_ptr<RawFrame> frame(raw, [this](RawFrame* f) {
            this->releaseFrame(std::shared_ptr<RawFrame>(f, [](RawFrame*){})); 
        });
        // Wait, the above is circular. Let's use a simpler way.
        // We'll manage raw pointers in the pool and wrap them in shared_ptr with a deleter 
        // that pushes back to free_queue_.
    }
    // REVISED STRATEGY: 
    for (size_t i = 0; i < POOL_SIZE; ++i) {
        pool_.push_back(std::make_shared<RawFrame>());
        free_queue_.push(pool_.back());
    }
}

KinectSensor::~KinectSensor() {
    stop();
    if (device_) {
        freenect_close_device(device_);
        device_ = nullptr;
    }
    if (ctx_) {
        freenect_shutdown(ctx_);
        ctx_ = nullptr;
    }
}

bool KinectSensor::init() {
    if (freenect_init(&ctx_, nullptr) < 0) {
        std::cerr << "[Sensor] freenect_init failed\n";
        return false;
    }
    freenect_set_log_level(ctx_, FREENECT_LOG_ERROR);
    freenect_select_subdevices(ctx_,
        static_cast<freenect_device_flags>(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int num_devices = freenect_num_devices(ctx_);
    if (num_devices < 1) {
        std::cerr << "[Sensor] No Kinect devices found\n";
        return false;
    }

    if (freenect_open_device(ctx_, &device_, 0) < 0) {
        std::cerr << "[Sensor] freenect_open_device failed\n";
        return false;
    }

    freenect_set_user(device_, this);
    freenect_set_depth_callback(device_, depthCallback);
    freenect_set_video_callback(device_, rgbCallback);
    freenect_set_depth_mode(device_,
        freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
    freenect_set_video_mode(device_,
        freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));

    return true;
}

bool KinectSensor::start() {
    if (!device_) return false;
    if (running_.load()) return true;

    freenect_start_depth(device_);
    freenect_start_video(device_);

    running_.store(true);
    capture_thread_ = std::thread(&KinectSensor::captureLoop, this);
    return true;
}

void KinectSensor::stop() {
    if (!running_.load()) return;
    running_.store(false);

    if (device_) {
        freenect_stop_depth(device_);
        freenect_stop_video(device_);
    }

    if (capture_thread_.joinable())
        capture_thread_.join();
}

void KinectSensor::captureLoop() {
    while (running_.load()) {
        struct timeval timeout;
        timeout.tv_sec  = 0;
        timeout.tv_usec = 10000; // 10ms poll
        int ret = freenect_process_events_timeout(ctx_, &timeout);
        if (ret < 0 && running_.load()) {
            std::cerr << "[Sensor] freenect error: " << ret << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void KinectSensor::depthCallback(freenect_device* dev, void* depth, uint32_t timestamp) {
    static_cast<KinectSensor*>(freenect_get_user(dev))->onDepth(depth, timestamp);
}

void KinectSensor::rgbCallback(freenect_device* dev, void* rgb, uint32_t timestamp) {
    static_cast<KinectSensor*>(freenect_get_user(dev))->onRgb(rgb, timestamp);
}

void KinectSensor::onDepth(void* data, uint32_t timestamp) {
    std::lock_guard<std::mutex> lk(sync_mutex_);
    
    if (!depth_pending_) {
        depth_pending_ = acquireFreeFrame();
        if (!depth_pending_) return; // Pool exhausted
    }

    std::memcpy(depth_pending_->depth.data(), data, DEPTH_WIDTH * DEPTH_HEIGHT * 2);
    depth_pending_->timestamp_depth = timestamp / 1000.0;
    depth_pending_->depth_valid = true;

    // Check if we have an RGB frame to pair with
    // Simple pairing: if rgb_pending_ is valid, we merge. 
    // In a more robust system, we'd check timestamps.
    if (rgb_pending_ && rgb_pending_->rgb_valid) {
        // Copy RGB into depth_pending_ (or vice versa)
        std::memcpy(depth_pending_->rgb.data(), rgb_pending_->rgb.data(), RGB_WIDTH * RGB_HEIGHT * 3);
        depth_pending_->timestamp_rgb = rgb_pending_->timestamp_rgb;
        depth_pending_->rgb_valid = true;
        depth_pending_->frame_id = ++frame_counter_;

        if (frame_callback_) {
            frame_callback_(depth_pending_);
        } else {
            std::lock_guard<std::mutex> qlk(queue_mutex_);
            ready_queue_.push(depth_pending_);
        }
        
        // Return rgb_pending_ to pool
        releaseFrame(std::move(rgb_pending_));
        depth_pending_ = nullptr; 
        rgb_pending_ = nullptr;
    }
}

void KinectSensor::onRgb(void* data, uint32_t timestamp) {
    std::lock_guard<std::mutex> lk(sync_mutex_);
    
    if (!rgb_pending_) {
        rgb_pending_ = acquireFreeFrame();
        if (!rgb_pending_) return;
    }

    std::memcpy(rgb_pending_->rgb.data(), data, RGB_WIDTH * RGB_HEIGHT * 3);
    rgb_pending_->timestamp_rgb = timestamp / 1000.0;
    rgb_pending_->rgb_valid = true;

    if (depth_pending_ && depth_pending_->depth_valid) {
        std::memcpy(depth_pending_->rgb.data(), rgb_pending_->rgb.data(), RGB_WIDTH * RGB_HEIGHT * 3);
        depth_pending_->timestamp_rgb = rgb_pending_->timestamp_rgb;
        depth_pending_->rgb_valid = true;
        depth_pending_->frame_id = ++frame_counter_;

        if (frame_callback_) {
            frame_callback_(depth_pending_);
        } else {
            std::lock_guard<std::mutex> qlk(queue_mutex_);
            ready_queue_.push(depth_pending_);
        }
        
        releaseFrame(std::move(rgb_pending_));
        depth_pending_ = nullptr;
        rgb_pending_ = nullptr;
    }
}

std::shared_ptr<RawFrame> KinectSensor::getLatestFrame() {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    if (ready_queue_.empty()) return nullptr;
    
    auto frame = ready_queue_.front();
    ready_queue_.pop();
    
    // Note: The caller must eventually release this frame back to the pool.
    // However, the PipelineController currently holds it. 
    // We'll add an explicit release in the PipelineController or use the custom deleter.
    return frame;
}

std::shared_ptr<RawFrame> KinectSensor::acquireFreeFrame() {
    std::lock_guard<std::mutex> lk(queue_mutex_);
    if (free_queue_.empty()) return nullptr;
    auto f = free_queue_.front();
    free_queue_.pop();
    f->depth_valid = false;
    f->rgb_valid = false;
    return f;
}

void KinectSensor::releaseFrame(std::shared_ptr<RawFrame> frame) {
    if (!frame) return;
    std::lock_guard<std::mutex> lk(queue_mutex_);
    free_queue_.push(std::move(frame));
}

} // namespace sensor
} // namespace kfusion
