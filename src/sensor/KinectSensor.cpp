#include "sensor/KinectSensor.h"
#include <sys/time.h>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace kfusion {
namespace sensor {

KinectSensor::KinectSensor() = default;

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

    frame_cv_.notify_all();
}

void KinectSensor::captureLoop() {
    while (running_.load()) {
        struct timeval timeout;
        timeout.tv_sec  = 0;
        timeout.tv_usec = 100000; // 100ms timeout
        int ret = freenect_process_events_timeout(ctx_, &timeout);
        if (ret < 0 && running_.load()) {
            std::cerr << "[Sensor] freenect_process_events_timeout error: " << ret << "\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void KinectSensor::depthCallback(freenect_device* dev, void* depth, uint32_t timestamp) {
    auto* self = static_cast<KinectSensor*>(freenect_get_user(dev));
    self->onDepth(depth, timestamp);
}

void KinectSensor::rgbCallback(freenect_device* dev, void* rgb, uint32_t timestamp) {
    auto* self = static_cast<KinectSensor*>(freenect_get_user(dev));
    self->onRgb(rgb, timestamp);
}

void KinectSensor::onDepth(void* data, uint32_t timestamp) {
    std::lock_guard<std::mutex> lk(sync_mutex_);
    auto* raw = static_cast<uint16_t*>(data);
    std::memcpy(pending_frame_.depth.data(), raw,
                DEPTH_WIDTH * DEPTH_HEIGHT * sizeof(uint16_t));
    pending_frame_.timestamp_depth = static_cast<double>(timestamp) / 1000.0;
    pending_frame_.depth_valid     = true;

    if (pending_frame_.rgb_valid) {
        pending_frame_.frame_id = ++frame_counter_;
        {
            std::lock_guard<std::mutex> fk(frame_mutex_);
            front_buffer_    = pending_frame_;
            new_frame_ready_ = true;
        }
        frame_cv_.notify_one();
        if (frame_callback_) frame_callback_(pending_frame_);
        pending_frame_.depth_valid = false;
        pending_frame_.rgb_valid   = false;
    }
}

void KinectSensor::onRgb(void* data, uint32_t timestamp) {
    std::lock_guard<std::mutex> lk(sync_mutex_);
    auto* raw = static_cast<uint8_t*>(data);
    std::memcpy(pending_frame_.rgb.data(), raw,
                RGB_WIDTH * RGB_HEIGHT * 3);
    pending_frame_.timestamp_rgb = static_cast<double>(timestamp) / 1000.0;
    pending_frame_.rgb_valid     = true;
}

bool KinectSensor::getLatestFrame(RawFrame& out) const {
    std::unique_lock<std::mutex> lk(frame_mutex_);
    if (!new_frame_ready_) return false;
    out              = front_buffer_;
    new_frame_ready_ = false;
    return true;
}

} // namespace sensor
} // namespace kfusion
