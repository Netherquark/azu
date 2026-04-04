#include "sensor/KinectSensor.hpp"
#include "utils/Logger.hpp"
#include "utils/Math.hpp"
#include <cstring>
#include <chrono>

namespace kf {

// ============================================================================
// Global Callback Functions
// ============================================================================

void freenect_depth_callback(freenect_device* device, void* v_depth,
                            uint32_t timestamp) {
    KinectSensor* sensor = static_cast<KinectSensor*>(freenect_get_user(device));
    if (sensor) {
        sensor->process_depth_frame(static_cast<uint8_t*>(v_depth), timestamp);
    }
}

void freenect_rgb_callback(freenect_device* device, void* v_rgb,
                          uint32_t timestamp) {
    KinectSensor* sensor = static_cast<KinectSensor*>(freenect_get_user(device));
    if (sensor) {
        sensor->process_color_frame(static_cast<uint8_t*>(v_rgb), timestamp);
    }
}

// ============================================================================
// KinectSensor Implementation
// ============================================================================

KinectSensor::KinectSensor() {
    depth_frame_current_ = std::make_shared<DepthFrame>();
    depth_frame_next_ = std::make_shared<DepthFrame>();
    color_frame_current_ = std::make_shared<ColorFrame>();
    color_frame_next_ = std::make_shared<ColorFrame>();
    last_fps_check_ = std::chrono::high_resolution_clock::now();
}

KinectSensor::~KinectSensor() {
    shutdown();
}

bool KinectSensor::initialize() {
    if (initialized_) return true;

    KF_LOG_INFO("Initializing Kinect sensor...");

    if (freenect_init(&context_, NULL) < 0) {
        KF_LOG_ERROR("Failed to initialize libfreenect");
        return false;
    }

    freenect_set_log_level(context_, FREENECT_LOG_WARNING);
    freenect_select_subdevices(
        context_, (freenect_device_flags)(FREENECT_DEVICE_MOTOR |
                                          FREENECT_DEVICE_CAMERA));

    if (freenect_open_device(context_, &device_, 0) < 0) {
        KF_LOG_ERROR("Failed to open Kinect device");
        freenect_shutdown(context_);
        context_ = nullptr;
        return false;
    }

    // Set depth mode to 11-bit
    freenect_set_depth_mode(device_, freenect_find_depth_mode(
                                         FREENECT_RESOLUTION_MEDIUM,
                                         FREENECT_DEPTH_11BIT_PACKED));

    // Set video mode to RGB
    freenect_set_video_mode(
        device_, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM,
                                          FREENECT_VIDEO_RGB));

    // Set callbacks
    freenect_set_user(device_, this);
    freenect_set_depth_callback(device_, freenect_depth_callback);
    freenect_set_video_callback(device_, freenect_rgb_callback);

    initialized_ = true;
    KF_LOG_INFO("Kinect sensor initialized successfully");
    return true;
}

bool KinectSensor::shutdown() {
    stop_capture();

    if (device_) {
        freenect_close_device(device_);
        device_ = nullptr;
    }

    if (context_) {
        freenect_shutdown(context_);
        context_ = nullptr;
    }

    initialized_ = false;
    KF_LOG_INFO("Kinect sensor shut down");
    return true;
}

bool KinectSensor::start_capture() {
    if (!initialized_) {
        KF_LOG_ERROR("Sensor not initialized");
        return false;
    }

    if (capturing_) return true;

    KF_LOG_INFO("Starting Kinect capture...");

    stop_thread_ = false;

    if (freenect_start_depth(device_) < 0) {
        KF_LOG_ERROR("Failed to start depth stream");
        return false;
    }

    if (freenect_start_video(device_) < 0) {
        KF_LOG_ERROR("Failed to start video stream");
        freenect_stop_depth(device_);
        return false;
    }

    // Start capture thread
    capture_thread_ = std::thread([this]() { capture_thread_func(); });

    capturing_ = true;
    frame_counter_ = 0;
    KF_LOG_INFO("Kinect capture started");
    return true;
}

bool KinectSensor::stop_capture() {
    if (!capturing_) return true;

    KF_LOG_INFO("Stopping Kinect capture...");

    stop_thread_ = true;

    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    if (device_) {
        freenect_stop_depth(device_);
        freenect_stop_video(device_);
    }

    capturing_ = false;
    KF_LOG_INFO("Kinect capture stopped");
    return true;
}

bool KinectSensor::get_depth_frame(DepthFramePtr& frame) {
    return depth_queue_.try_pop(frame);
}

bool KinectSensor::get_color_frame(ColorFramePtr& frame) {
    return color_queue_.try_pop(frame);
}

DepthFramePtr KinectSensor::get_latest_depth() {
    DepthFramePtr frame;
    // Drain queue to get latest
    while (depth_queue_.try_pop(frame)) {
        // Continue until queue is empty
    }
    return frame;
}

ColorFramePtr KinectSensor::get_latest_color() {
    ColorFramePtr frame;
    // Drain queue to get latest
    while (color_queue_.try_pop(frame)) {
        // Continue until queue is empty
    }
    return frame;
}

void KinectSensor::set_led(freenect_led_options led) {
    if (device_) {
        freenect_set_led(device_, led);
    }
}

void KinectSensor::set_tilt_angle(double angle) {
    if (device_) {
        freenect_set_tilt_degs(device_, angle);
    }
}

double KinectSensor::get_tilt_angle() {
    if (device_) {
        return freenect_get_tilt_degs(device_);
    }
    return 0.0;
}

void KinectSensor::capture_thread_func() {
    while (!stop_thread_) {
        if (freenect_process_events(context_) < 0) {
            KF_LOG_WARN("libfreenect: USB disconnect or error");
            break;
        }
    }
}

void KinectSensor::process_depth_frame(uint8_t* data, uint32_t timestamp) {
    if (!data) return;

    // Convert packed 11-bit to 16-bit
    uint16_t* src = reinterpret_cast<uint16_t*>(data);
    uint16_t* dst = depth_frame_next_->data.data();

    // Unpack 11-bit packed format
    // libfreenect packs: 5 bits pad, 11 bits depth in each 16-bit word
    for (size_t i = 0; i < DepthFrame::TOTAL_PIXELS; ++i) {
        dst[i] = src[i] >> 5;  // Shift out padding bits
    }

    depth_frame_next_->timestamp_us = timestamp;

    // Queue for retrieval
    depth_queue_.push(depth_frame_next_);

    // Swap buffers
    std::swap(depth_frame_current_, depth_frame_next_);

    frame_counter_++;
    update_fps();
}

void KinectSensor::process_color_frame(uint8_t* data, uint32_t timestamp) {
    if (!data) return;

    // RGB is already in correct format
    std::memcpy(color_frame_next_->data.data(), data,
                ColorFrame::TOTAL_PIXELS * ColorFrame::CHANNELS);

    color_frame_next_->timestamp_us = timestamp;

    // Queue for retrieval
    color_queue_.push(color_frame_next_);

    // Swap buffers
    std::swap(color_frame_current_, color_frame_next_);
}

void KinectSensor::update_fps() {
    if (frame_counter_ % 30 == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_fps_check_)
                .count();
        if (elapsed > 0) {
            fps_estimate_ = 30000.0 / elapsed;
            last_fps_check_ = now;
        }
    }
}

}  // namespace kf
