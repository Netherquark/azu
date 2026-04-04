#pragma once

#include "utils/Types.hpp"
#include "utils/ThreadSafeQueue.hpp"
#include <libfreenect.h>
#include <thread>
#include <atomic>
#include <chrono>

namespace kf {

// ============================================================================
// Kinect Sensor Wrapper
// ============================================================================

class KinectSensor {
public:
    KinectSensor();
    ~KinectSensor();

    KinectSensor(const KinectSensor&) = delete;
    KinectSensor& operator=(const KinectSensor&) = delete;

    // Initialization and shutdown
    bool initialize();
    bool shutdown();
    bool is_initialized() const { return initialized_; }

    // Capture control
    bool start_capture();
    bool stop_capture();
    bool is_capturing() const { return capturing_; }

    // Frame retrieval (non-blocking)
    bool get_depth_frame(DepthFramePtr& frame);
    bool get_color_frame(ColorFramePtr& frame);

    // Latest frame (with timestamp check)
    DepthFramePtr get_latest_depth();
    ColorFramePtr get_latest_color();

    // LED control
    void set_led(freenect_led_options led);

    // Tilt control
    void set_tilt_angle(double angle);
    double get_tilt_angle();

    // Statistics
    int get_frame_count() const { return frame_counter_; }
    double get_fps_estimate() const { return fps_estimate_; }

private:
    static constexpr int DEPTH_QUEUE_SIZE = 3;
    static constexpr int COLOR_QUEUE_SIZE = 3;

    // Freenect device
    freenect_context* context_ = nullptr;
    freenect_device* device_ = nullptr;

    // Frame buffers (double-buffered)
    DepthFramePtr depth_frame_current_;
    DepthFramePtr depth_frame_next_;
    ColorFramePtr color_frame_current_;
    ColorFramePtr color_frame_next_;

    // Thread-safe queues for processed frames
    ThreadSafeQueue<DepthFramePtr> depth_queue_;
    ThreadSafeQueue<ColorFramePtr> color_queue_;

    // State
    std::atomic<bool> initialized_{false};
    std::atomic<bool> capturing_{false};

    // Capture thread
    std::thread capture_thread_;
    std::atomic<bool> stop_thread_{false};

    // Statistics
    std::atomic<int> frame_counter_{0};
    double fps_estimate_ = 0.0;
    std::chrono::high_resolution_clock::time_point last_fps_check_;

    void capture_thread_func();
    void process_depth_frame(uint8_t* data, uint32_t timestamp);
    void process_color_frame(uint8_t* data, uint32_t timestamp);
    void update_fps();

    // Freenect callbacks (static, will use device userdata)
    friend void freenect_depth_callback(freenect_device* device, void* v_depth,
                                       uint32_t timestamp);
    friend void freenect_rgb_callback(freenect_device* device, void* v_rgb,
                                     uint32_t timestamp);
};

// Global callback functions
void freenect_depth_callback(freenect_device* device, void* v_depth,
                            uint32_t timestamp);
void freenect_rgb_callback(freenect_device* device, void* v_rgb,
                          uint32_t timestamp);

}  // namespace kf