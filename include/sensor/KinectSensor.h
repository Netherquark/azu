#pragma once

#include <libfreenect/libfreenect.h>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <array>
#include <vector>
#include <memory>
#include <functional>
#include "utils/RingBuffer.h"

namespace kfusion {
namespace sensor {

// Raw Kinect v1 depth/rgb frame dimensions
static constexpr int DEPTH_WIDTH  = 640;
static constexpr int DEPTH_HEIGHT = 480;
static constexpr int RGB_WIDTH    = 640;
static constexpr int RGB_HEIGHT   = 480;

// Kinect v1 intrinsics
static constexpr double FX = 525.0;
static constexpr double FY = 525.0;
static constexpr double CX = 319.5;
static constexpr double CY = 239.5;

// Convert raw 11-bit depth to meters
inline float rawDepthToMeters(uint16_t raw) {
    if (raw == 0 || raw >= 2047) return 0.0f;
    return 1.0f / (static_cast<float>(raw) * -0.0030711016f + 3.3309495161f);
}

struct RawFrame {
    std::vector<uint16_t> depth;
    std::vector<uint8_t>  rgb;
    double timestamp_depth = 0.0;
    double timestamp_rgb   = 0.0;
    bool   depth_valid     = false;
    bool   rgb_valid       = false;
    uint64_t frame_id      = 0;

    RawFrame() {
        depth.resize(DEPTH_WIDTH * DEPTH_HEIGHT);
        rgb.resize(RGB_WIDTH * RGB_HEIGHT * 3);
    }
};

using FrameCallback = std::function<void(std::shared_ptr<RawFrame>)>;

class KinectSensor {
public:
    KinectSensor();
    ~KinectSensor();

    // Non-copyable
    KinectSensor(const KinectSensor&) = delete;
    KinectSensor& operator=(const KinectSensor&) = delete;

    bool init();
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }
    bool isConnected() const { return device_ != nullptr; }

    // Register callback invoked on UI/pipeline thread (posted from capture thread)
    void setFrameCallback(FrameCallback cb) { frame_callback_ = std::move(cb); }

    // Returns latest synchronized frame (zero-copy)
    std::shared_ptr<RawFrame> getLatestFrame();

private:
    // libfreenect state
    freenect_context* ctx_    = nullptr;
    freenect_device*  device_ = nullptr;

    // Lock-free pipeline state
    static constexpr size_t POOL_SIZE = 8;
    std::vector<std::shared_ptr<RawFrame>> pool_;
    utils::RingBuffer<std::shared_ptr<RawFrame>> ready_queue_;
    utils::RingBuffer<std::shared_ptr<RawFrame>> free_queue_;
    
    // Indices currently assigned to libfreenect
    std::atomic<int> current_depth_idx_{-1};
    std::atomic<int> current_rgb_idx_{-1};

    uint64_t frame_counter_ = 0;

    std::atomic<bool> running_{false};
    std::thread       capture_thread_;

    FrameCallback frame_callback_;

    void captureLoop();

    // libfreenect callbacks (static, forwarded via user data)
    static void depthCallback(freenect_device* dev, void* depth, uint32_t timestamp);
    static void rgbCallback(freenect_device* dev, void* rgb, uint32_t timestamp);

    void onDepth(void* data, uint32_t timestamp);
    void onRgb(void* data, uint32_t timestamp);
};

} // namespace sensor
} // namespace kfusion
