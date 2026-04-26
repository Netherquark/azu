#include "sensor/KinectSensor.h"
#include "utils/Logger.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <sys/time.h>

#ifdef HAVE_FREENECT
#include <libfreenect/libfreenect.h>
#endif

namespace kfusion {
namespace sensor {

KinectSensor::KinectSensor() {
  pool_state_ = std::make_shared<PoolState>();
  for (size_t i = 0; i < POOL_SIZE; ++i) {
    pool_state_->pool.push_back(std::make_shared<RawFrame>());
    pool_state_->free_queue.push(pool_state_->pool.back());
  }
}

KinectSensor::~KinectSensor() {
    stop();
#ifdef HAVE_FREENECT
    if (device_) {
        freenect_close_device(device_);
        device_ = nullptr;
    }
    if (ctx_) {
        freenect_shutdown(ctx_);
        ctx_ = nullptr;
    }
#endif
}

bool KinectSensor::init() {
#ifndef HAVE_FREENECT
    KFLOG_ERROR("Sensor", "Kinect support is unavailable in this build (libfreenect not found at configure time).");
    return false;
#else
    // After stop(), device/context stay open so the pipeline can start again (e.g. Reset scan).
    if (device_) return true;

    if (freenect_init(&ctx_, nullptr) < 0) {
        KFLOG_ERROR("Sensor", "freenect_init FAILED: Could not initialize libfreenect.");
        return false;
    }
    freenect_set_log_level(ctx_, FREENECT_LOG_ERROR);
    freenect_select_subdevices(ctx_,
        static_cast<freenect_device_flags>(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int num_devices = freenect_num_devices(ctx_);
    if (num_devices < 1) {
        KFLOG_ERROR("Sensor", "No Kinect devices detected. Please check USB and power.");
        return false;
    }

    if (freenect_open_device(ctx_, &device_, 0) < 0) {
        KFLOG_ERROR("Sensor", "freenect_open_device FAILED: Could not open Kinect index 0.");
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
#endif
}

bool KinectSensor::start() {
#ifndef HAVE_FREENECT
    return false;
#else
    if (!device_) return false;
    if (running_.load()) return true;

  freenect_start_depth(device_);
  freenect_start_video(device_);

    running_.store(true);
    capture_thread_ = std::thread(&KinectSensor::captureLoop, this);
    return true;
#endif
}

void KinectSensor::stop() {
#ifndef HAVE_FREENECT
    running_.store(false);
    return;
#else
    if (!running_.load()) return;
    running_.store(false);

  if (device_) {
    freenect_stop_depth(device_);
    freenect_stop_video(device_);
  }

    if (capture_thread_.joinable())
        capture_thread_.join();
#endif
}

void KinectSensor::captureLoop() {
#ifdef HAVE_FREENECT
    while (running_.load()) {
        struct timeval timeout;
        timeout.tv_sec  = 0;
        timeout.tv_usec = 10000; // 10ms poll
        int ret = freenect_process_events_timeout(ctx_, &timeout);
        if (ret < 0 && running_.load()) {
            KFLOGF_ERROR("Sensor", "libfreenect event processing error: %d", ret);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
#endif
}

#ifdef HAVE_FREENECT
void KinectSensor::depthCallback(freenect_device* dev, void* depth, uint32_t timestamp) {
    static_cast<KinectSensor*>(freenect_get_user(dev))->onDepth(depth, timestamp);
}

void KinectSensor::rgbCallback(freenect_device *dev, void *rgb,
                               uint32_t timestamp) {
  static_cast<KinectSensor *>(freenect_get_user(dev))->onRgb(rgb, timestamp);
}
#else
void KinectSensor::depthCallback(freenect_device*, void*, uint32_t) {}
void KinectSensor::rgbCallback(freenect_device*, void*, uint32_t) {}
#endif

void KinectSensor::onDepth(void* data, uint32_t timestamp) {
    std::lock_guard<std::mutex> lk(sync_mutex_);
    static int pair_log = 0;
    
    if (!depth_pending_) {
        depth_pending_ = acquireFreeFrame();
        if (!depth_pending_) return; // Pool exhausted
    }

    // Return rgb_pending_ to pool by clearing it (deleter will do the rest)
    rgb_pending_ = nullptr;
    depth_pending_ = nullptr;
  }
}

void KinectSensor::onRgb(void *data, uint32_t timestamp) {
  std::lock_guard<std::mutex> lk(sync_mutex_);

  if (!rgb_pending_) {
    rgb_pending_ = acquireFreeFrame();
    if (!rgb_pending_)
      return;
  }

  std::memcpy(rgb_pending_->rgb.data(), data, RGB_WIDTH * RGB_HEIGHT * 3);
  rgb_pending_->timestamp_rgb = timestamp / 1000.0;
  rgb_pending_->rgb_valid = true;

  if (depth_pending_ && depth_pending_->depth_valid) {
    std::memcpy(depth_pending_->rgb.data(), rgb_pending_->rgb.data(),
                RGB_WIDTH * RGB_HEIGHT * 3);
    depth_pending_->timestamp_rgb = rgb_pending_->timestamp_rgb;
    depth_pending_->rgb_valid = true;
    depth_pending_->frame_id = ++frame_counter_;

    if (frame_callback_) {
      if (++pair_log_ % 150 == 0) {
        KFLOGF_DEBUG("Sensor", "Frames synchronized: ID=%d",
                     depth_pending_->frame_id);
      }
      frame_callback_(depth_pending_);
    } else {
      std::lock_guard<std::mutex> qlk(pool_state_->mutex);
      pool_state_->ready_queue.push(depth_pending_);
    }

    rgb_pending_ = nullptr;
    depth_pending_ = nullptr;
  }
}

std::shared_ptr<RawFrame> KinectSensor::getLatestFrame() {
  std::lock_guard<std::mutex> lk(pool_state_->mutex);
  if (pool_state_->ready_queue.empty())
    return nullptr;

  auto frame = pool_state_->ready_queue.front();
  pool_state_->ready_queue.pop();

  // Note: The caller must eventually release this frame back to the pool.
  // However, the PipelineController currently holds it.
  // We'll add an explicit release in the PipelineController or use the custom
  // deleter.
  return frame;
}

std::shared_ptr<RawFrame> KinectSensor::acquireFreeFrame() {
  std::lock_guard<std::mutex> lk(pool_state_->mutex);
  if (pool_state_->free_queue.empty())
    return nullptr;

  // Get the base pointer from the pool (which owns the lifetime)
  auto base_frame = pool_state_->free_queue.front();
  pool_state_->free_queue.pop();

  base_frame->depth_valid = false;
  base_frame->rgb_valid = false;

  // Return a shared_ptr with a custom deleter that pushes it back to the free
  // queue Use a lambda that captures 'pool_state_' to perform the release
  auto state = pool_state_;
  return std::shared_ptr<RawFrame>(
      base_frame.get(), [state, base_frame](RawFrame *) {
        std::lock_guard<std::mutex> lk_inner(state->mutex);
        state->free_queue.push(base_frame);
      });
}

void KinectSensor::releaseFrame(std::shared_ptr<RawFrame>) {
  // No-op manually; handles by custom deleter now
}

} // namespace sensor
} // namespace kfusion
