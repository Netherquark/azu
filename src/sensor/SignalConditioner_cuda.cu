#ifdef CUDA_ENABLED

#include "sensor/SignalConditioner.h"

#include "sensor/KinectSensor.h"

namespace kfusion {
namespace sensor {

bool SignalConditioner::processCuda(RawFrame& raw,
                                    cudaStream_t cuda_stream,
                                    float min_depth_m,
                                    float max_depth_m) {
    (void)raw;
    (void)cuda_stream;
    (void)min_depth_m;
    (void)max_depth_m;

    // CUDA stream plumbing is now explicit in the API and controller.
    // The current implementation falls back to the OpenMP path until
    // project dependencies for the full CUDA image/depth stack are added.
    return false;
}

} // namespace sensor
} // namespace kfusion

#endif
