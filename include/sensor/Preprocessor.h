#pragma once

#include <memory>
#include <string>

#include "sensor/KinectSensor.h"
#include "sensor/SignalConditioner.h"

namespace kfusion {
namespace sensor {

enum class PreprocessBackend {
    Auto,
    CPU,
    CUDA
};

const char* backendName(PreprocessBackend backend);
PreprocessBackend parseBackendName(const std::string& name);

class Preprocessor {
public:
    virtual ~Preprocessor() = default;

    virtual void reset() = 0;
    virtual void resetTemporalState() = 0;
    virtual void process(RawFrame& frame, float min_depth_m, float max_depth_m) = 0;

    // GPU-native access (returns nullptr if CPU backend is used)
    virtual float*    getGPUDepthMeters() const { return nullptr; }
    virtual uint8_t*  getGPURgb()         const { return nullptr; }

    virtual PreprocessBackend backend() const = 0;
};

class CPUPreprocessor final : public Preprocessor {
public:
    CPUPreprocessor();

    void reset() override;
    void resetTemporalState() override;
    void process(RawFrame& frame, float min_depth_m, float max_depth_m) override;
    PreprocessBackend backend() const override { return PreprocessBackend::CPU; }

private:
    SignalConditioner conditioner_;
};

class CUDAPreprocessor final : public Preprocessor {
public:
    explicit CUDAPreprocessor(cudaStream_t stream);

    void reset() override;
    void resetTemporalState() override;
    void process(RawFrame& frame, float min_depth_m, float max_depth_m) override;
    PreprocessBackend backend() const override { return PreprocessBackend::CUDA; }

    float*   getGPUDepthMeters() const override { 
#ifdef CUDA_ENABLED
        return conditioner_.getGPUDepthMeters(); 
#elif defined(HIP_ENABLED)
        return conditioner_.getGPUDepthMeters(); 
#else
        return nullptr;
#endif
    }
    uint8_t* getGPURgb()         const override { 
#ifdef CUDA_ENABLED
        return conditioner_.getGPURgb(); 
#elif defined(HIP_ENABLED)
        return conditioner_.getGPURgb(); 
#else
        return nullptr;
#endif
    }

private:
    cudaStream_t stream_ = nullptr;
    SignalConditioner conditioner_;
};

std::unique_ptr<Preprocessor> makePreprocessor(PreprocessBackend requested_backend,
                                               bool cuda_available,
                                               cudaStream_t cuda_stream,
                                               PreprocessBackend* actual_backend = nullptr);

} // namespace sensor
} // namespace kfusion
