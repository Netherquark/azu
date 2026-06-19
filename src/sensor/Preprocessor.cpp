#include "sensor/Preprocessor.h"

#include <algorithm>
#include <cctype>

namespace kfusion {
namespace sensor {

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

const char* backendName(PreprocessBackend backend) {
    switch (backend) {
        case PreprocessBackend::CPU: return "cpu";
        case PreprocessBackend::CUDA: return "cuda";
        case PreprocessBackend::HIP: return "hip";
        case PreprocessBackend::Auto:
        default: return "auto";
    }
}

PreprocessBackend parseBackendName(const std::string& name) {
    const std::string lowered = toLower(name);
    if (lowered == "cpu") {
        return PreprocessBackend::CPU;
    }
    if (lowered == "cuda" || lowered == "gpu") {
        return PreprocessBackend::CUDA;
    }
    if (lowered == "hip") {
        return PreprocessBackend::HIP;
    }
    return PreprocessBackend::Auto;
}

CPUPreprocessor::CPUPreprocessor() = default;

void CPUPreprocessor::reset() {
    conditioner_.reset();
}

void CPUPreprocessor::resetTemporalState() {
    conditioner_.resetEMA();
}

void CPUPreprocessor::process(RawFrame& frame, float min_depth_m, float max_depth_m) {
    conditioner_.process(frame, nullptr, min_depth_m, max_depth_m);
}

CUDAPreprocessor::CUDAPreprocessor(cudaStream_t stream)
    : stream_(stream) {}

void CUDAPreprocessor::reset() {
    conditioner_.reset();
}

void CUDAPreprocessor::resetTemporalState() {
    conditioner_.resetEMA();
}

void CUDAPreprocessor::process(RawFrame& frame, float min_depth_m, float max_depth_m) {
    conditioner_.process(frame, stream_, min_depth_m, max_depth_m);
}

std::unique_ptr<Preprocessor> makePreprocessor(PreprocessBackend requested_backend,
                                               bool gpu_available,
                                               cudaStream_t cuda_stream,
                                               PreprocessBackend* actual_backend) {
    PreprocessBackend resolved = requested_backend;
    if (resolved == PreprocessBackend::Auto) {
        if (gpu_available) {
#if defined(HIP_ENABLED)
            resolved = PreprocessBackend::HIP;
#else
            resolved = PreprocessBackend::CUDA;
#endif
        } else {
            resolved = PreprocessBackend::CPU;
        }
    } else if ((resolved == PreprocessBackend::CUDA || resolved == PreprocessBackend::HIP) && !gpu_available) {
        resolved = PreprocessBackend::CPU;
    }

    if (actual_backend) {
        *actual_backend = resolved;
    }

    if (resolved == PreprocessBackend::CUDA || resolved == PreprocessBackend::HIP) {
        return std::make_unique<CUDAPreprocessor>(cuda_stream);
    }
    return std::make_unique<CPUPreprocessor>();
}

} // namespace sensor
} // namespace kfusion
