#pragma once

#include <cstdint>
#include <vector>

namespace kfusion {
namespace sensor {

struct RawFrame;

#ifdef CUDA_ENABLED
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#else
using cudaStream_t = void*;
#endif

class SignalConditioner {
public:
    SignalConditioner();

    void reset();
    void resetEMA();
    void process(RawFrame& raw, cudaStream_t cuda_stream, float min_depth_m, float max_depth_m);

private:
    std::vector<float>    ema_buf_m_;
    std::vector<uint8_t>  sr_rgb_;
    std::vector<float>    guidance_luma_;
    std::vector<uint16_t> depth_scratch_;

    void preprocessRgb(std::vector<uint8_t>& rgb);
    void buildSuperResolutionGuidance(const std::vector<uint8_t>& rgb);
    void applyDepthEma(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m);
    void fillDepthHoles(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m);
    void guidedDepthFilter(std::vector<uint16_t>& depth, float min_depth_m, float max_depth_m);
    void processCpu(RawFrame& raw, float min_depth_m, float max_depth_m);

#ifdef CUDA_ENABLED
    bool processCuda(RawFrame& raw, cudaStream_t cuda_stream, float min_depth_m, float max_depth_m);
#endif
};

} // namespace sensor
} // namespace kfusion
