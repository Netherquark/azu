#include "utils/Timer.h"

namespace kfusion {
namespace utils {

ScopedTimer::ScopedTimer(const std::string& name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    // Only print if > 1ms to reduce noise
    if (ms > 1.0)
        fprintf(stderr, "[Timer] %s: %.2f ms\n", name_.c_str(), ms);
}

} // namespace utils
} // namespace kfusion
