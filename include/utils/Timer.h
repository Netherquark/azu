#pragma once
#include <chrono>
#include <string>
#include <cstdio>

namespace kfusion {
namespace utils {

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace utils
} // namespace kfusion
