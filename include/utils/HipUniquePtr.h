#pragma once

#ifdef HIP_ENABLED

#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>

namespace kfusion {
namespace utils {

/**
 * @brief Custom deleter for HIP memory.
 */
struct HipDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            hipFree(ptr);
        }
    }
};

/**
 * @brief RAII wrapper for HIP memory.
 */
template <typename T>
using HipUniquePtr = std::unique_ptr<T, HipDeleter>;

/**
 * @brief Helper to allocate HIP memory into a HipUniquePtr.
 */
template <typename T>
HipUniquePtr<T> make_hip_unique(size_t count) {
    T* ptr = nullptr;
    size_t size = count * sizeof(T);
    if (size == 0) return HipUniquePtr<T>(nullptr);
    
    hipError_t err = hipMalloc(&ptr, size);
    if (err != hipSuccess) {
        throw std::runtime_error("hipMalloc failed: " + std::string(hipGetErrorString(err)));
    }
    return HipUniquePtr<T>(ptr);
}

} // namespace utils
} // namespace kfusion

#endif // HIP_ENABLED
