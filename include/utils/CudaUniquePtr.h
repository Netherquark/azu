#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>

namespace kfusion {
namespace utils {

/**
 * @brief Custom deleter for CUDA memory.
 */
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

/**
 * @brief RAII wrapper for CUDA memory.
 */
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

/**
 * @brief Helper to allocate CUDA memory into a CudaUniquePtr.
 */
template <typename T>
CudaUniquePtr<T> make_cuda_unique(size_t count) {
    T* ptr = nullptr;
    size_t size = count * sizeof(T);
    if (size == 0) return CudaUniquePtr<T>(nullptr);
    
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
    return CudaUniquePtr<T>(ptr);
}

} // namespace utils
} // namespace kfusion
