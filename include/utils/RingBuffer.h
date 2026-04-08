#pragma once
#include <atomic>
#include <vector>
#include <optional>

namespace kfusion {
namespace utils {

// Lock-free SPSC ring buffer
template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity + 1), data_(capacity + 1), head_(0), tail_(0) {}

    // Producer: returns false if full
    bool push(T&& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next = (head + 1) % capacity_;
        if (next == tail_.load(std::memory_order_acquire)) return false; // full
        data_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer: returns empty optional if empty
    std::optional<T> pop() {
        size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return std::nullopt;
        T item = std::move(data_[tail]);
        tail_.store((tail + 1) % capacity_, std::memory_order_release);
        return item;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    size_t size() const {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_acquire);
        return (h >= t) ? (h - t) : (capacity_ - t + h);
    }

private:
    size_t            capacity_;
    std::vector<T>    data_;
    std::atomic<size_t> head_, tail_;
};

} // namespace utils
} // namespace kfusion
