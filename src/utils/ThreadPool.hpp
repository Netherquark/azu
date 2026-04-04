#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

namespace kf {

class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Set the number of worker threads (call before any work)
    void init(int num_threads);

    // Submit a task to the queue
    template <typename F, typename... Args>
    auto submit(F&& func, Args&&... args) {
        using return_type = typename std::invoke_result<F, Args...>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(func), std::forward<Args>(args)...));

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            tasks_.push([task]() { (*task)(); });
        }
        condition_.notify_one();

        return task->get_future();
    }

    // Wait for all tasks to complete
    void wait_all();

    // Stop the thread pool
    void stop();

    size_t active_thread_count() const;
    size_t queue_size() const;

private:
    void worker_thread_func();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable finish_condition_;
    std::atomic<bool> stop_{false};
    std::atomic<int> active_tasks_{0};
};

}  // namespace kf
