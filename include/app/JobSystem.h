#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>

namespace kfusion {
namespace app {

// Simple thread pool for background jobs
class JobSystem {
public:
    explicit JobSystem(size_t num_threads = 0);
    ~JobSystem();

    // Submit a job, returns future for completion
    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using ReturnType = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        std::future<ReturnType> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lk(queue_mutex_);
            if (stop_) throw std::runtime_error("JobSystem: submit on stopped pool");
            jobs_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    size_t threadCount() const { return workers_.size(); }

private:
    std::vector<std::thread>           workers_;
    std::queue<std::function<void()>>  jobs_;
    std::mutex                         queue_mutex_;
    std::condition_variable            cv_;
    std::atomic<bool>                  stop_{false};

    void workerLoop();
};

} // namespace app
} // namespace kfusion
