#include "app/JobSystem.h"
#include <thread>

namespace kfusion {
namespace app {

JobSystem::JobSystem(size_t num_threads) {
    if (num_threads == 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i)
        workers_.emplace_back(&JobSystem::workerLoop, this);
}

JobSystem::~JobSystem() {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        stop_.store(true);
    }
    cv_.notify_all();
    for (auto& w : workers_)
        if (w.joinable()) w.join();
}

void JobSystem::workerLoop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            cv_.wait(lk, [this] { return stop_.load() || !jobs_.empty(); });
            if (stop_.load() && jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop();
        }
        job();
    }
}

} // namespace app
} // namespace kfusion
