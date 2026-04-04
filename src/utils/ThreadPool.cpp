#include "utils/ThreadPool.hpp"
#include "utils/Logger.hpp"
#include <thread>

namespace kf {

ThreadPool::ThreadPool(int num_threads) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    init(num_threads);
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::init(int num_threads) {
    if (!workers_.empty()) {
        KF_LOG_WARN("ThreadPool already initialized, ignoring");
        return;
    }

    KF_LOG_INFO("Initializing ThreadPool with ", num_threads, " threads");

    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { worker_thread_func(); });
    }
}

void ThreadPool::worker_thread_func() {
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] { return !tasks_.empty() || stop_; });

        if (stop_ && tasks_.empty()) {
            break;
        }

        if (tasks_.empty()) {
            continue;
        }

        auto task = std::move(tasks_.front());
        tasks_.pop();
        active_tasks_++;
        lock.unlock();

        try {
            task();
        } catch (const std::exception& e) {
            KF_LOG_ERROR("Exception in thread pool task: ", e.what());
        }

        active_tasks_--;
        finish_condition_.notify_all();
    }
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    finish_condition_.wait(lock,
                           [this] { return tasks_.empty() && active_tasks_ == 0; });
}

void ThreadPool::stop() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

size_t ThreadPool::active_thread_count() const {
    return workers_.size();
}

size_t ThreadPool::queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return tasks_.size() + active_tasks_;
}

}  // namespace kf
