#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <mutex>
#include <chrono>
#include <iomanip>

namespace kf {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        log_level_ = level;
    }

    template <typename... Args>
    void debug(Args... args) {
        if (log_level_ <= LogLevel::DEBUG) {
            log(LogLevel::DEBUG, args...);
        }
    }

    template <typename... Args>
    void info(Args... args) {
        if (log_level_ <= LogLevel::INFO) {
            log(LogLevel::INFO, args...);
        }
    }

    template <typename... Args>
    void warn(Args... args) {
        if (log_level_ <= LogLevel::WARN) {
            log(LogLevel::WARN, args...);
        }
    }

    template <typename... Args>
    void error(Args... args) {
        if (log_level_ <= LogLevel::ERROR) {
            log(LogLevel::ERROR, args...);
        }
    }

    template <typename... Args>
    void critical(Args... args) {
        log(LogLevel::CRITICAL, args...);
    }

private:
    Logger() : log_level_(LogLevel::INFO) {}

    template <typename T>
    void write_arg(std::stringstream& ss, const T& arg) {
        ss << arg;
    }

    template <typename T, typename... Rest>
    void write_args(std::stringstream& ss, const T& arg, Rest... rest) {
        write_arg(ss, arg);
        write_args(ss, rest...);
    }

    void write_args(std::stringstream& ss) {}

    template <typename... Args>
    void log(LogLevel level, Args... args) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::stringstream ss;

        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        ss << std::put_time(std::localtime(&time), "%H:%M:%S") << " ";

        // Level
        switch (level) {
            case LogLevel::DEBUG:
                ss << "[DEBUG] ";
                break;
            case LogLevel::INFO:
                ss << "[INFO] ";
                break;
            case LogLevel::WARN:
                ss << "[WARN] ";
                break;
            case LogLevel::ERROR:
                ss << "[ERROR] ";
                break;
            case LogLevel::CRITICAL:
                ss << "[CRITICAL] ";
                break;
        }

        // Message
        write_args(ss, args...);

        std::cout << ss.str() << "\n";
        std::cout.flush();

        if (level == LogLevel::CRITICAL || level == LogLevel::ERROR) {
            std::cerr << ss.str() << "\n";
            std::cerr.flush();
        }
    }

    std::mutex mutex_;
    LogLevel log_level_;
};

}  // namespace kf

// Convenience macro
#define KF_LOG_INFO(...) kf::Logger::instance().info(__VA_ARGS__)
#define KF_LOG_WARN(...) kf::Logger::instance().warn(__VA_ARGS__)
#define KF_LOG_ERROR(...) kf::Logger::instance().error(__VA_ARGS__)
#define KF_LOG_DEBUG(...) kf::Logger::instance().debug(__VA_ARGS__)
