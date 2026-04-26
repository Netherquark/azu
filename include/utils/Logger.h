#pragma once
#include <string>
#include <mutex>
#include <chrono>
#include <memory>
#include <vector>

namespace kfusion {
namespace utils {

enum class LogLevel { Debug = 0, Info, Warning, Error };

class Logger {
public:
    static Logger& instance();

    void setLevel(LogLevel level);
    void log(LogLevel level, const std::string& tag, const std::string& msg);
    
    // Variadic printf-style logging
    void logf(LogLevel level, const char* tag, const char* format, ...);

    void debug(const std::string& tag, const std::string& msg);
    void info (const std::string& tag, const std::string& msg);
    void warn (const std::string& tag, const std::string& msg);
    void error(const std::string& tag, const std::string& msg);

private:
    Logger();
    std::mutex mutex_;
    LogLevel   level_;
};

} // namespace utils
} // namespace kfusion

// Convenience macros
#define KFLOG_DEBUG(tag, msg) ::kfusion::utils::Logger::instance().debug(tag, msg)
#define KFLOG_INFO(tag, msg)  ::kfusion::utils::Logger::instance().info(tag, msg)
#define KFLOG_WARN(tag, msg)  ::kfusion::utils::Logger::instance().warn(tag, msg)
#define KFLOG_ERROR(tag, msg) ::kfusion::utils::Logger::instance().error(tag, msg)

// Formatted macros
#define KFLOGF_DEBUG(tag, fmt, ...) ::kfusion::utils::Logger::instance().logf(::kfusion::utils::LogLevel::Debug, tag, fmt, ##__VA_ARGS__)
#define KFLOGF_INFO(tag, fmt, ...)  ::kfusion::utils::Logger::instance().logf(::kfusion::utils::LogLevel::Info, tag, fmt, ##__VA_ARGS__)
#define KFLOGF_WARN(tag, fmt, ...)  ::kfusion::utils::Logger::instance().logf(::kfusion::utils::LogLevel::Warning, tag, fmt, ##__VA_ARGS__)
#define KFLOGF_ERROR(tag, fmt, ...) ::kfusion::utils::Logger::instance().logf(::kfusion::utils::LogLevel::Error, tag, fmt, ##__VA_ARGS__)
