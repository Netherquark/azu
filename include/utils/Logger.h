#pragma once
#include <string>
#include <mutex>
#include <chrono>

namespace kfusion {
namespace utils {

enum class LogLevel { Debug = 0, Info, Warning, Error };

class Logger {
public:
    static Logger& instance();

    void setLevel(LogLevel level);
    void log(LogLevel level, const std::string& tag, const std::string& msg);
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
