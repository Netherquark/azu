#include "utils/Logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace kfusion {
namespace utils {

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

Logger::Logger() : level_(LogLevel::Info) {}

void Logger::setLevel(LogLevel level) {
    std::lock_guard<std::mutex> lk(mutex_);
    level_ = level;
}

void Logger::log(LogLevel level, const std::string& tag, const std::string& msg) {
    if (level < level_) return;

    std::lock_guard<std::mutex> lk(mutex_);

    // Timestamp
    auto now    = std::chrono::system_clock::now();
    auto t      = std::chrono::system_clock::to_time_t(now);
    auto ms     = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;
    std::ostringstream ts;
    ts << std::put_time(std::localtime(&t), "%H:%M:%S")
       << "." << std::setw(3) << std::setfill('0') << ms.count();

    const char* lvl_str = "INFO";
    const char* color   = "\033[0m";
    switch (level) {
        case LogLevel::Debug:   lvl_str = "DBG";  color = "\033[36m"; break;
        case LogLevel::Info:    lvl_str = "INF";  color = "\033[32m"; break;
        case LogLevel::Warning: lvl_str = "WRN";  color = "\033[33m"; break;
        case LogLevel::Error:   lvl_str = "ERR";  color = "\033[31m"; break;
    }

    std::cerr << color
              << "[" << ts.str() << "] "
              << "[" << lvl_str << "] "
              << "[" << tag << "] "
              << msg
              << "\033[0m\n";
}

void Logger::debug(const std::string& tag, const std::string& msg) {
    log(LogLevel::Debug, tag, msg);
}

void Logger::info(const std::string& tag, const std::string& msg) {
    log(LogLevel::Info, tag, msg);
}

void Logger::warn(const std::string& tag, const std::string& msg) {
    log(LogLevel::Warning, tag, msg);
}

void Logger::error(const std::string& tag, const std::string& msg) {
    log(LogLevel::Error, tag, msg);
}

} // namespace utils
} // namespace kfusion
