#include "utils/Logger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cstdarg>

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

void Logger::logf(LogLevel level, const char* tag, const char* format, ...) {
    if (level < level_) return;

    va_list args;
    va_start(args, format);
    
    // Determine size
    va_list args_copy;
    va_copy(args_copy, args);
    int size = std::vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);

    if (size <= 0) {
        va_end(args);
        return;
    }

    std::vector<char> buffer(size + 1);
    std::vsnprintf(buffer.data(), buffer.size(), format, args);
    va_end(args);

    log(level, tag, std::string(buffer.data()));
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
        case LogLevel::Debug:   lvl_str = "DEBUG"; color = "\033[36m"; break; // Cyan
        case LogLevel::Info:    lvl_str = "INFO "; color = "\033[32m"; break; // Green
        case LogLevel::Warning: lvl_str = "WARN "; color = "\033[33m"; break; // Yellow
        case LogLevel::Error:   lvl_str = "ERROR"; color = "\033[31m"; break; // Red
    }

    // Fixed width tag for alignment (10 chars)
    std::string padded_tag = tag;
    if (padded_tag.length() > 10) padded_tag = padded_tag.substr(0, 7) + "...";
    else while (padded_tag.length() < 10) padded_tag += " ";

    std::cerr << color
              << "[" << ts.str() << "] "
              << "[" << lvl_str << "] "
              << "[" << padded_tag << "] "
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
