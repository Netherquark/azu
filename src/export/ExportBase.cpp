#include "export/ExportBase.hpp"
#include "utils/Logger.hpp"
#include <fstream>

namespace kf {

bool ExportBase::write_file(const std::string& path, const void* data,
                           size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        KF_LOG_ERROR("Failed to open file for writing: ", path);
        return false;
    }

    file.write(static_cast<const char*>(data), size);
    if (!file.good()) {
        KF_LOG_ERROR("Failed to write file: ", path);
        return false;
    }

    file.close();
    KF_LOG_INFO("Exported to: ", path, " (", size, " bytes)");
    return true;
}

}  // namespace kf
