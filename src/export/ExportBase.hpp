#pragma once

#include "utils/Types.hpp"
#include <string>
#include <memory>

namespace kf {

// ============================================================================
// Base Exporter
// ============================================================================

class ExportBase {
public:
    virtual ~ExportBase() = default;

    // Perform export
    virtual ExportResult export_mesh(const Mesh& mesh,
                                    const std::string& output_path) = 0;

    // Get supported format
    virtual ExportFormat get_format() const = 0;

    // Get file extension
    virtual std::string get_extension() const = 0;

protected:
    // Helper: write file
    bool write_file(const std::string& path, const void* data, size_t size);
};

}  // namespace kf
