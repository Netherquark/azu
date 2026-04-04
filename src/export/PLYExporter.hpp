#pragma once

#include "export/ExportBase.hpp"
#include <sstream>

namespace kf {

// ============================================================================
// PLY Exporter (ASCII FORMAT)
// ============================================================================

class PLYExporter : public ExportBase {
public:
    ExportFormat get_format() const override { return ExportFormat::PLY; }
    std::string get_extension() const override { return ".ply"; }

    ExportResult export_mesh(const Mesh& mesh,
                            const std::string& output_path) override;

private:
    // Generate PLY header
    std::string generate_header(size_t vertex_count, size_t face_count);

    // Export vertices and faces to PLY format
    std::string generate_ply_data(const Mesh& mesh);
};

}  // namespace kf
