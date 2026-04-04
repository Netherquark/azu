#pragma once

#include "export/ExportBase.hpp"
#include "third_party/tinygltf/tiny_gltf.h"

namespace kf {

// ============================================================================
// GLB Exporter (GLTF Binary Format)
// ============================================================================

class GLBExporter : public ExportBase {
public:
    ExportFormat get_format() const override { return ExportFormat::GLB; }
    std::string get_extension() const override { return ".glb"; }

    ExportResult export_mesh(const Mesh& mesh,
                            const std::string& output_path) override;

private:
    // Build glTF model from mesh
    tinygltf::Model build_gltf_model(const Mesh& mesh);

    // Add mesh to model
    void add_mesh_to_model(tinygltf::Model& model, const Mesh& mesh);

    // Create buffer view for positions
    int create_position_buffer(tinygltf::Model& model, const Mesh& mesh);

    // Create buffer view for normals
    int create_normal_buffer(tinygltf::Model& model, const Mesh& mesh);

    // Create buffer view for colors
    int create_color_buffer(tinygltf::Model& model, const Mesh& mesh);

    // Create buffer view for indices
    int create_indices_buffer(tinygltf::Model& model, const Mesh& mesh);
};

}  // namespace kf
