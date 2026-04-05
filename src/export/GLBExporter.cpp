#include "export/GLBExporter.hpp"
#include "utils/Logger.hpp"
#include "utils/Math.hpp"
#include <cstring>
#include <chrono>

namespace kf {

tinygltf::Model GLBExporter::build_gltf_model(const Mesh& mesh) {
    tinygltf::Model model;

    // Create scene and node
    {
        tinygltf::Scene scene;
        scene.nodes.push_back(0);
        model.scenes.push_back(scene);
        model.defaultScene = 0;
    }

    {
        tinygltf::Node node;
        node.mesh = 0;
        model.nodes.push_back(node);
    }

    // Create mesh
    add_mesh_to_model(model, mesh);

    return model;
}

void GLBExporter::add_mesh_to_model(tinygltf::Model& model,
                                   const Mesh& mesh) {
    tinygltf::Mesh gltf_mesh;
    gltf_mesh.name = "KinectMesh";

    // Create accessors for positions, normals, colors, indices
    // Each function adds its accessor and returns the accessor index
    int pos_accessor_idx = create_position_buffer(model, mesh);
    int nrm_accessor_idx = create_normal_buffer(model, mesh);
    int col_accessor_idx = create_color_buffer(model, mesh);
    int idx_accessor_idx = create_indices_buffer(model, mesh);

    // Create primitive with correct accessor indices
    tinygltf::Primitive primitive;
    primitive.attributes["POSITION"] = pos_accessor_idx;
    primitive.attributes["NORMAL"] = nrm_accessor_idx;
    if (col_accessor_idx >= 0) {
        primitive.attributes["COLOR_0"] = col_accessor_idx;
    }
    primitive.indices = idx_accessor_idx;
    primitive.mode = TINYGLTF_MODE_TRIANGLES;

    // Create material
    {
        tinygltf::Material material;
        material.name = "DefaultMaterial";
        material.pbrMetallicRoughness.baseColorFactor = {0.8f, 0.8f, 0.8f,
                                                        1.0f};
        model.materials.push_back(material);
        primitive.material = 0;
    }

    gltf_mesh.primitives.push_back(primitive);
    model.meshes.push_back(gltf_mesh);
}

int GLBExporter::create_position_buffer(tinygltf::Model& model,
                                       const Mesh& mesh) {
    // Create buffer with vertex positions
    std::vector<float> positions;
    positions.reserve(mesh.positions.size() * 3);

    for (const auto& pos : mesh.positions) {
        // Coordinate conversion for Unity: Y-flip for OpenGL vs DirectX
        positions.push_back(pos.x());
        positions.push_back(pos.y());  // Unity Y-up
        positions.push_back(pos.z());
    }

    // Create buffer
    tinygltf::Buffer buffer;
    buffer.data.resize(positions.size() * sizeof(float));
    std::memcpy(buffer.data.data(), positions.data(),
               positions.size() * sizeof(float));
    int buffer_idx = model.buffers.size();
    model.buffers.push_back(buffer);

    // Create buffer view
    tinygltf::BufferView buffer_view;
    buffer_view.buffer = buffer_idx;
    buffer_view.byteOffset = 0;
    buffer_view.byteLength = buffer.data.size();
    buffer_view.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    int buffer_view_idx = model.bufferViews.size();
    model.bufferViews.push_back(buffer_view);

    // Create accessor
    tinygltf::Accessor accessor;
    accessor.bufferView = buffer_view_idx;
    accessor.byteOffset = 0;
    accessor.type = TINYGLTF_TYPE_VEC3;
    accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    accessor.count = mesh.positions.size();

    // Compute bounds
    if (!mesh.positions.empty()) {
        Vector3f min_pos = mesh.positions[0];
        Vector3f max_pos = mesh.positions[0];
        for (const auto& p : mesh.positions) {
            min_pos = min_pos.cwiseMin(p);
            max_pos = max_pos.cwiseMax(p);
        }
        accessor.minValues = {min_pos.x(), min_pos.y(), min_pos.z()};
        accessor.maxValues = {max_pos.x(), max_pos.y(), max_pos.z()};
    }

    model.accessors.push_back(accessor);
    return model.accessors.size() - 1;
}

int GLBExporter::create_normal_buffer(tinygltf::Model& model,
                                     const Mesh& mesh) {
    // Create buffer with vertex normals
    std::vector<float> normals;
    normals.reserve(mesh.normals.size() * 3);

    for (const auto& nrm : mesh.normals) {
        normals.push_back(nrm.x());
        normals.push_back(nrm.y());
        normals.push_back(nrm.z());
    }

    // Create buffer
    tinygltf::Buffer buffer;
    buffer.data.resize(normals.size() * sizeof(float));
    std::memcpy(buffer.data.data(), normals.data(),
               normals.size() * sizeof(float));
    int buffer_idx = model.buffers.size();
    model.buffers.push_back(buffer);

    // Create buffer view
    tinygltf::BufferView buffer_view;
    buffer_view.buffer = buffer_idx;
    buffer_view.byteOffset = 0;
    buffer_view.byteLength = buffer.data.size();
    buffer_view.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    int buffer_view_idx = model.bufferViews.size();
    model.bufferViews.push_back(buffer_view);

    // Create accessor
    tinygltf::Accessor accessor;
    accessor.bufferView = buffer_view_idx;
    accessor.byteOffset = 0;
    accessor.type = TINYGLTF_TYPE_VEC3;
    accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    accessor.count = mesh.normals.size();

    model.accessors.push_back(accessor);
    return model.accessors.size() - 1;
}

int GLBExporter::create_color_buffer(tinygltf::Model& model,
                                    const Mesh& mesh) {
    if (mesh.colors.empty()) return -1;

    // Create buffer with vertex colors (RGB)
    std::vector<float> colors;
    colors.reserve(mesh.colors.size());

    for (uint8_t c : mesh.colors) {
        colors.push_back(c / 255.0f);
    }

    // Create buffer
    tinygltf::Buffer buffer;
    buffer.data.resize(colors.size() * sizeof(float));
    std::memcpy(buffer.data.data(), colors.data(),
               colors.size() * sizeof(float));
    int buffer_idx = model.buffers.size();
    model.buffers.push_back(buffer);

    // Create buffer view
    tinygltf::BufferView buffer_view;
    buffer_view.buffer = buffer_idx;
    buffer_view.byteOffset = 0;
    buffer_view.byteLength = buffer.data.size();
    buffer_view.target = TINYGLTF_TARGET_ARRAY_BUFFER;
    int buffer_view_idx = model.bufferViews.size();
    model.bufferViews.push_back(buffer_view);

    // Create accessor
    tinygltf::Accessor accessor;
    accessor.bufferView = buffer_view_idx;
    accessor.byteOffset = 0;
    accessor.type = TINYGLTF_TYPE_VEC3;  // RGB
    accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    accessor.count = mesh.colors.size() / 3;

    model.accessors.push_back(accessor);
    return model.accessors.size() - 1;
}

int GLBExporter::create_indices_buffer(tinygltf::Model& model,
                                      const Mesh& mesh) {
    // Create buffer with triangle indices
    std::vector<uint32_t> indices = mesh.indices;

    tinygltf::Buffer buffer;
    buffer.data.resize(indices.size() * sizeof(uint32_t));
    std::memcpy(buffer.data.data(), indices.data(),
               indices.size() * sizeof(uint32_t));
    int buffer_idx = model.buffers.size();
    model.buffers.push_back(buffer);

    // Create buffer view
    tinygltf::BufferView buffer_view;
    buffer_view.buffer = buffer_idx;
    buffer_view.byteOffset = 0;
    buffer_view.byteLength = buffer.data.size();
    buffer_view.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
    int buffer_view_idx = model.bufferViews.size();
    model.bufferViews.push_back(buffer_view);

    // Create accessor
    tinygltf::Accessor accessor;
    accessor.bufferView = buffer_view_idx;
    accessor.byteOffset = 0;
    accessor.type = TINYGLTF_TYPE_SCALAR;
    accessor.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
    accessor.count = mesh.indices.size();

    model.accessors.push_back(accessor);
    return model.accessors.size() - 1;
}

ExportResult GLBExporter::export_mesh(const Mesh& mesh,
                                     const std::string& output_path) {
    ExportResult result;

    if (mesh.is_empty()) {
        result.success = false;
        result.error_message = "Mesh is empty";
        KF_LOG_ERROR("GLB export: ", result.error_message);
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    KF_LOG_INFO("Exporting GLB: ", mesh.vertex_count(), " vertices, ",
               mesh.triangle_count(), " triangles");

    // Build glTF model
    tinygltf::Model model = build_gltf_model(mesh);

    // Save as GLB
    tinygltf::TinyGLTF gltf;
    std::string error_msg;
    bool success = gltf.WriteGltfSceneToFile(
        &model, output_path, true, true, true, false,
        error_msg);  // Save as GLB (binary)

    result.success = success;

    if (result.success) {
        result.output_path = output_path;

        // Get file size
        std::ifstream file(output_path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            result.file_size_bytes = file.tellg();
            file.close();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        result.processing_time_ms =
            std::chrono::duration<float, std::milli>(end_time - start_time)
                .count();

        KF_LOG_INFO("GLB export successful: ", result.file_size_bytes,
                   " bytes, ", result.processing_time_ms, " ms");
    } else {
        result.error_message = error_msg;
        KF_LOG_ERROR("GLB export failed: ", error_msg);
    }

    return result;
}

}  // namespace kf
