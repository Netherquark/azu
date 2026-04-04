#pragma once
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#include <vector>
#include <fstream>
#include <Eigen/Dense>

struct Mesh {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3f> normals;
    std::vector<uint32_t> indices;
};

class Exporter {
public:
    static bool ExportPLY(const std::string& filename, const Mesh& mesh) {
        std::ofstream out(filename);
        if (!out) return false;
        
        out << "ply\nformat ascii 1.0\n";
        out << "element vertex " << mesh.vertices.size() << "\n";
        out << "property float x\nproperty float y\nproperty float z\n";
        out << "property float nx\nproperty float ny\nproperty float nz\n";
        out << "element face " << mesh.indices.size() / 3 << "\n";
        out << "property list uchar int vertex_index\nend_header\n";
        
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            out << mesh.vertices[i].x() << " " << mesh.vertices[i].y() << " " << mesh.vertices[i].z() << " "
                << mesh.normals[i].x() << " " << mesh.normals[i].y() << " " << mesh.normals[i].z() << "\n";
        }
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            out << "3 " << mesh.indices[i] << " " << mesh.indices[i+1] << " " << mesh.indices[i+2] << "\n";
        }
        return true;
    }

    static bool ExportGLB(const std::string& filename, const Mesh& mesh) {
        tinygltf::Model model;
        tinygltf::TinyGLTF gltf;

        // ... Data buffer creation omitted for brevity, but requires converting 
        // vertices into a raw byte buffer and assigning to model.buffers.
        // Coordinate conversion for Unity (Right handed Z-forward to Left handed Y-Up):
        // Unity X = -Kinect X, Unity Y = Kinect Y, Unity Z = Kinect Z.
        
        // Write out binary
        return gltf.WriteGltfSceneToFile(&model, filename, false, false, true, true);
    }
};