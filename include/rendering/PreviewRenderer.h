#pragma once

#include <QOpenGLFunctions_3_3_Core>
#include "rendering/ShaderProgram.h"
#include "rendering/Camera.h"
#include "sensor/FrameData.h"
#include "meshing/MeshData.h"
#include <memory>

namespace kfusion {
namespace rendering {

enum class RenderMode {
    PointCloud,
    Mesh
};

class PreviewRenderer : protected QOpenGLFunctions_3_3_Core {
public:
    PreviewRenderer();
    ~PreviewRenderer();

    void initialize();
    void resize(int w, int h);
    void render();

    void setMode(RenderMode mode) { mode_ = mode; }
    RenderMode mode() const { return mode_; }

    // Upload point cloud from frame data
    void uploadPointCloud(const sensor::FrameData& frame);

    // Upload mesh for rendering
    void uploadMesh(const meshing::MeshData& mesh);

    OrbitCamera& camera() { return camera_; }

private:
    RenderMode   mode_      = RenderMode::PointCloud;
    int          viewport_w_ = 1;
    int          viewport_h_ = 1;
    bool         initialized_ = false;

    OrbitCamera  camera_;

    // Point cloud GL objects
    unsigned int pc_vao_ = 0, pc_vbo_pos_ = 0, pc_vbo_col_ = 0;
    int          pc_count_ = 0;
    ShaderProgram pc_shader_;

    // Mesh GL objects
    unsigned int mesh_vao_ = 0, mesh_vbo_pos_ = 0, mesh_vbo_norm_ = 0, mesh_ebo_ = 0;
    int          mesh_index_count_ = 0;
    ShaderProgram mesh_shader_;

    void initPointCloudBuffers();
    void initMeshBuffers();
    void renderPointCloud();
    void renderMesh();

    // Shader sources
    static const char* POINTCLOUD_VERT;
    static const char* POINTCLOUD_FRAG;
    static const char* MESH_VERT;
    static const char* MESH_FRAG;
};

} // namespace rendering
} // namespace kfusion
