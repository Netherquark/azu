#pragma once

#include "utils/Types.hpp"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>

namespace kf {

// ============================================================================
// OpenGL Render Widget
// ============================================================================

class GLRenderWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    explicit GLRenderWidget(QWidget* parent = nullptr);
    ~GLRenderWidget() override;

    enum RenderMode {
        POINT_CLOUD,
        MESH
    };

    // Update mesh for rendering
    void set_mesh(const MeshPtr& mesh);

    // Update point cloud (for live preview)
    void set_point_cloud(const std::vector<Vector3f>& points,
                        const std::vector<uint8_t>& colors);

    // Set render mode
    void set_render_mode(RenderMode mode) { render_mode_ = mode; }

    // Reset view
    void reset_view();

    // Take screenshot
    bool save_screenshot(const std::string& path);

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

    // Mouse events
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void setup_shaders();
    void update_matrices();
    void render_mesh();
    void render_point_cloud();

    // Shader program
    std::unique_ptr<QOpenGLShaderProgram> shader_program_;

    // Mesh rendering
    std::unique_ptr<QOpenGLBuffer> mesh_vertex_buffer_;
    std::unique_ptr<QOpenGLBuffer> mesh_index_buffer_;
    std::unique_ptr<QOpenGLVertexArrayObject> mesh_vao_;
    size_t mesh_index_count_ = 0;
    MeshPtr current_mesh_;

    // Point cloud rendering
    std::unique_ptr<QOpenGLBuffer> point_cloud_buffer_;
    std::unique_ptr<QOpenGLBuffer> color_buffer_;
    std::unique_ptr<QOpenGLVertexArrayObject> pc_vao_;
    size_t point_cloud_size_ = 0;

    // Camera control
    glm::mat4 projection_matrix_;
    glm::mat4 view_matrix_;
    glm::vec3 camera_pos_{0, 0, 2};
    glm::vec3 camera_target_{0, 0, 0};
    glm::vec3 camera_up_{0, -1, 0};

    // Rotation
    float rotation_x_ = 0.0f;
    float rotation_y_ = 0.0f;
    float zoom_ = 1.0f;

    // Mouse
    QPoint last_mouse_pos_;
    bool is_dragging_ = false;

    // Rendering state
    RenderMode render_mode_ = MESH;
    bool needs_update_ = true;
};

}  // namespace kf
