#include "rendering/GLRenderWidget.hpp"
#include "utils/Logger.hpp"
#include <QMouseEvent>
#include <QWheelEvent>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

namespace kf {

GLRenderWidget::GLRenderWidget(QWidget* parent)
    : QOpenGLWidget(parent) {
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
}

GLRenderWidget::~GLRenderWidget() {
    makeCurrent();
    mesh_vertex_buffer_.reset();
    mesh_index_buffer_.reset();
    mesh_vao_.reset();
    point_cloud_buffer_.reset();
    color_buffer_.reset();
    pc_vao_.reset();
    shader_program_.reset();
    doneCurrent();
}

void GLRenderWidget::initializeGL() {
    initializeOpenGLFunctions();

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    setup_shaders();
    reset_view();
    KF_LOG_INFO("OpenGL widget initialized");
}

void GLRenderWidget::setup_shaders() {
    shader_program_ = std::make_unique<QOpenGLShaderProgram>();

    // Vertex shader
    const char* vertex_src = R"(
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec3 color;
        
        out vec3 v_position;
        out vec3 v_normal;
        out vec3 v_color;
        
        uniform mat4 projection;
        uniform mat4 view;
        uniform mat4 model;
        
        void main() {
            v_position = vec3(model * vec4(position, 1.0));
            v_normal = mat3(transpose(inverse(model))) * normal;
            v_color = color;
            gl_Position = projection * view * vec4(v_position, 1.0);
        }
    )";

    // Fragment shader
    const char* fragment_src = R"(
        #version 330 core
        in vec3 v_position;
        in vec3 v_normal;
        in vec3 v_color;
        
        out vec4 FragColor;
        
        void main() {
            vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
            vec3 normal = normalize(v_normal);
            float intensity = max(dot(normal, light_dir), 0.3);
            FragColor = vec4(v_color * intensity, 1.0);
        }
    )";

    shader_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_src);
    shader_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_src);
    shader_program_->link();

    if (!shader_program_->isLinked()) {
        KF_LOG_ERROR("Shader linking failed: ", shader_program_->log().toStdString());
    }
}

void GLRenderWidget::set_mesh(const MeshPtr& mesh) {
    makeCurrent();

    current_mesh_ = mesh;
    if (!mesh || mesh->is_empty()) return;

    // Create VAO
    if (!mesh_vao_) {
        mesh_vao_ = std::make_unique<QOpenGLVertexArrayObject>();
        mesh_vao_->create();
    }

    mesh_vao_->bind();

    // Vertex data
    if (!mesh_vertex_buffer_) {
        mesh_vertex_buffer_ = std::make_unique<QOpenGLBuffer>();
        mesh_vertex_buffer_->create();
    }

    std::vector<float> vertex_data;
    for (size_t i = 0; i < mesh->positions.size(); ++i) {
        const auto& pos = mesh->positions[i];
        const auto& nrm = mesh->normals[i];
        uint8_t r = 128, g = 128, b = 128;
        if (i * 3 + 2 < mesh->colors.size()) {
            r = mesh->colors[i * 3];
            g = mesh->colors[i * 3 + 1];
            b = mesh->colors[i * 3 + 2];
        }

        vertex_data.push_back(pos.x());
        vertex_data.push_back(pos.y());
        vertex_data.push_back(pos.z());
        vertex_data.push_back(nrm.x());
        vertex_data.push_back(nrm.y());
        vertex_data.push_back(nrm.z());
        vertex_data.push_back(r / 255.0f);
        vertex_data.push_back(g / 255.0f);
        vertex_data.push_back(b / 255.0f);
    }

    mesh_vertex_buffer_->bind();
    mesh_vertex_buffer_->allocate(vertex_data.data(),
                                 vertex_data.size() * sizeof(float));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                         (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                         (void*)(3 * sizeof(float)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float),
                         (void*)(6 * sizeof(float)));

    // Index buffer
    if (!mesh_index_buffer_) {
        mesh_index_buffer_ = std::make_unique<QOpenGLBuffer>(
            QOpenGLBuffer::IndexBuffer);
        mesh_index_buffer_->create();
    }

    mesh_index_buffer_->bind();
    mesh_index_buffer_->allocate(mesh->indices.data(),
                                mesh->indices.size() * sizeof(uint32_t));

    mesh_index_count_ = mesh->indices.size();

    mesh_vao_->release();
    doneCurrent();
}

void GLRenderWidget::set_point_cloud(
    const std::vector<Vector3f>& points,
    const std::vector<uint8_t>& colors) {
    makeCurrent();

    if (points.empty()) return;

    if (!pc_vao_) {
        pc_vao_ = std::make_unique<QOpenGLVertexArrayObject>();
        pc_vao_->create();
    }

    pc_vao_->bind();

    // Position buffer
    if (!point_cloud_buffer_) {
        point_cloud_buffer_ = std::make_unique<QOpenGLBuffer>();
        point_cloud_buffer_->create();
    }

    std::vector<float> pos_data;
    for (const auto& p : points) {
        pos_data.push_back(p.x());
        pos_data.push_back(p.y());
        pos_data.push_back(p.z());
    }

    point_cloud_buffer_->bind();
    point_cloud_buffer_->allocate(pos_data.data(),
                                 pos_data.size() * sizeof(float));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Color buffer
    if (!color_buffer_) {
        color_buffer_ = std::make_unique<QOpenGLBuffer>();
        color_buffer_->create();
    }

    color_buffer_->bind();
    color_buffer_->allocate(colors.data(),
                           colors.size() * sizeof(uint8_t));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, nullptr);

    point_cloud_size_ = points.size();

    pc_vao_->release();
    doneCurrent();
}

void GLRenderWidget::reset_view() {
    rotation_x_ = -30.0f;
    rotation_y_ = 45.0f;
    zoom_ = 1.0f;
    camera_pos_ = glm::vec3(0, 0, 2);
}

void GLRenderWidget::update_matrices() {
    int width = this->width();
    int height = this->height();
    float aspect = width > 0 && height > 0 ? float(width) / float(height) : 1.0f;

    projection_matrix_ =
        glm::perspective(45.0f, aspect, 0.1f, 100.0f);

    // Rotate around origin
    glm::mat4 rotation = glm::mat4(1.0f);
    rotation = glm::rotate(rotation, glm::radians(rotation_x_),
                          glm::vec3(1, 0, 0));
    rotation = glm::rotate(rotation, glm::radians(rotation_y_),
                          glm::vec3(0, 1, 0));

    glm::vec3 cam_dist = glm::vec3(0, 0, 2.0f / zoom_);
    camera_pos_ = glm::vec3(rotation * glm::vec4(cam_dist, 1.0f));

    view_matrix_ =
        glm::lookAt(camera_pos_, camera_target_, camera_up_);
}

void GLRenderWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!shader_program_ || !shader_program_->isLinked()) return;

    update_matrices();

    shader_program_->bind();
    shader_program_->setUniformValue("projection",
                                   QMatrix4x4(glm::value_ptr(projection_matrix_)));
    shader_program_->setUniformValue("view",
                                   QMatrix4x4(glm::value_ptr(view_matrix_)));

    QMatrix4x4 model;
    shader_program_->setUniformValue("model", model);

    if (render_mode_ == MESH && current_mesh_) {
        render_mesh();
    } else if (render_mode_ == POINT_CLOUD && point_cloud_size_ > 0) {
        render_point_cloud();
    }

    shader_program_->release();
}

void GLRenderWidget::render_mesh() {
    if (!mesh_vao_ || mesh_index_count_ == 0) return;

    mesh_vao_->bind();
    glDrawElements(GL_TRIANGLES, mesh_index_count_, GL_UNSIGNED_INT, nullptr);
    mesh_vao_->release();
}

void GLRenderWidget::render_point_cloud() {
    if (!pc_vao_ || point_cloud_size_ == 0) return;

    glPointSize(2.0f);
    pc_vao_->bind();
    glDrawArrays(GL_POINTS, 0, point_cloud_size_);
    pc_vao_->release();
    glPointSize(1.0f);
}

void GLRenderWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
}

void GLRenderWidget::mousePressEvent(QMouseEvent* event) {
    last_mouse_pos_ = event->pos();
    is_dragging_ = true;
}

void GLRenderWidget::mouseMoveEvent(QMouseEvent* event) {
    if (!is_dragging_) return;

    QPoint delta = event->pos() - last_mouse_pos_;
    rotation_y_ += delta.x() * 0.5f;
    rotation_x_ += delta.y() * 0.5f;

    last_mouse_pos_ = event->pos();
    update();
}

void GLRenderWidget::wheelEvent(QWheelEvent* event) {
    float delta = event->angleDelta().y() > 0 ? 1.1f : 0.9f;
    zoom_ *= delta;
    zoom_ = std::max(0.1f, std::min(10.0f, zoom_));
    update();
}

bool GLRenderWidget::save_screenshot(const std::string& path) {
    QImage screenshot = grabFramebuffer();
    return screenshot.save(QString::fromStdString(path));
}

}  // namespace kf
