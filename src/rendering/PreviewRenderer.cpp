#include "rendering/PreviewRenderer.h"
#include <Eigen/Core>
#include <cstring>
#include <iostream>

namespace kfusion {
namespace rendering {

namespace {

// Kinect camera space: +X right, +Y down (image), +Z forward. GL view: +Y up, −Z forward.
// Equivalent to 180° about X: (x,y,z) → (x,−y,−z). Keeps right-handed volume.
Eigen::Matrix4f kinectToOpenGL() {
    Eigen::Matrix4f F = Eigen::Matrix4f::Identity();
    F(1, 1) = -1.0f;
    F(2, 2) = -1.0f;
    return F;
}

} // namespace

// ---------------------------------------------------------------------------
// Shader sources
// ---------------------------------------------------------------------------

const char* PreviewRenderer::POINTCLOUD_VERT = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    gl_PointSize = 1.5;
    vColor = aColor;
}
)glsl";

const char* PreviewRenderer::POINTCLOUD_FRAG = R"glsl(
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(vColor, 1.0);
}
)glsl";

const char* PreviewRenderer::MESH_VERT = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModelView;
uniform mat3 uNormalMatrix;
out vec3 vNormal;
out vec3 vFragPos;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vFragPos = vec3(uModelView * vec4(aPos, 1.0));
    vNormal  = normalize(uNormalMatrix * aNormal);
}
)glsl";

const char* PreviewRenderer::MESH_FRAG = R"glsl(
#version 330 core
in vec3 vNormal;
in vec3 vFragPos;
out vec4 FragColor;
void main() {
    vec3 lightDir = normalize(vec3(1.0, 2.0, 3.0));
    float diff = max(dot(vNormal, lightDir), 0.0);
    vec3 ambient = vec3(0.15);
    vec3 diffuse = vec3(0.6) * diff;
    float spec_k = pow(max(dot(normalize(-vFragPos), reflect(-lightDir, vNormal)), 0.0), 32.0);
    vec3 specular = vec3(0.2) * spec_k;
    vec3 color = vec3(0.72, 0.78, 0.85);
    FragColor = vec4((ambient + diffuse + specular) * color, 1.0);
}
)glsl";

// ---------------------------------------------------------------------------

PreviewRenderer::PreviewRenderer() = default;

PreviewRenderer::~PreviewRenderer() {
    if (!initialized_) return;
    if (pc_vao_) { glDeleteVertexArrays(1, &pc_vao_); }
    if (pc_vbo_pos_) { glDeleteBuffers(1, &pc_vbo_pos_); }
    if (pc_vbo_col_) { glDeleteBuffers(1, &pc_vbo_col_); }
    if (mesh_vao_) { glDeleteVertexArrays(1, &mesh_vao_); }
    if (mesh_vbo_pos_) { glDeleteBuffers(1, &mesh_vbo_pos_); }
    if (mesh_vbo_norm_) { glDeleteBuffers(1, &mesh_vbo_norm_); }
    if (mesh_ebo_) { glDeleteBuffers(1, &mesh_ebo_); }
}

void PreviewRenderer::initialize() {
    initializeOpenGLFunctions();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClearColor(0.08f, 0.09f, 0.11f, 1.0f);

    pc_shader_.load(POINTCLOUD_VERT, POINTCLOUD_FRAG);
    mesh_shader_.load(MESH_VERT, MESH_FRAG);

    initPointCloudBuffers();
    initMeshBuffers();

    initialized_ = true;
}

void PreviewRenderer::resize(int w, int h) {
    viewport_w_ = std::max(w, 1);
    viewport_h_ = std::max(h, 1);
    glViewport(0, 0, viewport_w_, viewport_h_);
}

void PreviewRenderer::render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = static_cast<float>(viewport_w_) / static_cast<float>(viewport_h_);
    Eigen::Matrix4f V   = camera_.viewMatrix();
    Eigen::Matrix4f P   = camera_.projectionMatrix(aspect);
    Eigen::Matrix4f MVP = P * V;

    if (mode_ == RenderMode::PointCloud) {
        renderPointCloud();
    } else if (mesh_index_count_ > 0) {
        renderMesh();
    } else {
        // Mesh mode selected but no extraction yet (or empty) — keep showing live depth cloud.
        renderPointCloud();
    }

    (void)MVP;
}

void PreviewRenderer::initPointCloudBuffers() {
    glGenVertexArrays(1, &pc_vao_);
    glGenBuffers(1, &pc_vbo_pos_);
    glGenBuffers(1, &pc_vbo_col_);

    glBindVertexArray(pc_vao_);

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_pos_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_col_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void PreviewRenderer::initMeshBuffers() {
    glGenVertexArrays(1, &mesh_vao_);
    glGenBuffers(1, &mesh_vbo_pos_);
    glGenBuffers(1, &mesh_vbo_norm_);
    glGenBuffers(1, &mesh_ebo_);

    glBindVertexArray(mesh_vao_);

    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_pos_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_norm_);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ebo_);

    glBindVertexArray(0);
}

void PreviewRenderer::uploadPointCloud(const sensor::FrameData& frame) {
    if (!initialized_) return;

    const int N = frame.width * frame.height;
    std::vector<float> positions, colors;
    positions.reserve(N * 3);
    colors.reserve(N * 3);
    int count = 0;

    for (int i = 0; i < N; ++i) {
        if (frame.depth_meters[i] <= 0.0f) continue;
        const auto& v = frame.vertices[i];
        positions.push_back(v.x());
        positions.push_back(v.y());
        positions.push_back(v.z());
        colors.push_back(frame.rgb[i*3+0] / 255.0f);
        colors.push_back(frame.rgb[i*3+1] / 255.0f);
        colors.push_back(frame.rgb[i*3+2] / 255.0f);
        ++count;
    }

    pc_count_ = count;

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_pos_);
    glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(float),
                 positions.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, pc_vbo_col_);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float),
                 colors.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PreviewRenderer::clearGeometry() {
    pc_count_         = 0;
    mesh_index_count_ = 0;
}

void PreviewRenderer::uploadMesh(const meshing::MeshData& mesh) {
    if (!initialized_ || mesh.empty()) return;

    glBindVertexArray(mesh_vao_);

    // Positions
    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_pos_);
    glBufferData(GL_ARRAY_BUFFER,
                 mesh.positions.size() * sizeof(Eigen::Vector3f),
                 mesh.positions.data(), GL_DYNAMIC_DRAW);

    // Normals
    glBindBuffer(GL_ARRAY_BUFFER, mesh_vbo_norm_);
    glBufferData(GL_ARRAY_BUFFER,
                 mesh.normals.size() * sizeof(Eigen::Vector3f),
                 mesh.normals.data(), GL_DYNAMIC_DRAW);

    // Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh_ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 mesh.indices.size() * sizeof(uint32_t),
                 mesh.indices.data(), GL_DYNAMIC_DRAW);

    mesh_index_count_ = static_cast<int>(mesh.indices.size());
    glBindVertexArray(0);
}

void PreviewRenderer::renderPointCloud() {
    if (!pc_shader_.isValid() || pc_count_ == 0) return;

    float aspect = static_cast<float>(viewport_w_) / static_cast<float>(viewport_h_);
    Eigen::Matrix4f V   = camera_.viewMatrix();
    Eigen::Matrix4f P   = camera_.projectionMatrix(aspect);
    const Eigen::Matrix4f F   = kinectToOpenGL();
    const Eigen::Matrix4f MVP = P * V * F;

    pc_shader_.use();
    pc_shader_.setUniformMat4("uMVP", MVP.data());

    glBindVertexArray(pc_vao_);
    glDrawArrays(GL_POINTS, 0, pc_count_);
    glBindVertexArray(0);

    pc_shader_.disuse();
}

void PreviewRenderer::renderMesh() {
    if (!mesh_shader_.isValid() || mesh_index_count_ == 0) return;

    float aspect = static_cast<float>(viewport_w_) / static_cast<float>(viewport_h_);
    Eigen::Matrix4f V   = camera_.viewMatrix();
    Eigen::Matrix4f P   = camera_.projectionMatrix(aspect);
    const Eigen::Matrix4f F    = kinectToOpenGL();
    const Eigen::Matrix4f MV   = V * F;
    const Eigen::Matrix4f MVP  = P * MV;

    mesh_shader_.use();
    mesh_shader_.setUniformMat4("uMVP", MVP.data());
    mesh_shader_.setUniformMat4("uModelView", MV.data());

    Eigen::Matrix3f normalMatrix = MV.block<3,3>(0,0).inverse().transpose();
    mesh_shader_.setUniformMat3("uNormalMatrix", normalMatrix.data());

    glBindVertexArray(mesh_vao_);
    glDrawElements(GL_TRIANGLES, mesh_index_count_, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);

    mesh_shader_.disuse();
}

} // namespace rendering
} // namespace kfusion
