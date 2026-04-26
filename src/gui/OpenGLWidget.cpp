#include "gui/OpenGLWidget.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QSurfaceFormat>

namespace kfusion {
namespace gui {

OpenGLWidget::OpenGLWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    setFormat(fmt);

    setMinimumSize(640, 480);
}

OpenGLWidget::~OpenGLWidget() {
    makeCurrent();
    renderer_.reset();
    doneCurrent();
}

void OpenGLWidget::setRenderMode(rendering::RenderMode mode) {
    if (renderer_) renderer_->setMode(mode);
    update();
}

void OpenGLWidget::updatePointCloud(const sensor::FrameData& frame) {
    makeCurrent();
    if (renderer_) renderer_->uploadPointCloud(frame);
    doneCurrent();
    update();
}

void OpenGLWidget::updateMesh(const meshing::MeshData& mesh) {
    makeCurrent();
    if (renderer_) renderer_->uploadMesh(mesh);
    doneCurrent();
    update();
}

void OpenGLWidget::clearGeometry() {
    makeCurrent();
    if (renderer_) renderer_->clearGeometry();
    doneCurrent();
    update();
}

void OpenGLWidget::initializeGL() {
    initializeOpenGLFunctions();
    renderer_ = std::make_unique<rendering::PreviewRenderer>();
    renderer_->initialize();
}

void OpenGLWidget::resizeGL(int w, int h) {
    if (renderer_) renderer_->resize(w, h);
}

void OpenGLWidget::paintGL() {
    if (renderer_) renderer_->render();
}

void OpenGLWidget::mousePressEvent(QMouseEvent* e) {
    last_mouse_pos_  = e->pos();
    mouse_pressed_   = true;
    pressed_button_  = e->button();
}

void OpenGLWidget::mouseMoveEvent(QMouseEvent* e) {
    if (!mouse_pressed_ || !renderer_) return;
    if (renderer_->mode() == rendering::RenderMode::PointCloud) return;

    QPoint delta = e->pos() - last_mouse_pos_;
    last_mouse_pos_ = e->pos();

    if (pressed_button_ == Qt::LeftButton) {
        renderer_->camera().orbit(
            static_cast<float>(delta.x()),
            static_cast<float>(delta.y())
        );
    } else if (pressed_button_ == Qt::RightButton) {
        renderer_->camera().pan(
            static_cast<float>(delta.x()),
            static_cast<float>(delta.y())
        );
    }
    update();
}

void OpenGLWidget::wheelEvent(QWheelEvent* e) {
    if (!renderer_) return;
    if (renderer_->mode() == rendering::RenderMode::PointCloud) return;
    float delta = static_cast<float>(e->angleDelta().y()) / 120.0f;
    renderer_->camera().zoom(delta);
    update();
}

} // namespace gui
} // namespace kfusion
