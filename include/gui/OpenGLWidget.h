#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QPoint>
#include <memory>
#include "rendering/PreviewRenderer.h"
#include "meshing/MeshData.h"

namespace kfusion {
namespace gui {

class OpenGLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit OpenGLWidget(QWidget* parent = nullptr);
    ~OpenGLWidget() override;

    void setRenderMode(rendering::RenderMode mode);
    void updatePointCloud(const sensor::FrameData& frame);
    void updateMesh(const meshing::MeshData& mesh);
    void clearGeometry();

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

private:
    std::unique_ptr<rendering::PreviewRenderer> renderer_;
    QPoint last_mouse_pos_;
    bool   mouse_pressed_ = false;
    Qt::MouseButton pressed_button_ = Qt::NoButton;
};

} // namespace gui
} // namespace kfusion
