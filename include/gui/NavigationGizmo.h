#pragma once

#include <QWidget>
#include <QPoint>

namespace kfusion {
namespace gui {

class NavigationGizmo : public QWidget {
    Q_OBJECT
public:
    explicit NavigationGizmo(QWidget* parent = nullptr);

signals:
    void cameraRotationChanged(int pitch, int yaw, int roll);

public slots:
    void setCameraRotation(int pitch, int yaw, int roll);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

private:
    int pitch_ = 0;
    int yaw_ = 0;
    int roll_ = 0;
    QPoint last_mouse_pos_;
};

} // namespace gui
} // namespace kfusion
