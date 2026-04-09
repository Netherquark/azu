#pragma once

#include <QWidget>

QT_BEGIN_NAMESPACE
class QPushButton;
class QComboBox;
class QSpinBox;
class QLabel;
QT_END_NAMESPACE

namespace kfusion {
namespace gui {

class ControlPanel : public QWidget {
    Q_OBJECT
public:
    explicit ControlPanel(QWidget* parent = nullptr);

signals:
    void startClicked();
    void stopClicked();
    void resetClicked();
    void exportPLYClicked();
    void exportGLBClicked();
    void modeChanged(int index); // 0=PointCloud, 1=Mesh
    void threadsChanged(int n);

public slots:
    void onPipelineStarted();
    void onPipelineStopped();
    void setExportEnabled(bool enabled);

private:
    QPushButton* btn_start_  = nullptr;
    QPushButton* btn_stop_   = nullptr;
    QPushButton* btn_reset_  = nullptr;
    QPushButton* btn_ply_    = nullptr;
    QPushButton* btn_glb_    = nullptr;
    QComboBox*   combo_mode_ = nullptr;
    QSpinBox*    spin_threads_ = nullptr;
    QLabel*      lbl_status_ = nullptr;

    void setupUI();
    void connectSignals();
};

} // namespace gui
} // namespace kfusion
