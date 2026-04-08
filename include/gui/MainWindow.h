#pragma once

#include <QMainWindow>
#include <QTimer>
#include <memory>
#include "app/PipelineController.h"

QT_BEGIN_NAMESPACE
class QSplitter;
class QLabel;
class QPushButton;
class QProgressBar;
class QComboBox;
class QStatusBar;
QT_END_NAMESPACE

namespace kfusion {
namespace gui {

class OpenGLWidget;
class MetricsPanel;
class ControlPanel;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onStartClicked();
    void onStopClicked();
    void onResetClicked();
    void onExportPLY();
    void onExportGLB();
    void onModeChanged(int index);
    void onFrameReady(const sensor::FrameData& frame);
    void onMeshReady();
    void onMetricsTimer();

private:
    std::unique_ptr<app::PipelineController> pipeline_;

    OpenGLWidget*  gl_widget_       = nullptr;
    MetricsPanel*  metrics_panel_   = nullptr;
    ControlPanel*  control_panel_   = nullptr;

    QTimer* metrics_timer_ = nullptr;

    // Thread-safe metrics copy for timer refresh
    app::PipelineMetrics cached_metrics_;
    std::mutex           metrics_copy_mutex_;

    void setupUI();
    void connectSignals();
};

} // namespace gui
} // namespace kfusion
