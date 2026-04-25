#include "gui/MainWindow.h"
#include "gui/OpenGLWidget.h"
#include "gui/MetricsPanel.h"
#include "gui/ControlPanel.h"

#include <QSplitter>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QCloseEvent>
#include <QStatusBar>
#include <QLabel>

namespace kfusion {
namespace gui {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("KinectFusionQt");
    resize(1280, 720);

    setStyleSheet(R"css(
        QMainWindow, QWidget { background-color: #1e1f24; color: #ddd; }
        QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #aaa; }
        QProgressBar { border: 1px solid #555; border-radius: 3px; background: #333; }
        QProgressBar::chunk { background: #3a7fcf; border-radius: 2px; }
        QComboBox { background: #333; border: 1px solid #555; border-radius: 3px; padding: 3px; }
        QLabel { color: #ccc; }
        QStatusBar { background: #16171c; color: #aaa; }
    )css");

    pipeline_ = std::make_unique<app::PipelineController>();
    setupUI();
    connectSignals();

    // Timer for UI metrics refresh
    metrics_timer_ = new QTimer(this);
    connect(metrics_timer_, &QTimer::timeout, this, &MainWindow::onMetricsTimer);
    metrics_timer_->start(200); // 5 Hz UI refresh
}

MainWindow::~MainWindow() {
    if (pipeline_) pipeline_->stop();
}

void MainWindow::setupUI() {
    auto* central = new QWidget(this);
    setCentralWidget(central);

    auto* hbox = new QHBoxLayout(central);
    hbox->setSpacing(4);
    hbox->setContentsMargins(4, 4, 4, 4);

    // Left: control panel
    control_panel_ = new ControlPanel(this);
    hbox->addWidget(control_panel_);

    // Center: OpenGL view
    gl_widget_ = new OpenGLWidget(this);
    hbox->addWidget(gl_widget_, 1);

    // Right: metrics panel
    metrics_panel_ = new MetricsPanel(this);
    hbox->addWidget(metrics_panel_);

    statusBar()->showMessage("Ready — Connect Kinect and press Start.");
}

void MainWindow::connectSignals() {
    // Control panel → pipeline
    connect(control_panel_, &ControlPanel::startClicked, this, &MainWindow::onStartClicked);
    connect(control_panel_, &ControlPanel::stopClicked,  this, &MainWindow::onStopClicked);
    connect(control_panel_, &ControlPanel::resetClicked, this, &MainWindow::onResetClicked);
    connect(control_panel_, &ControlPanel::exportPLYClicked, this, &MainWindow::onExportPLY);
    connect(control_panel_, &ControlPanel::exportGLBClicked, this, &MainWindow::onExportGLB);
    connect(control_panel_, &ControlPanel::modeChanged, this, &MainWindow::onModeChanged);
    connect(control_panel_, &ControlPanel::threadsChanged, this, [this](int n) {
        if (pipeline_) pipeline_->setNumThreads(n);
    });

    connect(control_panel_, &ControlPanel::hyperparamsApplyClicked, this, [this]() {
        auto h = control_panel_->hyperparamsFromUi();
        if (h.min_depth >= h.max_depth) {
            QMessageBox::warning(this, "Invalid depth range",
                "Depth minimum must be less than depth maximum.");
            return;
        }
        pipeline_->setHyperparams(h);
        control_panel_->setHyperparams(pipeline_->hyperparamsSnapshot());
        statusBar()->showMessage("Hyperparameters applied.");
    });

    pipeline_->setFrameReadyCallback([this](const sensor::FrameData& frame) {
        // Must call on UI thread — use invokeMethod
        // Copy frame (safe because FrameData is value type)
        auto frame_copy = std::make_shared<sensor::FrameData>(frame);
        QMetaObject::invokeMethod(this, [this, frame_copy]() {
            onFrameReady(*frame_copy);
        }, Qt::QueuedConnection);
    });

    pipeline_->setMeshReadyCallback([this]() {
        QMetaObject::invokeMethod(this, [this]() {
            onMeshReady();
        }, Qt::QueuedConnection);
    });

    control_panel_->setHyperparams(pipeline_->hyperparamsSnapshot());
}

void MainWindow::onStartClicked() {
    if (!pipeline_->start()) {
        QMessageBox::critical(this, "Error",
            "Failed to start pipeline.\n"
            "Ensure Kinect is connected and libfreenect is installed.\n"
            "You may need to run with appropriate device permissions (udev rules).");
        return;
    }
    control_panel_->onPipelineStarted();
    statusBar()->showMessage("Capturing...");
}

void MainWindow::onStopClicked() {
    pipeline_->stop();
    control_panel_->onPipelineStopped();
    statusBar()->showMessage("Stopped.");
}

void MainWindow::onResetClicked() {
    const bool was_capturing = pipeline_->isRunning();

    pipeline_->reset();

    if (gl_widget_) gl_widget_->clearGeometry();

    if (was_capturing) {
        if (!pipeline_->start()) {
            QMessageBox::critical(this, "Error",
                "Scan was reset but failed to restart capture.\n"
                "Check the Kinect connection and press Start again.");
            control_panel_->onPipelineStopped();
            statusBar()->showMessage("Reset failed to restart — press Start.");
        } else {
            control_panel_->onPipelineStarted();
            statusBar()->showMessage("Scan reset — capturing again.");
        }
    } else {
        control_panel_->onPipelineStopped();
        statusBar()->showMessage("Volume cleared. Press Start to capture.");
    }

    if (metrics_panel_) metrics_panel_->update(pipeline_->metricsSnapshot());
}

void MainWindow::onExportPLY() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export PLY", "scan.ply", "PLY Files (*.ply)");
    if (path.isEmpty()) return;

    statusBar()->showMessage("Exporting PLY...");
    bool ok = pipeline_->exportPLY(path.toStdString());
    statusBar()->showMessage(ok ? ("PLY exported: " + path) : "PLY export FAILED.");
    if (!ok) {
        QMessageBox::warning(
            this,
            "Export Error",
            "PLY export failed.\nScan a little more so a mesh can be extracted, then try again.");
    }
}

void MainWindow::onExportGLB() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export GLB", "scan.glb", "GLB Files (*.glb)");
    if (path.isEmpty()) return;

    statusBar()->showMessage("Exporting GLB...");
    bool ok = pipeline_->exportGLB(path.toStdString());
    statusBar()->showMessage(ok ? ("GLB exported: " + path) : "GLB export FAILED.");
    if (!ok) {
        QMessageBox::warning(
            this,
            "Export Error",
            "GLB export failed.\nScan a little more so a mesh can be extracted, then try again.");
    }
}

void MainWindow::onModeChanged(int index) {
    rendering::RenderMode mode = (index == 0)
        ? rendering::RenderMode::PointCloud
        : rendering::RenderMode::Mesh;
    if (gl_widget_) gl_widget_->setRenderMode(mode);
}

void MainWindow::onFrameReady(const sensor::FrameData& frame) {
    // Always refresh GPU point cloud so "Mesh" preview can fall back until a mesh exists.
    if (gl_widget_) gl_widget_->updatePointCloud(frame);
}

void MainWindow::onMeshReady() {
    // Snapshot mesh from shared
    uint64_t ver;
    auto mesh = pipeline_->sharedMesh().snapshot(ver);

    if (gl_widget_ && mesh) {
        gl_widget_->updateMesh(*mesh);
    }

}

void MainWindow::onMetricsTimer() {
    if (!pipeline_ || !metrics_panel_) return;
    app::PipelineMetrics m = pipeline_->metricsSnapshot();
    metrics_panel_->update(m);
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (pipeline_) pipeline_->stop();
    event->accept();
}

} // namespace gui
} // namespace kfusion
