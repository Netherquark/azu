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

    // Pipeline → UI (via Qt::QueuedConnection for thread safety)
    pipeline_->setMetricsCallback([this](const app::PipelineMetrics& m) {
        std::lock_guard<std::mutex> lk(metrics_copy_mutex_);
        cached_metrics_ = m;
        cached_metrics_.state = pipeline_->state();
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
    pipeline_->reset();
    control_panel_->onPipelineStopped();
    control_panel_->setExportEnabled(false);
    statusBar()->showMessage("Reset. Ready.");
    if (metrics_panel_) metrics_panel_->update(app::PipelineMetrics{});
}

void MainWindow::onExportPLY() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export PLY", "scan.ply", "PLY Files (*.ply)");
    if (path.isEmpty()) return;

    statusBar()->showMessage("Exporting PLY...");
    bool ok = pipeline_->exportPLY(path.toStdString());
    statusBar()->showMessage(ok ? ("PLY exported: " + path) : "PLY export FAILED.");
    if (!ok) QMessageBox::warning(this, "Export Error", "PLY export failed. Is there a mesh?");
}

void MainWindow::onExportGLB() {
    QString path = QFileDialog::getSaveFileName(
        this, "Export GLB", "scan.glb", "GLB Files (*.glb)");
    if (path.isEmpty()) return;

    statusBar()->showMessage("Exporting GLB...");
    bool ok = pipeline_->exportGLB(path.toStdString());
    statusBar()->showMessage(ok ? ("GLB exported: " + path) : "GLB export FAILED.");
    if (!ok) QMessageBox::warning(this, "Export Error", "GLB export failed. Is there a mesh?");
}

void MainWindow::onModeChanged(int index) {
    rendering::RenderMode mode = (index == 0)
        ? rendering::RenderMode::PointCloud
        : rendering::RenderMode::Mesh;
    if (gl_widget_) gl_widget_->setRenderMode(mode);
}

void MainWindow::onFrameReady(const sensor::FrameData& frame) {
    // Only update point cloud if in point cloud mode
    if (gl_widget_ && gl_widget_->property("renderMode").toInt() == 0)
        gl_widget_->updatePointCloud(frame);
    else if (gl_widget_)
        gl_widget_->updatePointCloud(frame); // always keep it updated
}

void MainWindow::onMeshReady() {
    // Snapshot mesh from shared
    uint64_t ver;
    auto mesh = pipeline_->sharedMesh().snapshot(ver);

    if (gl_widget_) {
        gl_widget_->updateMesh(mesh);
    }

    // Enable export once we have a mesh
    if (!mesh.empty()) {
        control_panel_->setExportEnabled(true);
    }
}

void MainWindow::onMetricsTimer() {
    app::PipelineMetrics m;
    {
        std::lock_guard<std::mutex> lk(metrics_copy_mutex_);
        m = cached_metrics_;
    }
    if (metrics_panel_) metrics_panel_->update(m);
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (pipeline_) pipeline_->stop();
    event->accept();
}

} // namespace gui
} // namespace kfusion
