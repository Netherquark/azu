#include "gui/MainWindow.hpp"
#include "utils/Logger.hpp"
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QTimer>
#include <QComboBox>
#include <QFileDialog>
#include <iostream>

namespace kf {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle("KinectFusionQt - Kinect v1 Offline Scanning Pipeline");
    resize(1600, 900);

    // Initialize components
    sensor_ = std::make_unique<KinectSensor>();
    thread_pool_ = std::make_unique<ThreadPool>();
    ply_exporter_ = std::make_unique<PLYExporter>();
    glb_exporter_ = std::make_unique<GLBExporter>();

    // Initialize tracker and TSDF
    ICPTracker::Config tracker_config;
    tracker_config.camera = CameraModel();
    tracker_ = std::make_unique<ICPTracker>(tracker_config);

    TSDFVolume::Config tsdf_config;
    tsdf_config.voxel_size = 0.005f;
    tsdf_ = std::make_unique<TSDFVolume>(tsdf_config);

    setup_ui();
    setup_timers();

    KF_LOG_INFO("MainWindow initialized");
}

MainWindow::~MainWindow() {
    on_stop_capture();
    sensor_->shutdown();
    thread_pool_.reset();
}

void MainWindow::setup_ui() {
    // Central widget
    QWidget* central_widget = new QWidget(this);
    QHBoxLayout* main_layout = new QHBoxLayout(central_widget);

    // ===== LEFT PANEL: Controls =====
    QWidget* left_panel = new QWidget();
    QVBoxLayout* left_layout = new QVBoxLayout(left_panel);

    QGroupBox* capture_group = new QGroupBox("Capture Control", this);
    QVBoxLayout* capture_layout = new QVBoxLayout(capture_group);

    start_button_ = new QPushButton("Start Capture", this);
    stop_button_ = new QPushButton("Stop Capture", this);
    stop_button_->setEnabled(false);
    reset_button_ = new QPushButton("Reset Scan", this);

    capture_layout->addWidget(start_button_);
    capture_layout->addWidget(stop_button_);
    capture_layout->addWidget(reset_button_);

    connect(start_button_, &QPushButton::clicked, this, &MainWindow::on_start_capture);
    connect(stop_button_, &QPushButton::clicked, this, &MainWindow::on_stop_capture);
    connect(reset_button_, &QPushButton::clicked, this, &MainWindow::on_reset_scan);

    left_layout->addWidget(capture_group);

    // Export group
    QGroupBox* export_group = new QGroupBox("Export", this);
    QVBoxLayout* export_layout = new QVBoxLayout(export_group);

    export_ply_button_ = new QPushButton("Export PLY", this);
    export_glb_button_ =
        new QPushButton("Export GLB (Unity)", this);

    export_layout->addWidget(export_ply_button_);
    export_layout->addWidget(export_glb_button_);

    connect(export_ply_button_, &QPushButton::clicked, this,
           &MainWindow::on_export_ply);
    connect(export_glb_button_, &QPushButton::clicked, this,
           &MainWindow::on_export_glb);

    left_layout->addWidget(export_group);

    // Render mode group
    QGroupBox* render_group = new QGroupBox("Rendering", this);
    QVBoxLayout* render_layout = new QVBoxLayout(render_group);

    render_mode_combo_ = new QComboBox(this);
    render_mode_combo_->addItem("Mesh");
    render_mode_combo_->addItem("Point Cloud");

    render_layout->addWidget(render_mode_combo_);
    connect(render_mode_combo_,
           QOverload<int>::of(&QComboBox::currentIndexChanged), this,
           &MainWindow::on_render_mode_changed);

    left_layout->addWidget(render_group);

    left_layout->addStretch();

    // ===== CENTER: OpenGL Viewer =====
    render_widget_ = new GLRenderWidget(this);

    // ===== RIGHT PANEL: Metrics =====
    QWidget* right_panel = new QWidget();
    QVBoxLayout* right_layout = new QVBoxLayout(right_panel);

    status_label_ = new QLabel("Status: Idle", this);
    fps_label_ = new QLabel("FPS: 0", this);
    frame_count_label_ = new QLabel("Frames: 0", this);
    tracking_status_label_ = new QLabel("Tracking: ---", this);
    icp_error_label_ = new QLabel("ICP Error: ---", this);
    voxel_usage_label_ = new QLabel("Voxel Usage: 0%", this);
    triangle_count_label_ = new QLabel("Triangles: 0", this);
    memory_label_ = new QLabel("Memory: 0 MB", this);

    progress_bar_ = new QProgressBar(this);
    progress_bar_->setRange(0, 100);
    progress_bar_->setValue(0);

    QGroupBox* metrics_group = new QGroupBox("Reconstruction Stats", this);
    QVBoxLayout* metrics_layout = new QVBoxLayout(metrics_group);

    metrics_layout->addWidget(status_label_);
    metrics_layout->addWidget(fps_label_);
    metrics_layout->addWidget(frame_count_label_);
    metrics_layout->addWidget(tracking_status_label_);
    metrics_layout->addWidget(icp_error_label_);
    metrics_layout->addWidget(voxel_usage_label_);
    metrics_layout->addWidget(triangle_count_label_);
    metrics_layout->addWidget(memory_label_);
    metrics_layout->addWidget(progress_bar_);
    metrics_layout->addStretch();

    right_layout->addWidget(metrics_group);

    // Assemble main layout
    main_layout->addWidget(left_panel, 1);
    main_layout->addWidget(render_widget_, 4);
    main_layout->addWidget(right_panel, 1);

    setCentralWidget(central_widget);
}

void MainWindow::setup_timers() {
    QTimer* update_timer = new QTimer(this);
    connect(update_timer, &QTimer::timeout, this, &MainWindow::update_metrics);
    update_timer->start(250);  // Update metrics 4x/sec

    QTimer* capture_timer = new QTimer(this);
    connect(capture_timer, &QTimer::timeout, this, &MainWindow::capture_frame);
    capture_timer->start(33);  // ~30 FPS
}

void MainWindow::on_start_capture() {
    if (!sensor_->initialize()) {
        update_status("ERROR: Failed to initialize sensor");
        return;
    }

    if (!sensor_->start_capture()) {
        update_status("ERROR: Failed to start capture");
        return;
    }

    is_capturing_ = true;
    start_button_->setEnabled(false);
    stop_button_->setEnabled(true);
    update_status("Capturing...");
    frame_count_total_ = 0;

    KF_LOG_INFO("Capture started");
}

void MainWindow::on_stop_capture() {
    if (sensor_) {
        sensor_->stop_capture();
    }
    is_capturing_ = false;
    start_button_->setEnabled(true);
    stop_button_->setEnabled(false);
    update_status("Stopped");

    KF_LOG_INFO("Capture stopped");
}

void MainWindow::on_reset_scan() {
    if (tsdf_) {
        tsdf_->reset();
    }
    if (tracker_) {
        tracker_->reset_pose();
    }

    frame_count_total_ = 0;
    update_status("Scan reset");

    KF_LOG_INFO("Scan reset");
}

void MainWindow::on_export_ply() {
    QString file_path =
        QFileDialog::getSaveFileName(this, "Export PLY", "", "PLY Files (*.ply)");
    if (file_path.isEmpty()) return;

    MeshPtr mesh = tsdf_->extract_mesh();
    if (!mesh || mesh->is_empty()) {
        update_status("ERROR: Mesh is empty");
        return;
    }

    ExportResult result =
        ply_exporter_->export_mesh(*mesh, file_path.toStdString());

    if (result.success) {
        update_status("PLY exported: " + result.output_path);
    } else {
        update_status("PLY export failed: " + result.error_message);
    }
}

void MainWindow::on_export_glb() {
    QString file_path =
        QFileDialog::getSaveFileName(this, "Export GLB", "", "GLB Files (*.glb)");
    if (file_path.isEmpty()) return;

    MeshPtr mesh = tsdf_->extract_mesh();
    if (!mesh || mesh->is_empty()) {
        update_status("ERROR: Mesh is empty");
        return;
    }

    ExportResult result =
        glb_exporter_->export_mesh(*mesh, file_path.toStdString());

    if (result.success) {
        update_status("GLB exported: " + result.output_path);
    } else {
        update_status("GLB export failed: " + result.error_message);
    }
}

void MainWindow::on_toggle_render_mode() {
    if (current_render_mode_ == GLRenderWidget::MESH) {
        current_render_mode_ = GLRenderWidget::POINT_CLOUD;
    } else {
        current_render_mode_ = GLRenderWidget::MESH;
    }
    render_widget_->set_render_mode(current_render_mode_);
}

void MainWindow::on_render_mode_changed(int index) {
    if (index == 0) {
        current_render_mode_ = GLRenderWidget::MESH;
    } else {
        current_render_mode_ = GLRenderWidget::POINT_CLOUD;
    }
    render_widget_->set_render_mode(current_render_mode_);
}

void MainWindow::capture_frame() {
    if (!is_capturing_) return;

    DepthFramePtr depth_frame;
    ColorFramePtr color_frame;

    if (sensor_->get_depth_frame(depth_frame)) {
        frame_count_total_++;

        // Depth to vertex map
        VertexMapPtr vertex_map = tracker_->depth_to_vertex_map(*depth_frame);

        // Extract point cloud for preview
        if (current_render_mode_ == GLRenderWidget::POINT_CLOUD) {
            std::vector<Vector3f> points;
            std::vector<uint8_t> colors;

            for (int y = 0; y < DepthFrame::HEIGHT; ++y) {
                for (int x = 0; x < DepthFrame::WIDTH; ++x) {
                    Vector4f v = vertex_map->vertex(x, y);
                    if (v.w() > 0.5f) {
                        points.push_back(v.head<3>());
                        colors.push_back(100);
                        colors.push_back(150);
                        colors.push_back(200);
                    }
                }
            }

            render_widget_->set_point_cloud(points, colors);
        }

        // Queue for async processing
        thread_pool_->submit([this, vertex_map, depth_frame]() {
            process_tracking();
            process_integration();
        });
    }
}

void MainWindow::process_tracking() {
    // ICP tracking against model (simplified for now)
    // In production, would raycast and track
}

void MainWindow::process_integration() {
    DepthFramePtr depth_frame;
    ColorFramePtr color_frame;

    if (sensor_->get_depth_frame(depth_frame)) {
        CameraPose pose = tracker_->get_pose();
        tsdf_->integrate(*depth_frame, pose, color_frame.get());

        // Extract mesh periodically
        if (frame_count_total_ % 30 == 0) {
            thread_pool_->submit([this]() { process_mesh_extraction(); });
        }
    }
}

void MainWindow::process_mesh_extraction() {
    MeshPtr mesh = tsdf_->extract_mesh();
    if (mesh && !mesh->is_empty()) {
        render_widget_->set_mesh(mesh);
    }
}

void MainWindow::update_metrics() {
    fps_label_->setText(
        QString::fromStdString("FPS: " + std::to_string(
            static_cast<int>(sensor_->get_fps_estimate()))));
    frame_count_label_->setText(
        QString::fromStdString("Frames: " + std::to_string(frame_count_total_)));

    stats_.voxel_grid_usage_percent = tsdf_->get_voxel_grid_usage();
    voxel_usage_label_->setText(
        QString::fromStdString("Voxel Usage: " +
                              std::to_string(static_cast<int>(
                                  stats_.voxel_grid_usage_percent)) +
                              "%"));

    tracking_status_label_->setText(
        QString::fromStdString(stats_.tracking_ok ? "Tracking: OK" :
                                                   "Tracking: LOST"));
}

void MainWindow::update_status(const std::string& status) {
    if (status_label_) {
        status_label_->setText(QString::fromStdString("Status: " + status));
    }
    KF_LOG_INFO("Status: ", status);
}

void MainWindow::closeEvent(QCloseEvent* event) {
    on_stop_capture();
    QMainWindow::closeEvent(event);
}

}  // namespace kf
