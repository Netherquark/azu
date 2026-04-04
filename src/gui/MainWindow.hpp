#pragma once

#include "utils/Types.hpp"
#include "sensor/KinectSensor.hpp"
#include "tracking/ICPTracker.hpp"
#include "tsdf/TSDFVolume.hpp"
#include "export/PLYExporter.hpp"
#include "export/GLBExporter.hpp"
#include "rendering/GLRenderWidget.hpp"
#include "utils/ThreadPool.hpp"
#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QProgressBar>
#include <QComboBox>
#include <memory>

namespace kf {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    void on_start_capture();
    void on_stop_capture();
    void on_reset_scan();
    void on_export_ply();
    void on_export_glb();
    void on_toggle_render_mode();
    void on_render_mode_changed(int index);

    void update_metrics();
    void capture_frame();
    void process_tracking();
    void process_integration();
    void process_mesh_extraction();

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void setup_ui();
    void setup_timers();
    void update_status(const std::string& status);

    // Sensor and processing
    std::unique_ptr<KinectSensor> sensor_;
    std::unique_ptr<ICPTracker> tracker_;
    std::unique_ptr<TSDFVolume> tsdf_;
    std::unique_ptr<ThreadPool> thread_pool_;

    // Exporters
    std::unique_ptr<PLYExporter> ply_exporter_;
    std::unique_ptr<GLBExporter> glb_exporter_;

    // UI Components
    GLRenderWidget* render_widget_ = nullptr;
    QPushButton* start_button_ = nullptr;
    QPushButton* stop_button_ = nullptr;
    QPushButton* reset_button_ = nullptr;
    QPushButton* export_ply_button_ = nullptr;
    QPushButton* export_glb_button_ = nullptr;
    QComboBox* render_mode_combo_ = nullptr;

    // Metrics display
    QLabel* fps_label_ = nullptr;
    QLabel* frame_count_label_ = nullptr;
    QLabel* tracking_status_label_ = nullptr;
    QLabel* icp_error_label_ = nullptr;
    QLabel* voxel_usage_label_ = nullptr;
    QLabel* triangle_count_label_ = nullptr;
    QLabel* memory_label_ = nullptr;
    QLabel* status_label_ = nullptr;
    QProgressBar* progress_bar_ = nullptr;

    // State
    bool is_capturing_ = false;
    ReconstructionStats stats_;
    float fps_counter_ = 0.0f;
    int frame_count_total_ = 0;

    GLRenderWidget::RenderMode current_render_mode_ = GLRenderWidget::MESH;
};

}  // namespace kf;