#include "gui/ControlPanel.h"
#include "gui/UiScale.h"
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QFrame>
#include <QScrollArea>

namespace kfusion {
namespace gui {

namespace {

QDoubleSpinBox* makeDoubleSpin(double minV, double maxV, double step, double val, int decimals, QWidget* parent) {
    auto* s = new QDoubleSpinBox(parent);
    s->setRange(minV, maxV);
    s->setSingleStep(step);
    s->setDecimals(decimals);
    s->setValue(val);
    s->setButtonSymbols(QAbstractSpinBox::PlusMinus);
    return s;
}

} // namespace

ControlPanel::ControlPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
    connectSignals();
}

void ControlPanel::setupUI() {
    auto* root = new QVBoxLayout(this);
    root->setSpacing(6);
    root->setContentsMargins(6, 6, 6, 6);

    lbl_status_ = new QLabel("Status: Idle", this);
    lbl_status_->setStyleSheet("color: #aaa; font-size: 9pt;");
    root->addWidget(lbl_status_);

    auto* line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    root->addWidget(line);

    auto* grp_capture = new QGroupBox("Capture", this);
    auto* v_cap = new QVBoxLayout(grp_capture);
    btn_start_ = new QPushButton("▶  Start Capture", this);
    btn_stop_  = new QPushButton("■  Stop Capture",  this);
    btn_reset_ = new QPushButton("↺  Reset Scan",    this);
    btn_start_->setStyleSheet("QPushButton { background: #2a7a2a; color: white; padding: 0.35em 0.5em; border-radius: 4px; font-size: 9pt; }");
    btn_stop_ ->setStyleSheet("QPushButton { background: #7a2a2a; color: white; padding: 0.35em 0.5em; border-radius: 4px; font-size: 9pt; }");
    btn_reset_->setStyleSheet("QPushButton { background: #444; color: white; padding: 0.35em 0.5em; border-radius: 4px; font-size: 9pt; }");
    btn_stop_->setEnabled(false);
    v_cap->addWidget(btn_start_);
    v_cap->addWidget(btn_stop_);
    v_cap->addWidget(btn_reset_);
    root->addWidget(grp_capture);

    auto* scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* inner = new QWidget(scroll);
    auto* inner_layout = new QVBoxLayout(inner);
    inner_layout->setSpacing(6);
    inner_layout->setContentsMargins(0, 0, 4, 0);

    auto* grp_hp = new QGroupBox("Hyperparameters (Apply)", inner);
    auto* g = new QGridLayout(grp_hp);
    int r = 0;

    g->addWidget(new QLabel("Depth min (m)", grp_hp), r, 0);
    spin_depth_min_ = makeDoubleSpin(0.01, 3.0, 0.01, 0.05, 3, grp_hp);
    g->addWidget(spin_depth_min_, r++, 1);

    g->addWidget(new QLabel("Depth max (m)", grp_hp), r, 0);
    spin_depth_max_ = makeDoubleSpin(0.2, 12.0, 0.1, 8.0, 2, grp_hp);
    g->addWidget(spin_depth_max_, r++, 1);

    g->addWidget(new QLabel("Voxel size (m)", grp_hp), r, 0);
    spin_voxel_ = makeDoubleSpin(0.003, 0.05, 0.001, 0.01, 3, grp_hp);
    g->addWidget(spin_voxel_, r++, 1);

    g->addWidget(new QLabel("Truncation (m)", grp_hp), r, 0);
    spin_trunc_ = makeDoubleSpin(0.01, 0.25, 0.005, 0.03, 3, grp_hp);
    g->addWidget(spin_trunc_, r++, 1);

    g->addWidget(new QLabel("Max weight", grp_hp), r, 0);
    spin_max_weight_ = makeDoubleSpin(1.0, 512.0, 1.0, 128.0, 0, grp_hp);
    g->addWidget(spin_max_weight_, r++, 1);

    g->addWidget(new QLabel("Resolution (³)", grp_hp), r, 0);
    spin_resolution_ = new QSpinBox(grp_hp);
    spin_resolution_->setRange(64, 512);
    spin_resolution_->setSingleStep(32);
    spin_resolution_->setValue(256);
    g->addWidget(spin_resolution_, r++, 1);

    g->addWidget(new QLabel("Origin X", grp_hp), r, 0);
    spin_origin_x_ = makeDoubleSpin(-4.0, 4.0, 0.05, -1.28, 2, grp_hp);
    g->addWidget(spin_origin_x_, r++, 1);
    g->addWidget(new QLabel("Origin Y", grp_hp), r, 0);
    spin_origin_y_ = makeDoubleSpin(-4.0, 4.0, 0.05, -1.28, 2, grp_hp);
    g->addWidget(spin_origin_y_, r++, 1);
    g->addWidget(new QLabel("Origin Z", grp_hp), r, 0);
    spin_origin_z_ = makeDoubleSpin(-2.0, 4.0, 0.05, 0.0, 2, grp_hp);
    g->addWidget(spin_origin_z_, r++, 1);

    g->addWidget(new QLabel("ICP dist (m)", grp_hp), r, 0);
    spin_icp_dist_ = makeDoubleSpin(0.02, 0.5, 0.01, 0.1, 3, grp_hp);
    g->addWidget(spin_icp_dist_, r++, 1);

    g->addWidget(new QLabel("ICP angle (°)", grp_hp), r, 0);
    spin_icp_angle_ = makeDoubleSpin(5.0, 90.0, 1.0, 30.0, 0, grp_hp);
    g->addWidget(spin_icp_angle_, r++, 1);

    g->addWidget(new QLabel("ICP iters coarse", grp_hp), r, 0);
    spin_icp_it2_ = new QSpinBox(grp_hp);
    spin_icp_it2_->setRange(1, 40);
    spin_icp_it2_->setValue(10);
    g->addWidget(spin_icp_it2_, r++, 1);

    g->addWidget(new QLabel("ICP iters mid", grp_hp), r, 0);
    spin_icp_it1_ = new QSpinBox(grp_hp);
    spin_icp_it1_->setRange(1, 40);
    spin_icp_it1_->setValue(5);
    g->addWidget(spin_icp_it1_, r++, 1);

    g->addWidget(new QLabel("ICP iters fine", grp_hp), r, 0);
    spin_icp_it0_ = new QSpinBox(grp_hp);
    spin_icp_it0_->setRange(1, 40);
    spin_icp_it0_->setValue(4);
    g->addWidget(spin_icp_it0_, r++, 1);

    btn_apply_hyper_ = new QPushButton("Apply hyperparameters", grp_hp);
    btn_apply_hyper_->setStyleSheet("QPushButton { background: #3a5a7a; color: white; padding: 0.35em 0.5em; border-radius: 4px; font-size: 9pt; }");
    g->addWidget(btn_apply_hyper_, r++, 0, 1, 2);

    inner_layout->addWidget(grp_hp);

    auto* grp_mode = new QGroupBox("Preview Mode", inner);
    auto* v_mode = new QVBoxLayout(grp_mode);
    combo_mode_ = new QComboBox(grp_mode);
    combo_mode_->addItem("Point Cloud");
    combo_mode_->addItem("Mesh");
    v_mode->addWidget(combo_mode_);
    inner_layout->addWidget(grp_mode);

    auto* grp_export = new QGroupBox("Export", inner);
    auto* v_exp = new QVBoxLayout(grp_export);
    btn_ply_ = new QPushButton("Export PLY", grp_export);
    btn_glb_ = new QPushButton("Export GLB (Unity)", grp_export);
    btn_ply_->setStyleSheet("QPushButton { background: #2a4a7a; color: white; padding: 0.35em 0.5em; border-radius: 4px; font-size: 9pt; }");
    btn_glb_->setStyleSheet("QPushButton { background: #4a2a7a; color: white; padding: 0.35em 0.5em; border-radius: 4px; font-size: 9pt; }");
    btn_ply_->setEnabled(false);
    btn_glb_->setEnabled(false);
    v_exp->addWidget(btn_ply_);
    v_exp->addWidget(btn_glb_);
    inner_layout->addWidget(grp_export);

    auto* thread_layout = new QHBoxLayout();
    thread_layout->addWidget(new QLabel("Threads (0=Auto):"));
    spin_threads_ = new QSpinBox(inner);
    spin_threads_->setRange(0, 256);
    spin_threads_->setValue(0);
    thread_layout->addWidget(spin_threads_);
    inner_layout->addLayout(thread_layout);

    inner_layout->addStretch();
    scroll->setWidget(inner);
    root->addWidget(scroll, 1);

    setPanelWidthInChars(*this, 17, 22);
}

void ControlPanel::connectSignals() {
    connect(btn_start_, &QPushButton::clicked, this, &ControlPanel::startClicked);
    connect(btn_stop_,  &QPushButton::clicked, this, &ControlPanel::stopClicked);
    connect(btn_reset_, &QPushButton::clicked, this, &ControlPanel::resetClicked);
    connect(btn_ply_,   &QPushButton::clicked, this, &ControlPanel::forwardExportPLY);
    connect(btn_glb_,   &QPushButton::clicked, this, &ControlPanel::forwardExportGLB);
    connect(combo_mode_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ControlPanel::modeChanged);
    connect(spin_threads_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ControlPanel::threadsChanged);
    connect(btn_apply_hyper_, &QPushButton::clicked, this, &ControlPanel::hyperparamsApplyClicked);
}

app::FusionHyperparams ControlPanel::hyperparamsFromUi() const {
    app::FusionHyperparams h;
    h.min_depth = static_cast<float>(spin_depth_min_->value());
    h.max_depth = static_cast<float>(spin_depth_max_->value());
    h.tsdf.voxel_size     = static_cast<float>(spin_voxel_->value());
    h.tsdf.truncation     = static_cast<float>(spin_trunc_->value());
    h.tsdf.max_weight     = static_cast<float>(spin_max_weight_->value());
    h.tsdf.resolution     = spin_resolution_->value();
    h.tsdf.origin         = Eigen::Vector3f(
        static_cast<float>(spin_origin_x_->value()),
        static_cast<float>(spin_origin_y_->value()),
        static_cast<float>(spin_origin_z_->value()));
    h.icp.dist_threshold  = static_cast<float>(spin_icp_dist_->value());
    h.icp.angle_threshold = static_cast<float>(spin_icp_angle_->value());
    h.icp.max_iterations[2] = spin_icp_it2_->value();
    h.icp.max_iterations[1] = spin_icp_it1_->value();
    h.icp.max_iterations[0] = spin_icp_it0_->value();
    return h;
}

void ControlPanel::setHyperparams(const app::FusionHyperparams& h) {
    spin_depth_min_->setValue(h.min_depth);
    spin_depth_max_->setValue(h.max_depth);
    spin_voxel_->setValue(h.tsdf.voxel_size);
    spin_trunc_->setValue(h.tsdf.truncation);
    spin_max_weight_->setValue(h.tsdf.max_weight);
    spin_resolution_->setValue(h.tsdf.resolution);
    spin_origin_x_->setValue(h.tsdf.origin.x());
    spin_origin_y_->setValue(h.tsdf.origin.y());
    spin_origin_z_->setValue(h.tsdf.origin.z());
    spin_icp_dist_->setValue(h.icp.dist_threshold);
    spin_icp_angle_->setValue(h.icp.angle_threshold);
    spin_icp_it2_->setValue(h.icp.max_iterations[2]);
    spin_icp_it1_->setValue(h.icp.max_iterations[1]);
    spin_icp_it0_->setValue(h.icp.max_iterations[0]);
}

void ControlPanel::onPipelineStarted() {
    btn_start_->setEnabled(false);
    btn_stop_->setEnabled(true);
    lbl_status_->setText("Status: Running");
    lbl_status_->setStyleSheet("color: #4f4; font-size: 9pt;");
    // Allow export during capture; PipelineController::export* waits for mesh if needed.
    setExportEnabled(true);
}

void ControlPanel::onPipelineStopped() {
    btn_start_->setEnabled(true);
    btn_stop_->setEnabled(false);
    lbl_status_->setText("Status: Stopped");
    lbl_status_->setStyleSheet("color: #fa4; font-size: 9pt;");
}

void ControlPanel::setExportEnabled(bool enabled) {
    btn_ply_->setEnabled(enabled);
    btn_glb_->setEnabled(enabled);
}

void ControlPanel::forwardExportPLY() {
    emit exportPLYClicked();
}

void ControlPanel::forwardExportGLB() {
    emit exportGLBClicked();
}

} // namespace gui
} // namespace kfusion
