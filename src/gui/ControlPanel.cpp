#include "gui/ControlPanel.h"
#include <QPushButton>
#include <QComboBox>
#include <QSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>

namespace kfusion {
namespace gui {

ControlPanel::ControlPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
    connectSignals();
}

void ControlPanel::setupUI() {
    auto* root = new QVBoxLayout(this);
    root->setSpacing(8);
    root->setContentsMargins(8, 8, 8, 8);

    // Status label
    lbl_status_ = new QLabel("Status: Idle", this);
    lbl_status_->setStyleSheet("color: #aaa; font-size: 11px;");
    root->addWidget(lbl_status_);

    // Separator
    auto* line = new QFrame(this);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    root->addWidget(line);

    // Capture group
    auto* grp_capture = new QGroupBox("Capture", this);
    auto* v_cap = new QVBoxLayout(grp_capture);

    btn_start_ = new QPushButton("▶  Start Capture", this);
    btn_stop_  = new QPushButton("■  Stop Capture",  this);
    btn_reset_ = new QPushButton("↺  Reset Scan",    this);

    btn_start_->setStyleSheet("QPushButton { background: #2a7a2a; color: white; padding: 6px; border-radius: 4px; }");
    btn_stop_ ->setStyleSheet("QPushButton { background: #7a2a2a; color: white; padding: 6px; border-radius: 4px; }");
    btn_reset_->setStyleSheet("QPushButton { background: #444; color: white; padding: 6px; border-radius: 4px; }");

    btn_stop_->setEnabled(false);

    v_cap->addWidget(btn_start_);
    v_cap->addWidget(btn_stop_);
    v_cap->addWidget(btn_reset_);
    root->addWidget(grp_capture);

    // Preview mode group
    auto* grp_mode = new QGroupBox("Preview Mode", this);
    auto* v_mode = new QVBoxLayout(grp_mode);
    combo_mode_ = new QComboBox(this);
    combo_mode_->addItem("Point Cloud");
    combo_mode_->addItem("Mesh");
    v_mode->addWidget(combo_mode_);
    root->addWidget(grp_mode);

    // Export group
    auto* grp_export = new QGroupBox("Export", this);
    auto* v_exp = new QVBoxLayout(grp_export);

    btn_ply_ = new QPushButton("Export PLY", this);
    btn_glb_ = new QPushButton("Export GLB (Unity)", this);

    btn_ply_->setStyleSheet("QPushButton { background: #2a4a7a; color: white; padding: 6px; border-radius: 4px; }");
    btn_glb_->setStyleSheet("QPushButton { background: #4a2a7a; color: white; padding: 6px; border-radius: 4px; }");
    btn_ply_->setEnabled(false);
    btn_glb_->setEnabled(false);

    v_exp->addWidget(btn_ply_);
    v_exp->addWidget(btn_glb_);
    root->addWidget(grp_export);

    // Threads config
    auto* thread_layout = new QHBoxLayout();
    thread_layout->addWidget(new QLabel("Threads (0=Auto):"));
    spin_threads_ = new QSpinBox(this);
    spin_threads_->setRange(0, 256);
    spin_threads_->setValue(0);
    thread_layout->addWidget(spin_threads_);
    root->addLayout(thread_layout);

    root->addStretch();
    setFixedWidth(200);
}

void ControlPanel::connectSignals() {
    connect(btn_start_, &QPushButton::clicked, this, &ControlPanel::startClicked);
    connect(btn_stop_,  &QPushButton::clicked, this, &ControlPanel::stopClicked);
    connect(btn_reset_, &QPushButton::clicked, this, &ControlPanel::resetClicked);
    connect(btn_ply_,   &QPushButton::clicked, this, &ControlPanel::exportPLYClicked);
    connect(btn_glb_,   &QPushButton::clicked, this, &ControlPanel::exportGLBClicked);
    connect(combo_mode_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &ControlPanel::modeChanged);
    connect(spin_threads_, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ControlPanel::threadsChanged);
}

void ControlPanel::onPipelineStarted() {
    btn_start_->setEnabled(false);
    btn_stop_->setEnabled(true);
    lbl_status_->setText("Status: Running");
    lbl_status_->setStyleSheet("color: #4f4; font-size: 11px;");
}

void ControlPanel::onPipelineStopped() {
    btn_start_->setEnabled(true);
    btn_stop_->setEnabled(false);
    lbl_status_->setText("Status: Stopped");
    lbl_status_->setStyleSheet("color: #fa4; font-size: 11px;");
}

void ControlPanel::setExportEnabled(bool enabled) {
    btn_ply_->setEnabled(enabled);
    btn_glb_->setEnabled(enabled);
}

} // namespace gui
} // namespace kfusion
