#include <QApplication>
#include <QSurfaceFormat>
#include <QStyleFactory>
#include <iostream>
#include "gui/MainWindow.h"

int main(int argc, char* argv[]) {
    // Set OpenGL surface format globally before creating QApplication
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    fmt.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    QSurfaceFormat::setDefaultFormat(fmt);

    QApplication app(argc, argv);
    app.setApplicationName("KinectFusionQt");
    app.setApplicationVersion("1.0.0");
    app.setOrganizationName("KinectFusion");

    // Apply Fusion dark style
    app.setStyle(QStyleFactory::create("Fusion"));
    QPalette dark;
    dark.setColor(QPalette::Window,          QColor(30, 31, 36));
    dark.setColor(QPalette::WindowText,      QColor(220, 220, 220));
    dark.setColor(QPalette::Base,            QColor(22, 23, 28));
    dark.setColor(QPalette::AlternateBase,   QColor(35, 36, 42));
    dark.setColor(QPalette::Text,            QColor(220, 220, 220));
    dark.setColor(QPalette::Button,          QColor(45, 46, 54));
    dark.setColor(QPalette::ButtonText,      QColor(220, 220, 220));
    dark.setColor(QPalette::Highlight,       QColor(58, 127, 207));
    dark.setColor(QPalette::HighlightedText, QColor(255, 255, 255));
    dark.setColor(QPalette::ToolTipBase,     QColor(50, 50, 60));
    dark.setColor(QPalette::ToolTipText,     QColor(200, 200, 200));
    app.setPalette(dark);

    std::cout << "KinectFusionQt v1.0 starting...\n";

    kfusion::gui::MainWindow window;
    window.show();

    return app.exec();
}
