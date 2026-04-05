#include "gui/MainWindow.hpp"
#include "utils/Logger.hpp"
#include <QApplication>
#include <iostream>

using namespace kf;

int main(int argc, char** argv) {
    // Initialize Qt Application
    QApplication app(argc, argv);

    // Configure logging
    Logger::instance().set_level(LogLevel::INFO);

    KF_LOG_INFO("========================================");
    KF_LOG_INFO("  KinectFusionQt - Offline Scanning");
    KF_LOG_INFO("  Kinect v1 → TSDF → GLB/PLY Export");
    KF_LOG_INFO("  Platform: Fedora 43 (Linux x86_64)");
    KF_LOG_INFO("========================================");

    try {
        // Create and show main window
        MainWindow window;
        window.show();

        KF_LOG_INFO("Application started");
        return app.exec();
    } catch (const std::exception& e) {
        KF_LOG_ERROR("Fatal exception: ", e.what());
        return 1;
    }
    } catch (const std::exception& e) {
        KF_LOG_ERROR("Fatal exception: ", e.what());
        return 1;
    }
}