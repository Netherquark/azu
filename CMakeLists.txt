cmake_minimum_required(VERSION 3.18)
project(KinectFusionQt VERSION 1.0.0 LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Optional CUDA
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_AVAILABLE TRUE)
    add_compile_definitions(CUDA_ENABLED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 86) # RTX 5070 (Ampere/Ada - use 89 for Ada Lovelace)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found - using CPU fallback paths")
endif()

# Qt5
find_package(Qt5 REQUIRED COMPONENTS Widgets OpenGL)
# Do NOT use AUTOMOC globally — we use qt5_wrap_cpp explicitly for GUI headers
# to avoid AUTOMOC missing Q_OBJECT headers that live in include/ not src/
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# libfreenect
find_package(PkgConfig REQUIRED)
pkg_check_modules(FREENECT REQUIRED libfreenect)

# OpenGL
find_package(OpenGL REQUIRED)

# Eigen (bundled or system)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "System Eigen3 not found, using bundled")
    set(EIGEN3_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/third_party/eigen)
endif()

# OpenMP
find_package(OpenMP)

# Threads
find_package(Threads REQUIRED)

# tinygltf (header-only, bundled)
set(TINYGLTF_DIR ${CMAKE_SOURCE_DIR}/third_party/tinygltf)

# ---- Source files ----

set(SENSOR_SOURCES
    src/sensor/KinectSensor.cpp
    src/sensor/FrameData.cpp
)

set(TRACKING_SOURCES
    src/tracking/ICPTracker.cpp
    src/tracking/PyramidLevel.cpp
)

set(TSDF_SOURCES
    src/tsdf/TSDFVolume.cpp
)

set(MESHING_SOURCES
    src/meshing/MarchingCubes.cpp
    src/meshing/MeshData.cpp
)

set(RENDERING_SOURCES
    src/rendering/PreviewRenderer.cpp
    src/rendering/ShaderProgram.cpp
    src/rendering/Camera.cpp
)

set(EXPORT_SOURCES
    src/export/PLYExporter.cpp
    src/export/GLBExporter.cpp
)

set(GUI_SOURCES
    src/gui/MainWindow.cpp
    src/gui/OpenGLWidget.cpp
    src/gui/MetricsPanel.cpp
    src/gui/ControlPanel.cpp
)

set(APP_SOURCES
    src/app/Application.cpp
    src/app/PipelineController.cpp
    src/app/JobSystem.cpp
)

set(UTILS_SOURCES
    src/utils/Logger.cpp
    src/utils/RingBuffer.cpp
    src/utils/Timer.cpp
)

set(MAIN_SOURCE src/main.cpp)

set(ALL_SOURCES
    ${MAIN_SOURCE}
    ${SENSOR_SOURCES}
    ${TRACKING_SOURCES}
    ${TSDF_SOURCES}
    ${MESHING_SOURCES}
    ${RENDERING_SOURCES}
    ${EXPORT_SOURCES}
    ${GUI_SOURCES}
    ${APP_SOURCES}
    ${UTILS_SOURCES}
)

if(CUDA_AVAILABLE)
    set(CUDA_SOURCES
        src/tsdf/TSDFVolume_cuda.cu
        src/tracking/ICPTracker_cuda.cu
    )
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

# ---- Target ----

add_executable(KinectFusionQt ${ALL_SOURCES})

# Force Qt AUTOMOC to process GUI headers containing Q_OBJECT
set_source_files_properties(
    include/gui/MainWindow.h
    include/gui/OpenGLWidget.h
    include/gui/MetricsPanel.h
    include/gui/ControlPanel.h
    PROPERTIES SKIP_AUTOMOC OFF
)

# Explicit MOC generation for all Q_OBJECT headers (reliable across include paths)
set(MOC_HEADERS
    include/gui/MainWindow.h
    include/gui/OpenGLWidget.h
    include/gui/MetricsPanel.h
    include/gui/ControlPanel.h
)
qt5_wrap_cpp(MOC_SOURCES ${MOC_HEADERS}
    TARGET KinectFusionQt
    OPTIONS --no-notes
)
target_sources(KinectFusionQt PRIVATE ${MOC_SOURCES})

# Include dirs
target_include_directories(KinectFusionQt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${FREENECT_INCLUDE_DIRS}
    ${TINYGLTF_DIR}
    ${OPENGL_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(KinectFusionQt PRIVATE
    Qt5::Widgets
    Qt5::OpenGL
    ${OPENGL_LIBRARIES}
    ${FREENECT_LIBRARIES}
    Threads::Threads
)

if(CUDA_AVAILABLE)
    target_link_libraries(KinectFusionQt PRIVATE cuda cudart)
    target_compile_options(KinectFusionQt PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -diag-suppress=20012
            -diag-suppress=20015
        >
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(KinectFusionQt PRIVATE OpenMP::OpenMP_CXX)
endif()

# Eigen is header-only
if(Eigen3_FOUND)
    target_link_libraries(KinectFusionQt PRIVATE Eigen3::Eigen)
endif()

# tinygltf defines
target_compile_definitions(KinectFusionQt PRIVATE
    TINYGLTF_IMPLEMENTATION
    STB_IMAGE_IMPLEMENTATION
    STB_IMAGE_WRITE_IMPLEMENTATION
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA enabled: ${CUDA_AVAILABLE}")
