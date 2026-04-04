cmake_minimum_required(VERSION 3.16)
project(KinectFusionQt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Qt5 setup
find_package(Qt5 COMPONENTS 
    Core 
    Gui 
    Widgets 
    OpenGL 
    REQUIRED
)

find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

# System libfreenect
find_library(FREENECT_LIB freenect REQUIRED)
find_path(FREENECT_INCLUDE_DIR libfreenect.h REQUIRED)

# CUDA support (optional)
find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17 -arch=sm_86 -O3")
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinygltf
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
    ${FREENECT_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
)

# Source files
set(KINECT_SOURCES
    src/main.cpp
    src/sensor/KinectSensor.cpp
    src/tracking/ICPTracker.cpp
    src/tracking/PointToPlaneICP.cpp
    src/tsdf/TSDFVolume.cpp
    src/meshing/MarchingCubes.cpp
    src/export/ExportBase.cpp
    src/export/PLYExporter.cpp
    src/export/GLBExporter.cpp
    src/rendering/GLRenderWidget.cpp
    src/gui/MainWindow.cpp
    src/utils/CameraModel.cpp
    src/utils/ThreadPool.cpp
    src/utils/Logger.cpp
)

set(KINECT_HEADERS
    src/sensor/KinectSensor.hpp
    src/tracking/ICPTracker.hpp
    src/tracking/PointToPlaneICP.hpp
    src/tsdf/TSDFVolume.hpp
    src/meshing/MarchingCubes.hpp
    src/export/ExportBase.hpp
    src/export/PLYExporter.hpp
    src/export/GLBExporter.hpp
    src/rendering/GLRenderWidget.hpp
    src/gui/MainWindow.hpp
    src/utils/CameraModel.hpp
    src/utils/ThreadSafeQueue.hpp
    src/utils/ThreadPool.hpp
    src/utils/Logger.hpp
    src/utils/Types.hpp
    src/utils/Math.hpp
)

if(CUDA_FOUND)
    set(KINECT_CUDA_SOURCES
        src/tsdf/TSDFVolume.cu
    )
    list(APPEND KINECT_SOURCES ${KINECT_CUDA_SOURCES})
endif()

# Create executable
add_executable(KinectFusionQt ${KINECT_SOURCES} ${KINECT_HEADERS})

# Link libraries
target_link_libraries(KinectFusionQt
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::OpenGL
    OpenGL::OpenGL
    Eigen3::Eigen
    Threads::Threads
    ${FREENECT_LIB}
)

if(CUDA_FOUND)
    target_link_libraries(KinectFusionQt
        cuda_runtime
        cublas
    )
endif()

# Compiler flags
if(MSVC)
    target_compile_options(KinectFusionQt PRIVATE /W4 /permissive-)
else()
    target_compile_options(KinectFusionQt PRIVATE 
        -Wall -Wextra -Wpedantic
        -ffast-math -march=native
    )
    if(NOT APPLE)
        target_compile_options(KinectFusionQt PRIVATE -fopenmp)
        target_link_libraries(KinectFusionQt gomp)
    endif()
endif()

# Release optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG_MODE")

message(STATUS "KinectFusionQt configured successfully")
message(STATUS "  Qt5: ${Qt5_DIR}")
message(STATUS "  OpenGL: ${OPENGL_FOUND}")
message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  libfreenect: ${FREENECT_LIB}")
message(STATUS "  CUDA: ${CUDA_FOUND}")