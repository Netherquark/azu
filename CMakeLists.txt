cmake_minimum_required(VERSION 3.18)
project(KinectFusionQt VERSION 1.0.0 LANGUAGES CXX)

# ---- General Configuration ----

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for LSP support (clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimized build flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
    # Enable LTO (Interprocedural Optimization) if supported
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "LTO/IPO enabled for optimized performance")
    endif()
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")
endif()

# ---- Optional CUDA Configuration ----

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_AVAILABLE TRUE)
    add_compile_definitions(CUDA_ENABLED)
    set(CMAKE_CUDA_STANDARD 17)
    
    # Target architectures: 86 (Ampere), 89 (Ada), 100 (Blackwell/RTX 50-series)
    set(CMAKE_CUDA_ARCHITECTURES 86 89 100) 
    
    # Modern CUDA detection for include paths
    find_package(CUDAToolkit REQUIRED)
    
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER} (Archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found - falling back to CPU-only implementation")
endif()

# ---- Dependency Resolution ----

# Qt5 - Try system paths first, then common Fedora/RHEL paths
list(APPEND CMAKE_PREFIX_PATH "/usr/lib64/cmake/Qt5" "/usr/lib/cmake/Qt5")
find_package(Qt5 QUIET COMPONENTS Widgets OpenGL)

if(NOT Qt5_FOUND)
    # Final fallback via pkg-config
    pkg_check_modules(QT5_COMPONENTS QUIET Qt5Widgets Qt5OpenGL)
    if(QT5_COMPONENTS_FOUND)
        message(STATUS "Qt5 found via pkg-config")
        set(Qt5_FOUND TRUE)
    else()
        message(FATAL_ERROR "Qt5 not found! Ensure 'qt5-qtbase-devel' is installed. Error: ${Qt5_DIR}")
    endif()
endif()

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
# Note: We use explicit qt5_wrap_cpp below for specific GUI headers to manage complex include paths.

# libfreenect (Kinect v1)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FREENECT REQUIRED libfreenect)

# OpenGL
find_package(OpenGL REQUIRED)

# Eigen (Check system first, then local fallback)
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/eigen")
        message(STATUS "System Eigen3 not found, using bundled copy in third_party/eigen")
        set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen")
    else()
        message(FATAL_ERROR "Eigen3 not found! Please run 'bash scripts/fetch_deps.sh' to download it.")
    endif()
endif()

# OpenMP for CPU parallelism
find_package(OpenMP)

# Threads
find_package(Threads REQUIRED)

# tinygltf (bundle path)
set(TINYGLTF_DIR "${CMAKE_SOURCE_DIR}/third_party/tinygltf")
if(NOT EXISTS "${TINYGLTF_DIR}/tiny_gltf.h")
    message(FATAL_ERROR "tinygltf headers missing! Please run 'bash scripts/fetch_deps.sh'")
endif()

# ---- Source Organization ----

# Grouping by module
file(GLOB_RECURSE SENSOR_SOURCES CONFIGURE_DEPENDS "src/sensor/*.cpp")
file(GLOB_RECURSE TRACKING_SOURCES CONFIGURE_DEPENDS "src/tracking/*.cpp")
file(GLOB_RECURSE TSDF_SOURCES CONFIGURE_DEPENDS "src/tsdf/*.cpp")
file(GLOB_RECURSE MESHING_SOURCES CONFIGURE_DEPENDS "src/meshing/*.cpp")
file(GLOB_RECURSE RENDERING_SOURCES CONFIGURE_DEPENDS "src/rendering/*.cpp")
file(GLOB_RECURSE EXPORT_SOURCES CONFIGURE_DEPENDS "src/export/*.cpp")
file(GLOB_RECURSE GUI_SOURCES CONFIGURE_DEPENDS "src/gui/*.cpp")
file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS "src/app/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES CONFIGURE_DEPENDS "src/utils/*.cpp")

set(ALL_SOURCES
    src/main.cpp
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
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "src/*.cu")
    list(APPEND ALL_SOURCES ${CUDA_SOURCES})
endif()

# ---- Build Target ----

add_executable(KinectFusionQt ${ALL_SOURCES})

# Qt MOC generation
# We explicitly list headers that require MOC to ensure they are found regardless of search depth
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

# Include Directories
target_include_directories(KinectFusionQt PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src" # Needed for some CUDA internal includes
    "${TINYGLTF_DIR}"
    ${EIGEN3_INCLUDE_DIR}
    ${FREENECT_INCLUDE_DIRS}
)

if(CUDA_AVAILABLE)
    target_include_directories(KinectFusionQt PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Link Libraries
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

if(Eigen3_FOUND)
    target_link_libraries(KinectFusionQt PRIVATE Eigen3::Eigen)
endif()

# tinygltf implementation moved to GLBExporter.cpp to avoid multiple definition errors

# ---- Diagnostic Summary ----

message(STATUS "--------------------------------------------")
message(STATUS "KinectFusionQt Configuration Summary:")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Support: ${CUDA_AVAILABLE}")
if(CUDA_AVAILABLE)
    message(STATUS "  CUDA Arch:    ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  OpenMP:       ${OpenMP_CXX_FOUND}")
message(STATUS "  Qt Version:   ${Qt5_VERSION}")
message(STATUS "--------------------------------------------")

