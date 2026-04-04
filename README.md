# KinectFusionQt

**Offline 3D Scanning Application using Kinect v1 Sensor**

A complete, production-grade C++ application for volumetric 3D reconstruction from Kinect v1 depth streams. Captures live depth/RGB data, performs real-time ICP tracking, integrates frames into a TSDF volume, and exports to PLY/GLB format for Unity.

## Features

- **Live Kinect v1 Capture** (libfreenect)
- **Real-time ICP Tracking** (frame-to-model, multi-resolution pyramid)
- **TSDF Volumetric Reconstruction** (512³ voxels, ~2.6GB VRAM optimized)
- **Marching Cubes Mesh Extraction** (incremental, background thread)
- **Multi-threaded Architecture** (capture, tracking, integration, mesh extraction on separate threads)
- **OpenGL Live Preview** (mesh + point cloud modes)
- **Export Formats** (PLY, GLB/Unity-ready via tinygltf)
- **Qt5 GUI** with real-time metrics and controls
- **CUDA Support** (optional, for TSDF integration acceleration)

## Requirements

### Hardware
- AMD Ryzen CPU (multicore recommended)
- NVIDIA RTX 5070+ GPU (8GB VRAM, optional for CUDA acceleration)
- 32GB+ RAM
- Kinect v1 (USB 2.0/3.0)

### OS
- Fedora 43 (tested)
- Other Linux distributions (may require package name adjustments)

### System Libraries

```bash
sudo dnf install -y \
    cmake \
    qt5-qtbase-devel \
    qt5-qtdeclarative-devel \
    qt5-qttools-devel \
    eigen3-devel \
    libfreenect-devel \
    glm-devel \
    opengl-devel \
    libusb1-devel \
    openmp-devel \
    gcc-c++ \
    make
```

### Optional: CUDA Support

```bash
sudo dnf install -y \
    cuda-toolkit-12-* \
    cuda-runtime-*
```

## Build Instructions

### 1. Clone / Navigate to Project

```bash
cd /home/vs/Code/CV/kinect_asset
```

### 2. Create Build Directory

```bash
mkdir -p build
cd build
```

### 3. Configure CMake

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

For Debug build:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

With CUDA (if available):
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
# CUDA is auto-detected if present
```

### 4. Build

```bash
make -j$(nproc)
```

### 5. Run

```bash
./KinectFusionQt
```

## Usage

### Start Capture
1. Click **"Start Capture"** button
2. Kinect sensor initializes and begins streaming depth/RGB
3. Frames are processed in real-time:
   - Depth → 3D vertex map (camera coordinates)
   - ICP tracking against TSDF model
   - Frame integrated into TSDF volume
   - Mesh extracted periodically

### Live Preview
- **Mesh Mode**: Shows reconstructed surface (updated ~0.5-2 sec intervals)
- **Point Cloud Mode**: Shows live depth point cloud per-frame (fast)
- **Mouse Controls**:
  - Drag to orbit camera
  - Scroll wheel to zoom

### Export
- **Export PLY**: Saves ASCII PLY mesh (all platforms, larger file)
- **Export GLB**: Saves binary GLB asset (Unity-ready, smaller, optimized)
- Exports automatically convert coordinates to Unity space (Y-up)

### Reset Scan
- Clears TSDF volume and resets pose tracking
- Useful for starting a new capture session

## Architecture

### Module Overview

```
src/
├── main.cpp                       # Application entry point
├── sensor/
│   ├── KinectSensor.*             # libfreenect wrapper, frame capture
├── tracking/
│   ├── ICPTracker.*               # Multi-resolution ICP tracker
│   ├── PointToPlaneICP.*          # ICP solver (core algorithm)
├── tsdf/
│   ├── TSDFVolume.*               # TSDF volumetric reconstruction
├── meshing/
│   ├── MarchingCubes.*            # Mesh extraction algorithm
├── export/
│   ├── ExportBase.*               # Base exporter interface
│   ├── PLYExporter.*              # ASCII PLY format
│   ├── GLBExporter.*              # Binary GLB/glTF format
├── rendering/
│   ├── GLRenderWidget.*           # OpenGL preview widget
├── gui/
│   ├── MainWindow.*               # Qt5 main UI
└── utils/
    ├── Types.hpp                  # Core data structures
    ├── Math.hpp                   # Math utilities
    ├── Logger.*                   # Logging system
    ├── CameraModel.*              # Camera intrinsics
    ├── ThreadPool.*               # Thread pool
    └── ThreadSafeQueue.hpp        # Concurrent queue
```

### Threading Model

1. **Capture Thread**: Event loop with libfreenect
2. **Main/UI Thread**: Qt event loop, rendering
3. **Compute Threads**: ThreadPool for ICP, TSDF integration, mesh extraction

### Data Flow

```
Sensor (libfreenect)
  ↓
Capture Thread
  ↓ (Depth + RGB frames)
Main Thread
  ├→ GLRenderWidget (OpenGL preview)
  ├→ ICP Tracker (async)
  ├→ TSDF Integration (async)
  └→ Mesh Extraction (async, periodic)
       ↓
     Export (on demand)
```

## Performance Targets

- **Capture**: 30 FPS (Kinect v1)
- **Tracking**: 10–20 FPS per frame
- **Integration**: ~50 FPS
- **Mesh Updates**: 1 every 2 seconds
- **Memory**: < 16GB system RAM, < 8GB VRAM

## Configuration

Kinect v1 hardcoded intrinsics (adjust in `src/utils/CameraModel.hpp`):
```cpp
fx = 525.0, fy = 525.0
cx = 319.5, cy = 239.5
```

TSDF Volume (in `src/tsdf/TSDFVolume.hpp`):
- Resolution: 512³ voxels
- Voxel size: 5mm (default)
- Truncation distance: 10mm (default)

ICP Tracking (in `src/tracking/PointToPlaneICP.hpp`):
- Distance threshold: 50mm
- Normal angle threshold: 45°
- Max iterations: 50

## File Formats

### PLY Export
- ASCII format with per-vertex positions, normals, colors
- Compatible with: MeshLab, Blender, CloudCompare, Unity

### GLB Export
- Binary glTF 2.0 format (tinygltf)
- Single mesh asset
- Automatic Y-flip for Unity (right-handed → left-handed)
- Ready-to-import into Unity

## Coordinate System

- **Kinect**: X-right, Y-down, Z-forward
- **TSDF**: X-right, Y-up, Z-forward  
- **Unity**: Auto-converted on export

## Troubleshooting

### Kinect Not Found
```bash
lsusb | grep Kinect
pkg-config --cflags --libs libfreenect
```

### Build Errors
- Ensure CMake 3.16+
- Qt5, Eigen3, libfreenect-devel installed
- Check `build/` directory is clean

### Poor Mesh Quality
- Increase scan duration (more frames)
- Improve lighting for depth capture
- Reduce TSDF truncation distance for finer details

## References

- libfreenect: https://github.com/OpenKinetic/libfreenect
- Eigen: https://eigen.tuxfamily.org/
- tinygltf: https://github.com/syoyo/tinygltf
- Qt5: https://www.qt.io/
- KinectFusion Paper: Newcombe et al., ISMAR 2011

---

**Status**: Production-ready  
**Platform**: Fedora 43 (Linux x86_64)  
**Build Date**: 2026-04-04