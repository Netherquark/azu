# KinectFusionQt Knowledge Base

A technical reference for the real-time 3D scanning application using Kinect v1 and volumetric TSDF reconstruction.

## 1. System Architecture

The application is built around a decoupled, multi-threaded pipeline designed to maximize throughput and minimize UI latency.

### 1.1 Threading Model
The system orchestrates four primary threads managed by `PipelineController`:

| Thread | Responsibility | Synchronization |
|---|---|---|
| **Capture Thread** | Interfacing with `libfreenect`. Polls hardware and populates circular pool. | **Zero-Copy circular pool** + `std::atomic` frame index. |
| **Tracking Thread** | Processes raw depth/RGB into `FrameData` and `FramePyramid`. Performs pose-aware ICP. | Polls `ready_queue_` via `RingBuffer`. |
| **Integration Thread** | Integrates tracked frames into the `TSDFVolume`. Performs raycasting. | Shared/Unique locking on voxel grid. |
| **Meshing Thread**| Extracts smoothly-shaded indexed mesh from TSDF. | Parallelized weighted accumulation. |

### 1.2 Data Flow
1. **Sensor**: `KinectSensor` produces `RawFrame` (depth 11-bit, RGB).
2. **Preprocessing**: `buildFrameData` converts raw depth to meters and computes camera-space vertices/normals.
3. **Tracking**: `ICPTracker` compares live `FramePyramid` (3 levels) against a reference `ModelFrame` (raycasted from TSDF) to find the 6-DOF camera pose.
4. **Integration**: `TSDFVolume` updates voxel grid using the computed pose.
5. **Meshing**: `MarchingCubes` extract `MeshData` from the TSDF volume.
6. **Rendering**: `PreviewRenderer` (OpenGL 3.3) visualizes either the live point cloud or the global mesh.

---

## 2. Module Documentation

### 2.1 Sensor Module (`include/sensor/`, `src/sensor/`)

#### KinectSensor.h / .cpp
**Class**: `KinectSensor`
- **Purpose**: Low-level interface to Kinect v1 hardware via `libfreenect`.
- **Key Functions**:
    - `init()`: Initializes `libfreenect` context and circular buffer pool.
    - `start()`/`stop()`: Controls the 30 FPS capture thread.
    - `captureLoop()`: Continuously polls `freenect_process_events`.
- **Concurrency**: Uses a **circular pool of 8 `shared_ptr<RawFrame>`** buffers to decouple hardware interrupts from pipeline processing. This achieves zero-copy frame passage.

#### FrameData.h / .cpp
**Structs**: `FrameData`, `FramePyramid`
- **Purpose**: Storage for processed per-frame geometry.
- **Key Functions**:
    - `buildFrameData()`: Converts 11-bit depth to meters, back-projects to 3D camera space, and initializes RGB.
    - `computeNormals()`: Approximates surface normals using cross-products of neighboring vertices.
    - `buildFramePyramid()`: Generates 3-level downsampled pyramid (1x, 0.5x, 0.25x) for multi-resolution ICP.

---

### 2.2 Tracking Module (`include/tracking/`, `src/tracking/`)

#### ICPTracker.h / .cpp
**Class**: `ICPTracker`
- **Purpose**: Implements Point-to-Plane Iterative Closest Point algorithm.
- **Key Functions**:
    - `track()`: Multiresolution entry point. Processes coarse-to-fine pyramids.
    - `trackLevel()`: Executes Point-to-Plane ICP iterations at a specific pyramid level.
    - `buildLinearSystem()`: **Pose-Aware Correspondence**. Projects live points through the previous pose and finds closest points in the raycasted model. Constructs the $6 \times 6$ Gauss-Newton Hessian.
- **Optimization**: All inner loops are parallelized using `OpenMP` for real-time tracking (30ms budget for CPU).

#### ICPTracker_cuda.cu
- **Purpose**: GPU-accelerated preprocessing kernels.
- **Functions**:
    - `computeNormalsGPU()`: Parallelized normal computation using vertex map neighbors.
    - `downsampleKernel()`: GPU-based downsampling for pyramid generation.

---

### 2.3 TSDF Module (`include/tsdf/`, `src/tsdf/`)

#### TSDFVolume.h / .cpp
**Class**: `TSDFVolume`
- **Purpose**: Global 3D model representation using a Voxel Grid.
- **Parameters**: 256³ resolution (default), 1cm voxel size, 3cm truncation distance.
- **Key Functions**:
    - `integrate()`: **Image-Centric Integration**. Directly ray-marches from valid depth pixels $(O(W \times H))$ into the volume, significantly faster than voxel-centric $(O(N^3))$ methods.
    - `raycast()`: Casts rays through the volume to find surface zero-crossings.
    - `interpolate()`: Trilinear interpolation of TSDF values for smooth gradients.
- **Concurrency**: Guarded by `std::shared_mutex`. Readers (Raycast/Mesh) use `shared_lock`, while Integrator uses `unique_lock`.
- **Acceleration**: Systematic use of `OpenMP` pragmas.

#### TSDFVolume_cuda.cu
- **Purpose**: High-performance GPU integration.
- **Kernel**: `integrationKernel`
    - Parallelizes over every voxel in the grid.
    - Projects voxel center into camera space and updates TSDF if within truncation distance.
    - Includes atomic-style weight averaging.

---

### 2.4 Meshing Module (`include/meshing/`, `src/meshing/`)

#### MarchingCubes.h / .cpp
- **Purpose**: Polygonization of the TSDF volume.
- **Key Functions**:
    - `extract()`: Polygonization kernel. Parallelized by voxel slices.
- **Quality**:
    - **Smooth Shading**: Computes normals using central differences of the TSDF field interpolated at specific vertices.
    - **Indexed Geometry**: Produces indexed vertex buffers (EBO) to reduce VRAM duplication and improve cache locality.

#### MeshData.h
- **Struct**: `MeshData` (Positions, Normals, Colors, Indices).
- **Class**: `SharedMesh`: A thread-safe container with versioning to pass new meshes from the extraction thread to the rendering thread without stalling.

---

### 2.5 Rendering Module (`include/rendering/`, `src/rendering/`)

#### PreviewRenderer.h
- **Purpose**: Core OpenGL 3.3 renderer.
- **Modes**: `PointCloud` (live-tracking debug) and `Mesh` (global reconstruction).
- **Optimization**: Normal matrices are pre-computed on the CPU to avoid expensive `transpose(inverse())` calls in the vertex shader.

#### OrbitCamera.h
- Implements mouse-driven rotation, zoom, and pan for the 3D viewport.

---

### 2.6 Application & GUI Modules

#### PipelineController.h
- **The Orchestrator**: Contains the loop logic for all pipeline threads.
- Handles the state machine: `Idle` -> `Running` -> `TrackingLost` -> `Error`.
- Manages inter-thread communication via `std::queue` and `std::condition_variable`.

#### MainWindow.h
- Qt5-based main window.
- Uses a `QTimer` to refresh metrics (FPS, error, usage) at a fixed rate from the `PipelineController`.

---

## 3. Coordinate Systems

| Frame | Axes Definition |
|---|---|
| **Kinect Camera** | X: Right, Y: Down, Z: Forward (Hand: Right) |
| **TSDF World** | Same as Initial Camera Frame |
| **GLB Export** | X: Right, Y: Up, Z: Forward (Standard GLTF) |

> [!NOTE]
> During GLB export, the `GLBExporter` applies a `Y -> -Y` flip to convert from Kinect space (Y-down) to the industry-standard Y-up orientation.

---

## 4. Pending Tasks & Stubs

Current state based on codebase audit:

- **TODO**: CUDA Parity. CPU logic (image-centric, pose-aware) is now ahead of legacy CUDA kernels.
- **TODO**: Multi-Kinect support is stubbed but not implemented in `KinectSensor`.
- **TODO**: Global Loop Closure (Pose Graph optimization).
- **FIXME**: `MarchingCubes` extraction is slightly redundant if volume hasn't changed.

---

## 5. Usage Context for Agents
When starting a new session with this codebase:
- **Build**: Ensure `libfreenect`, `Eigen3`, and `Qt5` are installed. Use `scripts/fetch_deps.sh` for tinygltf.
- **Debugging**: Inspect `Logger.h` outputs. Tracking issues are usually visible in `MetricsPanel.h`.
- **Modifying Kernels**: Edit `TSDFVolume_cuda.cu` for integration changes.
