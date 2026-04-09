# KinectFusionQt Knowledge Base

A technical reference for the real-time 3D scanning application using Kinect v1 and volumetric TSDF reconstruction.

## 1. System Architecture

The application is built around a decoupled, multi-threaded pipeline designed to maximize throughput and minimize UI latency.

### 1.1 Threading Model
The system orchestrates four primary threads managed by `PipelineController`:

| Thread | Responsibility | Synchronization |
|---|---|---|
| **Capture Thread** | Interfacing with `libfreenect`. Polls hardware and populates a raw frame buffer. | Double-buffering + `std::condition_variable`. |
| **Tracking Thread** | Processes raw depth/RGB into `FrameData` and `FramePyramid`. Performs ICP against the model frame. | Polls `raw_queue_` via `std::condition_variable`. |
| **Integration Thread** | Integrates successfully tracked frames into the `TSDFVolume`. Performs raycasting to update the `ModelFrame`. | Polls `integration_queue_` via `std::condition_variable`. |
| **Meshing Thread** | Periodically extracts a mesh from the TSDF volume using Marching Cubes. | Triggered every N integrated frames or on-demand. |

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
    - `init()`: Initializes `libfreenect` context and opens device.
    - `start()`/`stop()`: Controls the 30 FPS capture thread.
    - `captureLoop()`: Continuously polls `freenect_process_events`.
    - `depthCallback`/`rgbCallback`: Static callbacks from libfreenect that populate `back_buffer_`.
- **Details**: Uses a `std::mutex` protected double-buffer (`back_buffer_` vs `front_buffer_`) to safely transfer frames to the pipeline.

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
    - `track()`: Entry point. Orchestrates tracking across pyramid levels (coarsest to finest).
    - `trackLevel()`: Performs fixed-count iterations at a specific resolution level.
    - `buildLinearSystem()`: Finds correspondences between live frame and raycasted model. Constructs $6 \times 6$ coefficient matrix $A$ and $6 \times 1$ vector $b$ for the Gauss-Newton step.
- **Implementation**: Uses `Eigen::LLT` to solve the linear system for incremental rotation (Euler approximation) and translation.

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
    - `integrate()`: Projects live depth into the volume and updates Signed Distance Function (SDF) and weight.
    - `raycast()`: Casts rays from a camera pose through the volume to find zero-crossings (surface hits). Produces `ModelFrame`.
    - `interpolate()`: Trilinear interpolation of TSDF values for smooth normal computation.
- **Acceleration**: Uses `OpenMP` for CPU-based integration/raycasting.

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
    - `extract()`: Iterates through the voxel grid (Z -> Y -> X). Identifies active edges in $2 \times 2 \times 2$ voxel cubes.
- **Implementation**: Uses standard 256-entry lookup tables (`edge_table`, `tri_table`) to generate triangles based on zero-crossing edges.

#### MeshData.h
- **Struct**: `MeshData` (Positions, Normals, Colors, Indices).
- **Class**: `SharedMesh`: A thread-safe container with versioning to pass new meshes from the extraction thread to the rendering thread without stalling.

---

### 2.5 Rendering Module (`include/rendering/`, `src/rendering/`)

#### PreviewRenderer.h
- **Purpose**: Core OpenGL 3.3 renderer.
- **Modes**: `PointCloud` (live-tracking debug) and `Mesh` (global reconstruction).
- **Details**: Uses VAO/VBO for efficient buffer management. Point cloud rendering uses `GL_POINTS` with camera-space attributes.

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

- **TODO**: Multi-Kinect support is stubbed but not implemented in `KinectSensor`.
- **TODO**: CUDA raycasting for tracking acceleration (currently tracking uses CPU raycast).
- **TODO**: Color integration in `TSDFVolume::integrate` is simplified.
- **FIXME**: `MarchingCubes` extraction is slightly redundant if volume hasn't changed significantly.

---

## 5. Usage Context for Agents
When starting a new session with this codebase:
- **Build**: Ensure `libfreenect`, `Eigen3`, and `Qt5` are installed. Use `scripts/fetch_deps.sh` for tinygltf.
- **Debugging**: Inspect `Logger.h` outputs. Tracking issues are usually visible in `MetricsPanel.h`.
- **Modifying Kernels**: Edit `TSDFVolume_cuda.cu` for integration changes.
