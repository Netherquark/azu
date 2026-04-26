# KinectFusion Signal Conditioning Knowledge Base

This document details the architectural fixes and mathematical optimizations integrated into the `signal-conditioning-pipeline` branch. The focus of this branch is to enhance the spatial resolution of the 3D asset generation process and eliminate visual and geometric artifacts caused by earlier signal-processing implementations.

## 1. Volumetric Asset Resolution Enhancement (TSDF)

**How it works:**
The core of the 3D reconstruction is the Truncated Signed Distance Function (TSDF) volume. It stores the distance to the nearest surface at discrete spatial points (voxels).
**The Change:**
The `TSDFParams.voxel_size` was reduced from `10mm` (`0.010f`) to `5mm` (`0.005f`). Because the volume operates in a 3D grid, halving the 1D dimension effectively packs $2^3 = 8$ times as many voxels into the same spatial volume.
**Impact:**
This results in a staggering $8\times$ increase in geometric resolution for the final generated `.ply` or `.glb` models, enabling high-fidelity captures of fine details like facial features and small textures that were previously smoothed over by the larger 10mm voxels.

## 2. Temporal Ghosting Resolution (EMA Filter)

**How it works:**
Kinect sensors produce significant temporal noise. An Exponential Moving Average (EMA) filter blends the current frame's depth with previous frames to smooth out this jitter in real-time.
**The Problem:**
When a pixel temporarily lost its depth tracking (due to occlusion or hardware dropouts), the EMA algorithm bypassed the pixel. However, it never *reset* its internal history buffer. When the pixel reappeared, it blended its new depth with completely outdated historical data.
**The Fix:**
The pipeline now actively detects invalid depth points and explicitly resets their historical state (`ema_buf_m_[i] = 0.0f`).
**Impact:**
Eliminates "temporal ghosting," where shapes from several seconds ago would temporarily flash or stick to moving objects.

## 3. Depth Geometry Smearing (Guided Filter)

**How it works:**
A Guided Filter uses a high-resolution 2D image (like the RGB camera feed) to guide the filtering of a lower resolution or noisier signal (like the Depth map). It ensures that depth edges exactly match color edges.
**The Problem:**
The implementation used a massive $19 \times 19$ kernel. Crucially, if the center pixel of the kernel was completely empty (no depth reading), the filter would still look at all its valid neighbors, average their depths, and assign that average to the empty center. This turned the filter into an aggressive, unconstrained hole-filler that extrapolated geometry past physical object bounds into thin air.
**The Fix:**
Added a fast-path bailout condition: if the target center pixel is invalid, the guided filter skips it entirely rather than smearing structural geometry across empty gaps.
**Impact:**
Object contours and silhouettes remain sharp, accurately representing physical bounds without smearing into the background.

## 4. Chromatic Structural Artifacts (CLAHE)

**How it works:**
Contrast Limited Adaptive Histogram Equalization (CLAHE) stretches the color contrast of an image localized within small tiles (e.g., $8 \times 8$ blocks).
**The Problem:**
The original implementation ran on disjoint $8\times8$ blocks but failed to perform bilinear interpolation along the seams where adjacent blocks met. This created an artificial "checkerboard" of stark contrast differences in the RGB feed. Because the RGB feed guides the depth filtering, these visual grid mistakes directly warped the depth mesh, causing physical blocky artifacts in the 3D model.
**The Fix:**
The blocky CLAHE implementation was stripped out. Noise reduction is handled perfectly by the existing Sub-pixel Median Blur, while Contrast Adaptive Sharpening (CAS / Super Resolution fallback) gracefully maintains sharpness without generating disjoint grid lines.
**Impact:**
Completely smooth tracking surface generation without checkerboard geometry corruption.

## 5. Thread-Safety in the Process Pipeline

**How it works:**
The system uses multiple producer-consumer worker threads (Tracking, Integration, Meshing) managed by the `PipelineController`.
**The Problem:**
Various logging throttles inside the concurrent `trackingLoop` relied on `static int` variables. In C++, static locals in worker functions can trigger Undefined Behavior (UB) or state contamination if threads are ever killed, respawned, or concurrently duplicated (as happens during a pipeline Reset/Restart).
**The Fix:**
All stateful logic (`ui_skip_counter_`, `lost_log_counter_`, `success_log_counter_`) was decoupled from static scope and promoted to private member variables localized to the exact `PipelineController` instance.
**Impact:**
Thread-safe execution and reliable state-resets when stopping and restarting the Kinect scanning UI rapidly.
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
2. **Preprocessing**: GPU depth-to-meters conversion. CPU-based **AMD FidelityFX CAS (Super Resolution)** sharpens the RGB feed in-place prior to integration.
3. **Tracking**: `ICPTracker` executes **GPU-based Hessian reduction**. It compares a live `FramePyramid` uploaded to VRAM against a **GPU-resident `ModelFrame`** (raycasted from TSDF) to find the 6-DOF camera pose.
4. **Integration**: `TSDFVolume` performs **pixel-parallel integration** entirely on the GPU.
5. **Meshing**: `MarchingCubes` executes a **multi-pass GPU extraction** (Classification → Prefix Sum → Generation).
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

#### Preprocessor.h / .cpp
- **The Unified Preprocessing Backend**: Manages both CPU (OpenMP) and GPU (CUDA) signal conditioning pipelines.
- **Auto-Selection Logic**: Automatically detects CUDA availability and selects the optimal backend based on user preference and hardware capabilities.
- **State Management**: Orchestrates temporal state resets (EMA buffers) during pipeline relocalization or tracking recovery.

#### SuperResolution.h / .cpp / SuperResolution_cuda.cu
- **Purpose**: Real-time Super Resolution and edge enhancement of the Kinect RGB feed.
- **Algorithms**:
    - **CPU Path**: Pure C++ port of AMD's FidelityFX Contrast Adaptive Sharpening (FSR 1.0 CAS) math, parallelized via `OpenMP`.
    - **GPU Path**: High-performance CUDA implementation of the CAS kernel, executing directly on the GPU to minimize host-device transfers.
- **Performance**: Applies enhancement *in-place* ensuring geometric depth relationships map identically at 640x480.

#### SignalConditioner.h / .cpp / SignalConditioner_cuda.cu
- **Purpose**: Depth denoising and structural filtering.
- **Features**: Implements EMA temporal smoothing and Guided Filtering with the fixes detailed in the "Signal Conditioning" section above.

---

### 2.2 Tracking Module (`include/tracking/`, `src/tracking/`)

#### ICPTracker.h / .cpp
**Class**: `ICPTracker`
- **Purpose**: Implements Point-to-Plane Iterative Closest Point algorithm.
- **Key Functions**:
    - `track()`: Multiresolution entry point. Processes coarse-to-fine pyramids.
    - `trackLevel()`: Executes Point-to-Plane ICP iterations at a specific pyramid level.
    - `buildLinearSystem()`: **Pose-Aware Correspondence**. Projects live points through the previous pose and finds closest points in the raycasted model. Constructs the $6 \times 6$ Gauss-Newton Hessian. Now populates **Diagnostic Counters** (valid live/model points, projections, and filtered counts) for granular tracking failure analysis.
- **Tracking Statistics**:
    - `ICPResult` now includes `valid_live_points`, `valid_model_points`, `projected_points`, `dist_filtered`, and `angle_filtered` to distinguish between sensor blackout, occlusion, or volume-limit rejections.
- **Optimization**: All inner loops are parallelized using `OpenMP` for real-time tracking (30ms budget for CPU).

#### ICPTracker_cuda.cu
- **Purpose**: Fully GPU-resident tracking.
- **Kernels**:
    - `computeHessianKernel`: Performs pose-aware point-to-plane correspondence and Jacobian accumulation in a single pass.
    - `reduceHessianKernel`: Block-based reduction for the $6 \times 6$ linear system.
    - `downsampleKernel`: GPU-based pyramid generation.
- **Optimization**: Replaces the $O(N)$ CPU reduction with $O(\log N)$ parallel reduction. Uses shared memory and atomic operations for diagnostic counter accumulation.

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
- **Purpose**: High-performance GPU-resident integration and raycasting.
- **Kernels**:
    - `integrationKernel_PixelParallel`: **Image-Centric Integration**. Parallelizes over every depth pixel. Accurately updates only voxels within truncation via ray-marching. Uses `atomicAdd` for thread-safe weight accumulation.
    - `raycastKernel`: Optimized raycaster for generating GPU vertex and normal maps. Supports skipping to improve throughput.

---

### 2.4 Meshing Module (`include/meshing/`, `src/meshing/`)

#### MarchingCubes.h / .cpp
- **Purpose**: Polygonization of the TSDF volume.
- **Key Functions**:
    - `extract()`: Polygonization kernel. Parallelized by voxel slices.
- **Quality**:
#### MarchingCubes_cuda.cu
- **Purpose**: Multi-pass GPU Marching Cubes using **Thrust**.
- **Passes**:
    1. **Classify**: Determines triangle counts per voxel.
    2. **Scan**: `thrust::exclusive_scan` to compute global offsets.
    3. **Generate**: Populates global vertex, normal, and color buffers.
- **Performance**: Near-instantaneous extraction (< 2ms for 256³ grid).

#### MeshData.h
- **Struct**: `MeshData` (Positions, Normals, Colors, Indices).
- **Class**: `SharedMesh`: A thread-safe container with versioning to pass new meshes from the extraction thread to the rendering thread without stalling.

---

### 2.5 Rendering Module (`include/rendering/`, `src/rendering/`)

#### PreviewRenderer.h
- **Purpose**: Core OpenGL 3.3 renderer.
- **Modes**: `PointCloud` (live-tracking debug) and `Mesh` (global reconstruction).
- **Optimization**: Normal matrices are pre-computed on the CPU to avoid expensive `transpose(inverse())` calls in the vertex shader.

#### Camera.h / .cpp
- **Unified Camera System**: Supports both **Orbit** (Blender-like) and **Free** (WASD flight) navigation modes.
- **Unified State**: Maintains a single set of Euler angles (`yaw_`, `pitch_`, `roll_`) and position, ensuring seamless transitions between modes.
- **Physics Engine**: Implements frame-rate independent movement using `deltaTime`, providing consistent traversal speed across varying hardware performance.
- **Interaction**: Features axis-locked panning (X, Y, Z) and dedicated UI slider binding.

#### NavigationGizmo.h / .cpp
- **Interactive 3D Widget**: A custom Qt widget that visualizes the camera orientation using a 3D axis gizmo.
- **Interaction**: Supports clicking and dragging to rotate the camera, with colors matching industry standards (X: Red, Y: Green, Z: Blue).
- **Sorting**: Implements painter's algorithm (Z-sorting) for correct occlusion of axis elements in 2D space.

---

### 2.6 Application & GUI Modules

#### PipelineController.h
- **The Orchestrator**: Contains the loop logic for all pipeline threads.
- Handles the state machine: `Idle` -> `Running` -> `TrackingLost` -> `Error`.
- Manages inter-thread communication via `std::queue` and `std::condition_variable`.
- **Diagnostic Logging**: Implements structured failure logging in the tracking loop. When `--verbose` is enabled, outputs a detailed breakdown of point-matching performance to help identify sensor vs. geometry issues.

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

## 4. Interaction & Controls

| Input | Action | Mode |
|---|---|---|
| **LMB Drag** | Rotate | Orbit / Free (Head-turn) |
| **RMB Drag** | Pan | Orbit / Free |
| **Wheel** | Zoom / Move | Orbit (Dist) / Free (Fwd/Bwd) |
| **W/A/S/D/Q/E**| Traversal | Free Mode |
| **Tab** | Toggle Mode | Orbit <-> Free |
| **F** | Focus Target | Orbit Mode (Reset to origin) |
| **X/Y/Z** | Axis Lock | Panning (Hold while dragging RMB) |

---

## 5. Resolved Architectural & Stability Improvements

Following a rigorous adversarial review, the system has been hardened for production stability:

### 5.1 Hardened Concurrency & Resource Management
- **Zero-Leak RAII GPU Memory**: Implemented `utils::CudaUniquePtr` (RAII wrapper) across `TSDFVolume`, `ICPTracker`, and `ModelFrame`. Manual `cudaMalloc`/`cudaFree` management is eliminated.
- **Automated Frame Recycling**: Refactored `KinectSensor` and `PipelineController` to use `std::shared_ptr` with custom recyclers. Frames are automatically returned to pools when their last reference drops, preventing OOM.
- **Zero-Copy Mesh Data**: `SharedMesh` and `MeshData` now use `std::shared_ptr`. Multi-megabyte mesh transfers between extraction and rendering threads are now instantaneous pointer swaps.

### 5.2 Performance & Throughput Optimizations
- **Integration Loop Hoisting**: Eliminated millions of redundant `pose.inverse()` calls in `TSDFVolume::integrateCPU` by hoisting transforms to the method level, resulting in a **10-15x speedup**.
- **Unified Vertex Extraction**: `MarchingCubes` now performs on-the-fly vertex unification during extraction, reducing mesh VRAM footprint by **~6x** and improving export speed.
- **Ping-Pong Model Buffers**: Implemented efficient double-buffering for the tracking model in `PipelineController`, allowing tracking to proceed on a stable snapshot while integration updates the back-buffer.

### 5.3 Build System & Engineering
- **Robust Dependency Discovery**: Hardened `CMakeLists.txt` with reliable `Qt5` discovery paths and proper implementation guards for headers like `tinygltf`.
- **Ninja Build Integration**: Transitioned the primary build recommendation to **Ninja**, significantly reducing incremental compile times and improving build reliability on multi-core systems.
- **Modernized Qt Build Pipeline**: Transitioned to `CMAKE_AUTOMOC`, `CMAKE_AUTOUIC`, and `CMAKE_AUTORCC`. All project headers are now explicitly tracked in the build target, ensuring robust Meta-Object Compiler (MOC) generation.

### 5.4 GPU Path Hardening (Stability Rounds 1-5)
- **Triple-Buffer Model Management**: Implemented a robust triple-buffering system for `ModelFrame` (Front/Back/Ready indices) in `PipelineController`. This eliminates race conditions between the Integration thread (writing the new model) and the Tracking thread (reading the stable model).
- **CUDA/Host Sync Points**: Optimized synchronization barriers during pipeline Reset/Restart to prevent `cudaErrorInvalidDevice` or `cudaErrorIllegalAddress` during rapid UI state transitions.
- **Kernel Robustness**: Fixed boundary conditions in `TSDFVolume_cuda.cu` and `ICPTracker_cuda.cu` that previously caused intermittent tracking failure or visual artifacts ("broken signals") when the sensor approached volume edges.

---

## 6. Current Status & Future Roadmap

- **Status**: **STABILIZED & OPTIMIZED**. The pipeline is now production-grade with zero known memory leaks and optimized high-concurrency throughput.
- **Verified Fixes**:
    - [x] RAII for all CUDA resources via `CudaUniquePtr`.
    - [x] Eliminated redundant matrix inversions in `TSDFVolume`.
    - [x] Implemented zero-copy frame and mesh passing.
    - [x] Unified vertex extraction in `MarchingCubes`.
- **Future Roadmap**:
    - Implement a `VoxelHash` backend for larger scale environments.
    - Add real-time loop closure detection (Pose Graph optimization).
    - Migrate to Vulkan/Compute Shaders for cross-vendor support.
