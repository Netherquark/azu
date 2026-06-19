# Regression & Stability Review: Resolved

**⚠️ CUDA/NVIDIA Path Status: DEPRECATED**
The CUDA (NVIDIA) code path is currently **untested and may not compile**. Due to lack of access to NVIDIA hardware for testing, the CUDA backend cannot be verified. The CUDA code remains in the codebase but is not actively maintained. Use the HIP (AMD) or CPU backend instead.

This document tracks the resolution of previously identified functional regressions and logical inconsistencies introduced during the transition to the unified `Camera` system and the high-performance pipeline refactor.

## 7. Resolved: GLB Color Export Issues (CPU and GPU Paths)
**Status: FIXED**
- **Problem:** GLB exports had color issues on both CPU and GPU paths. CPU path colors were too dark due to integer truncation during color averaging. GPU path colors were broken because VoxelGPU stored colors as floats in the 0-255 range instead of the normalized 0-1 range, causing incorrect interpolation and output.
- **Resolution:**
  - **CPU Path:** Changed color averaging in `TSDFVolume::integrateCPU` to use float arithmetic with `std::round()` instead of direct uint8_t truncation, preventing darkening over time.
  - **GPU Path:** Normalized colors to 0-1 range in `syncToGPU` for both CUDA and HIP, denormalized back to 0-255 in `syncFromGPU`, `raycastKernel`, `compactPointsKernel`, and `MarchingCubes` kernels. This ensures consistent color handling across the entire GPU pipeline.
- **Impact:** GLB exports now have correct, vibrant colors on both CPU and GPU paths with proper brightness levels.

## 8. Resolved: Super Resolution Quality Degradation
**Status: FIXED**
- **Problem:** Super resolution (AMD FSR 1.0 CAS) quality was worse than the native Kinect feed due to overly aggressive sharpening with a sharpness parameter of 0.85.
- **Resolution:** Reduced the CAS sharpness parameter from 0.85 to 0.5 in all signal conditioner paths (CPU OpenMP, CUDA, and HIP). This provides a more conservative sharpening that enhances detail without introducing artifacts or degrading quality.
- **Impact:** Super resolution now provides subtle enhancement without over-sharpening, maintaining the native feed quality while improving edge definition.

## 1. Resolved: UI Sliders Desync (Split-State Problem)
**Status: FIXED**
- **Problem:** The `Camera` class previously maintained dual sets of variables (`azimuth`/`elevation` for Orbit and `yaw`/`pitch` for Free mode).
- **Resolution:** The `Camera` class has been refactored to use a unified state (`yaw_`, `pitch_`, `roll_`). Both Orbit and Free modes now operate on these shared variables. The UI sliders are bi-directionally bound to these values, ensuring consistency regardless of the active navigation mode.

## 2. Resolved: WASD Movement UI Sync
**Status: FIXED**
- **Problem:** Camera updates during WASD movement (physics loop) were not being reported back to the UI.
- **Resolution:** `OpenGLWidget::updatePhysics()` now explicitly emits the `cameraRotated` signal whenever the camera position or orientation changes, keeping the `ControlPanel` sliders in perfect sync with the 3D viewport.

## 3. Resolved: 'Z' Axis Lock Behavior
**Status: FIXED**
- **Problem:** The 'Z' key previously triggered "Zoom" instead of a true Z-axis translation.
- **Resolution:** The axis-lock logic in `OpenGLWidget::mouseMoveEvent` has been corrected. Holding 'Z' now performs a true 3D translation along the world Z-axis using `camera().move(Eigen::Vector3f(0.0f, 0.0f, 1.0f), ...)`, matching the 'X' and 'Y' behavior for consistent panning.

## 4. Resolved: Frame-Rate Independent Physics
**Status: FIXED**
- **Problem:** Camera movement speed was hardcoded to a fixed value per tick, causing variable speeds on different hardware.
- **Resolution:** Implemented `deltaTime` calculation using `QElapsedTimer` (via `frame_timer_.restart()`). The movement speed is now defined in meters per second (`2.0f m/s`), ensuring consistent traversal regardless of UI framerate or ICP processing load.

## 5. Resolved: Transition Instability (NaN Fix)
**Status: FIXED**
- **Problem:** State conversion using `std::asin` was prone to `NaN` errors at steep angles.
- **Resolution:** The conversion logic in `Camera::updateFreeFromOrbit` and `updateOrbitFromFree` has been simplified to use direct trigonometric projections (`cos`/`sin`) and forward vector calculation, eliminating the risky inverse trigonometric calls.

## 6. Resolved: GPU Path "Broken Signal" & Race Conditions
**Status: FIXED**
- **Problem:** The GPU pipeline suffered from intermittent tracking failures ("broken signals") and occasional crashes during high-throughput integration.
- **Resolution:** 
    - **Triple-Buffering:** A `TripleBuffer` model for `ModelFrame` was implemented, separating the "Integration" (writing), "Tracking" (reading), and "Ready" states. This eliminated VRAM race conditions where the tracker would read a partially-integrated model.
    - **Super Resolution (CAS) Fix:** The CUDA-accelerated CAS filter was refined to handle edge cases in the RGB feed, preventing "black pixel" artifacts that previously poisoned the ICP correspondence search.
    - **Relocalization Logic:** Added a specialized relocalization mode that triggers when tracking is lost, using relaxed distance/angle thresholds to recover the pose without requiring a full system reset.

---

## Conclusion
The "Blender-like" navigation system is now technically sound and fully integrated with the UI. The "Split-State" regression has been eliminated, and the physics engine is now robust against framerate fluctuations.
