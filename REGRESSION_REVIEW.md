# Regression & Stability Review: Resolved

This document tracks the resolution of previously identified functional regressions and logical inconsistencies introduced during the transition to the unified `Camera` system and the high-performance pipeline refactor.

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

---

## Conclusion
The "Blender-like" navigation system is now technically sound and fully integrated with the UI. The "Split-State" regression has been eliminated, and the physics engine is now robust against framerate fluctuations.

