# Adversarial Review & Regression Report: Preview Navigation Update

This document provides a critical analysis of the recent camera navigation changes, identifying functional regressions and logical inconsistencies introduced during the transition from `OrbitCamera` to the unified `Camera` system.

## 1. Regression: UI Sliders Desync in Free Mode
**Status: MAJOR ERROR**
- **Problem:** The newly added XYZ Sliders in the `ControlPanel` are logically tied to `azimuth_`, `elevation_`, and `roll_`. While these variables are updated and used in `Orbit` mode, they are **completely ignored** in `Free` mode, which relies on `yaw_` and `pitch_`.
- **Symptoms:** 
  1. Dragging a slider while in "Free Mode" (WASD flight) results in no visual change in the viewport.
  2. Looking around with the mouse in "Free Mode" does not update the sliders, as mouse-look updates `yaw_`/`pitch_` while the sliders continue to reflect the stale `azimuth_`/`elevation_` values from the last Orbit state.
- **Rationale:** The `Camera` class maintains two separate sets of state variables for rotation. This "Dual-State" architecture creates a split-brain problem where UI components only talk to one half of the camera's logic.

## 2. Regression: WASD Movement Does Not Update UI
**Status: MINOR ERROR**
- **Problem:** `OpenGLWidget::updatePhysics()` (the WASD loop) updates the camera position/rotation but never emits the `cameraRotated` signal.
- **Symptoms:** Even if the logic were synced (Issue #1), flying through space would leave the UI sliders static, leading to a disconnect between the 3D state and the 2D control surface.
- **Rationale:** The signal emission was only added to the `mouseMoveEvent`, neglecting the asynchronous nature of the `QTimer`-driven physics loop.

## 3. Inconsistency: 'Z' Axis Lock Behavior
**Status: DESIGN FLAW**
- **Problem:** In `Orbit` mode, holding 'X' or 'Y' while right-clicking pans the **Target**. However, holding 'Z' switches the behavior to **Zooming** (`distance_`).
- **Symptoms:** This violates the principle of least astonishment. A user holding 'Z' expects to translate the scene along the depth axis (panning the target), not to change the focal distance of the orbit.
- **Rationale:** The implementation shortcutted "depth movement" to "zoom" instead of performing a true 3D translation along the world Z-axis.

## 4. Technical Debt: Lack of Delta-Time in Physics
**Status: PERFORMANCE REGRESSION**
- **Problem:** WASD movement speed is hardcoded to `0.05f` meters per tick.
- **Symptoms:** On a high-refresh monitor or during heavy CPU load (ICP processing), the "flight" speed will vary wildly.
- **Rationale:** The `QTimer` interval is assumed to be constant, but Qt timers are not real-time guaranteed. This leads to frame-rate-dependent gameplay/navigation.

## 5. Potential Bug: Numerical Instability in Transitions
**Status: EDGE CASE**
- **Problem:** `Camera::updateFreeFromOrbit()` uses `std::asin(forward.y())`. 
- **Symptoms:** If `forward.y()` slightly exceeds 1.0 due to floating point drift, `asin` returns `NaN`, potentially "breaking" the camera (black screen) when switching from Orbit to Free mode at steep angles.
- **Rationale:** Although a `clamp` was added in a later iteration, the dependency on `std::atan2` and `std::asin` for state conversion is prone to precision issues compared to a unified Transform or Quaternion representation.

---

## Conclusion
While the "Blender-like" features significantly improve the potential for navigation, the current implementation has introduced a **Split-State Regression**. The UI is currently a "read-only" or "orbit-only" display that loses its meaning as soon as the user enters Free mode. 

**Recommendation:** The `Camera` class should be refactored to use a single set of rotation variables (Pitch/Yaw/Roll) that both modes share, or the conversion functions must be called more aggressively to keep the UI in sync.
