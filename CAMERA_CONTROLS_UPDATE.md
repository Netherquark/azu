# Camera Navigation Update

This document details the newly introduced camera controls in the KinectFusionQt application, mimicking the Blender-like navigation while extending the system to support free-flight. 

## Features Added & Rationality

### 1. Unified `Camera` Class supporting both Orbit and Free modes
- **What was changed:** 
  The existing `OrbitCamera` was renamed to `Camera` and expanded to track not only the target and spherical coordinates (azimuth, elevation) but also explicit position and Euler angles (pitch, yaw, roll). An internal mode toggle (`Mode::Orbit` vs `Mode::Free`) dictates how inputs are processed and how the `viewMatrix()` is calculated.
- **Rationality:**
  Supporting two distinct paradigms of navigation (revolving vs. flying) within a single UI view requires shared state or smooth transitions. By embedding both modes inside a single `Camera` class, the viewport seamlessly recalculates coordinates (e.g., inferring target position from free position, or vice versa) whenever the user switches context without breaking the view matrix.

### 2. Blender-like Orbiting & Target Focus
- **What was changed:**
  Pressing the `F` key forces the application into Orbit Mode and centers the camera's target back onto the object/origin (`[0, 0, 1]`). Orbit mode remains the default, where dragging the mouse revolves the viewpoint spherically around this target.
- **Rationality:**
  When checking the quality of a recreation, the focus is the 3D volume (the reconstructed mesh). By revolving around a static point (Target), the user can inspect the generated geometry from all angles intuitively without drifting out into empty space. 

### 3. Axis Locks for X, Y, and Z Panning
- **What was changed:**
  Added constraint handling in `OpenGLWidget` that listens for the `X`, `Y`, or `Z` keys. Holding one of these keys while dragging with the Right Mouse Button locks the panning to that specific local axis (Z acts as an explicit zoom lock). 
- **Rationality:**
  Modeled after Blender's transform shortcuts (G + X/Y/Z), giving the user precise, predictable translations when they are trying to align their view or inspect a specific cross-section of the reconstructed mesh.

### 4. XYZ UI Sliders (Control Panel)
- **What was changed:**
  Added three `QSlider` widgets inside the `ControlPanel` corresponding to Pitch (X), Yaw (Y), and Roll (Z). These UI elements are bi-directionally bound to the `OpenGLWidget`.
- **Rationality:**
  Providing dedicated visual sliders fulfills the requirement of discrete UI interaction for rotation. It makes exact alignments accessible for users preferring UI constraints over free-hand mouse drag, particularly useful for inspecting misalignments in the 3D scan on an explicit axis. Sliders are generally more intuitive and standard in 3D applications compared to dials.

### 5. Fly / Free Mode (WASD)
- **What was changed:**
  A QTimer-driven physics loop was added to `OpenGLWidget` operating at ~60fps. Pressing `Tab` deselects the origin and switches the camera into Free Mode. The `W, A, S, D, Q, E` keys traverse the 3D space relative to the camera's current viewing angle (First Person control), and Left-Click + Drag rotates the head (yaw/pitch).
- **Rationality:**
  Orbiting is counter-intuitive if the user accidentally pans too far away from the model or needs to explore interior spaces (like a reconstructed room). The First Person / Free flight navigation solves this by anchoring traversal to the camera's origin rather than an external target point.

---

## Adversarial Review

An adversarial review highlighting potential flaws, oversights, and technical debt introduced by these changes:

1. **Gimbal Lock in Free Mode:**
   - **Critique:** The Free mode implements pitch and yaw via Euler angles with clamping (`std::clamp(pitch_, -1.5f, 1.5f)`). While the clamping mostly avoids the catastrophic zenith flip, utilizing Quaternions natively for rotation mapping would offer computationally robust transformations and prevent interpolation hitches.

2. **Transition Jitters (Orbit <-> Free):**
   - **Critique:** When hitting `Tab` to switch between Orbit and Free modes, `Camera::updateOrbitFromFree()` casts a ray forward by `distance_` to declare a new target. If the user is facing empty space and switches to Orbit, the target is defined arbitrarily `3.0m` away. This means trying to orbit immediately afterward may feel like orbiting a ghost pivot rather than the actual mesh. 
   - **Mitigation Needed:** Implementing a Raycast against the actual TSDF volume / Meshing data to assign the *Target* point dynamically based on what geometry is centrally visible, rather than arbitrary empty space.

3. **Input Handling / Event Monopolization:**
   - **Critique:** The `OpenGLWidget` now implements `setFocusPolicy(Qt::StrongFocus)` to catch keyboard events. If the user is trying to type inside a `QSpinBox` in the `ControlPanel` and presses `W`, `A`, or `F`, focus might conflict. Qt event bubbling must be strictly monitored to prevent WASD keys from hijacking UI input boxes when Free mode is active.

4. **Hardcoded Frame Rates & Timer Dependency:**
   - **Critique:** Free mode relies on a `QTimer` running at `16ms` intervals. This means camera velocity is tied directly to the timer tick rate. If the UI thread stutters due to heavy point cloud rendering or ICP solving, the traversal speed will unpredictably lag. A proper `deltaTime` calculation using `std::chrono` inside the `updatePhysics()` tick is necessary to guarantee framerate-independent camera speed.

5. **Lack of Roll (Z) Interaction:**
   - **Critique:** Roll is implemented and functional via the UI sliders, but there are no mouse/keyboard hooks to manipulate Roll dynamically within the 3D viewport. It is strictly a UI-driven novelty unless the user manually operates the `QSlider`. 

6. **Axis Panning Constraints:**
   - **Critique:** The `X, Y, Z` locks restrict 2D mouse deltas (`dx, dy`) to the panning function. However, screen-space panning does not perfectly map 1:1 to World/Local X/Y/Z axes depending on the camera angle. A true axis lock should project the 2D mouse vector onto the 3D viewport axis ray, ensuring movement feels consistent regardless of view orientation.
