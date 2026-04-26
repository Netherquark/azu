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
