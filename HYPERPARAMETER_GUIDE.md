# KinectFusion Hyperparameter Guide

This document explains the various hyperparameters available in the KinectFusion reconstruction pipeline, their significance, and recommended configurations for different scanning scenarios.

## 0. Signal Conditioning Parameters
These parameters control the preprocessing of RGB and depth data before tracking and integration.

| Parameter | UI Label | Range | Default | Significance |
| :--- | :--- | :--- | :--- | :--- |
| `cas_sharpness` | CAS Sharpness | 0.0 - 1.0 | 0.5 | Controls the strength of Contrast Adaptive Sharpening (AMD FSR 1.0). Lower values provide subtle enhancement, higher values increase edge definition but may introduce artifacts. |

---

## 1. Global Depth Gating
These parameters filter the raw depth data before it enters the tracking or integration stages.

| Parameter | UI Label | Range | Default | Significance |
| :--- | :--- | :--- | :--- | :--- |
| `min_depth` | Depth min (m) | 0.01 - 3.0 | 0.30 | Filters out sensor noise very close to the lens. |
| `max_depth` | Depth max (m) | 0.2 - 12.0 | 2.50 | Filters out background objects to focus on the target. |

---

## 2. TSDF Volume Parameters
The TSDF (Truncated Signed Distance Function) volume is where the 3D model is stored.

| Parameter | UI Label | Range | Default | Significance |
| :--- | :--- | :--- | :--- | :--- |
| `resolution` | Resolution (³) | 64 - 512 | 256 | Number of voxels per axis. Total memory scales as $O(N^3)$. |
| `voxel_size` | Voxel size (m) | 0.003 - 0.05 | 0.005 | Physical size of one voxel. `resolution * voxel_size` = physical volume width. |
| `truncation` | Truncation (m) | 0.01 - 0.25 | 0.03 | The "thickness" of the surface update region. Usually set to 3-4x the voxel size. |
| `max_weight` | Max weight | 1 - 512 | 128 | Limits temporal averaging. Lower values allow the model to adapt faster to movement or drift. |
| `origin` | Origin X/Y/Z | -4.0 - 4.0 | -1.28... | World coordinates of the bottom-front-left corner of the volume. |

---

## 3. ICP Tracking Parameters
The Iterative Closest Point (ICP) algorithm tracks the camera pose by aligning live depth frames with the raycasted model.

| Parameter | UI Label | Range | Default | Significance |
| :--- | :--- | :--- | :--- | :--- |
| `dist_threshold`| ICP dist (m) | 0.02 - 0.5 | 0.1 | Max distance between corresponding points. Too large = drift; Too small = lost tracking. |
| `angle_threshold`| ICP angle (°) | 5 - 90 | 30 | Max normal angle difference for point pairs. Helps filter out edges and noise. |
| `iterations` | ICP iters | 1 - 40 | 10, 5, 4 | Iteration count for Coarse (L2), Mid (L1), and Fine (L0) pyramid levels. |

---

## Recommended Scanning Presets

### 1. A Helmet (Small, High Detail)
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Voxel Size** | `0.003m` | High resolution is required to capture small features like vents or straps. |
| **Resolution** | `512` | Coupled with 3mm voxels, this provides a ~1.5m³ capture area, enough for the object and surroundings. |
| **Truncation** | `0.010m` | Tight truncation prevents thin surfaces (like the shell) from "bleeding" through. |
| **Depth Max** | `1.5m` | Keeps the sensor close to the object to maximize the depth precision. |
| **ICP Dist** | `0.05m` | Tighter threshold ensures high precision tracking on the detailed surface. |

### 2. A Chair (Medium, Structural)
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Voxel Size** | `0.008m` | Standard balance between detail and volume size. |
| **Resolution** | `256` | Provides a ~2.0m³ volume, perfect for encompassing a standard office chair. |
| **Truncation** | `0.025m` | Handles the standard noise levels of the Kinect v2 at 1.5-2.0m distance. |
| **Depth Max** | `3.0m` | Allows capturing the chair while standing far enough to see the whole object. |

### 3. A Room (Large Environment)
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Voxel Size** | `0.030m` | Prioritizes volume coverage over fine surface detail. |
| **Resolution** | `256` | Provides a ~7.7m³ volume, suitable for a medium-sized office or bedroom. |
| **Truncation** | `0.100m` | Large truncation helps "pull" the surface into the volume across noisy long-range depth data. |
| **ICP Dist** | `0.20m` | Looser threshold allows for faster camera movement during large-scale scanning. |
| **Depth Max** | `8.0m` | Maximize the sensor's range to capture distant walls and ceilings. |

### 4. A Human (Medium, Detail-Oriented, Dynamic)
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Voxel Size** | `0.005m` | Good for capturing facial features and cloth folds. |
| **Resolution** | `256` | Provides ~1.3m³ volume (head/torso) or `512` (~2.5m³) for full body. |
| **Max Weight** | `64` | **Critical:** Lower weight allows the model to "forget" old data faster, adapting to minor body sways or breathing. |
| **Truncation** | `0.015m` | Balanced for medium-range scanning. |
| **Depth Range** | `0.5m - 2.5m` | Standard ergonomic range for scanning a person. |
