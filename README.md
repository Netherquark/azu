# KinectFusionQt

Real-time 3D scanning application using Kinect v1, volumetric TSDF reconstruction,
and Unity-ready GLB export. Runs on Fedora 43 with HIP/ROCm (AMD) acceleration.

**⚠️ CUDA/NVIDIA Path Status: DEPRECATED**
The CUDA (NVIDIA) code path is currently **untested and may not compile**. Due to lack of access to NVIDIA hardware for testing, the CUDA backend cannot be verified. The CUDA code remains in the codebase but is not actively maintained. Use the HIP (AMD) or CPU backend instead.

---

## Features

- Live Kinect v1 capture via **libfreenect** (no OpenNI)
- **High-Performance GPU Pipeline**: Fully GPU-resident architecture optimized for AMD (HIP/ROCm) GPUs.
- **Image-Centric TSDF**: $O(W \times H)$ pixel-parallel integration for real-time fidelity.
- **Real-Time Super Resolution**: HIP-accelerated or OpenMP-parallelized AMD FidelityFX CAS filters for enhanced RGB clarity.
- **GPU-Resident ICP**: Multi-resolution tracking with block-reduced Hessian construction.
- **Multi-Pass Marching Cubes**: Parallel mesh extraction via HIP/Host-Scan (< 2ms per scan).
- **Navigation Gizmo**: Blender-style interactive 3D axis gizmo for orientation control.
- **OpenGL 3.3** real-time preview (point cloud + mesh modes)
- **PLY** (binary) and **GLB** (Unity-ready) export via tinygltf
- Qt5 GUI with live metrics panel

---

## Getting Started (Fedora 43+)

This guide covers the full setup for Fedora-based systems.

### 1. Prerequisites

Install core development tools and library dependencies via `dnf`:

```bash
sudo dnf install -y \
    cmake \
    git \
    ninja-build \
    gcc-c++ \
    qt5-qtbase-devel \
    qt5-qtbase-gui \
    libfreenect-devel \
    eigen3-devel \
    mesa-libGL-devel \
    mesa-libGLU-devel \
    libXrandr-devel \
    libXi-devel \
    libgomp \
    pkgconf-pkg-config
```

#### Optional: HIP Acceleration (AMD GPUs)
**AMD GPUs (ROCm/HIP):**
```bash
sudo dnf install rocm-hip rocm-opencl
```

**⚠️ CUDA (NVIDIA) - DEPRECATED:**
The CUDA backend is untested and may not compile. No NVIDIA hardware is available for testing. The CUDA code remains in the codebase but is not actively maintained. Use HIP (AMD) or CPU backend instead.

### 2. Cloning and Setup

```bash
git clone https://github.com/Netherquark/azu.git
cd azu

# Fetch bundled dependencies (tinygltf, stb)
bash scripts/fetch_deps.sh
```

### 3. Hardware Setup (Kinect v1)

Configure udev rules to allow non-root access to the Kinect hardware:

```bash
# Copy and reload rules
sudo cp udev/99-kinect.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Add your user to the plugdev group
sudo groupadd -f plugdev
sudo usermod -aG plugdev $USER
# NOTE: You must log out and back in for group changes to take effect!
```

---

### 4. Building the Project

The build system supports auto-detecting your GPU, or explicitly forcing a specific backend. It uses **CMake 3.18+** and is optimized for speed using **Ninja**.

**Using the build helper (Recommended):**
```bash
# Auto-detects HIP (AMD), then CPU fallback
./scripts/build.sh

# Force a specific backend
./scripts/build.sh --hip
./scripts/build.sh --cpu
```

**⚠️ CUDA build option removed:**
The `--cuda` build option has been removed from the build script since the CUDA backend is untested and deprecated. Use HIP or CPU backend instead.
The script will output the compiled binary in `build-hip/`, `build-cpu/`, or `build/` respectively.

**Manual CMake:**
```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=AUTO
ninja
```

**Note:** If you experience "undefined reference" errors after a `git pull`, it is highly recommended to perform a clean build (`./scripts/build.sh --clean`) to refresh the automated Qt metadata.

**Note:** The CMake configuration will output a diagnostic summary at the end of the `cmake` step, showing which GPU backend was selected.

---

## Usage

1. **Connect Kinect v1** via USB before launching the application.
2. Launch the scanner from your target build directory (e.g., `build-hip/`): `QT_QPA_PLATFORM=xcb ./build-hip/KinectFusionQt --verbose`
3. Click **▶ Start Capture** — the pipeline will begin live tracking and volume integration.
4. **Scan**: Move the Kinect slowly and steadily around your target object.
5. **View**: Toggle between **Point Cloud** and **Mesh** modes to inspect quality in real-time.
6. **Navigate**: Use **Blender-like** controls for inspection:
   - **LMB**: Orbit around target.
   - **RMB + X/Y/Z**: Axis-locked panning.
   - **Interactive Gizmo**: Click and drag the XYZ gizmo in the bottom-right to rotate.
   - **Tab**: Switch to **Free Flight** (WASD + Q/E to fly).
   - **F**: Focus back on the origin.
7. **Export**: Once satisfied, click **Export PLY** or **Export GLB**.

### Optimization Tips

- **Range**: Maintain a distance of **0.3m to 2.5m** for optimal depth precision (tunable via **Depth min/max** sliders).
- **Lighting**: Ensure consistent, non-flickering lighting for robust RGB-based ICP tracking.
- **Volume**: The default reconstruction cube is 2.56m. You can adjust the `origin` and `voxel_size` in `include/tsdf/TSDFVolume.h` for smaller objects (e.g., set `voxel_size` to 0.005 for 5mm precision).
- **Tracking Lost**: If tracking is lost (indicated in the status panel), click **Reset** to clear the volume and start a new scan.
- **Diagnostics**: Start with `--verbose` to see a detailed breakdown of tracking failures (correspondences, projections, filtering).

---

## Project Structure

```text
KinectFusionQt/
├── include/           # Header files (.h)
├── src/               # Implementation files (.cpp, .cu)
├── scripts/           # Dependency & utility scripts
├── third_party/       # External libraries (populated by fetch_deps.sh)
├── udev/              # Linux hardware rules
└── CMakeLists.txt     # Main build configuration
```

---

## Architecture

```
Kinect HW
   │
   ▼
KinectSensor (libfreenect, capture thread)
   │  RawFrame (depth 11-bit + RGB 640×480)
   ▼
Preprocessor (Unified Backend: CPU/CUDA)
   │  AMD FSR CAS Super Resolution + Denoising
   ▼
Pipeline (GPU Resident)
   │
   ├──► Tracking ──────────► GPU ICP (Hessian reduction, pose-to-pose)
   │         │                     │ updated pose
   │         └──────────────────────┤
   │                                ▼
   └──► Integration ───────► Pixel-Parallel TSDF (CUDA / HIP)
                 │                   │
                 │               Raycast (GPU) → ModelFrame (VRAM)
                 │
                 ▼
         Meshing Thread ──► GPU Marching Cubes ──► SharedMesh
                                                             │
                                                      ┌──────┴──────┐
                                                      ▼             ▼
                                               PLYExporter    GLBExporter
                                                                    │
                                                              Unity-ready .glb
```

---

## Coordinate Systems

| Space        | Axes                     | Notes                        |
|--------------|--------------------------|------------------------------|
| Kinect depth | X right, Y down, Z fwd   | Right-handed                 |
| TSDF world   | Same as Kinect at origin | Pose tracked via ICP         |
| GLB export   | X right, Y up, Z fwd     | Right-handed (GLTF standard) |
| Unity import | X right, Y up, Z fwd     | Left-handed (Z flipped auto) |

The GLBExporter applies `Y → -Y` to convert from Kinect Y-down to GLTF Y-up.
Unity's built-in GLTF importer then handles the right-to-left-handed flip automatically.
Scale is **1 unit = 1 metre** throughout.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No Kinect devices found" | Check udev rules; run `lsusb` to confirm device visible |
| Permission denied on USB | Add user to `plugdev`; re-login |
| Tracking immediately lost | Ensure scene has enough texture/geometry; reduce motion speed; **Check `--verbose` logs for `inliers` and `model_pts`** |
| Low FPS | Ensure Release build; try CPU backend if GPU init fails |
| GLB doesn't import to Unity | Ensure Unity 2019.4+ which includes built-in GLTF support, or use GLTFast package |
| Linker / Undefined Reference errors | Run `./scripts/build.sh --clean` to refresh Qt meta-object data |
| HIP build fails | Check GPU_BACKEND logs in CMake. Run `./scripts/build.sh --cpu` for software fallback |
| GLB colors appear dark or incorrect | This should be fixed with proper color normalization. If issues persist, check that RGB integration is enabled and volume has sufficient weight |
| CUDA build requested | CUDA backend is deprecated and untested. Use HIP (AMD) or CPU backend instead |

---

## License

MIT — see LICENSE file.
