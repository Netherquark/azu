# KinectFusionQt

Real-time 3D scanning application using Kinect v1, volumetric TSDF reconstruction,
and Unity-ready GLB export. Runs on Fedora 43 with optional CUDA acceleration.

---

## Features

- Live Kinect v1 capture via **libfreenect** (no OpenNI)
- **High-Performance GPU Pipeline**: Fully GPU-resident architecture optimized for NVIDIA RTX 30/40/50 series GPUs.
- **Image-Centric TSDF**: $O(W \times H)$ pixel-parallel integration for real-time fidelity.
- **GPU-Resident ICP**: Multi-resolution tracking with block-reduced Hessian construction.
- **Multi-Pass Marching Cubes**: Parallel mesh extraction via CUDA/Thrust (< 2ms per scan).
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

#### Optional: CUDA Acceleration
If you have an NVIDIA GPU (RTX 30/40/50 series), install the CUDA toolkit:
```bash
sudo dnf install cuda
```
Ensure `/usr/local/cuda/bin` is in your `PATH`.

### 2. Cloning and Setup

```bash
git clone https://github.com/1vedantshinde/kinect_asset.git
cd kinect_asset

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

## Building the Project

The build system uses CMake 3.18+ and supports optimized Release builds with LTO (Link Time Optimization).

```bash
mkdir build && cd build

# Strategy A: Standard Release Build (Recommended)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Strategy B: Build with CUDA enabled
# (Automatically detected if nvcc is in PATH)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Note:** The CMake configuration will output a diagnostic summary at the end of the `cmake ..` step, showing which features (CUDA, OpenMP, etc.) are enabled.

---

## Usage

1. **Connect Kinect v1** via USB before launching the application.
2. Launch the scanner: `./KinectFusionQt` (from the `build` directory).
3. Click **▶ Start Capture** — the pipeline will begin live tracking and volume integration.
4. **Scan**: Move the Kinect slowly and steadily around your target object.
5. **View**: Toggle between **Point Cloud** and **Mesh** modes to inspect quality in real-time.
6. **Export**: Once satisfied, click **Export PLY** or **Export GLB**.

### Optimization Tips

- **Range**: Maintain a distance of 0.5m to 2.5m for optimal depth precision.
- **Lighting**: Ensure consistent, non-flickering lighting for robust RGB-based ICP tracking.
- **Volume**: The default reconstruction cube is 2.56m. You can adjust the `origin` and `voxel_size` in `include/tsdf/TSDFVolume.h` for smaller objects (e.g., set `voxel_size` to 0.005 for 5mm precision).
- **Reset**: If tracking is lost (indicated in the status panel), click **Reset** to clear the volume and start a new scan.

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
Pipeline (GPU Resident)
   │
   ├──► Tracking ──────────► GPU ICP (Hessian reduction, pose-to-pose)
   │         │                     │ updated pose
   │         └──────────────────────┤
   │                                ▼
   └──► Integration ───────► Pixel-Parallel TSDF (CUDA)
                 │                   │
                 │               Raycast (GPU) → ModelFrame (VRAM)
                 │
                 ▼
         Meshing Thread ──► GPU Marching Cubes (Thrust) ──► SharedMesh
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
| Tracking immediately lost | Ensure scene has enough texture/geometry; reduce motion speed |
| Low FPS | Disable CUDA if GPU init fails; ensure Release build |
| GLB doesn't import to Unity | Ensure Unity 2019.4+ which includes built-in GLTF support, or use GLTFast package |
| CUDA build fails | Check `nvcc --version`; set `CMAKE_CUDA_ARCHITECTURES=86` for RTX 5070 |

---

## License

MIT — see LICENSE file.
