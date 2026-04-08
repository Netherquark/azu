# KinectFusionQt

Real-time 3D scanning application using Kinect v1, volumetric TSDF reconstruction,
and Unity-ready GLB export. Runs on Fedora 43 with optional CUDA acceleration.

---

## Features

- Live Kinect v1 capture via **libfreenect** (no OpenNI)
- **TSDF volume** reconstruction (512³ voxels, ~2.56 m³ scene)
- **ICP tracking** (frame-to-model, multi-resolution pyramid)
- **Marching Cubes** mesh extraction in background thread
- **OpenGL 3.3** real-time preview (point cloud + mesh modes)
- **PLY** (binary) and **GLB** (Unity-ready) export via tinygltf
- Qt5 GUI with live metrics panel

---

## Requirements

### System packages (Fedora 43)

```bash
sudo dnf install -y \
    cmake ninja-build gcc-c++ \
    qt5-qtbase-devel qt5-qtbase-gui \
    libfreenect-devel \
    eigen3-devel \
    mesa-libGL-devel mesa-libGLU-devel \
    libXrandr-devel libXi-devel \
    openmp \
    pkgconf-pkg-config
```

### Optional: CUDA (RTX 5070)

```bash
# Install CUDA toolkit from NVIDIA repo (Fedora instructions):
# https://developer.nvidia.com/cuda-downloads
# Choose: Linux → x86_64 → Fedora → 37 → rpm(network)
# Then:
sudo dnf install cuda
```

Ensure `nvcc` is on your `PATH`:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Build

### 1. Fetch header-only dependencies

```bash
bash scripts/fetch_deps.sh
```

This downloads:
- `tinygltf` v2.8.21 → `third_party/tinygltf/`
- `stb_image` headers → `third_party/tinygltf/`
- Bundled Eigen 3.4.0 → `third_party/eigen/` (only if system Eigen not found)

### 2. Set up Kinect udev rules

```bash
sudo cp udev/99-kinect.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG plugdev $USER
# Log out and back in for group to take effect
```

### 3. Configure and build

```bash
mkdir build && cd build

# Release build (recommended)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# With CUDA explicitly enabled (auto-detected if nvcc found):
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make -j$(nproc)
```

### 4. Run

```bash
./KinectFusionQt
```

---

## Usage

1. **Connect Kinect v1** via USB before launching.
2. Click **▶ Start Capture** — the pipeline begins capturing, tracking, and integrating.
3. Move the Kinect slowly around your target object.
4. Switch between **Point Cloud** and **Mesh** preview modes at any time.
5. Click **Export PLY** or **Export GLB** once a mesh has been extracted.
6. Import the `.glb` directly into Unity (Assets → Import New Asset).

### Tips

- Keep the Kinect 0.5–3 m from the subject for best depth quality.
- Good initial lighting helps RGB matching for ICP.
- The TSDF volume covers a **2.56 m cube** centred at (0, 0, 1.28 m) by default.
  Adjust `TSDFParams::origin` and `voxel_size` in `include/tsdf/TSDFVolume.h`
  for larger/smaller scenes.
- If tracking is lost (shown in metrics panel), stop, reset, and rescan.

---

## Project Structure

```
KinectFusionQt/
├── CMakeLists.txt
├── README.md
├── scripts/
│   └── fetch_deps.sh          # Downloads tinygltf + stb
├── udev/
│   └── 99-kinect.rules        # Kinect USB permissions
├── include/
│   ├── sensor/                # KinectSensor, FrameData, intrinsics
│   ├── tracking/              # ICPTracker, ModelFrame
│   ├── tsdf/                  # TSDFVolume (512³, CPU+CUDA)
│   ├── meshing/               # MarchingCubes, MeshData
│   ├── rendering/             # PreviewRenderer, ShaderProgram, Camera
│   ├── export/                # PLYExporter, GLBExporter
│   ├── gui/                   # MainWindow, OpenGLWidget, panels
│   ├── app/                   # PipelineController, JobSystem
│   └── utils/                 # Logger, Timer, RingBuffer
├── src/                       # Implementations mirror include/
└── third_party/
    ├── tinygltf/              # tiny_gltf.h, stb headers (after fetch)
    └── eigen/                 # Bundled Eigen (if system not found)
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
FrameData builder (backproject → vertex map, compute normals)
   │
   ├──► Tracking Thread ──► ICPTracker (3-level pyramid, point-to-plane)
   │         │                     │ updated pose
   │         └──────────────────────┤
   │                                ▼
   └──► Integration Thread ──► TSDFVolume::integrate (CPU/CUDA)
                │                    │
                │               Raycast → ModelFrame (for next ICP)
                │
                ▼
         Meshing Thread ──► MarchingCubes::extract ──► SharedMesh
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
