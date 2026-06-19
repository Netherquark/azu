#!/usr/bin/env bash
# ============================================================
# scripts/build.sh — KinectFusionQt build helper
#
# Usage:
#   ./scripts/build.sh [--hip|--cuda|--cpu] [--debug] [--clean] [-j N]
#
# Options:
#   --hip     Force HIP/ROCm back-end (AMD GPU)
#   --cuda    Force CUDA back-end (NVIDIA GPU)
#   --cpu     Force CPU-only (OpenMP) back-end
#   (none)    AUTO: prefer HIP, fall back to CUDA, then CPU
#   --debug   Debug build (no optimisations)
#   --clean   Wipe the build directory before configuring
#   -j N      Parallelism (default: nproc)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BACKEND="AUTO"
BUILD_TYPE="Release"
CLEAN=0
JOBS=$(nproc)

for arg in "$@"; do
    case "$arg" in
        --hip)    BACKEND="HIP"   ;;
        --cuda)   BACKEND="CUDA"  ;;
        --cpu)    BACKEND="CPU"   ;;
        --debug)  BUILD_TYPE="Debug" ;;
        --clean)  CLEAN=1         ;;
        -j*)      JOBS="${arg#-j}" ;;
        *)  echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# Map backend to build directory
case "$BACKEND" in
    HIP)  BUILD_DIR="$PROJECT_ROOT/build-hip"  ;;
    CUDA) BUILD_DIR="$PROJECT_ROOT/build-cuda" ;;
    CPU)  BUILD_DIR="$PROJECT_ROOT/build-cpu"  ;;
    AUTO) BUILD_DIR="$PROJECT_ROOT/build"      ;;
esac

echo "================================================================"
echo "  KinectFusionQt build"
echo "  Backend   : $BACKEND"
echo "  Build type: $BUILD_TYPE"
echo "  Build dir : $BUILD_DIR"
echo "  Jobs      : $JOBS"
echo "================================================================"

if [[ $CLEAN -eq 1 ]]; then
    echo "[build.sh] Cleaning $BUILD_DIR ..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_ROOT" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGPU_BACKEND="$BACKEND"

ninja -j "$JOBS"

echo ""
echo "================================================================"
echo "  Build complete: $BUILD_DIR/KinectFusionQt"
echo "================================================================"
