#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TINYGLTF_DIR="${REPO_ROOT}/third_party/tinygltf"
EIGEN_DIR="${REPO_ROOT}/third_party/eigen"

echo "=== KinectFusionQt dependency fetcher ==="
echo "Repo root: ${REPO_ROOT}"

# ---- tinygltf (single header) ----
echo ""
echo "[1/4] Fetching tinygltf..."
mkdir -p "${TINYGLTF_DIR}"
curl -fsSL "https://raw.githubusercontent.com/syoyo/tinygltf/v2.8.21/tiny_gltf.h" \
     -o "${TINYGLTF_DIR}/tiny_gltf.h"
echo "      -> tiny_gltf.h OK"

# ---- nlohmann/json (required by tinygltf) ----
echo "[2/4] Fetching nlohmann/json..."
curl -fsSL "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp" \
     -o "${TINYGLTF_DIR}/json.hpp"
echo "      -> json.hpp OK"

# ---- stb (required by tinygltf) ----
echo "[3/4] Fetching stb headers..."
curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" \
     -o "${TINYGLTF_DIR}/stb_image.h"
curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" \
     -o "${TINYGLTF_DIR}/stb_image_write.h"
echo "      -> stb_image.h, stb_image_write.h OK"

# ---- Eigen (check if system has it, else fetch) ----
echo "[4/4] Checking Eigen3..."
if pkg-config --exists eigen3 2>/dev/null; then
    EIGEN_INC="$(pkg-config --cflags-only-I eigen3 | sed 's/-I//')"
    echo "      -> System Eigen3 found at ${EIGEN_INC}"
    echo "         (Using system Eigen; bundled copy not needed)"
else
    echo "      -> System Eigen3 not found, fetching bundled copy..."
    mkdir -p "${EIGEN_DIR}"
    EIGEN_VERSION="3.4.0"
    EIGEN_ARCHIVE="/tmp/eigen-${EIGEN_VERSION}.tar.gz"
    curl -fsSL "https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz" \
         -o "${EIGEN_ARCHIVE}"
    tar -xzf "${EIGEN_ARCHIVE}" -C "${EIGEN_DIR}" \
        --strip-components=1 \
        "eigen-${EIGEN_VERSION}/Eigen" \
        "eigen-${EIGEN_VERSION}/unsupported"
    rm "${EIGEN_ARCHIVE}"
    echo "      -> Bundled Eigen ${EIGEN_VERSION} extracted to ${EIGEN_DIR}"
fi

echo ""
echo "=== All dependencies fetched successfully ==="
echo "You can now run: mkdir build && cd build && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release && ninja"
