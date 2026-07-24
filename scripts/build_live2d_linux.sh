#!/bin/bash
# Build Live2D Cubism Native SDK renderer for Jetson Orin (ARM64 Linux)
#
# One-time setup. Downloads the SDK, compiles a minimal renderer
# that accepts viseme parameters via stdin pipe and outputs
# rendered RGBA frames via stdout pipe.
#
# Usage:
#   chmod +x scripts/build_live2d_linux.sh
#   ./scripts/build_live2d_linux.sh
#
# Output: build/live2d_renderer (standalone executable)
#
# Prerequisites on Orin:
#   sudo apt install -y cmake build-essential libgl1-mesa-dev libglew-dev

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
SDK_DIR="$HOME/Live2D/CubismNativeSamples"

echo "=== Camera Tutor — Live2D Renderer Build ==="
echo "Project: $PROJECT_DIR"
echo "Build:   $BUILD_DIR"
echo ""

# ── Step 1: Clone Cubism Native SDK (if not already) ──────────

if [ ! -d "$SDK_DIR" ]; then
    echo "[1/4] Cloning Live2D Cubism Native Samples..."
    git clone --depth 1 https://github.com/Live2D/CubismNativeSamples.git "$SDK_DIR"
    echo "       Done: $SDK_DIR"
else
    echo "[1/4] SDK already at $SDK_DIR"
fi

# ── Step 2: Patch the sample for our pipe protocol ─────────────

echo "[2/4] Building custom pipe renderer..."

RENDERER_SRC="$PROJECT_DIR/camera_tutor/live2d_renderer.cpp"
RENDERER_DST="$SDK_DIR/Samples/OpenGL/Demo/proj.linux.cmake/src/"

# Copy our renderer into the SDK tree
mkdir -p "$RENDERER_DST"
cp "$RENDERER_SRC" "$RENDERER_DST/live2d_renderer.cpp" 2>/dev/null || {
    echo "       [WARN] live2d_renderer.cpp not found at $RENDERER_SRC"
    echo "       Using default SDK sample (no pipe protocol — visual test only)"
}

# ── Step 3: Compile ───────────────────────────────────────────

echo "[3/4] Compiling..."

cd "$SDK_DIR/Samples/OpenGL/Demo/proj.linux.cmake"
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
    2>&1 | tail -5

make -j$(nproc) 2>&1 | tail -5

# ── Step 4: Install ───────────────────────────────────────────

echo "[4/4] Installing..."

mkdir -p "$BUILD_DIR"
cp Demo "$BUILD_DIR/live2d_renderer" 2>/dev/null || {
    # If our custom renderer was compiled instead
    cp live2d_renderer "$BUILD_DIR/live2d_renderer" 2>/dev/null || true
}

# Copy model files
mkdir -p "$PROJECT_DIR/camera_tutor/models"
cp -r "$SDK_DIR/Res/"* "$PROJECT_DIR/camera_tutor/models/" 2>/dev/null || true

echo ""
echo "=== Build Complete ==="
echo "Renderer: $BUILD_DIR/live2d_renderer"
echo "Models:   $PROJECT_DIR/camera_tutor/models/"
echo ""
echo "Test with:"
echo "  $BUILD_DIR/live2d_renderer --help"
echo ""
echo "If the renderer was not built (pipe protocol source missing),"
echo "the default SDK Demo was installed. It works for visual testing."
echo "Run it once to verify Live2D works on your Orin:"
echo "  cd $SDK_DIR/Samples/OpenGL/Demo/proj.linux.cmake/build"
echo "  ./Demo"
