#!/usr/bin/env bash
# Carnage v1 - build & run setup for Google Colab (T4 GPU)
#
# Prerequisites:
#   1. Upload this project to /content/Carnage-v1 (or adapt REPO_ROOT below)
#   2. Runtime: Python 3, GPU T4 (torch CUDA build is preinstalled on Colab)
# Then run:  bash colab_setup.sh
set -e

export REPO_ROOT="${REPO_ROOT:-/content/Carnage-v1}"
export BUILD_DIR="$REPO_ROOT/build"

echo "=== [1/4] System dependencies ==="
apt-get update -qq
apt-get install -y -qq cmake g++ make python3-dev

echo "=== [2/4] GPU + PyTorch check ==="
nvidia-smi
python3 -c "import torch; print('torch', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"

echo "=== [3/4] Check repo ==="
if [ ! -d "$REPO_ROOT" ]; then
    echo "ERROR: Project not found at $REPO_ROOT"
    echo "Upload the project first (right click the folder in Colab) and re-run."
    exit 1
fi

echo "=== [4/4] Build ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j"$(nproc)"

echo ""
echo "Done! Start training with:"
echo "  cd $BUILD_DIR && ./Carnage $REPO_ROOT/collision_meshes"
echo "You should see 'Observation size: 94' right after startup."