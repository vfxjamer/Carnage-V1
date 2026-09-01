#!/usr/bin/env bash
set -euo pipefail

repo_root="${CARNAGE_REPO_ROOT:-$(pwd)}"
build_dir="${CARNAGE_BUILD_DIR:-$repo_root/build-kaggle}"

apt-get update -qq
apt-get install -y -qq cmake g++ make python3-dev

python3 -m pip install -q --upgrade "wandb>=0.17" kaggle kagglehub psutil
python3 -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'devices', torch.cuda.device_count())"
nvidia-smi

cmake -S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build "$build_dir" --config Release -j"$(nproc)"
ctest --test-dir "$build_dir" -C Release --output-on-failure

echo "Carnage build and tests completed: $build_dir/Carnage"
