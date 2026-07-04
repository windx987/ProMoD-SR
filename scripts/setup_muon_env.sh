#!/usr/bin/env bash
# =============================================================================
# setup_muon_env.sh
#
# Build the SISR29 conda env: PyTorch 2.9.1+cu126 with official torch.optim.Muon,
# and rebuild the smm_cuda extension against it. Idempotent — safe to rerun.
#
# Requirements: CUDA 12.x driver (cu126 runs on 12.4+ via minor-version compat),
# ~15 GB free on the env disk. See set_MUON.md for background and caveats.
#
# Usage:
#   bash scripts/setup_muon_env.sh [CONDA_ROOT] [ENV_NAME]
#   (defaults: glider's shared miniconda3, env "SISR29")
# =============================================================================
set -euo pipefail

CONDA="${1:-/mnt/pvc-shared-pvc-environment-ff3ed7c7/miniconda3}"
ENV_NAME="${2:-SISR29}"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=$CONDA/envs/$ENV_NAME/bin/python
export PIP_NO_CACHE_DIR=1   # env disks are often near-full; don't cache wheels

echo "=== [1/5] conda env ($ENV_NAME, python 3.11) ==="
$CONDA/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
$CONDA/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
if [ ! -x "$PY" ]; then
    $CONDA/bin/conda create -y -n "$ENV_NAME" python=3.11
fi

echo "=== [2/5] python deps (torch 2.9.1+cu126 + BasicSR runtime) ==="
$PY -m pip install -r "$PROJ/requirements_sisr29.txt"

echo "=== [3/5] verify torch.optim.Muon + CUDA ==="
$PY - <<'EOF'
import torch
from torch.optim import Muon
print(f'torch {torch.__version__} | cuda available: {torch.cuda.is_available()}')
print('torch.optim.Muon: OK')
EOF

echo "=== [4/5] CUDA build toolchain (nvcc + math-lib dev headers) ==="
# The pod has no system nvcc, and torch's headers include cusparse.h/cublas.h,
# so the dev packages are required — cuda-nvcc alone is NOT enough.
$CONDA/bin/conda install -y -n "$ENV_NAME" -c nvidia/label/cuda-12.6.3 \
    cuda-nvcc cuda-cudart-dev cuda-crt \
    libcusparse-dev libcublas-dev libcusolver-dev cuda-nvrtc-dev

echo "=== [5/5] build smm_cuda against torch 2.9 ==="
cd "$PROJ/ops_smm"
rm -rf build
CUDA_HOME=$CONDA/envs/$ENV_NAME $PY setup.py build_ext --inplace
# NB: import torch first — the extension links against torch's libc10.so
$PY -c "
import sys, torch
sys.path.insert(0, '.')
import smm_cuda
print('smm_cuda OK with torch', torch.__version__)
"

echo "=== DONE: $ENV_NAME ready. Launch with scripts/train_promod_glider_x2_muon.sh ==="
