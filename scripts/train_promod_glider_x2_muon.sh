#!/usr/bin/env bash
# =============================================================================
# train_promod_glider_x2.sh
#
# Launch ProMoD-light ×2 training on the glider HPC (2×A100, no Slurm).
#
# Usage:
#   bash scripts/train_promod_glider_x2.sh [OPT_FILE]
#
# Default config: options/train/311_ProMoD_light_SRx2_muon.yml
# Override:       bash scripts/train_promod_glider_x2.sh options/train/other.yml
# =============================================================================
set -euo pipefail

CONDA=/mnt/pvc-shared-pvc-environment-ff3ed7c7/miniconda3
PYTHON=$CONDA/envs/SISR29/bin/python
PROJ=$HOME/research-sisr/ProMoD-SR
OPT="${1:-options/train/311_ProMoD_light_SRx2_muon.yml}"

[ -f "$PYTHON" ] || { echo "ERROR: SISR python not found at $PYTHON"; exit 1; }
[ -d "$PROJ" ]   || { echo "ERROR: $PROJ not found — run setup_glider.sh first"; exit 1; }

cd "$PROJ"

# ── env ───────────────────────────────────────────────────────────────────
# Use SISR conda binaries without activating (avoids shell sourcing issues)
export PATH="$CONDA/envs/SISR29/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA/envs/SISR29/lib:${LD_LIBRARY_PATH:-}"

# ProMoD-SR source (basicsr + archs) and smm_cuda via PYTHONPATH
SMM_SO=$(find "$PROJ/ops_smm" -name "smm_cuda*.so" 2>/dev/null | head -1)
if [ -z "$SMM_SO" ]; then
    echo "WARNING: smm_cuda not built — run setup_glider.sh first"
    SMM_PATH=""
else
    SMM_PATH="$(dirname "$SMM_SO")"
fi

export PYTHONPATH="$PROJ${SMM_PATH:+:$SMM_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

# ── verify ────────────────────────────────────────────────────────────────
echo "========================================"
echo "  ProMoD-SR training  (glider HPC)"
echo "========================================"
echo "  Config : $OPT"
echo "  Project: $PROJ"
$PYTHON -c "import torch; print(f'  PyTorch: {torch.__version__}  |  GPUs: {torch.cuda.device_count()}  |  CUDA: {torch.version.cuda}')"
echo ""

# ── launch ────────────────────────────────────────────────────────────────
LOG_FILE="scripts/logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p scripts/logs

torchrun \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT:-4321} \
    basicsr/train.py \
    -opt "$OPT" \
    --launcher pytorch \
    --auto_resume \
    2>&1 | tee "$LOG_FILE"
