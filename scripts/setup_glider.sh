#!/usr/bin/env bash
# =============================================================================
# setup_glider.sh
#
# One-time setup for the glider HPC (2×A100-SXM4-40GB, no Slurm).
#
# What it does:
#   1. Removes old conda envs (lotus, torch) to free space on env PVC
#   2. Clones ProMoD-SR from GitHub (or updates if already present)
#   3. Builds smm_cuda extension in-place (writes to ~/research-sisr/, not PVC)
#   4. Applies gradient accumulation patch to BasicSR source
#
# Usage:
#   bash scripts/setup_glider.sh
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()     { echo -e "${RED}[ERR]${NC}   $*"; exit 1; }
section() { echo -e "\n${BLUE}════════════════════════════════════════${NC}";
            echo -e "${BLUE}  $*${NC}";
            echo -e "${BLUE}════════════════════════════════════════${NC}"; }

CONDA=/mnt/pvc-shared-pvc-environment-ff3ed7c7/miniconda3
PYTHON=$CONDA/envs/SISR/bin/python
PROJ=$HOME/research-sisr/ProMoD-SR
REPO=https://github.com/windx987/ProMoD-SR.git

[ -f "$PYTHON" ] || err "SISR python not found at $PYTHON"
info "Python:  $PYTHON  ($(${PYTHON} --version))"
info "Project: $PROJ"

# ── 1. Remove old conda envs ───────────────────────────────────────────────
section "1/4  Clean old conda envs"

CONDA_BIN=$CONDA/bin/conda
for env in lotus torch; do
    if [ -d "$CONDA/envs/$env" ]; then
        info "Removing env: $env"
        "$CONDA_BIN" env remove -n "$env" -y 2>/dev/null || {
            warn "conda remove failed — deleting directory directly"
            rm -rf "$CONDA/envs/$env"
        }
        info "Removed: $env"
    else
        info "Already gone: $env"
    fi
done

# ── 2. Clone / update ProMoD-SR ────────────────────────────────────────────
section "2/4  Clone ProMoD-SR"

mkdir -p "$HOME/research-sisr"
if [ -d "$PROJ/.git" ]; then
    info "Repo exists — pulling latest"
    git -C "$PROJ" pull --ff-only
else
    info "Cloning from GitHub"
    git clone "$REPO" "$PROJ"
fi

cd "$PROJ"
info "HEAD: $(git log --oneline -1)"

# ── 3. Build smm_cuda in-place ─────────────────────────────────────────────
section "3/4  Build smm_cuda"

SMM_DIR="$PROJ/ops_smm"
[ -d "$SMM_DIR/src" ] || err "ops_smm/src not found — is the repo complete?"

cd "$SMM_DIR"
# Build in-place so the .so lands in ops_smm/ (home dir, not env PVC)
$PYTHON setup.py build_ext --inplace 2>&1 | tail -5
info "Build done"

# Verify the .so is there
SO=$(find "$SMM_DIR" -name "smm_cuda*.so" 2>/dev/null | head -1)
[ -n "$SO" ] || err "smm_cuda .so not found after build"
info "smm_cuda: $SO"

# ── 4. Apply gradient accumulation patch ──────────────────────────────────
section "4/4  Apply grad accum patch"

cd "$PROJ"
bash scripts/apply_grad_accum_patch.sh

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
info "Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. bash scripts/preprocess_df2k.sh"
echo "    2. bash scripts/train_promod_glider_x2.sh"
echo ""
echo "  Conda env to activate:"
echo "    source $CONDA/etc/profile.d/conda.sh && conda activate SISR"
