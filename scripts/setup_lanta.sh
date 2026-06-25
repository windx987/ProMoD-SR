#!/bin/bash
# Run this on the LANTA login node ONCE to set up the project.
# ssh -i ~/.ssh/mac_ed25519 ub086@transfer.lanta.nstda.or.th
# then: bash /project/zz992000-zdevb/zz992004/ub086/research-sisr/PFT-SR/scripts/setup_lanta.sh

set -e

# Auto-detect environment (LANTA vs local WSL)
if [ -f ~/.conda/envs/demo/bin/python ]; then
    PYTHON=~/.conda/envs/demo/bin/python
    PIP=~/.conda/envs/demo/bin/pip
    PROJ=${PROJ:-/project/zz992000-zdevb/zz992004/ub086/research-sisr}
elif [ -f ~/miniconda3/envs/PFT/bin/python ]; then
    PYTHON=~/miniconda3/envs/PFT/bin/python
    PIP=~/miniconda3/envs/PFT/bin/pip
    PROJ=${PROJ:-/mnt/c/Users/teraw/Developer/research-sisr}
else
    echo "ERROR: No known conda env found. Set PYTHON and PROJ manually."
    exit 1
fi

PFT=$PROJ/PFT-SR
RAW=$PROJ/raw_data
echo "Env:  $PYTHON"
echo "PROJ: $PROJ"

echo "=== [1/5] Installing pip dependencies ==="
$PIP install addict fairscale future lmdb opencv-python pyyaml \
             requests scikit-image scipy tqdm yapf tb-nightly \
             --quiet

echo "=== [2/5] Installing basicsr package ==="
cd $PFT
$PYTHON setup.py develop --quiet

extract_zip() {
    local src=$1 dst=$2
    if command -v unzip &>/dev/null; then
        unzip -q "$src" -d "$dst"
    else
        $PYTHON -c "import zipfile; zipfile.ZipFile('$src').extractall('$dst')"
    fi
}

echo "=== [3/5] Extracting DIV2K training data ==="
mkdir -p $PFT/datasets/DIV2K

if [ ! -d "$PFT/datasets/DIV2K/DIV2K_train_HR" ]; then
    echo "  Extracting DIV2K_train_HR.zip ..."
    extract_zip $RAW/DIV2K_train_HR.zip $PFT/datasets/DIV2K
else
    echo "  DIV2K_train_HR already extracted, skipping."
fi

if [ ! -d "$PFT/datasets/DIV2K/DIV2K_train_LR_bicubic" ]; then
    echo "  Extracting DIV2K_train_LR_bicubic_X2.zip ..."
    extract_zip $RAW/DIV2K_train_LR_bicubic_X2.zip $PFT/datasets/DIV2K
    echo "  Extracting DIV2K_train_LR_bicubic_X3.zip ..."
    extract_zip $RAW/DIV2K_train_LR_bicubic_X3.zip $PFT/datasets/DIV2K
    echo "  Extracting DIV2K_train_LR_bicubic_X4.zip ..."
    extract_zip $RAW/DIV2K_train_LR_bicubic_X4.zip $PFT/datasets/DIV2K
else
    echo "  DIV2K_train_LR_bicubic already extracted, skipping."
fi

echo "=== [4/5] Extracting TestDataSR ==="
mkdir -p $PFT/datasets
if [ ! -d "$PFT/datasets/TestDataSR" ]; then
    echo "  Extracting TestDataSR.zip ..."
    extract_zip $RAW/TestDataSR.zip $PFT/datasets
else
    echo "  TestDataSR already extracted, skipping."
fi

echo "=== [5/5] Verifying data paths ==="
check() { [ -d "$1" ] && echo "  OK: $1" || echo "  MISSING: $1"; }
check $PFT/datasets/DIV2K/DIV2K_train_HR
check $PFT/datasets/DIV2K/DIV2K_train_LR_bicubic/X2
check $PFT/datasets/TestDataSR/HR/Set5/x2
check $PFT/datasets/TestDataSR/LR/LRBI/Set5/x2

echo ""
echo "Setup done. Next: sbatch $PFT/scripts/build_smm.sh"
