#!/usr/bin/env bash
# =============================================================================
# preprocess_df2k.sh
#
# Builds LMDB datasets for both DIV2K (light model) and DF2K (normal model).
#
# Input sources (read-only):
#   DIV2K folder : /mnt/pvc-shared-pvc-datasets-04a32ccd/DIV2K/
#   DF2K folder  : /mnt/pvc-shared-pvc-datasets-04a32ccd/DF2K/
#   Flickr2K.tar : ~/rawdata/Flickr2K.tar  (only needed if DF2K HR < 3450 imgs)
#
# Output (written to home Lustre — 280 TB free):
#   ~/datasets/DIV2K/DIV2K_train_HR.lmdb                                (~1.5 GB)  ← light model
#   ~/datasets/DIV2K/DIV2K_train_LR_bicubic_X2.lmdb                     (~400 MB)  ← light model
#   /mnt/pvc-shared-pvc-datasets-04a32ccd/DF2K/DF2K_train_HR.lmdb       (~10 GB)   ← normal model
#   /mnt/pvc-shared-pvc-datasets-04a32ccd/DF2K/DF2K_train_LR_bicubic_X2.lmdb (~2.5 GB)
#   /mnt/pvc-shared-pvc-datasets-04a32ccd/DF2K/DF2K_train_LR_bicubic_X3.lmdb (~1.2 GB)
#   /mnt/pvc-shared-pvc-datasets-04a32ccd/DF2K/DF2K_train_LR_bicubic_X4.lmdb (~0.7 GB)
#
# Key design:
#   LR images named 0001x2.png → LMDB key '0001' (strip xN suffix).
#   Keys in LQ and GT LMDBs must match for BasicSR's paired loader.
#
# Usage:
#   bash scripts/preprocess_df2k.sh [REPO_ROOT]
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${BLUE}════════════════════════════════════════${NC}";
            echo -e "${BLUE}  $*${NC}";
            echo -e "${BLUE}════════════════════════════════════════${NC}"; }

REPO="${1:-.}"
REPO="$(realpath "$REPO")"

# ── paths ──────────────────────────────────────────────────────────────────
PVC_DATA=/mnt/pvc-shared-pvc-datasets-04a32ccd

# Primary source: pre-assembled DF2K in datasets PVC
DF2K_HR_SRC="$PVC_DATA/DF2K/DF2K_train_HR"
DF2K_LR_SRC="$PVC_DATA/DF2K/DF2K_train_LR_bicubic"

# Fallback components if DF2K is incomplete
DIV2K_HR_SRC="$PVC_DATA/DIV2K/DIV2K_train_HR"
DIV2K_LR_SRC="$PVC_DATA/DIV2K/DIV2K_train_LR_bicubic"
FLICKR2K_TAR="${FLICKR2K_TAR:-$HOME/rawdata/Flickr2K.tar}"

# Output: datasets PVC (~59 GB free)
OUT="/mnt/pvc-shared-pvc-datasets-04a32ccd/DF2K"
WORK="$HOME/datasets/DF2K_assembled"   # staging area when assembling from parts

# Auto-detect python
CONDA=/mnt/pvc-shared-pvc-environment-ff3ed7c7/miniconda3
if   [ -f "$CONDA/envs/SISR/bin/python" ]; then PYTHON=$CONDA/envs/SISR/bin/python
elif [ -f "$HOME/miniconda3/envs/SISR/bin/python" ]; then PYTHON=$HOME/miniconda3/envs/SISR/bin/python
else PYTHON=python3
fi

info "Python:  $PYTHON"
info "Output:  $OUT"

# ── 1. Ensure DF2K HR is available ────────────────────────────────────────
section "1/5  Verify / assemble DF2K HR"

mkdir -p "$OUT"

DF2K_HR_FINAL="$DF2K_HR_SRC"
DF2K_LR_FINAL="$DF2K_LR_SRC"

# Check assembled staging area first (avoids re-extracting Flickr2K.tar)
if [ -d "$WORK/HR" ] && [ "$(ls "$WORK/HR" | wc -l)" -ge 3000 ]; then
    ASSEMBLED_COUNT=$(ls "$WORK/HR" | wc -l)
    info "Assembled folder already has $ASSEMBLED_COUNT images — skipping extraction"
    _need_assemble=0
    DF2K_HR_FINAL="$WORK/HR"
    DF2K_LR_FINAL="$WORK/LR_bicubic"
elif [ -d "$DF2K_HR_SRC" ]; then
    HR_COUNT=$(ls "$DF2K_HR_SRC" | wc -l)
    info "DF2K HR found: $HR_COUNT images at $DF2K_HR_SRC"

    if [ "$HR_COUNT" -lt 3000 ]; then
        warn "Only $HR_COUNT HR images (expected ~3450) — assembling from DIV2K + Flickr2K"
        _need_assemble=1
    else
        info "DF2K HR is complete ($HR_COUNT images). Using directly."
        _need_assemble=0
    fi
else
    warn "DF2K HR folder not found — assembling from DIV2K + Flickr2K"
    _need_assemble=1
fi

if [ "${_need_assemble:-0}" -eq 1 ]; then
    mkdir -p "$WORK/HR" "$WORK/LR_bicubic/X2" "$WORK/LR_bicubic/X3" "$WORK/LR_bicubic/X4"

    # Copy DIV2K
    [ -d "$DIV2K_HR_SRC" ] || { echo "ERROR: $DIV2K_HR_SRC not found"; exit 1; }
    info "Copying DIV2K HR (800 images)..."
    cp -rn "$DIV2K_HR_SRC/." "$WORK/HR/"
    for scale in X2 X3 X4; do
        [ -d "$DIV2K_LR_SRC/$scale" ] && cp -rn "$DIV2K_LR_SRC/$scale/." "$WORK/LR_bicubic/$scale/" || warn "DIV2K LR $scale missing"
    done

    # Extract and copy Flickr2K
    if [ -f "$FLICKR2K_TAR" ]; then
        info "Extracting Flickr2K from $FLICKR2K_TAR ..."
        FTMP="$HOME/datasets/_flickr2k_tmp"
        mkdir -p "$FTMP"
        tar -xf "$FLICKR2K_TAR" -C "$FTMP" --strip-components=1 2>/dev/null || tar -xf "$FLICKR2K_TAR" -C "$FTMP"

        # Find HR images in extracted tree
        FLICKR_HR=$(find "$FTMP" -maxdepth 3 -name "Flickr2K_HR" -type d | head -1)
        if [ -n "$FLICKR_HR" ]; then
            info "Copying Flickr2K HR from $FLICKR_HR"
            cp -rn "$FLICKR_HR/." "$WORK/HR/"
        else
            warn "Could not locate Flickr2K_HR in extracted archive"
        fi

        # LR (if present)
        for scale in X2 X3 X4; do
            FLICKR_LR=$(find "$FTMP" -maxdepth 4 -name "$scale" -type d | head -1)
            [ -n "$FLICKR_LR" ] && cp -rn "$FLICKR_LR/." "$WORK/LR_bicubic/$scale/" || true
        done
        rm -rf "$FTMP"
    else
        warn "Flickr2K.tar not found at $FLICKR2K_TAR"
        warn "Set FLICKR2K_TAR=/path/to/Flickr2K.tar and re-run, or ensure DF2K HR has 3450+ images"
    fi

    DF2K_HR_FINAL="$WORK/HR"
    DF2K_LR_FINAL="$WORK/LR_bicubic"
    info "Assembled HR count: $(ls $DF2K_HR_FINAL | wc -l)"
fi

# ── 2. Write shared builder ────────────────────────────────────────────────
section "2/5  Prepare builder"

TMPDIR_PY="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_PY"' EXIT

cat > "$TMPDIR_PY/build_lmdb.py" << 'PYEOF'
"""Build a single LMDB from a folder of PNG images.

Usage:
    python build_lmdb.py <img_folder> <lmdb_path> [strip_suffix]

strip_suffix: optional string to strip from key (e.g. 'x2').
              HR: 000001.png  → key '000001'
              LR: 000001x2.png → key '000001'  (strip 'x2')
"""
import sys
import os
import cv2
import lmdb
import numpy as np
from pathlib import Path

folder    = sys.argv[1]
lmdb_path = sys.argv[2]
strip_suf = sys.argv[3] if len(sys.argv) > 3 else ''

img_paths = sorted(p for p in os.listdir(folder)
                   if p.lower().endswith('.png'))
keys = [Path(p).stem.removesuffix(strip_suf) for p in img_paths]

# Estimate map size from first file size
first_bytes = os.path.getsize(os.path.join(folder, img_paths[0]))
map_size = max(int(first_bytes * len(img_paths) * 3), 10 * 1024**3)

print(f"  Images : {len(img_paths)}")
print(f"  MapSz  : {map_size/1e9:.1f} GB")
print(f"  Dest   : {lmdb_path}")

env = lmdb.open(lmdb_path, map_size=map_size)
txn = env.begin(write=True)
meta = []
skipped = []

for i, (img_file, key) in enumerate(zip(img_paths, keys)):
    fpath = os.path.join(folder, img_file)
    # Store raw file bytes — avoids decode+reencode corruption on NFS
    with open(fpath, 'rb') as f:
        raw = f.read()
    # Verify the image is readable
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"  [WARN] Skipping corrupt image: {img_file}", flush=True)
        skipped.append(img_file)
        continue
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    txn.put(key.encode(), raw)
    meta.append(f"{key}.png ({h},{w},{c}) 1")
    if (i + 1) % 200 == 0:
        txn.commit()
        txn = env.begin(write=True)
        print(f"  Committed {i+1}/{len(img_paths)}", flush=True)

if skipped:
    print(f"  [WARN] Skipped {len(skipped)} corrupt files: {skipped}")

txn.commit()
env.close()

with open(os.path.join(lmdb_path, 'meta_info.txt'), 'w') as f:
    f.write('\n'.join(meta) + '\n')

print(f"  Done → {lmdb_path}")
PYEOF

build() {
    local src=$1 dst=$2 strip=${3:-}
    if [ -f "$dst/meta_info.txt" ]; then
        warn "Already exists, skipping: $(basename $dst)"
        return
    fi
    [ -d "$src" ] || { warn "Source not found: $src — skipping"; return; }
    info "Building $(basename $dst) from $src ..."
    $PYTHON "$TMPDIR_PY/build_lmdb.py" "$src" "$dst" "$strip"
}

# ── 3-4. DIV2K LMDBs (for ProMoD-light) ──────────────────────────────────
OUT_DIV2K="$HOME/datasets/DIV2K"
mkdir -p "$OUT_DIV2K"

section "3/9  DIV2K HR LMDB  (light model)"
build "$PVC_DATA/DIV2K/DIV2K_train_HR" \
      "$OUT_DIV2K/DIV2K_train_HR.lmdb"

section "4/9  DIV2K LR x2 LMDB  (light x2)"
build "$PVC_DATA/DIV2K/DIV2K_train_LR_bicubic/X2" \
      "$OUT_DIV2K/DIV2K_train_LR_bicubic_X2.lmdb" "x2"

section "5/9  DIV2K LR x3 LMDB  (light x3 finetune)"
build "$PVC_DATA/DIV2K/DIV2K_train_LR_bicubic/X3" \
      "$OUT_DIV2K/DIV2K_train_LR_bicubic_X3.lmdb" "x3"

section "6/9  DIV2K LR x4 LMDB  (light x4 finetune)"
build "$PVC_DATA/DIV2K/DIV2K_train_LR_bicubic/X4" \
      "$OUT_DIV2K/DIV2K_train_LR_bicubic_X4.lmdb" "x4"

# ── 5-7. DF2K LMDBs (for ProMoD normal) ──────────────────────────────────
section "7/9  DF2K HR LMDB  (normal model)"
build "$DF2K_HR_FINAL" "$OUT/DF2K_train_HR.lmdb"

section "8/9  DF2K LR x2 LMDB  (normal x2)"
build "$DF2K_LR_FINAL/X2" "$OUT/DF2K_train_LR_bicubic_X2.lmdb" "x2"

section "9/9  DF2K LR x3 and x4 LMDB  (normal x3/x4 finetune)"
build "$DF2K_LR_FINAL/X3" "$OUT/DF2K_train_LR_bicubic_X3.lmdb" "x3"
build "$DF2K_LR_FINAL/X4" "$OUT/DF2K_train_LR_bicubic_X4.lmdb" "x4"

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}═══════════ Verification ═══════════${NC}"

echo "  DIV2K (light model):"
for lmdb in "$OUT_DIV2K/DIV2K_train_HR.lmdb" \
            "$OUT_DIV2K/DIV2K_train_LR_bicubic_X2.lmdb" \
            "$OUT_DIV2K/DIV2K_train_LR_bicubic_X3.lmdb" \
            "$OUT_DIV2K/DIV2K_train_LR_bicubic_X4.lmdb"; do
    if [ -f "$lmdb/meta_info.txt" ]; then
        n=$(wc -l < "$lmdb/meta_info.txt")
        [ "$n" -ge 800 ] \
            && echo -e "    ${GREEN}✓${NC} $(basename $lmdb)  ($n entries)" \
            || echo -e "    ${RED}✗${NC} $(basename $lmdb)  ($n entries — expected 800)"
    else
        echo -e "    ${RED}✗${NC} $(basename $lmdb)  (missing)"
    fi
done

echo "  DF2K (normal model):"
DF2K_HR_COUNT=$(ls "$DF2K_HR_FINAL" | wc -l)
# Allow up to 5 skipped corrupt images without failing verification
DF2K_MIN=$(( DF2K_HR_COUNT - 5 ))
for lmdb in "$OUT/DF2K_train_HR.lmdb" "$OUT/DF2K_train_LR_bicubic_X2.lmdb"; do
    if [ -f "$lmdb/meta_info.txt" ]; then
        n=$(wc -l < "$lmdb/meta_info.txt")
        [ "$n" -ge "$DF2K_MIN" ] \
            && echo -e "    ${GREEN}✓${NC} $(basename $lmdb)  ($n entries)" \
            || echo -e "    ${RED}✗${NC} $(basename $lmdb)  ($n entries — expected ~$DF2K_HR_COUNT)"
    else
        echo -e "    ${RED}✗${NC} $(basename $lmdb)  (missing)"
    fi
done

echo ""
info "Done. YAML paths:"
echo "  Light (DIV2K):  dataroot_gt: $OUT_DIV2K/DIV2K_train_HR.lmdb"
echo "                  dataroot_lq: $OUT_DIV2K/DIV2K_train_LR_bicubic_X2.lmdb"
echo "  Normal (DF2K):  dataroot_gt: $OUT/DF2K_train_HR.lmdb"
echo "                  dataroot_lq: $OUT/DF2K_train_LR_bicubic_X2.lmdb"
