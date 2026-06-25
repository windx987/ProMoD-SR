#!/usr/bin/env bash
# =============================================================================
# preprocess_div2k.sh
#
# Converts DIV2K PNG folders to LMDB for fast training I/O.
#
# Produces:
#   datasets/DIV2K/DIV2K_train_HR.lmdb         (HR,  ~1.5 GB)
#   datasets/DIV2K/DIV2K_train_LR_bicubic_X2.lmdb (LR x2, ~400 MB)
#   datasets/DIV2K/DIV2K_train_LR_bicubic_X3.lmdb (LR x3, ~200 MB)
#   datasets/DIV2K/DIV2K_train_LR_bicubic_X4.lmdb (LR x4, ~120 MB)
#
# Key design:
#   LR images are named 0001x2.png but lmdb keys must match HR keys (0001).
#   The Python builder strips the xN suffix when writing meta_info.txt.
#
# Usage:
#   bash scripts/preprocess_div2k.sh [REPO_ROOT]
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
DIV2K="$REPO/datasets/DIV2K"

# Auto-detect python
if   [ -f ~/.conda/envs/demo/bin/python ];   then PYTHON=~/.conda/envs/demo/bin/python
elif [ -f ~/miniconda3/envs/PFT/bin/python ]; then PYTHON=~/miniconda3/envs/PFT/bin/python
else PYTHON=python3
fi
info "Python: $PYTHON"
info "DIV2K:  $DIV2K"

[ -d "$DIV2K/DIV2K_train_HR" ]            || { echo "ERROR: $DIV2K/DIV2K_train_HR not found"; exit 1; }
[ -d "$DIV2K/DIV2K_train_LR_bicubic/X2" ] || { echo "ERROR: $DIV2K/DIV2K_train_LR_bicubic/X2 not found"; exit 1; }

TMPDIR_PY="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_PY"' EXIT

# ── shared builder script ──────────────────────────────────────────────────────
cat > "$TMPDIR_PY/build_lmdb.py" << 'PYEOF'
"""Build a single LMDB from a folder of PNG images.

Usage:
    python build_lmdb.py <img_folder> <lmdb_path> [strip_suffix]

strip_suffix: optional string to remove from key (e.g. 'x2').
              HR images: 0001.png  → key '0001'
              LR images: 0001x2.png → key '0001'  (strip 'x2')
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import lmdb
from tqdm import tqdm
from basicsr.utils import scandir

folder     = sys.argv[1]
lmdb_path  = sys.argv[2]
strip_suf  = sys.argv[3] if len(sys.argv) > 3 else ''

img_paths = sorted(scandir(folder, suffix='png', recursive=False))
keys      = [p.split('.png')[0].removesuffix(strip_suf) for p in img_paths]

# estimate map size: read one image, multiply
sample = cv2.imread(os.path.join(folder, img_paths[0]))
est_bytes = int(sample.nbytes * len(img_paths) * 1.2)
map_size  = max(est_bytes, 10 * 1024**3)  # at least 10 GB

print(f"  Images : {len(img_paths)}")
print(f"  Sample : {sample.shape}  ({sample.nbytes/1e6:.1f} MB uncompressed)")
print(f"  Map    : {map_size/1e9:.1f} GB")
print(f"  Dest   : {lmdb_path}")

env = lmdb.open(lmdb_path, map_size=map_size)
txn = env.begin(write=True)
meta = []

for i, (img_path, key) in enumerate(tqdm(zip(img_paths, keys), total=len(img_paths))):
    img = cv2.imread(os.path.join(folder, img_path))
    h, w, c = img.shape
    _, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    txn.put(key.encode(), buf.tobytes())
    meta.append(f"{key}.png ({h},{w},{c}) 1")
    if (i + 1) % 200 == 0:
        txn.commit()
        txn = env.begin(write=True)

txn.commit()
env.close()

with open(os.path.join(lmdb_path, 'meta_info.txt'), 'w') as f:
    f.write('\n'.join(meta) + '\n')

print(f"  Done → {lmdb_path}")
PYEOF

# ── build function ─────────────────────────────────────────────────────────────
build() {
    local src=$1 dst=$2 strip=${3:-}
    if [ -f "$dst/meta_info.txt" ]; then
        warn "Already exists, skipping: $(basename $dst)"
        return
    fi
    info "Building $(basename $dst) ..."
    cd "$REPO"
    $PYTHON "$TMPDIR_PY/build_lmdb.py" "$src" "$dst" "$strip"
}

section "1/4  HR"
build "$DIV2K/DIV2K_train_HR" \
      "$DIV2K/DIV2K_train_HR.lmdb"

section "2/4  LR x2"
build "$DIV2K/DIV2K_train_LR_bicubic/X2" \
      "$DIV2K/DIV2K_train_LR_bicubic_X2.lmdb" "x2"

section "3/4  LR x3"
if [ -d "$DIV2K/DIV2K_train_LR_bicubic/X3" ]; then
    build "$DIV2K/DIV2K_train_LR_bicubic/X3" \
          "$DIV2K/DIV2K_train_LR_bicubic_X3.lmdb" "x3"
else
    warn "X3 folder not found — skipping"
fi

section "4/4  LR x4"
if [ -d "$DIV2K/DIV2K_train_LR_bicubic/X4" ]; then
    build "$DIV2K/DIV2K_train_LR_bicubic/X4" \
          "$DIV2K/DIV2K_train_LR_bicubic_X4.lmdb" "x4"
else
    warn "X4 folder not found — skipping"
fi

section "Verification"
for lmdb in \
    "$DIV2K/DIV2K_train_HR.lmdb" \
    "$DIV2K/DIV2K_train_LR_bicubic_X2.lmdb"; do
    n=$(wc -l < "$lmdb/meta_info.txt" 2>/dev/null || echo 0)
    if [ "$n" -ge 800 ]; then
        echo -e "  ${GREEN}✓${NC} $(basename $lmdb)  ($n entries)"
    else
        echo -e "  ${RED}✗${NC} $(basename $lmdb)  ($n entries — expected 800)"
    fi
done

echo ""
info "LMDB ready. Update your YAML:"
echo ""
echo "    dataroot_gt: datasets/DIV2K/DIV2K_train_HR.lmdb"
echo "    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X2.lmdb"
echo "    io_backend:"
echo "      type: lmdb"
