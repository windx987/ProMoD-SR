#!/usr/bin/env bash
# =============================================================================
# apply_grad_accum_patch.sh
#
# Adds gradient accumulation to a BasicSR-based repo (PFT-SR compatible).
#
# Design:
#   - Each Python patcher is written to a temp file and run with python3.
#     This avoids all heredoc quoting issues with {}, quotes, and backslashes.
#   - Every generated file is validated with ast.parse before being written.
#   - train.py patch rewrites the ENTIRE while loop body (not surgical replace),
#     so it is robust to both clean files and previously broken patches.
#   - Idempotent: already-patched files are skipped (except train.py, always re-run).
#   - Full backup before any file is touched.
#
# Usage:
#   bash apply_grad_accum_patch.sh [REPO_ROOT]
#
# Examples:
#   bash apply_grad_accum_patch.sh .
#   bash apply_grad_accum_patch.sh /mnt/c/Users/teraw/Developer/research-sisr/PFT-SR
#
# YAML convention (keys under datasets.train, NOT under train):
#   datasets:
#     train:
#       batch_size_per_gpu: 8
#       accum_iters: 4          # effective batch = 8 x 4 x num_gpu
#       use_grad_clip: true
#       grad_clip_norm: 1.0
#
# Files patched:
#   basicsr/models/base_model.py  — accum_iters default + helper methods
#   basicsr/models/sr_model.py    — new optimize_parameters with accum logic
#   basicsr/train.py              — guarded LR scheduler + extended stats log
# =============================================================================

set -euo pipefail

# ── colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${BLUE}════════════════════════════════════════${NC}";
            echo -e "${BLUE}  $*${NC}";
            echo -e "${BLUE}════════════════════════════════════════${NC}"; }

# ── resolve repo root ─────────────────────────────────────────────────────────
REPO="${1:-.}"
REPO="$(realpath "$REPO")"

BASE_MODEL="$REPO/basicsr/models/base_model.py"
SR_MODEL="$REPO/basicsr/models/sr_model.py"
TRAIN_PY="$REPO/basicsr/train.py"

# ── validate ──────────────────────────────────────────────────────────────────
section "Validating repo"
info "Repo root: $REPO"
[[ -f "$BASE_MODEL" ]] || error "Not found: $BASE_MODEL"
[[ -f "$SR_MODEL"   ]] || error "Not found: $SR_MODEL"
[[ -f "$TRAIN_PY"   ]] || error "Not found: $TRAIN_PY"

if ls "$REPO/basicsr/archs/"*pft* 2>/dev/null | grep -qi .; then
    info "PFT-SR repo confirmed"
else
    warn "Could not confirm PFT-SR — continuing as generic BasicSR"
fi

CUSTOM=$(grep -rl "def optimize_parameters" "$REPO/basicsr/models/" 2>/dev/null \
    | grep -v "__pycache__" | grep -v "sr_model.py" \
    | grep -v "srgan_model.py" | grep -v "esrgan_model.py" || true)
if [[ -n "$CUSTOM" ]]; then
    warn "These files also define optimize_parameters — patch manually if needed:"
    echo "$CUSTOM" | sed 's/^/         /'
fi

# ── backup ────────────────────────────────────────────────────────────────────
section "Backing up originals"
BACKUP="$REPO/.grad_accum_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP"
cp "$BASE_MODEL" "$BACKUP/"
cp "$SR_MODEL"   "$BACKUP/"
cp "$TRAIN_PY"   "$BACKUP/"
info "Backups → $BACKUP"

already_patched() { grep -q "$1" "$2" 2>/dev/null; }

# ── write Python patcher scripts to temp files ────────────────────────────────
TMPDIR_PATCH="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_PATCH"' EXIT

# =============================================================================
# patcher 1: base_model.py
# =============================================================================
cat > "$TMPDIR_PATCH/p1_base_model.py" << 'PYEOF'
import sys, ast

path = sys.argv[1]
with open(path) as f:
    src = f.read()

ANCHOR = "        self.optimizers = []"
if ANCHOR not in src:
    print(f"ERROR: anchor 'self.optimizers = []' not found in {path}", file=sys.stderr)
    sys.exit(1)

REPLACEMENT = (
    "        self.optimizers = []\n"
    "        # gradient accumulation: forward passes per optimizer.step()\n"
    "        # set by subclass from opt['datasets']['train']['accum_iters']\n"
    "        self.accum_iters = 1"
)
src = src.replace(ANCHOR, REPLACEMENT, 1)

HELPERS = (
    "\n"
    "    # -- Gradient accumulation helpers ------------------------------------\n"
    "    def _should_update(self, current_iter):\n"
    '        """True on iterations where optimizer.step() fires."""\n'
    "        return (((current_iter - 1) % self.accum_iters) + 1) == self.accum_iters\n"
    "\n"
    "    def _loss_scale(self):\n"
    '        """Divide loss by this before backward() to keep gradient scale."""\n'
    "        return 1.0 / self.accum_iters\n"
)
src = src.rstrip() + "\n" + HELPERS + "\n"

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"ERROR: syntax error in generated base_model.py: {e}", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print("  base_model.py OK")
PYEOF

# =============================================================================
# patcher 2: sr_model.py
# =============================================================================
cat > "$TMPDIR_PATCH/p2_sr_model.py" << 'PYEOF'
import sys, re, ast

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# 2a: inject accum_iters read before setup_optimizers()
ANCHOR = "        self.setup_optimizers()"
if ANCHOR not in src:
    print(f"ERROR: 'self.setup_optimizers()' not found in {path}", file=sys.stderr)
    sys.exit(1)

REPLACEMENT = (
    "        # gradient accumulation -- reads from opt['datasets']['train']\n"
    "        _ds_train = self.opt.get('datasets', {}).get('train', {})\n"
    "        self.accum_iters = _ds_train.get('accum_iters', 1)\n"
    "        self.setup_optimizers()"
)
src = src.replace(ANCHOR, REPLACEMENT, 1)

# 2b: replace optimize_parameters (captures ema block too)
PATTERN = re.compile(
    r"    def optimize_parameters\(self, current_iter\):.*?(?=\n    def |\Z)",
    re.DOTALL
)

NEW_METHOD = (
    "    def optimize_parameters(self, current_iter):\n"
    '        """Forward + backward with gradient accumulation.\n'
    "\n"
    "        Accumulates gradients over `accum_iters` consecutive iterations,\n"
    "        then calls optimizer.step() once at the end of each window.\n"
    "\n"
    "        Window position is 1-based so the logic is correct for:\n"
    "          - fresh training  (current_iter starts at 1 after increment)\n"
    "          - resuming        (any arbitrary checkpoint iter)\n"
    "          - accum_iters = 1 (identical to no accumulation)\n"
    "\n"
    "        Keys are read from opt['datasets']['train']:\n"
    "          accum_iters, use_grad_clip, grad_clip_norm\n"
    "\n"
    "        DDP no_sync() skips cross-GPU all-reduce on non-update steps,\n"
    "        saving ~20-30 pct comms overhead on multi-GPU / Slurm setups.\n"
    '        """\n'
    "        import contextlib\n"
    "\n"
    "        # 1-based position within the accumulation window\n"
    "        window_pos     = ((current_iter - 1) % self.accum_iters) + 1\n"
    "        is_first       = (window_pos == 1)\n"
    "        is_update_step = (window_pos == self.accum_iters)\n"
    "\n"
    "        # Zero gradients at the START of each window\n"
    "        if is_first:\n"
    "            self.optimizer_g.zero_grad()\n"
    "\n"
    "        # Skip all-reduce on non-update steps (DDP only)\n"
    "        if hasattr(self.net_g, 'no_sync') and not is_update_step:\n"
    "            sync_ctx = self.net_g.no_sync()\n"
    "        else:\n"
    "            sync_ctx = contextlib.nullcontext()\n"
    "\n"
    "        with sync_ctx:\n"
    "            self.output = self.net_g(self.lq)\n"
    "\n"
    "            l_total = 0\n"
    "            loss_dict = OrderedDict()\n"
    "\n"
    "            # pixel loss (L1 / Charbonnier)\n"
    "            if self.cri_pix:\n"
    "                l_pix = self.cri_pix(self.output, self.gt)\n"
    "                l_total += l_pix\n"
    "                loss_dict['l_pix'] = l_pix\n"
    "\n"
    "            # perceptual loss (SRGAN variants)\n"
    "            if hasattr(self, 'cri_perceptual') and self.cri_perceptual:\n"
    "                l_percep, l_style = self.cri_perceptual(self.output, self.gt)\n"
    "                if l_percep is not None:\n"
    "                    l_total += l_percep\n"
    "                    loss_dict['l_percep'] = l_percep\n"
    "                if l_style is not None:\n"
    "                    l_total += l_style\n"
    "                    loss_dict['l_style'] = l_style\n"
    "\n"
    "            # Scale loss so accumulated gradient == single large-batch gradient\n"
    "            (l_total / self.accum_iters).backward()\n"
    "\n"
    "        # optimizer.step() fires at the END of each accumulation window\n"
    "        if is_update_step:\n"
    "            _ds_train = self.opt.get('datasets', {}).get('train', {})\n"
    "            if _ds_train.get('use_grad_clip', False):\n"
    "                torch.nn.utils.clip_grad_norm_(\n"
    "                    self.net_g.parameters(),\n"
    "                    _ds_train.get('grad_clip_norm', 1.0)\n"
    "                )\n"
    "            self.optimizer_g.step()\n"
    "\n"
    "        self.log_dict = self.reduce_loss_dict(loss_dict)\n"
    "\n"
    "        if self.ema_decay > 0 and is_update_step:\n"
    "            self.model_ema(decay=self.ema_decay)\n"
    "\n"
)

m = PATTERN.search(src)
if not m:
    print(f"ERROR: optimize_parameters not found in {path}", file=sys.stderr)
    sys.exit(1)

src = src[:m.start()] + NEW_METHOD + src[m.end():]

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"ERROR: syntax error in generated sr_model.py: {e}", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print("  sr_model.py OK")
PYEOF

# =============================================================================
# patcher 3: train.py
# =============================================================================
cat > "$TMPDIR_PATCH/p3_train.py" << 'PYEOF'
import sys, ast, re

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# ── 3a: stats log in create_train_val_dataloader ─────────────────────────────
# Replace the logger.info('Training statistics:' ...) block regardless of
# whether it was previously patched (use regex so it's always idempotent).
STATS_PATTERN = re.compile(
    r"( {12}logger\.info\('Training statistics:'.*?)(?=\n {8}elif)",
    re.DOTALL
)
NEW_STATS = (
    "            _accum_log = dataset_opt.get('accum_iters', 1)\n"
    "            logger.info('Training statistics:'\n"
    "                        f'\\n\\tNumber of train images : {len(train_set)}'\n"
    "                        f'\\n\\tDataset enlarge ratio  : {dataset_enlarge_ratio}'\n"
    "                        f'\\n\\tBatch size per gpu     : {dataset_opt[\"batch_size_per_gpu\"]}'\n"
    "                        f'\\n\\tWorld size (gpu number): {opt[\"world_size\"]}'\n"
    "                        f'\\n\\tIter per epoch         : {num_iter_per_epoch}'\n"
    "                        f'\\n\\tGradient accum iters   : {_accum_log}'\n"
    "                        f'\\n\\tEffective batch size   : {dataset_opt[\"batch_size_per_gpu\"] * _accum_log * opt[\"world_size\"]}'\n"
    "                        f'\\n\\tOptimizer steps (YAML) : {total_iters}'\n"
    "                        f'\\n\\tRaw loop iters         : {total_iters * _accum_log}')"
)
m_stats = STATS_PATTERN.search(src)
if not m_stats:
    print("ERROR: Training statistics logger.info block not found", file=sys.stderr)
    sys.exit(1)
src = src[:m_stats.start()] + NEW_STATS + src[m_stats.end():]

# ── 3b: for-epoch + while loop block ─────────────────────────────────────────
# Anchor: from `start_time = time.time()` through `# end of epoch`
# This captures the scaling injection point + entire training loop.
epoch_start = re.search(r'^    start_time = time\.time\(\)\n', src, re.MULTILINE)
epoch_end   = re.search(r'^    # end of epoch\n', src, re.MULTILINE)
if not epoch_start:
    print("ERROR: 'start_time = time.time()' not found", file=sys.stderr)
    sys.exit(1)
if not epoch_end:
    print("ERROR: '# end of epoch' not found", file=sys.stderr)
    sys.exit(1)

# Rewrite from start_time through # end of epoch.
# YAML total_iter / warmup_iter are in OPTIMIZER-STEP units.
# We scale to raw-iter units here so the rest of the loop is unchanged.
LOOP_BLOCK = (
    "    start_time = time.time()\n"
    "\n"
    "    # Scale YAML iter counts (optimizer steps) → raw loop iterations\n"
    "    _accum = opt.get('datasets', {}).get('train', {}).get('accum_iters', 1)\n"
    "    if _accum > 1:\n"
    "        total_iters  *= _accum\n"
    "        total_epochs *= _accum\n"
    "        msg_logger.max_iters = total_iters  # fix ETA to use raw iter count\n"
    "    _warmup_raw = opt['train'].get('warmup_iter', -1)\n"
    "    if _warmup_raw > 0 and _accum > 1:\n"
    "        _warmup_raw *= _accum\n"
    "\n"
    "    for epoch in range(start_epoch, total_epochs + 1):\n"
    "        train_sampler.set_epoch(epoch)\n"
    "        prefetcher.reset()\n"
    "        train_data = prefetcher.next()\n"
    "\n"
    "        while train_data is not None:\n"
    "            data_timer.record()\n"
    "\n"
    "            current_iter += 1\n"
    "            if current_iter > total_iters:\n"
    "                break\n"
    "\n"
    "            # 1-based position in the accumulation window\n"
    "            _window_pos = ((current_iter - 1) % _accum) + 1\n"
    "            is_update_step = (_window_pos == _accum)\n"
    "\n"
    "            # LR scheduler only advances on actual optimizer.step() iterations\n"
    "            if is_update_step:\n"
    "                model.update_learning_rate(current_iter, warmup_iter=_warmup_raw)\n"
    "\n"
    "            # training\n"
    "            model.feed_data(train_data)\n"
    "            model.optimize_parameters(current_iter)\n"
    "            iter_timer.record()\n"
    "            if current_iter == 1:\n"
    "                # reset start time in msg_logger for more accurate eta_time\n"
    "                # not work in resume mode\n"
    "                msg_logger.reset_start_time()\n"
    "\n"
    "            # log\n"
    "            if current_iter % opt['logger']['print_freq'] == 0:\n"
    "                log_vars = {'epoch': epoch, 'iter': current_iter}\n"
    "                log_vars.update({'lrs': model.get_current_learning_rate()})\n"
    "                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})\n"
    "                log_vars.update(model.get_current_log())\n"
    "                msg_logger(log_vars)\n"
    "                if _accum > 1:\n"
    "                    logger.info(f'  accum [{_window_pos}/{_accum}]')\n"
    "\n"
    "            # save models and training states\n"
    "            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:\n"
    "                logger.info('Saving models and training states.')\n"
    "                model.save(epoch, current_iter)\n"
    "\n"
    "            # validation\n"
    "            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):\n"
    "                if len(val_loaders) > 1:\n"
    "                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')\n"
    "                for val_loader in val_loaders:\n"
    "                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])\n"
    "\n"
    "            data_timer.start()\n"
    "            iter_timer.start()\n"
    "            train_data = prefetcher.next()\n"
    "        # end of iter\n"
    "\n"
    "    # end of epoch\n"
)

src = src[:epoch_start.start()] + LOOP_BLOCK + src[epoch_end.end():]

try:
    ast.parse(src)
except SyntaxError as e:
    print(f"ERROR: syntax error in generated train.py: {e}", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(src)
print("  train.py OK")
PYEOF

# =============================================================================
# PATCH 1 — base_model.py
# =============================================================================
section "Patch 1/3 — base_model.py"
if already_patched "accum_iters" "$BASE_MODEL"; then
    warn "Already patched — skipping"
else
    python3 "$TMPDIR_PATCH/p1_base_model.py" "$BASE_MODEL"
fi

# =============================================================================
# PATCH 2 — sr_model.py
# =============================================================================
section "Patch 2/3 — sr_model.py"
if already_patched "accum_iters" "$SR_MODEL"; then
    warn "Already patched — skipping"
else
    python3 "$TMPDIR_PATCH/p2_sr_model.py" "$SR_MODEL"
fi

# =============================================================================
# PATCH 3 — train.py
# Always re-run: rewrites the while loop body which may be broken from
# a previous partial patch, even if accum_iters is already present.
# =============================================================================
section "Patch 3/3 — train.py"
python3 "$TMPDIR_PATCH/p3_train.py" "$TRAIN_PY"

# =============================================================================
# VERIFY — accum_iters present + syntax clean on all three files
# =============================================================================
section "Verification"
ALL_OK=true
for F in "$BASE_MODEL" "$SR_MODEL" "$TRAIN_PY"; do
    FNAME=$(basename "$F")
    if ! grep -q "accum_iters" "$F"; then
        echo -e "  ${RED}✗${NC} $FNAME — accum_iters NOT found"
        ALL_OK=false
        continue
    fi
    if python3 -m py_compile "$F" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $FNAME — accum_iters present, syntax OK"
    else
        echo -e "  ${RED}✗${NC} $FNAME — SYNTAX ERROR"
        ALL_OK=false
    fi
done

$ALL_OK || error "One or more patches failed. Originals are in: $BACKUP"

# =============================================================================
# DONE
# =============================================================================
section "Done"
cat << EOF

  Add these keys under datasets.train in your YAML
  (NOT under the top-level train: section):

    datasets:
      train:
        batch_size_per_gpu: 8
        accum_iters: 4        # effective batch = 8 x 4 x num_gpu
        use_grad_clip: true
        grad_clip_norm: 1.0

  Effective batch reference:
    PFT light (LANTA): 4 GPUs x 8 imgs        = 32  (no accum needed)
    Single-GPU local : 1 GPU  x 8 imgs x 4    = 32  (simulate 4-GPU run)
    Single-GPU local : 1 GPU  x 4 imgs x 8    = 32  (half VRAM)

  To undo:
    cp $BACKUP/base_model.py  $BASE_MODEL
    cp $BACKUP/sr_model.py    $SR_MODEL
    cp $BACKUP/train.py       $TRAIN_PY

  Re-running is safe:
    base_model + sr_model are skipped if already patched.
    train.py is always re-applied (idempotent — safe to re-run).
  Backups: $BACKUP
EOF
