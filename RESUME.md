# Resume protocol — snapshot taken 2026-08-05 after migrating off the full results PVC

## Current state (2026-08-05)

`pvc-shared-pvc-results-8da1bd63` (the old results PVC) filled to 100%
(3.3T/3.5T used, 0 avail) on 2026-08-03 ~20:52 — turned out **the entire
NFS backend it sits on was full** (backup/datasets/environment PVCs on
the same `172.16.101.195:/mnt/nfs/*` server all showed the identical
3.3T/3.5T/0-avail numbers simultaneously), not just this one PVC. 601
died outright from it (uncaught exception); 402/505 survived the initial
hit but every one of the three runs had a **corrupt trailing checkpoint**
(0-byte or truncated `.state`/`.pth` from the write that was in progress
when the disk filled: 402 @290000, 505 @355000, 601 @330000).

**User provided a new PVC, `pvc-shared-pvc-pj-storage-9ec66456`** — same
healthy Lustre backend as `/home/glider` (434T size, ~275T avail, 37%
used at migration time). All `experiments`/`tb_logger`/`logs` data (only
~14GB real) was copied there, corrupt trailing checkpoints deleted, all
4 nodes' symlinks re-pointed, and 402/505/601 relaunched cleanly:

| Run | Node/Port | Resumed from (post-migration) | Target |
|---|---|---|---|
| **402** (ProSATv2) | 1 / 2200 | iter 285,000 | 500,000 |
| **505** (MoE e=4) | 2 / 2202 | iter 350,000 | 500,000 |
| **601** (CLF) | 4 / 2206 | iter 325,000 | 500,000 |

**`pvc-shared-pvc-results-8da1bd63` is no longer used** — all `experiments`/
`tb_logger`/`logs` symlinks on all 4 nodes now point at
`/mnt/pvc-shared-pvc-pj-storage-9ec66456/results/{experiments,tb_logger,logs}`.
If a future session finds symlinks pointing at the old `results` PVC,
that means a node got rebuilt from the old recipe — re-point it to
`pj-storage` instead, not back to `results`.

## Rebuild-from-zero recipe (node wipe recovery)

Per node, everything below is idempotent — safe to re-run. **Uses
`pj-storage`, not the old `results` PVC:**

```bash
CONDA=/mnt/pvc-shared-pvc-environment-ff3ed7c7/miniconda3
PY29=$CONDA/envs/SISR29/bin/python
PROJ=$HOME/research-sisr/ProMoD-SR
REPO=https://github.com/windx987/ProMoD-SR.git
RESULTS=/mnt/pvc-shared-pvc-pj-storage-9ec66456/results

# 1. clone (conda envs SISR/SISR29 + tmux live on the shared environment
#    PVC and survive node wipes -- only the per-node repo clone is lost)
mkdir -p "$HOME/research-sisr"
[ -d "$PROJ/.git" ] && git -C "$PROJ" pull --ff-only || git clone "$REPO" "$PROJ"
cd "$PROJ"

# 2. symlink experiments/tb_logger into the results PVC -- use mv/rm -f,
#    NOT a live swap under a running process (see "What NOT to do" below)
mkdir -p "$RESULTS"/{experiments,tb_logger,logs}
for d in experiments tb_logger; do
  [ -L "$d" ] || { [ -e "$d" ] && mv "$d" "${d}.template-$(date +%s)"; ln -s "$RESULTS/$d" "$d"; }
done

# 3. build smm_cuda for SISR29 (only env needed -- 402/505/601 all use
#    Muon optimizer, which requires SISR29; skip SISR/py3.9 build unless
#    a non-Muon run needs it)
cd ops_smm && rm -rf build
CUDA_HOME="$CONDA/envs/SISR29" "$PY29" setup.py build_ext --inplace
```

Then launch/resume (per run, `--auto_resume` finds the latest
`training_states/*.state` automatically via the symlinked `experiments/`
path — no yml edits needed). **Before relying on `--auto_resume`, always
check the latest `.state`/`.pth` pair isn't 0-byte or truncated** (sort
numerically, not alphabetically — `ls ... | sed -E 's#.*/([0-9]+)\.(state|pth)#\1#' | sort -n | tail -3`
— a disk-full or killed-mid-write event leaves exactly this kind of
corrupt trailing file, and `--auto_resume` will pick it first and crash
with a `torch.load` `EOFError`. Delete the corrupt pair before launching
if found):

```bash
cd $PROJ
export PATH="$CONDA/envs/SISR29/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA/envs/SISR29/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJ:$(dirname $(find ops_smm -name 'smm_cuda*.so'))"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
LOGFILE=/mnt/pvc-shared-pvc-pj-storage-9ec66456/results/logs/train_<NAME>.log
ln -sf "$LOGFILE" ~/train_<NAME>.log
nohup torchrun --nproc_per_node=2 --master_port=4321 basicsr/train.py \
  -opt options/train/<NAME>.yml --launcher pytorch --auto_resume \
  > "$LOGFILE" 2>&1 < /dev/null &
disown
```

## Step 1 — reconnect and verify the tunnel is back

Tunnel is initiated *from* the HPC side (`~/reverse-tunnel.sh` on each
node). If a plain SSH check hangs, the tunnel needs restarting from the
HPC side first.

```bash
ssh -p 2200 glider@localhost "echo alive"   # node 1 -- 402
ssh -p 2202 glider@localhost "echo alive"   # node 2 -- 505
ssh -p 2204 glider@localhost "echo alive"   # node 3 -- free
ssh -p 2206 glider@localhost "echo alive"   # node 4 -- 601
```

## Step 2 — confirm each run is still actually training

```bash
for port in 2200 2202 2206; do
  echo "=== port $port ==="
  ssh -p $port glider@localhost "ps aux | grep -c '[t]rain.py'"
done
```
Expect `3` on each (torchrun launcher + 2 DDP workers).

## Step 3 — check current progress

```bash
ssh -p 2200 glider@localhost "tail -5 /mnt/pvc-shared-pvc-pj-storage-9ec66456/results/logs/train_402.log"
ssh -p 2202 glider@localhost "tail -5 /mnt/pvc-shared-pvc-pj-storage-9ec66456/results/logs/train_505.log"
ssh -p 2206 glider@localhost "tail -5 /mnt/pvc-shared-pvc-pj-storage-9ec66456/results/logs/train_601.log"
```

Also spot-check `df -h /mnt/pvc-shared-pvc-pj-storage-9ec66456` occasionally
— if it's ever back near 0 avail, stop and investigate before anything
else (same failure mode as the 2026-08-03 incident could recur).

## Step 4 — re-arm the checkpoint monitors (do NOT survive a session restart)

```
Monitor: ssh -p 2200 glider@localhost "tail -f -n0 /mnt/pvc-shared-pvc-pj-storage-9ec66456/results/logs/train_402.log" | grep -E --line-buffered "Validation|# psnr|# ssim|Traceback|Error|error|CUDA out of memory|Killed|No space"
Monitor: ssh -p 2202 glider@localhost "tail -f -n0 /mnt/pvc-shared-pvc-pj-storage-9ec66456/results/logs/train_505.log" | grep -E --line-buffered "Validation|# psnr|# ssim|Traceback|Error|error|CUDA out of memory|Killed|No space"
Monitor: ssh -p 2206 glider@localhost "tail -f -n0 /mnt/pvc-shared-pvc-pj-storage-9ec66456/results/logs/train_601.log" | grep -E --line-buffered "Validation|# psnr|# ssim|Traceback|Error|error|CUDA out of memory|Killed|No space"
```
All three should be `persistent: true`.

## Context for a fresh session

- This session's plan file: `/home/unix/.claude/plans/mighty-zooming-marble.md`
  — holds the **ProSAT-v2** plan (already executed; run 402 is it).
- Persistent memory (auto-loaded every session): `experiment_queue.md`,
  `MEMORY.md`, `glider_hpc.md`, `results_pvc_disk_full.md` in the Claude
  memory directory — updated with this migration.
- `SUMUP.md` / `PROGRESS.md` / `ARCH.md` in this repo have fuller
  narrative history (CLF's design, PIR's postmortem, ProSAT-v2's fixes,
  the 2026-07-22/23 pod-wipe incidents, and this disk-full/migration
  incident).
- **402 background**: fixes ProSAT (401)'s two real bugs — a
  parameter-free router with no trainable gradient path, and a GDFN gate
  that corrupted neighboring active tokens via zero-fill (root cause of
  401's iter-50K stall). 402 passed iter 50K cleanly with new bests on all
  three benchmarks, unlike 401.
- **PIR**: designed, built, verified, then scrapped by explicit user
  request (too invasive, two real OOM bugs found during verification).
  File (`promod_pir_arch.py`) kept as historical record, no action needed.

## What NOT to do on resume

- Don't relaunch from scratch — all three runs resumed correctly from
  their last *good* checkpoint (corrupt trailing ones removed first);
  re-launching without `--auto_resume`, or before checking for a corrupt
  trailing checkpoint, risks either losing progress or crashing on launch.
- Don't swap the `experiments`/`tb_logger` symlinks while a training
  process is still running against the old path — kill it cleanly first
  (`kill` the torchrun launcher PID, wait for all `train.py` processes to
  exit), then swap, then relaunch. Swapping live corrupted 503/504 in the
  2026-07-23 incident (see `PROGRESS.md`).
- Don't `rm` the template `experiments`/`tb_logger` dirs on a fresh clone
  before symlinking — use `mv`.
- Don't touch `basicsr/train.py` on any node without checking `git status`
  first.
- Don't assume a quiet monitor means something's wrong — re-arm first,
  then check logs directly if still unsure.
