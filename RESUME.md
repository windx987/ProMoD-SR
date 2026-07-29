# Resume protocol — snapshot taken 2026-07-29 ~12:02 before a PC restart

Nothing needs to be relaunched. All three training runs below are detached
(`nohup ... & disown`) processes running independently on their Glider HPC
nodes — restarting the local PC does **not** touch them. What *does* break
is anything local to this machine: the SSH reverse tunnel and this Claude
Code session's background `Monitor` tasks (they're just `ssh ... | grep`
pipes owned by this local process — no local process, no pipe, but the
remote training keeps running regardless).

## Step 1 — reconnect and verify the tunnel is back

The reverse tunnel is initiated *from* the HPC side (`~/reverse-tunnel.sh`
on each node), reaching back to this machine. If a plain SSH check hangs or
refuses, the tunnel itself may need restarting from the HPC side — check
that before assuming a node died.

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
Expect `3` on each (torchrun launcher + 2 DDP worker processes). If any
show `0`, that run died — check its log's tail for the actual error before
doing anything else (don't just relaunch blind).

## Step 3 — check current progress

```bash
ssh -p 2200 glider@localhost "tail -5 ~/train_402.log"
ssh -p 2202 glider@localhost "tail -5 ~/train_505.log"
ssh -p 2206 glider@localhost "tail -5 ~/train_601.log"
```

## Step 4 — re-arm the checkpoint monitors (they do NOT survive a session restart)

```
Monitor: ssh -p 2200 glider@localhost "tail -f -n0 ~/train_402.log" | grep -E --line-buffered "Validation|# psnr|# ssim|Traceback|Error|error|CUDA out of memory|Killed"
Monitor: ssh -p 2202 glider@localhost "tail -f -n0 ~/train_505.log" | grep -E --line-buffered "Validation|# psnr|# ssim|Traceback|Error|error|CUDA out of memory|Killed"
Monitor: ssh -p 2206 glider@localhost "tail -f -n0 ~/train_601.log" | grep -E --line-buffered "Validation|# psnr|# ssim|Traceback|Error|error|CUDA out of memory|Killed"
```
All three should be `persistent: true`.

## Snapshot at time of writing

| Run | Node/Port | Arch | Iter | % done | ETA | Latest best (Set5/Set14/BSD100 PSNR) |
|---|---|---|---|---|---|---|
| **402** | 1 / 2200 | ProSATv2 (`prosat_v2_arch.py`) | 118,500 / 500,000 | 23.7% | ~2d 9h | 37.99 / 33.62 / 32.17 |
| **505** | 2 / 2202 | PMDMoEModel, e=4 | 269,500 / 500,000 | 53.9% | ~2d 20h | 38.29 / 34.14 / 32.43 |
| **601** | 4 / 2206 | PMDCLFModel | 244,700 / 500,000 | 48.9% | ~2d 23h | 38.26 / 34.11 / 32.41 |

**Node 3 (2204) is free.**

## Context for a fresh session

- This session's plan file: `/home/unix/.claude/plans/mighty-zooming-marble.md`
  — currently holds the **ProSAT-v2** plan (already executed; run 402 is
  it). If you re-plan something new, that file gets overwritten — this
  RESUME.md and the memory files below are the durable record, not the
  plan file.
- Persistent memory (auto-loaded every session): `experiment_queue.md` and
  `MEMORY.md` in the Claude memory directory — both already updated with
  402's launch, PIR's scrap, and full experiment history through this point.
- `SUMUP.md` / `PROGRESS.md` / `ARCH.md` in this repo have the fuller
  narrative history and architecture design docs if deeper context is
  needed (CLF's design, PIR's postmortem, ProSAT-v2's router/GDFN fixes).
- **402 background**: fixes ProSAT (401)'s two real bugs — a parameter-free
  router with no trainable gradient path, and a GDFN gate that corrupted
  neighboring active tokens via zero-fill before its depthwise conv (root
  cause of 401's iter-50K stall). 402 has already cleanly passed iter 50K
  with new bests on all three benchmarks, unlike 401.
- **PIR**: designed, built, verified, then scrapped by explicit user
  request (too invasive a change to PFT's attention internals, two real
  OOM bugs found during verification). File (`promod_pir_arch.py`) and its
  commits are kept as historical record, same treatment as v1.0 — no
  action needed, don't resurrect it without being asked.

## What NOT to do on resume

- Don't relaunch anything — all three runs are healthy and mid-flight.
- Don't touch `basicsr/train.py` on any node without checking `git status`
  first — a stale uncommitted debug diff has shown up on multiple nodes
  before (see `experiment_queue.md`).
- Don't assume a quiet monitor means something's wrong — re-arm first,
  then check logs directly if still unsure.
