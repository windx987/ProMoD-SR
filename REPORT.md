# ProMoD-SR Training Collapse — Investigation Report

**Period:** 2026-07-01 → 2026-07-04 (RESOLVED)
**Symptom:** 301 (ProMoD-light ×2) validation PSNR peaks at 25–30K iters around
31.9–33.0 dB Set5 — *below bicubic interpolation (33.66 dB)* — then plateaus or
declines while training loss keeps improving. Baseline PFT-light reaches ~38 dB.

**Resolution:** the primary cause was root cause **#6** — the benchmark val LR
images on the dataset PVC were generated with the wrong kernel. After
regenerating them, the 300 control (PFT-light reproduction) scores
**38.21 dB Set5 @ 215K iters**, above the published 38.10. All logged val
numbers in the run-history table below are invalid as quality measurements;
they reflect the corrupted benchmark, not the models.

## Run history

| # | Config | Optimizer | Routing | Recipe | Set5 trajectory | Outcome |
|---|--------|-----------|---------|--------|-----------------|---------|
| A | 301 | custom Muon 5e-4 | A-MoD (attention scores), min cap 0.4 | b4×accum4, long warmup | peak 33.01 @ 30K | frozen through 120K |
| B | 301 | custom Muon 5e-4 | A-MoD, min cap 0.625 | b4×accum4, short warmup* | 25.29→30.03→32.39→**33.03 @ 30K** | slow decline to ~32.4 |
| C | 301 | custom Muon 5e-4 | *(intended LR-MoD)* | as B | identical to B ±0.003 dB | **never ran the new code** — deployed to a stale second clone |
| D | 301 | AdamW 5e-4 | LR-MoD (inert — no gradient) | b4×accum4, short warmup* | 24.35@5K, **31.22@10K**, dip 28.98@15K, 31.87@25K | monotonic decline to ~30.7 @ 215K |
| E | 300 control v1 | AdamW 5e-4 | **MoD disabled** | short warmup* | 22.86@5K, 31.49@10K, **dip 29.16@15K** | killed — reproduced the dip with no MoD |
| F | 300 control v2 | AdamW 5e-4 | MoD disabled | **exact upstream: b8, no accum, warmup 20K steps** | 32.75@5K, 32.62@10K, 31.14@15K | **in flight** — mild decline during warmup, verdict at 25–30K |

\* "short warmup" = the warmup-unit regression described below (5K optimizer steps instead of 20K).

## Root causes found (in discovery order)

### 1. Custom Muon implementation was mis-scaled
The hand-rolled Muon normalized each orthogonalized update to the raw gradient
norm (`scale = g.norm()/update.norm()`) instead of the official
`sqrt(max(1, rows/cols))` adjustment. With global grad-clip 1.0 spread over
~200 tensors, 2D weight matrices received updates ~100× smaller than intended —
the network learned mostly through biases/norms and hit a ceiling.
**Fix:** replaced by the official `torch.optim.Muon` (PyTorch 2.9.1) via a
wrapper — see `set_MUON.md`. Runs A–C were all affected.

### 2. MoD router received zero gradient
Routing used a hard binary top-k mask; `topk` indices are non-differentiable
and the router score was never multiplied into the output. The router
(`nn.Linear(dim,1)`) stayed at random init forever; token selection flickered
with feature drift, injecting noise that scaled with lr.
**Fix:** selected tokens' attention/FFN outputs are now scaled by
`sigmoid(router_score)` (standard MoD, Raposo et al. 2024); skipped tokens
remain exactly 0. Verified: router gradient flows every step.

### 3. Routing top-k was global → image-size dependent
`topk` ran over all N tokens: N = 4,096 at train (64×64 LQ patches) but tens of
thousands at validation (full images). Train and val saw different routing
distributions — explaining train loss improving while val declined.
**Fix:** top-k now runs within each 32×32 attention window (1,024 tokens,
constant at any input size). Verified: per-window sparsity exactly equals the
capacity ratio at 64×64 and 256×256, both shift configurations.

### 4. Warmup unit regression (self-inflicted)
All YAML iteration counts (`total_iter`, scheduler milestones, `warmup_iter`)
are in **optimizer steps**; with grad accumulation, `train.py` converts to raw
loop iterations. The `warmup_iter *= accum` line was misdiagnosed as a bug and
removed mid-investigation, silently cutting warmup from 20,000 to 5,000
optimizer steps. Both AdamW runs (D, E) dipped ~2 dB at optimizer step ~3,750,
right as the too-fast ramp crossed lr ≈ 3.7e-4.
**Fix:** conversion restored; 300/301 additionally switched to the exact
upstream recipe (2 GPU × batch 8, no accumulation) so every YAML count is
trivially an optimizer step.

### 6. PRIMARY: validation LR images used the wrong degradation kernel
The `Evaluation/*/LRbicx{2,3,4}` dirs on the dataset PVC were generated with
OpenCV cubic (no antialiasing) instead of MATLAB `imresize` — the universal SR
benchmark standard, and the kernel the (correct) DIV2K training LR uses.
Kernel fingerprints: train LR vs MATLAB bicubic ≈ 0.28 (PNG rounding); val LR
vs MATLAB ≈ 1.59, vs cv2 ≈ 0.23. Every model was trained on one degradation
and scored on another — as a model sharpens toward the true kernel, its score
on the mismatched one *falls*, producing the universal "peak then decline
below bicubic" signature. The faster the optimizer, the earlier the false peak.

Proof: the same 300-control checkpoint scores 30.87 dB (provided files) vs
37.69 dB (correct MATLAB-bicubic LR) at 20K, and 38.21 dB at 215K.
**Fix:** regenerated `LRbicx{2,3,4}_matlab/` for Set5/Set14/B100/U100/M109
from GTmod12 via `basicsr.utils.matlab_functions.imresize` (script:
`~/regen_val_lr.py` on glider; originals untouched); all train configs point
at the new dirs. **Anything ever validated against the old dirs has
systematically depressed numbers and should be re-scored.**

### 5. Operational: deployment and infrastructure traps
- **Two repo clones on glider.** Training runs from `~/research-sisr/ProMoD-SR`;
  a stale `~/ProMoD-SR` clone absorbed one deployment — run C was a byte-identical
  rerun of run B (curves matched to 0.003 dB, which is what exposed it).
- **Pod restarts** kill tmux/processes and wipe container-layer installs.
  Mitigations: tmux installed on the PVC (conda base env); both launch scripts
  now pass `--auto_resume`; training resumed cleanly from the 10K state after
  the 2026-07-03 restart.
- **Reverse tunnel outages** (hours-long) are now handled by a state-change
  monitor rather than per-poll warnings.

## Final status (2026-07-04)

- **300 control / PFT-light baseline reproduction:** stopped at 215K
  (checkpoints kept, resumable). True score on corrected Set5: **38.21 dB** —
  above the published 38.10. Pipeline, recipe and arch fully validated.
- **301 (ProMoD soft per-window routing, AdamW upstream recipe):** launched
  from scratch 2026-07-04 on both GPUs, validating against the corrected
  benchmark. Expected healthy band at 5–20K: ~35–37.5 dB (control was 37.69 @
  20K). If it lands markedly below the control's curve, the gap measures the
  real cost of MoD token skipping — tune capacity schedule / warmup layers,
  not the training setup.

## Follow-up queued

- 302/303 (×3/×4 finetunes) and the 20x normal-model series follow once 301
  validates.

## Muon A/B and the effective-batch-32 consolidation (2026-07-04 → 07-06)

**Phase 1 — 301 (AdamW) vs 311 (Muon), both effective batch 16, MoD active.**
Ran in parallel on shared GPUs against the corrected benchmark. Near-identical
curves throughout — Muon led by ~+0.05–0.12 dB at 5–10K, the gap closed to a
statistical tie by 15K and stayed there. 301 was paused at iter 100K
(Set5 38.158) to give 311 the full GPUs; 311 continued alone through **210K**
(Set5 **38.2038**, Set14 **34.0017**, BSD100 32.3849) — both optimizers work
equally well for this model; Muon shows no clear advantage.

Reconstructed 311 (batch-16) Set5 trajectory (from conversation record — the
original `~/train_311.log` and its experiment/checkpoint folder were later
overwritten/deleted during the batch-32 consolidation below, so this table is
the only surviving record of that run; treat exact digits as approximate,
the trend and milestones are solid):

| Iter | Set5 | Iter | Set5 | Iter | Set5 |
|------|------|------|------|------|------|
| 5K | 34.67 | 75K | 38.10 | 145K | 38.16 |
| 10K | 35.68 | 80K | 38.11 | 150K | 38.16 |
| 15K | 37.13 | 85K | 38.10 | 155K | 38.17 |
| 20K | 37.64 | 90K | 38.12 | 160K | 38.17 |
| 25K | 37.82 | 95K | 38.13 | 165K | 38.18 |
| 30K | 37.90 | 100K | 38.14 | 170K | 38.18 |
| 35K | 37.93 | 105K | 38.14 | 175K | 38.18 |
| 40K | 37.98 | 110K | 38.13 | 180K | 38.18 |
| 45K | 38.01 | 115K | 38.14 | 185K | 38.19 |
| 50K | 38.04 | 120K | 38.16 | 190K | 38.18 |
| 55K | 38.05 | 125K | 38.17 | 195K | 38.19 |
| 60K | 38.08 | 130K | 38.16 | 200K | 38.19 |
| 65K | 38.08 | 135K | 38.17 | 205K | 38.20 |
| 70K | 38.10 | 140K | 38.17 | 210K | **38.20** |

**Phase 2 — batch-32 correction.** The published PFT-light numbers (776K
params, 278.3G FLOPs, ×2: **Set5 38.36 / Set14 34.19 / BSD100 32.43**, effective
batch **32**) surfaced a recipe mismatch: our reproduction trained at effective
batch 16 (2 GPU × 8), not 32. At iter 210K the gap to target was Set5 −0.16 dB,
Set14 −0.19 dB, BSD100 −0.04 dB — plausibly attributable to the smaller batch's
noisier gradient estimate.

**Phase 3 — consolidation.** Rather than maintain two named experiments, `301`
was reconfigured to `type: Muon` + `batch_size_per_gpu: 16` (effective 32,
matching the paper) and both prior experiment folders (301's paused AdamW/
batch-16 run at 100K, and a brief 311 batch-32 restart that only reached
~100 iters) were wiped. `301` is now the single canonical light-model ×2 run:
Muon optimizer, effective batch 32, corrected val data, launched fresh
2026-07-06 via the SISR29 env. First checkpoints tracked slightly ahead of the
batch-16 curve at matched iterations (e.g. 34.77 vs ~34.6 @ 5K), consistent
with the batch-size hypothesis. Progress continues under `~/train_301.log`.
