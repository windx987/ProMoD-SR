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

- **311** (`options/train/311_ProMoD_light_SRx2_muon.yml`): Muon-vs-AdamW A/B,
  identical to 301 except the optimizer, running in the SISR29 env
  (PyTorch 2.9.1, official `torch.optim.Muon`). Setup verified end-to-end —
  see `set_MUON.md`.
- 302/303 (×3/×4 finetunes) and the 20x normal-model series follow once 301
  validates.
