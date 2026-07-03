# ProMoD-SR Training Collapse — Investigation Report

**Period:** 2026-07-01 → 2026-07-03 (ongoing)
**Symptom:** 301 (ProMoD-light ×2) validation PSNR peaks at 25–30K iters around
31.9–33.0 dB Set5 — *below bicubic interpolation (33.66 dB)* — then plateaus or
declines while training loss keeps improving. Baseline PFT-light reaches ~38 dB.

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

## Current status & decision tree

Control v2 (run F) is the pipeline verdict: stock-PFT-light forward
(`mod_disable: true`), exact upstream recipe, healthy 5K start (32.75 dB — the
strongest early checkpoint of any run). A mild decline appeared at 15K while lr
ramps through 3.75e-4 — the same lr level as the earlier dips, which is
suspicious, but mid-warmup wobble is not yet pathological.

- **If F recovers and climbs past ~34 by 25–30K** → pipeline exonerated →
  launch 301 with fixes #2–#4 (soft per-window routing, AdamW upstream recipe).
- **If F shows the peak-then-decline signature** → run the last isolation
  control: upstream `PFT` arch + `PFTModel` classes (still untouched in this
  repo) through our env/data. Healthy → bug hides in the `PMD*` classes even
  with MoD disabled; sick → env/data-level problem.

## Follow-up queued

- **311** (`options/train/311_ProMoD_light_SRx2_muon.yml`): Muon-vs-AdamW A/B,
  identical to 301 except the optimizer, running in the SISR29 env
  (PyTorch 2.9.1, official `torch.optim.Muon`). Setup verified end-to-end —
  see `set_MUON.md`.
- 302/303 (×3/×4 finetunes) and the 20x normal-model series follow once 301
  validates.
