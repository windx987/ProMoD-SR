# ProMoD-SR / ProSAT Progress Log

Running log of experiment state, decisions, and open issues. Updated as runs
complete or milestones land. See `REPORT.md` for the deeper training-collapse
investigation history and the published PFT-light target numbers.

## Current state (2026-07-14)

Two training runs live in parallel across two glider nodes, both GPUs busy
on each (no free slot for a new experiment right now):

| Run | Node | Port | Iter | Status |
|---|---|---|---|---|
| 401_ProSAT_light_SRx2_scratch | main | 2200 | ~179K / 500K | **stalled — see below** |
| 304_PFTlight_muon_dense_SRx2 | new (`c16g2-02-...`) | 2202 | ~58K / 500K | healthy, climbing |

301 (ProMoD-light, Muon, eff. batch 32) completed 2026-07-12 — see
`promod_training_recipe` memory / earlier REPORT.md entries for final numbers
(Set5 38.3198, Set14 34.1400, BSD100 32.4369; BSD100 exceeded the published
PFT-light target, Set5/Set14 landed ~0.04–0.05 dB short).

**ProMoDv1.1 (PMDGSModel) implemented 2026-07-14** — a new arch
(`basicsr/archs/promod_v1_1_arch.py`) queued as run 5 (`501_ProMoDv1_1_light_
SRx2_scratch.yml`), not yet launched (both nodes occupied). Real
gather/scatter execution of v1.0's already-correct MoD routing math (v1.0 is
mask-multiply — dense compute, zeroed output; see benchmark.py's header).
Verified locally on CPU: forward pass at 2 sizes, full gradient coverage
(including non-zero router gradients), and equivalence against v1.0 —
bit-exact at the single-layer level, ~6.5e-4 max diff at the full 24-layer
level, explained by PFA's progressive value-cascade (a per-layer Hadamard
attention-multiply, active regardless of key-shrink) being dropped in
routed layers, an expected consequence of the design (see file docstring
and commit `1a8419f`), not a bug. Honest FLOPs reduction at the 301
schedule: 7.85% (256.21G vs 278.04G dense @640×360) — v1.0's `flops()`
optimistically claimed 10.4% by assuming the FFN's depthwise conv and fc1
could be routed; they can't (same zero-fill risk as ProSAT's GDFN bug, see
below), so v1.1's corrected formula only routes fc2 and attention's Q side.
**Still needs**: a real GPU run to confirm actual latency speedup
(benchmark.py extended with a v1.1 column) and a short smoke train before
committing to the full 500K run — blocked on a node freeing up.

## Goal

Map FLOPs-vs-PSNR tradeoff for MoD-style routing (20–50% target reduction
while holding baseline PSNR), plus two supporting tracks: a dense PFT-light +
Muon reproduction (isolates optimizer effect from MoD cost) and ProSAT
(SAT's K/V-compressed attention + parameter-free importance routing).

## Run queue

1. **304_PFTlight_muon_dense_SRx2** — dense (mod_disable) + Muon + eff. batch
   32. Now running on the new node (2202). At iter 44K: Set5 38.0605 (@40K),
   Set14 33.7378 (@40K), BSD100 32.2669 (@40K) — all still climbing, tracking
   toward the published target (38.36 / 34.19 / 32.43) with the first lr
   decay (250K) still far off.
2. **321_ProMoD_light_SRx2_r0500** (`mod_capacity: 0.5` → 210.77G FLOPs
   @640×360, −24.2% vs dense 278.04G) — not yet started, no node assigned.
3. **322_ProMoD_light_SRx2_r0250** (`mod_capacity: 0.25` → 177.13G, −36.3%)
   — not yet started. Early-kill rule: if val peaks ~25–30K then declines
   while train loss improves (known routing-collapse signature), kill and
   move to the next setting.
4. **401_ProSAT_light_SRx2_scratch** — implemented, running, **currently
   stalled** (see Open Issues).

`mod_capacity` is now a `network_g` config knob in `promod_arch.py`
(`build_capacity_schedule(..., capacity=...)`); default `None` preserves the
301/302/303 hard-coded schedule exactly (regression-checked).

## ProSAT: implementation + stall investigation

Implemented from scratch (no prior code existed) per `ProSAT.md`:
- `basicsr/archs/prosat_arch.py` — PSAA attention (Q from routed-active
  tokens, K/V from SAT's `cluster_and_merge` DTA aggregation over ALL
  tokens), GDFN gated FFN, PSAB blocks with **real gather/scatter** token
  skipping (unlike ProMoD's mask-multiply — skipped tokens are not computed
  at all), progressive importance routing (active `*= 0.5+0.5*score`,
  skipped `*= 0.9` decay).
- `options/train/401_ProSAT_light_SRx2_scratch.yml` — C=60, 4 groups × 4
  blocks, 6 heads, 0.7736M params, 4.02G FLOPs @64×64 (16.6% below its own
  dense variant — well short of ProSAT.md's claimed ~34%, since DTA/convs
  don't scale with capacity r).
- Switched from ProSAT.md's spec'd Adam to **Muon** per user request
  (transparent — the Muon wrapper auto-splits params by `ndim`, no arch
  change needed).

**First attempt (with iteration-driven `mod_ramp`, dense until 50K, ramp to
target capacity by 100K)**: stalled hard exactly at iter 50K — all three
benchmarks AND train loss (`l_pix`) went flat simultaneously, right when
routing began engaging. Not the classic "train improves while val degrades"
overfit signature — a genuine stall.

**Root cause hypothesis**: `GDFN.forward` scatters skipped tokens' gate
features into a **zero-filled** buffer before the gate's 3×3 depthwise conv.
The conv mixes spatial neighbors, so real zeros at skipped positions corrupt
neighboring active tokens' gate values — worse as capacity drops. ProMoD
never has this because its mask-multiply approach always computes the true
dense conv and only zeroes the *output* afterward.

**Fix attempted**: removed the temporal `mod_ramp` mechanism entirely —
routing now active from iteration 0 at the target capacity schedule
(matching ProMoD's convention: no ramp, just structural
`mod_warmup_layers`). Deleted `ProSATModel` (existed only to drive the
ramp); config's `model_type` switched to `PMDModel`. Wiped the old
checkpoint/tb_logger and reran from scratch (old log archived as
`~/train_401_ramp_attempt.log`).

**Result: the fix did NOT resolve the stall.** Rerun trajectory 105K→155K:
best Set5 crept 37.9521 → 37.9775 (+0.025 dB over 50K iterations) — same
flat plateau as before, just without the sharp transition at 50K (since
there's no ramp to transition at anymore). This is strong evidence the
**GDFN zero-fill artifact, not the ramp, is the actual root cause** — it was
present throughout this whole rerun (routing active from iter 0) and the
stall still happened.

### Next step (not yet implemented)

Fix the GDFN zero-fill artifact directly: keep `fc1`/`act` (the gate's input
projection) dense for all tokens — cheap, just linears — and only route
`fc2`'s output application to active tokens. This removes the zero-fill
from the depthwise conv's input entirely, at the cost of some of the FFN's
FLOPs savings (fc1 becomes non-scaled by r). Alternative: fill skipped
positions from the pre-FFN residual instead of zero (cheaper edit, less
clean). Should re-run a short (~20-30K iter) check before committing another
full 500K run.

## Second compute node (port 2202)

Discovered/set up 2026-07-13: a separate pod (hostname `c16g2-02-...`),
own 2×A100-40GB, idle. Shares the same `/mnt/pvc-shared-pvc-{datasets,
environment,results,backup}` PVCs as the main node (verified identical NFS
export UUIDs) but has its own separate `/home/glider` PVC — repo cloned
fresh, `smm_cuda` built for **both** `SISR` (py3.9) and `SISR29` (py3.11)
envs (each Python ABI needs its own `.so`), grad-accum patch applied,
`train_x2`/`work` tmux sessions created. Now running 304 in parallel with
401 on the main node — doubles effective throughput on the run queue.

## Key FLOPs-accounting caveat

Analytical FLOPs reductions are smaller than the design docs claim, because
`wqkv`/LePE/convs/upsampler (ProMoD) and DTA/convs (ProSAT) don't scale with
the capacity ratio r:
- ProMoD's 301 schedule (avg r≈0.76): only 10.4% reduction, not ~24%.
- ProMoD r→0 caps at ~48% reduction (never approaches 100%).
- ProSAT's light schedule (avg r≈0.66): 16.6% reduction, not ProSAT.md's
  claimed ~34%.
- ProSAT's DTA is quadratic in N (m≈0.03N), so its FLOPs are only reported
  at the SAT-paper 64×64 convention — never at 640×360 like ProMoD.
