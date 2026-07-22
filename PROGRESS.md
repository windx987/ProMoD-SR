# ProMoD-SR / ProSAT Progress Log

Running log of experiment state, decisions, and open issues. Updated as runs
complete or milestones land. See `REPORT.md` for the deeper training-collapse
investigation history and the published PFT-light target numbers.

## Current state (2026-07-22)

**Main node (port 2200) pod restarted 2026-07-22 — 502's progress
(iter 110K+, exact last iteration unknown since the tunnel had been down
since 2026-07-17) is lost.** The pod restart wiped `/home/glider` entirely
(fresh hostname, empty home dir) — everything that lived only on
pod-local storage (the git clone, the `smm_cuda` build, tmux sessions,
and critically `experiments/`/`tb_logger/`/the training log, all written
relative to the project root by default) is gone. The shared PVCs
(datasets, environment/conda, results, backup) survived intact, confirming
this is a pod-local wipe, not a PVC failure.

**Root-cause fix applied before relaunching**: `basicsr/train.py` writes
`experiments/<name>/` (checkpoints, training state) and `tb_logger/<name>/`
relative to the project root by default — pod-local, so a future restart
would wipe them again. Fixed by symlinking both into the persistent
results PVC (`/mnt/pvc-shared-pvc-results-8da1bd63/{experiments,tb_logger}`)
before relaunching, and by redirecting the training log directly to that
PVC too (`.../results-8da1bd63/logs/train_502.log`, with `~/train_502.log`
symlinked to it so existing monitor commands keep working unchanged). This
fix was applied on the main node only so far — node 2 (2202, running 503)
and node 3 (2204, running 504) are still writing to pod-local storage and
would lose progress on a similar restart; worth applying the same symlink
fix there at a safe pause point, without disrupting their live runs.

**502 relaunched from scratch** (iter 0) on the rebuilt main node: fresh
clone, `smm_cuda` rebuilt for SISR29 (env itself survived on the PVC —
only the pod-local `.so` needed rebuilding), grad-accum patch reapplied,
tmux sessions recreated. First 100 iterations clean (l_pix in normal
warmup-lr range). Unfortunately the prior ~110K+ iterations of progress
on this run are unrecoverable — no checkpoint survived the wipe.

## Previous state (2026-07-21)

501 (ProMoDv1.1, default progressive schedule) has **completed** its full
500K-iteration run (finished 2026-07-20, 6 days 3:39:28 total training
time). Final best results:

| Benchmark | PSNR (dB) | SSIM |
|---|---|---|
| Set5 | 38.2597 @500K | 0.9620 @465K |
| Set14 | 34.1848 @310K | 0.9227 @310K |
| BSD100 | 32.4095 @440K | 0.9033 @445K |

Lands within ~0.05–0.09 dB of both 301's v1.0 mask-multiply baseline
(Set5 38.3198, Set14 34.1400, BSD100 32.4369) and 304's dense-Muon
reproduction (Set5 38.3497, Set14 34.2352, BSD100 32.4626) — actually
slightly exceeds 301 on Set14 — despite the real gather/scatter
execution and the 7.85% honest FLOPs cut. Confirms v1.1's routing
mechanism costs essentially nothing in quality at this (small) capacity
reduction. Node 2204 (node 3) is now free.

304 (dense PFT-light + Muon reproduction) has **completed** its full
500K-iteration run (finished 2026-07-19, 5 days 15:07:34 total training
time). Final best results:

| Benchmark | PSNR (dB) | SSIM |
|---|---|---|
| Set5 | 38.3497 @410K | 0.9623 @410K |
| Set14 | 34.2352 @435K | 0.9232 @435K |
| BSD100 | 32.4626 @490K | 0.9040 @500K |

Solidly matches/exceeds both 301's ProMoD baseline (Set5 38.3198, Set14
34.1400, BSD100 32.4369) and the PFT-light published targets across all
three benchmarks and both metrics — confirms the Muon-optimizer dense
reproduction is sound. Node 2202 (node 2) is now free.

401 (ProSAT) **completed** 2026-07-16, 3 days 4:36:03 total. Final best
results:

| Benchmark | PSNR (dB) | SSIM |
|---|---|---|
| Set5 | 38.0303 @450K | 0.9612 @325K |
| Set14 | 33.6887 @365K | 0.9194 @385K |
| BSD100 | 32.2202 @500K | 0.9008 @500K |

The stall diagnosed below (flat plateau from iter 50K) never fully
resolved after the mod_ramp removal — final numbers sit noticeably below
301's ProMoD baseline and below the PFT-light published targets. The
GDFN zero-fill fix (see "Next step" below) remains unimplemented; node
2200 (main) was freed and reused for 502 (below).

**502_ProMoDv1_1_light_SRx2_r0480 launched 2026-07-16** on the main node
(port 2200): `PMDGSModel` with `mod_capacity: 0.48` (uniform r override
for all non-warmup layers, via the existing `mod_capacity` knob in
`build_capacity_schedule` — no code change needed). Verified directly
via `model.flops([640, 360])` (not estimated): 221.92G vs 278.04G dense
= **20.18% honest FLOPs reduction**, a substantially more aggressive cut
than 501's default progressive schedule (7.85%). Same architecture/code
as 501 (already validated: CPU equivalence, gradient coverage, GPU smoke
train, healthy real training trajectory) — only the capacity value
differs, so no new correctness verification was needed before launching
the full 500K run.

**503_ProMoDv1_1_light_SRx2_r0500_nowarmup launched 2026-07-19** on node
2 (port 2202, freed by 304's completion): `PMDGSModel` with
`mod_capacity: 0.5` AND `mod_warmup_layers: 0` — the first run to drop
the 2-layer dense-warmup exception entirely, so all 24 layers are routed
at r=0.5 (previous runs 301/501/502 always kept the first 2 layers dense).
Verified via `model.flops([640,360])`: 198.29G vs 278.04G dense = **28.68%
honest FLOPs reduction**, the most aggressive cut attempted yet. This is
untested territory — watching early iterations closely for the ProSAT
flat-plateau stall signature or the routing-collapse signature (early
PSNR peak then decline while train loss keeps improving), since no prior
run has removed the warmup exception. First 100 iters clean (l_pix in
normal warmup-lr range, no crash).

**504_ProMoD_MoE_light_SRx2_e2 launched 2026-07-21** on node 3 (port
2204, freed by 501's completion): a new architecture, `PMDMoEModel`
(`basicsr/archs/promod_moe_arch.py`), adding MoE (Mixture-of-Experts,
width-sparsity) as a capacity/quality axis orthogonal to MoD's existing
depth-sparsity. Unlike every MoD variant so far, **this is NOT a
FLOPs-reduction technique** — it's a soft, fully-dense multi-expert FFN:
`ConvFFN`'s `fc2` is replicated into `num_experts` (2) independent
branches, computed for every token (no top-k, no gather/scatter, no
auxiliary load-balancing loss) and combined by a per-token softmax gate.
Composes cleanly with MoD's existing `active_mask` since the expert
combination lives entirely inside the dense `convffn()` call — the two
routers never interact. Design was informed by literature research this
session: width-MoE's payoff at ~1M-param scale is genuinely uncertain
(ViMoE), so the safe "dense/soft combination" pattern was chosen over
top-k+aux-loss or the more novel "integrated MoD+MoE null-expert router"
(deferred — see plan file). Verified: CPU equivalence at `num_experts=1`
(bit-identical to `ConvFFN`), gradient coverage across all experts +
both routers (CPU and, separately, on real GPU with `smm_cuda`), and
honest FLOPs/params — `num_experts=2` costs **+68.6K params (0.778M →
0.845M) and +10.46% FLOPs (249.25G → 275.32G @640×360)**, reported
honestly since this trades compute for quality rather than reducing it.
First 500 iterations clean (loss 0.258→0.050, monotonic, no crash/NaN).

One MoD run live; **main node (port 2200, running 502) is still
unreachable** — the reverse tunnel itself is down (nothing listening
locally on 2200, confirmed via verbose SSH; not just a slow remote), a
known "pod restart" failure mode for this HPC setup. Last confirmed 502
data point is from 2026-07-17 13:29 (iter 110,000) — the persistent
monitor watching it has been silently stalled since then (no further
validation lines arrived, and none of the down-detection logic fired
because the connection attempt itself times out around the same ~10s
threshold the detector uses, a gap in that script worth fixing). Needs a
manual tunnel restart from the user's side before 502's current
iteration/status can be confirmed.

| Run | Node | Port | Iter | Status |
|---|---|---|---|---|
| 504_ProMoD_MoE_light_SRx2_e2 | node 3 (`c16g2-03-...`) | 2204 | ~100K / 500K | healthy, clean upward trend, no gate-collapse/stall signature |
| 503_ProMoDv1_1_light_SRx2_r0500_nowarmup | node 2 (`c16g2-02-...`) | 2202 | ~275K / 500K | healthy, no stall/collapse signature through the 50K mark and beyond |
| 502_ProMoDv1_1_light_SRx2_r0480 | main | 2200 | **restarted from iter 0** 2026-07-22 (pod wipe lost prior 110K+) | lower-r (0.48) variant, ~20% FLOPs cut; now checkpointing to persistent PVC |

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

**GPU verification completed 2026-07-14 on a third node** (`c16g2-03-...`,
port 2204 — set up identically to node 2: fresh clone, `smm_cuda` built for
both `SISR`/`SISR29` envs, grad-accum patch, tmux sessions). `benchmark.py`
(now with a third PMDGSModel column) on a real A100:

| Resolution | PFT (ms) | ProMoD v1.0 (ms) | v1.1 (ms) | v1.1 vs PFT |
|---|---|---|---|---|
| 64×64 (train patch) | 124.1±51.7 | 127.4±57.7 | 67.4±10.8 | **1.84× faster** |
| 128×128 | 172.7±36.9 | 204.3±32.8 | 163.3±9.7 | 1.06× faster |
| 256×256 | 480.2±20.8 | 523.5±45.0 | 519.1±42.3 | 0.92× (slower) |
| 320×180 | 501.9±31.2 | 452.3±17.6 | 538.5±44.9 | 0.93× (slower) |
| 640×360 | 1784.0±163.6 | 1794.9±146.0 | 2180.5±134.7 | 0.82× (slower) |

Peak GPU memory is *higher* for v1.1 at every resolution too (e.g. 20.7GB
vs 16.7GB @640×360). **Verdict: real gather/scatter is only a net win at
the training-patch size (64×64); at anything larger it's slower and uses
more memory than mask-multiply**, despite doing genuinely less arithmetic
(confirmed correct via the CPU equivalence tests above). This is a hardware-
efficiency result, not a correctness one — the many small, irregular-memory
gather/scatter/topk ops (one extra global topk per routed layer just to
reindex for the FFN, on top of the windowed ones) don't parallelize on a
GPU as well as PFT's original large regular dense matmuls. Matches why the
original DeepMind MoD paper and most real MoD deployments need dedicated
fused/custom kernels to realize wall-clock speedup — naive eager-mode
PyTorch gather/scatter isn't enough on its own. A short smoke train (501,
node 3) confirmed training-time correctness separately: 100 iters clean, no
crash/NaN, real `smm_cuda` invoked successfully for the dense warmup
layers; per-iteration time (~1.13s) is slower than 304's dense-Muon
baseline (~0.92s/iter), consistent with the benchmark finding.

**Next step, if pursued further**: this POC validates the *routing decision
and gradient math* end-to-end (that part is solid) but shows real
gather/scatter needs custom/fused kernels (or batching many windows'
gathers into fewer, larger ops) to pay off in wall-clock terms — plain
`torch.gather`/`torch.scatter_`/`torch.topk` per-layer isn't sufficient.
Not blocking the rest of the queue; 501 keeps running as a real training
signal since the mechanism is correct even though the speed goal wasn't met.

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
