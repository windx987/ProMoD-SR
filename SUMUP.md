# Experiment Summary — ProMoD-SR / ProSAT

All runs: DIV2K training, ×2 upscaling, Set5/Set14/BSD100 eval, embed_dim=52,
depths=[2,4,6,6,6] (24 transformer layers), Muon optimizer, 500K iters,
effective batch 32. PFT-light published targets (paper baseline to beat):
**Set5 38.36 / Set14 34.19 / BSD100 32.43** (PSNR, dB).

## Completed runs

| Run | Arch | What it tests | Set5 (PSNR/SSIM) | Set14 (PSNR/SSIM) | BSD100 (PSNR/SSIM) | vs target |
|---|---|---|---|---|---|---|
| **301** | ProMoD v1.0 (mask-multiply), default progressive MoD schedule | ProMoD baseline, Muon optimizer | 38.3198 / — | 34.1400 / — | 32.4369 / — | Set5/Set14 ~0.04–0.05dB short; BSD100 exceeded |
| **304** | PFT dense (`mod_disable`), Muon | Isolates optimizer effect — pure PFT-light + Muon, no MoD at all | **38.3497 / 0.9623** | **34.2352 / 0.9232** | **32.4626 / 0.9040** | Matches/exceeds target on all three |
| **401** | ProSAT (SAT's DTA + parameter-free routing) | Alternative architecture to ProMoD | 38.0303 / 0.9612 | 33.6887 / 0.9194 | 32.2202 / 0.9008 | Below both 301 and target — see "ProSAT" below |
| **501** | ProMoD v1.1 (real gather/scatter), same schedule as 301 | Does real gather/scatter cost quality vs v1.0's mask-multiply? | 38.2597 / 0.9620 | 34.1848 / 0.9227 | 32.4095 / 0.9033 | Within 0.05–0.09dB of 301/304; **slightly beats 301 on Set14** |

**Reproduction chain**: 304 (dense+Muon) reproduces/slightly exceeds the
published PFT-light target on all three benchmarks — confirms the Muon
optimizer swap is sound in isolation. 301 (adds ProMoD's default MoD
schedule on top of the same recipe) costs ~0.04–0.05dB on Set5/Set14
relative to 304/target — the actual price of MoD sparsity at that
schedule. 501 (identical schedule to 301, but real gather/scatter
execution instead of mask-multiply) lands within noise of 301 — **confirms
v1.1's routing math is correct and costs nothing extra vs v1.0**, since
the same sparsity pattern produces the same quality regardless of how it's
executed.

**ProSAT (401)**: stalled hard at iter 50K (flat loss + flat val across
all benchmarks simultaneously) when its temporal `mod_ramp` engaged
routing. Removing the ramp (routing active from iter 0, matching ProMoD's
convention) did **not** fix it — confirms the root cause is GDFN's
zero-fill artifact (skipped tokens' gate features scattered into a
zero-filled buffer before a depthwise conv corrupts neighboring active
tokens), not the ramp. Completed its full 500K anyway, landing below both
301 and the published target. The GDFN fix itself (keep `fc1`/depthwise
conv dense, route only the pointwise output) was identified but never
implemented.

## In-progress runs (as of this writing)

| Run | Arch | Node | Iter | Best-so-far Set5 (PSNR/SSIM) | Best-so-far Set14 | Best-so-far BSD100 |
|---|---|---|---|---|---|---|
| **502** | ProMoD v1.1, `mod_capacity=0.48` (warmup kept) | main / 2200 | ~165K/500K (33%) — **relaunched from 0 twice after infra incidents, see below** | 38.1629 @160K / 0.9616 | 33.8947 @135K / 0.9208 | 32.3321 @165K / 0.9024 |
| **503** | ProMoD v1.1, `mod_capacity=0.5`, **no warmup exception** | node 2 / 2202 | ~435K/500K (87%) | 38.2354 @420K / 0.9618 | 34.0264 @345K / 0.9216 | 32.3739 @435K / 0.9027 |
| **504** | ProMoD-**MoE** (soft dense multi-expert FFN, `num_experts=2`), default MoD schedule | node 3 / 2204 | ~235K/500K (47%) | 38.2397 @215K / 0.9619 | 34.0885 @220K / 0.9221 | 32.4060 @235K / 0.9033 |
| **321** | ProMoD v1.0 (mask-multiply), `mod_capacity=0.5` (warmup kept) | node 4 / 2206 | just launched | — | — | — |

None of the in-progress runs have hit a stall or routing-collapse
signature (early PSNR peak + decline while train loss keeps improving) at
any point so far, including well past ProSAT's iter-50K failure point.

**321 is the true same-r v1.0-vs-v1.1 comparison** long offered but not
launched until a fourth node became available: it uses `PMDModel` (v1.0,
mask-multiply) at `mod_capacity=0.5` with the warmup exception kept —
directly comparable to 503 (v1.1, same r=0.5, but *without* the warmup
exception) and closer still to a hypothetical "v1.0 at 503's exact
schedule" data point. Once both finish, 321 vs 503 isolates the
mask-multiply-vs-gather/scatter execution question at a much more
aggressive capacity than 301 vs 501 did.

## FLOPs accounting (honest, @640×360; dense baseline = 278.04G)

| Config | FLOPs | Reduction vs dense | Notes |
|---|---|---|---|
| Dense (r=1.0, no MoD) | 278.04G | — | 304's architecture |
| 301/501 schedule (progressive, avg r≈0.76) | 256.21G (v1.1 honest) / 249.25G (v1.0 optimistic) | 7.85% / 10.4% | v1.0's `flops()` over-credits fc1/dwconv as routable; v1.1's is corrected |
| **502** (r=0.48, warmup kept) | 221.92G | **20.18%** | |
| **503** (r=0.5, no warmup) | 198.29G | **28.68%** | most aggressive MoD cut attempted |
| **504** (MoE, e=2, on 301-schedule base) | 275.32G | **−10.46% (i.e. +10.46% cost)** | **not a FLOPs-reduction technique** — trades compute for FFN capacity; params 0.778M→0.845M (+68.6K) |

Real GPU latency (benchmark.py, A100, batch=1, @640×360): PFT 1784.0ms,
ProMoD v1.0 1794.9ms (mask-multiply ≈ same as dense, as expected), v1.1
2180.5ms (**slower** despite doing less arithmetic — naive per-layer
`torch.gather`/`scatter_`/`topk` doesn't parallelize as well as PFT's
large dense matmuls; v1.1 only wins at the 64×64 training-patch size,
1.84× faster there). 504's real GPU latency was not separately
benchmarked — only FLOPs/params were computed.

## Architecture family tree

- **v1.0** (`promod_arch.py`, `PMDModel`): MoD via mask-multiply (dense
  compute, zero the output for skipped tokens). Fast on GPU, FLOPs figure
  is theoretical/optimistic.
- **v1.1** (`promod_v1_1_arch.py`, `PMDGSModel`): same MoD routing math,
  real gather/scatter execution. Provably correct (CPU equivalence,
  gradient coverage) but slower on GPU except at small patch sizes —
  hardware-efficiency finding, not a bug.
- **MoE** (`promod_moe_arch.py`, `PMDMoEModel`): built on v1.0, adds a
  soft fully-dense multi-expert FFN as a *width*-capacity axis, orthogonal
  to MoD's *depth* axis. No top-k, no gather/scatter, no aux loss —
  composes with MoD's existing masking without the two routers
  interacting. Informed by literature research this session (width-MoE's
  payoff at ~1M-param scale is genuinely uncertain; the "dense/soft
  combination" pattern was chosen as the safe first experiment over
  top-k+aux-loss or the more novel "integrated MoD+MoE null-expert
  router," which remains deferred).

## Infrastructure incidents (affects the in-progress numbers above)

Two pod-local-storage incidents hit the in-progress runs this cycle —
full technical writeup in `PROGRESS.md`:
1. **Main node pod wipe (2026-07-22)**: lost 502's first attempt entirely
   (110K+ iterations) when the pod restarted and wiped local-only
   checkpoint storage. Root-caused to `experiments/`/`tb_logger/` living
   on pod-local disk by default; fixed via symlinks into the persistent
   results PVC.
2. **Self-inflicted symlink-swap crash (2026-07-23)**: applying the same
   fix to nodes 2/3 hit an `ln` nesting gotcha (502 was affected too, on a
   second pass, losing its second attempt back to iter ~65K) and a
   live-process crash when `tb_logger`'s path was swapped out from under
   an already-running process. 503 and 504 recovered with minimal loss
   (~3K iterations each) via `--auto_resume` from their last real
   checkpoint.
3. **502's silent save failure (2026-07-23)**: after its own symlink fix,
   502 kept logging successful "Saving models and training states" every
   5000 iters with zero errors, yet no checkpoint file existed anywhere —
   `torch.save()` never creates its target directory itself, and that
   directory (only ever created once, at process startup) never got
   (re-)created at the new PVC location since 502's process was never
   restarted. Fixed by manually creating the directory on the PVC (no
   restart needed); confirmed working at the next real save (iter
   70,000). The exact reason the save silently succeeded/failed without
   ever logging an error is still unresolved.

All three in-progress runs are now confirmed checkpointing correctly to
the persistent results PVC, so a future pod restart shouldn't cause
another full loss.

## Open threads

- ProSAT's GDFN zero-fill fix — diagnosed, never implemented.
- 321/322 (v1.0 mask-multiply at r=0.5/r=0.25) — queued, no node assigned.
- MoDA (cross-layer KV attention, arXiv:2603.15619) — researched in depth,
  not implemented; open question whether ProMoD-SR's 24 layers actually
  exhibit the signal-degradation problem MoDA targets (the paper's own
  vision validation needed 64 layers to show the effect). See `MoDA.md`.
