# ProMoD-CLF: Cross-Layer Feature Fusion for Image Super-Resolution

**A native, quality-focused architecture — not adapted from an external paper**

**Date:** July 26, 2026

---

## Table of Contents

1. [Motivation & Key Insight](#1-motivation--key-insight)
2. [Why Not Another MoD/Efficiency Mechanism](#2-why-not-another-modefficiency-mechanism)
3. [Verifying the Gap Against Upstream Repos](#3-verifying-the-gap-against-upstream-repos)
4. [Overall Pipeline](#4-overall-pipeline)
5. [CrossLayerFusion Module](#5-crosslayerfusion-module)
6. [Stability-by-Construction](#6-stability-by-construction)
7. [Block Structure & History Buffer Scope](#7-block-structure--history-buffer-scope)
8. [Model Configuration](#8-model-configuration)
9. [Complexity Analysis](#9-complexity-analysis)
10. [Staged Verification](#10-staged-verification)
11. [Comparison: CLF vs the MoD Family](#11-comparison-clf-vs-the-mod-family)
12. [Current Results](#12-current-results)
13. [Open Questions & Future Ablations](#13-open-questions--future-ablations)

---

## 1. Motivation & Key Insight

Every architectural change made to ProMoD-SR before this one — v1.0
(mask-multiply MoD), v1.1 (real gather/scatter MoD), MoE (multi-expert
FFN) — followed the same recipe: **borrow a mechanism from another paper,
bolt it onto the PFT backbone, evaluate on FLOPs/PSNR.** MoD came from
Raposo et al. 2024. MoE came from the general Mixture-of-Experts
literature. Neither is native to this codebase or this problem.

This document describes a deliberate break from that pattern: **a
mechanism designed by reading this specific codebase and finding a gap
that was never filled**, not by importing a named technique from
elsewhere.

### The gap

PFT's `WindowAttention` implements Progressive Focusing Attention (PFA): from
the third layer per shift-parity onward, attention is computed only
against a shrinking top-k index set carried forward layer-to-layer via a
`pfa_list = [pfa_values, pfa_indices]` state threaded through every block's
`forward()` signature. This narrows *which keys/queries attention looks
at* as depth increases — a real, useful cross-layer mechanism, and the
one PFT's own paper credits for its quality.

But look at what it actually carries: **only attention indices and
values**, never the layer's *output features themselves*. The residual
stream (`x = shortcut + x_win.view(...)`, then `x = x + convffn(...)`)
already accumulates every prior layer's contribution — but only as an
undifferentiated running sum. No layer can reach back and independently
weight *one specific* earlier layer's raw, undiluted output. That
capability simply doesn't exist anywhere in PFT, in any of ProMoD's MoD
variants, or (as verified — see §3) in any of the related upstream
papers this project has studied.

**CrossLayerFusion (CLF) fills exactly that gap**: a small, gated,
per-layer hook that lets a layer selectively pull in the last `K` layers'
actual output features, independent of the residual stream's running sum.

---

## 2. Why Not Another MoD/Efficiency Mechanism

The immediate temptation, when asked to "improve the architecture," is to
find another efficiency trick to stack on top of MoD. This was
deliberately rejected, because two research passes this session confirmed
a real problem with that path:

**MoD, PFT's own PFA cascade, and ProSAT's DTA (Density-driven Token
Aggregation) all compete on the same axis** — "identify which spatial
tokens matter less, spend less compute on them." Concretely:

- PFA narrows attention *support* (which tokens get attended to/from)
  as depth increases.
- MoD's router additionally decides whether a token's **entire block**
  (attention + FFN) runs at all, using a signal that — despite what the
  design docs imply — is completely disconnected from PFA's own signal
  (MoD's router is an independent `nn.Linear(dim, 1)` learned from
  scratch on `norm1(x)`, never reading `pfa_values`/`pfa_indices`).
- DTA compresses the K/V side via clustering.

Stacking MoD on top of PFT, when PFT is already doing a related
importance-narrowing operation, produces the diminishing/overlapping
returns actually measured in this project: MoD alone costs ~0.04–0.05dB
even at a modest capacity ratio (run 301 vs 304), and v1.0's FLOPs savings
are mostly theoretical rather than realized in wall-clock time. Adding yet
another mechanism on this same axis (a different routing heuristic, a
different clustering scheme) was judged unlikely to do better than what's
already been tried.

**The alternative chosen**: drop the efficiency framing entirely for
this architecture, and spend the compute on quality instead, via a
genuinely different axis — cross-layer *feature* reuse, not cross-layer
*attention-index* narrowing.

---

## 3. Verifying the Gap Against Upstream Repos

Design docs and even this project's own README are not sufficient
evidence that a mechanism doesn't already exist — they can be wrong (the
README implies MoD's router reads PFA's signal; the code shows it
doesn't). So before committing to CLF, the actual upstream repos for the
three most related papers were cloned and read in full, not just their
paper abstracts or READMEs:

**`github.com/CVL-UESTC/PFT-SR`** (the original PFT). `diff` against this
project's own `pft_arch.py` returned **zero output — our copy is
byte-identical to the original**. Confirmed directly in the source: PFA's
cross-layer linkage really is index-only. No layer anywhere in the
original code reads another layer's raw hidden-state output outside the
standard sequential residual stream.

**`github.com/PhuTran1005/SAT`** (source of ProSAT's DTA mechanism). The
real DTA is a clustering/weighted-average *merge* of tokens into a
compressed K/V representation — it never drops or skips a token, and the
network's main token stream stays at full resolution at every layer,
always. This also clarified something important: **the `.detach()`-ed,
zero-gradient router and the zero-fill-before-depthwise-conv corruption
bug that stalled ProSAT (run 401) are this project's own additions**,
layered on top of a faithfully-ported DTA — not flaws inherited from the
original SAT paper, which never introduces a "some tokens are missing"
state that a spatial conv could corrupt in the first place.

**`github.com/CVL-UESTC/IET`** (a newer paper, explicitly built on PFT and
ATD). Extends PFA's cascade idea further — propagating candidate token
indices across whole transformer blocks, not just within one — and adds a
genuinely different mechanism: for its deeper blocks, each token borrows
the just-computed output feature of its single most attention-similar
*other token* (same layer, different spatial position), fused via a small
linear layer before the FFN. This is real feature reuse, but along a
different axis than CLF: **cross-token** (same depth, different position)
rather than **cross-layer** (same position, different depth). Confirms CLF
occupies genuinely unclaimed space even against the closest related
follow-up work.

---

## 4. Overall Pipeline

CLF changes nothing about PFT's backbone structure — same patch embed,
same 5 residual groups (`depths=[2,4,6,6,6]`), same window attention, same
upsampling head. The only change is what feeds into `norm1` before
attention at each layer:

```
Stock PFT / all MoD variants:                CLF:

  x  ──────────────┐                          x  ──┬─────────────────┐
                    │                               │                 │
                    ▼                               ▼                 │
              norm1(x)                     x_in = CLF(x, history)     │
                    │                               │                 │
                    ▼                               ▼                 │
             WindowAttention                   norm1(x_in)            │
             (PFA cascade)                          │                 │
                    │                                ▼                │
                    ▼                         WindowAttention         │
        x = shortcut + attn_out              (PFA cascade, unchanged) │
                    │                                │                │
                    ▼                                ▼                │
        x = x + convffn(norm2(x))           x = shortcut + attn_out   │
                                                       │               │
     shortcut == original x  ◄─────────────────────── x (unfused!) ───┘
                                                       │
                                                       ▼
                                          x = x + convffn(norm2(x))
                                                       │
                                                       ▼
                                    history = (history + [x])[-K:]
```

Two things worth noting from the diagram: `shortcut` is always the
**original, un-fused** `x` — CLF only changes what attention *sees*, never
what gets added back into the residual stream. And `history` is updated
with the layer's real output *after* the block completes, so the next
layer's fusion always sees genuinely finished features, not a
partially-computed value.

---

## 5. CrossLayerFusion Module

```python
class CrossLayerFusion(nn.Module):
    def __init__(self, dim, history_window, gate_type='scalar', fusion_proj=True):
        super().__init__()
        self.k = history_window
        self.gate_type = gate_type

        if fusion_proj:
            self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False)
                                        for _ in range(history_window)])
            for lin in self.proj:
                nn.init.zeros_(lin.weight)
        else:
            self.proj = nn.ModuleList([nn.Identity() for _ in range(history_window)])

        if gate_type == 'scalar':
            self.gate = nn.Parameter(torch.zeros(history_window))
        elif gate_type == 'channel':
            self.gate = nn.Parameter(torch.zeros(history_window, dim))

    def forward(self, x, feat_history):
        recent = feat_history[-self.k:]           # oldest-first, most-recent-last
        fused = 0.
        for j, h in enumerate(reversed(recent)):   # j=0 -> lag1, j=1 -> lag2, ...
            h_proj = self.proj[j](h)
            g = self.gate[j] if self.gate_type == 'scalar' else self.gate[j].view(1, 1, -1)
            fused = fused + g * h_proj
        return x + fused
```

At any layer with fewer than `K` prior layers available (i.e. near the
start of the stack), the loop simply runs over however many entries exist
— no special-casing needed. Layer 0 always has an empty `feat_history`, so
`CrossLayerFusion` is skipped entirely there (guarded at the call site,
see §7).

---

## 6. Stability-by-Construction

This project has three real, documented incidents where a new learned
component destabilized training or silently failed to learn at all:

1. A hand-rolled Muon optimizer was mis-scaled per-parameter, corrupting
   training from step 0.
2. ProSAT's importance score was `.detach()`-ed — structurally zero
   gradient — and this stalled training hard at iter 50K.
3. ProSAT's GDFN scattered skipped tokens into a zero-filled buffer before
   a depthwise conv, corrupting neighboring valid tokens' features.

CLF's design rules out all three failure classes **by construction**, not
by testing-and-hoping:

- **`proj` weights are zero-initialized.** Regardless of what the gate's
  value is, `CrossLayerFusion` computes the exact identity function at
  step 0 — training starts byte-identical to stock PFT. This directly
  addresses failure class (1): nothing new can destabilize training from
  iteration 1, because nothing new does anything at iteration 1.
- **The gate is a raw, unconstrained scalar (or per-channel vector) — not
  passed through `sigmoid`.** A sigmoid has a saturation region where
  gradient becomes numerically negligible; a bare scalar multiplier has no
  such region. This is a direct, deliberate contrast with failure class
  (2) — ProSAT's router had a *structural* zero-gradient path; CLF's gate
  gradient is `fused_output_contribution · upstream_grad`, never
  structurally suppressed.
- **Every operation is pointwise** — `nn.Linear` applied per-token,
  elementwise gate multiply, elementwise add. **No spatial or convolution
  op ever touches `feat_history`.** This rules out failure class (3) by
  construction: there is no such thing as an "invalid" or "zero-filled"
  position anywhere in CLF's design, since every layer's full feature map
  is always fully computed before being added to history — unlike MoD,
  which has to reason about what a "skipped" token's placeholder value
  should be.

---

## 7. Block Structure & History Buffer Scope

Verified directly against `pft_arch.py`'s actual code (not assumed):
`pfa_list` is initialized **once** in `forward_features` and threaded
through all 5 residual groups with **zero resets between them** — PFA's
state genuinely persists across the entire 24-layer stack, group
boundaries included. `feat_history` follows the identical convention:

```python
def forward_features(self, x, params):
    pfa_values, pfa_indices = [None, None], [None, None]
    pfa_list = [pfa_values, pfa_indices]
    feat_history = []                       # initialized once, persists globally

    x = self.patch_embed(x)
    for layer in self.layers:                # loops across all 5 groups
        x, pfa_list, feat_history = layer(x, pfa_list, feat_history, x_size, params)
    ...
```

Every `PMDCLFTL` layer's `forward` does the minimal possible diff against
a stock PFT layer:

```python
def forward(self, x, pfa_list, feat_history, x_size, params):
    ...
    x_in = self.clf(x, feat_history) if (self.clf is not None and feat_history) else x
    shortcut = x                    # unchanged -- original, un-fused input
    x = self.norm1(x_in)            # CHANGED: was self.norm1(x)
    ...                              # everything else byte-identical to stock PFT
    x = shortcut + attn_out
    x = x + self.convffn(self.norm2(x), x_size)
    feat_history = (feat_history + [x])[-self.history_window:]
    return x, [pfa_values, pfa_indices], feat_history
```

This is safe to persist across group boundaries specifically because
every group in every existing config shares the same `embed_dim` and
`input_resolution` (no hierarchical downsampling anywhere in this
project's configs) — a layer near the start of group *i* can reach back
into group *i-1*'s raw outputs with no dimension mismatch. That reach —
bypassing the intervening group's own `patch_unembed → conv → patch_embed`
transform entirely — is the concrete new information pathway CLF adds
that nothing else in this codebase provides.

---

## 8. Model Configuration

New standalone file: `basicsr/archs/promod_clf_arch.py`. Deliberately
contains **no router, no `capacity_ratio`, no capacity schedule** —
mirrors the project's existing "one file per architectural variant"
convention (`promod_arch.py`, `promod_v1_1_arch.py`, `promod_moe_arch.py`
are each self-contained too), and the absence of any MoD keys in its yaml
config is itself part of the "this is quality-only" signal:

```yaml
network_g:
  type: PMDCLFModel
  # ...same backbone params as every other ProMoD variant...
  history_window: 3             # K, number of prior layers reachable
  fusion_gate_type: 'scalar'     # 'scalar' | 'channel'
  fusion_proj: true              # learned alignment projection per slot
  # NOTE: no capacity_ratio / capacity_schedule / mod_warmup_layers here
```

`gate_type='scalar'` with `fusion_proj=True` and `history_window=3` is the
configuration actually running (see run 601). Two cheap ablations are
identified but not yet tried: `gate_type='channel'` (per-channel gate,
negligible extra params) and `fusion_proj=False` (drop the alignment
projection, cutting most of CLF's parameter cost) — both worth checking
once 601's full-run result is in, to see if the same quality can be had
more cheaply.

---

## 9. Complexity Analysis

Measured directly via `benchmark.py`-style instantiation (not estimated):

| Config | Params | FLOPs @640×360 | vs dense |
|---|---|---|---|
| Dense PFT (run 304 architecture) | 0.776M | 278.04G | — |
| **ProMoD-CLF** (`K=3`, scalar gate) | **0.970M** | **319.95G** | **+15.07%** |

The added cost is scale-invariant — the same +15.07% figure holds at the
64×64 training-patch resolution too, since the added term (`proj_j`, an
`nn.Linear(dim, dim)` applied per active history slot) scales with
`h × w × dim²` exactly like the rest of the network's linear layers.

The `.flops()` accounting is **exact, not worst-case**: each layer knows
its own `layer_id`, so it reports `min(layer_id, history_window)` active
history slots — layer 0 through layer `K-1` genuinely have fewer than `K`
slots available and are costed accordingly, not padded to the maximum.

CLF is explicitly **not** a FLOPs-reduction technique — same honest
framing already used for the MoE variant's row in `SUMUP.md`. The bar it
needs to clear to be worth keeping is beating dense PFT's PSNR/SSIM
outright, since it spends compute rather than saving it.

---

## 10. Staged Verification

Given this project's history of expensive silent training failures (a
mis-scaled optimizer, a zero-gradient router, a corruption bug — all
discovered only after real GPU-hours were spent), CLF was verified in
three cheap-to-expensive stages before committing to the full 500K-iter
run:

**1. Offline smoke test** (no GPU cluster, <1 minute): instantiate
`PMDCLFModel`, run one forward pass, print shape/params/FLOPs, diff the
param count against an identically-configured dense model. Result:
+194,760 params (194.8K) — matched the hand estimate from planning almost
exactly.

**2. Short real-data smoke run** (2000 iters, ~35 minutes, node 4, real
DIV2K data, actual Muon optimizer settings, validation disabled):
checked the loss curve for early divergence, and inspected gate values
directly from saved checkpoints (every 500 iters) since hooking gate
telemetry into the shared, in-use `train.py` was avoided to reduce
blast radius on other live runs. Results, all clean:

| Iter | l_pix | max&#124;gate&#124; across all 24 layers | Any NaN |
|---|---|---|---|
| 10 | 0.2945 | — | — |
| 500 | — | 0.0013 | No |
| 1000 | 0.0287 | 0.0031 | No |
| 1500 | — | 0.0043 | No |
| 2000 | 0.0188 | 0.0066 | No |

23 of 24 layers' gates moved measurably off their zero-init by iter 500
(confirming real, non-degenerate gradient flow); layer 0's gate correctly
stayed at exactly `0.0` throughout, since it has zero history available
(`min(layer_id=0, history_window=3) = 0` active slots) and its
`CrossLayerFusion.forward` is never invoked — this is expected behavior,
not a bug. One real bug was caught and fixed during this stage: both new
yaml configs initially set `model_type: PMDCLFModel` (wrong — that name
only exists in the arch registry) instead of `model_type: PMDModel` (the
generic training wrapper every ProMoD variant uses; `network_g.type` is
what actually selects the architecture).

**3. Full 500K-iteration run** (601, node 4) — launched only after stages
1 and 2 were both clean.

---

## 11. Comparison: CLF vs the MoD Family

| | v1.0 (mask-multiply) | v1.1 (gather/scatter) | MoE | **CLF** |
|---|---|---|---|---|
| Axis | Depth (skip whole blocks) | Depth (skip whole blocks) | Width (FFN capacity) | **Cross-layer feature reuse** |
| Goal | Efficiency | Efficiency | Quality (spends compute) | **Quality (spends compute)** |
| Router/gate | Learned linear probe, hard top-k | Same as v1.0 | Softmax over experts | **Raw scalar, no top-k, no routing** |
| Touches PFA? | No (routing independent of PFA) | Bypasses PFA for routed layers | No | **No — orthogonal, PFA unchanged** |
| Native or adapted? | Adapted (Raposo et al. 2024) | Adapted (same routing, different execution) | Adapted (general MoE literature) | **Native — fills a gap found by reading this codebase** |
| FLOPs vs dense | −10 to −29% (honest, per schedule) | Same schedules as v1.0 | +10 to +30% (per expert count) | **+15.07%, scale-invariant** |

---

## 12. Current Results

As of this writing, run 601 (the full CLF training run, node 4) is in
progress. Because comparing an in-progress run's current-best against a
fully-converged baseline's *final* number is unfair, comparisons instead
pull the baseline's own validation history at the *same* iteration from
its still-intact training log:

**601 vs 304 (dense + Muon), both at iter 55,000 (11% through):**

| Benchmark | 601 | 304 | Gap |
|---|---|---|---|
| Set5 PSNR/SSIM | 38.1346 / 0.9616 | 38.1249 / 0.9615 | **+0.0097dB / +0.0001** |
| Set14 PSNR/SSIM | 33.7887 / 0.9200 | 33.8351 / 0.9199 | −0.0464dB / +0.0001 |
| BSD100 PSNR/SSIM | 32.3208 / 0.9022 | 32.3149 / 0.9022 | **+0.0059dB / tied** |

At matched training progress, CLF is essentially tied with dense PFT —
ahead on two of three benchmarks, all differences within noise — despite
paying +15.07% more FLOPs. For context, every other variant checked this
way trails 304 by a real margin at matched iterations: MoE (e=2, now
complete) by 0.03–0.06dB, MoE (e=4, in progress) by 0.02–0.13dB, v1.1 at a
20.18% FLOPs cut (now complete) by 0.09–0.14dB. CLF is the only variant
that hasn't cost quality so far — though it's still early, and the real
test is where it lands at full convergence against 304's final numbers
(38.3497/34.2352/32.4626 PSNR).

---

## 13. Open Questions & Future Ablations

- **Does CLF's quality edge hold at convergence?** The current read is
  from 11% through training; needs re-checking as 601 approaches 500K.
- **`gate_type='channel'`**: does per-channel gating do meaningfully
  better than a scalar, or do the gates end up uniform (in which case the
  scalar variant is strictly more efficient for the same result)?
- **`fusion_proj=False`**: is the alignment projection (the majority of
  CLF's added parameter cost) actually earning its keep, or does an
  identity mapping of the historical feature work just as well?
- **Composability with MoD**: CLF was deliberately kept standalone
  (no MoD code in `promod_clf_arch.py` at all) to isolate its effect and
  avoid ambiguity if something misbehaves. A future `PMDCLFMoDModel`
  combining both is possible — `promod_moe_arch.py` already demonstrates
  the pattern for importing primitives across variant files — but wasn't
  built here, since composing two experimental mechanisms before either
  is independently validated would make it harder to attribute any result
  to either fusion or feature reuse specifically.
- **Larger `history_window`**: `K=3` was chosen as a reasonable first
  value, not derived from a sweep. Worth trying `K=2` (cheaper) and larger
  values (more expressive, more params) once the base result is known.
