# MoDA (Mixture-of-Depths Attention) — research notes

Source: arXiv:2603.15619 ("Mixture-of-Depths Attention", submitted 2026-03-16)
and github.com/hustvl/MoDA (274 stars, MIT license). Authors: Lianghui Zhu,
Yuxin Fang, Bencheng Liao, Shijie Wang, Tianheng Cheng, Zilong Huang, Chen
Chen, Lai Wei, Yutao Zeng, Ya Wang, Yi Lin, Yu Li, Xinggang Wang — School of
EIC, Huazhong University of Science & Technology, and ByteDance Seed.

**Update note**: this doc was first written from two `WebFetch` passes over
the arXiv abstract page and the GitHub README, which run fetched content
through a small summarizing model — fine for orientation, risky for exact
numbers/formulas. This revision re-fetched the paper's ar5iv HTML rendering
(preserves tables/equations far better than a PDF-to-text pass) and the
actual kernel/model source files directly from `raw.githubusercontent.com`,
and corrects/sharpens the doc accordingly. Every number and code excerpt
below is now traceable to one of those primary sources unless marked
otherwise.

## Important naming clarification

**This is a different mechanism from the "MoD" (Mixture-of-Depths) work we use
in ProMoD/ProSAT**, despite the overlapping name. Ours (Raposo et al. 2024,
arXiv:2404.02258) is **token routing**: a per-token/per-window decision to
skip or keep computing a token through a layer, to save FLOPs. This paper's
"MoDA" is **cross-layer attention**: every token still goes through every
layer as normal — the new thing is that a layer's attention heads can also
attend to the K/V produced by *earlier layers*, not just the current layer's
K/V. Nothing here proposes skipping computation; it targets model *quality*
in very deep stacks, not efficiency. Worth remembering before assuming this
paper is a direct extension of our existing MoD work.

## Problem the paper addresses

Verbatim abstract (from ar5iv): "Scaling depth is a key driver for large
language models (LLMs). Yet, as LLMs become deeper, they often suffer from
*signal degradation*: informative features formed in shallow layers are
gradually diluted by repeated residual updates, making them harder to
recover in deeper layers. We introduce *mixture-of-depths attention*
(MoDA), a mechanism that allows each attention head to attend to sequence
KV pairs at the current layer and depth KV pairs from preceding layers."

## Mechanism — verified against actual source (not paraphrased)

**Combined-softmax, from the repo's own reference implementation**
(`naive_mixture_of_depth_causal_ref`, `fla/ops/moda/moda_v14.py`):

```python
logits_space = (q_vec @ k_bh[:seq_kv_end].T) * scale
if use_depth:
    logits_depth = (kd_bh[base_t] @ q_vec) * scale
    logits = torch.cat([logits_space, logits_depth], dim=0)
w = torch.softmax(logits, dim=0)
```

So "combined-softmax" is exactly what it sounds like: current-layer
attention logits and depth-KV attention logits are concatenated along the
key axis, then **one** softmax normalizes across both — not two separate
attention passes blended afterward. Simple and confirms the earlier
paraphrase was accurate.

**Chunking**, from the real Triton kernel (`parallel_moda_fwd_kernel`,
`moda_v14.py`) — query rows are mapped to depth-KV rows via integer
division by the GQA group count, not a separate chunk-size hyperparameter
worked into the equations as cleanly as the ar5iv extraction of the prose
suggested:

```python
T_kv = T_q // moda_group_num
o_q = q_block_start + tl.arange(0, BT)
o_q_base = o_q // moda_group_num
base_rows_start = q_block_start // moda_group_num
base_rows_end = (q_block_start + valid_rows - 1) // moda_group_num + 1
depth_block_start = base_rows_start * L
depth_block_end = base_rows_end * L
```

The paper's prose (ar5iv extraction) frames this as a utilization ratio:
depth utilization η = `(T·L)/(T·(C·L)) = 1/C` without grouping, rising to
`G/C` with GQA grouping — algebraically consistent with the code's
group-based row-reduction, though the exact correspondence between the
paper's `C` (chunk size) and the code's `moda_group_num` wasn't fully
untangled in this pass; treat the *mechanism* (concatenate-then-softmax,
group-reduced depth indexing) as verified, the precise symbol mapping
between paper and code as approximate.

**Function signature** (`parallel_moda`, `moda_v14.py`):
```python
def parallel_moda(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    cached_k: Optional[torch.Tensor] = None,
    cached_v: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    moda_group_num: int = 1,
    is_causal: bool = True,
    ...
)
```
`is_causal` **defaults to `True` but is overridable to `False`** — and the
vision (DeiT) code actually calls it with `is_causal=False` (see "Vision
wiring" below). This resolves what was previously an open question in this
doc: MoDA does **not** require causal masking, so a non-causal SR model is
not blocked on that count.

**Post-norm vs pre-norm ablation** (Table 6, 48-layer models, per ar5iv
extraction — loss, lower is better):
- Post-norm + depth-KV: 3.4062 → 3.3653 (−0.0409)
- Pre-norm + depth-KV: 3.3800 → 3.3759 (−0.0041)

Post-norm gets roughly **10× more benefit** from adding depth-KV than
pre-norm does in this ablation — a much sharper effect than the original
"works better with post-norm" paraphrase conveyed.

**Kernel efficiency** (Table 2, per ar5iv extraction): 97.3% of
FlashAttention-2's throughput at T=65,536 (2.73% extra time), degrading to
25.86% extra time at T=4,096 — confirms the original summary's numbers were
accurate; the fixed per-call overhead of the depth-indexing logic amortizes
better over long sequences.

## Results (paper) — verified numbers, per-table

- Model scale: **700M and 1.5B parameter models, trained on 400B tokens**
  (this detail — training length — was missing from the first pass).
- Perplexity (Table 5, 1.5B): C4 validation 16.16 → 15.97; average across 10
  domains 13.67 → **13.47 (−0.20)** — confirms the "+0.2 average
  perplexity improvement" figure.
- Downstream tasks (Table 4, 1.5B): overall average **62.28% → 64.39%
  (+2.11 points)**, confirming the original figure; per-task detail now
  available — HellaSwag 65.86%→66.24%, WinoGrande 63.22%→65.59%,
  ARC-Challenge 42.47%→46.82% (the largest single-task gain).
- FLOPs: abstract states "negligible 3.7% FLOPs computational overhead."
  **Flagged inconsistency, not resolved in this pass**: Table 3 (per the
  ar5iv extraction) reports 8.41T → 8.33T FLOPs for a 700M variant — a
  *decrease*, not the +3.7% the abstract states. This could be two
  different comparisons (e.g. different configs, or units/scale mismatch
  in the extraction) rather than an actual contradiction in the paper, but
  it wasn't untangled here — treat the abstract's "+3.7%" as the
  headline figure and the Table 3 numbers as unverified until read more
  carefully.

## Repository (github.com/hustvl/MoDA)

- Requirements: PyTorch ≥2.5, Triton ≥3.0, einops, transformers ≥4.53.0,
  datasets ≥3.3.0, causal-conv1d ≥1.4.0.
- Install: `cd libs/moda_triton && pip install -e .`
- `libs/moda_triton/fla/ops/moda/` contains exactly: `__init__.py`,
  `moda_v14.py`, `moda_v16.py` ("Chunk-Visible MoDA"), `moda_v17.py` (main
  Triton kernel), `fda_v12.py` ("FDA" / Flash Depth Attention) — confirmed
  directly from the repo tree, not inferred from README prose.
- Ships both language-model training scripts (`language_tasks/BlaGPT`,
  FineWeb10B) and vision-task scripts (`vision_tasks/deit`) — see "Vision
  wiring" below for what was actually confirmed there.
- License: MIT.

## Vision wiring (vision_tasks/deit) — new section, confirmed from source

`vision_tasks/deit/models.py` contains real `MoDAAttention` and `MoDABlock`
classes (confirmed by direct fetch, not just inferred from the README
mentioning a `deit` directory):

```python
# MoDAAttention.forward
def forward(self, x, depth_k=None, depth_v=None, current_depth=None):
    o = parallel_moda_v17(
        q=q, k=k, v=v, cached_k=depth_k, cached_v=depth_v,
        moda_group_num=G, is_causal=False, current_depth=current_depth
    )
```
```python
# MoDABlock.forward
def forward(self, x, depth_k=None, depth_v=None, current_depth=None):
    attn_out, k_attn, v_attn = self.attn(
        self.norm1(x), depth_k=depth_k, depth_v=depth_v,
        current_depth=current_depth
    )
    return x, k_attn, v_attn, k_mlp, v_mlp
```

`MoDABlock` returns its own K/V alongside the block output specifically so
the next layer can read them as part of the depth buffer — a
`_WriteDepthSlotKV` custom autograd `Function` manages gradient flow
through that buffer via pre-allocated slot writes (not full cache
reconstruction each step), which is the actual engineering answer to "how
does this train efficiently."

**Training scripts confirm the exact experimental setup**
(`vision_tasks/deit/scripts/train/`, three files, all matched at the same
depth):
- `deit_t_64l_4gpu.sh` — dense baseline
- `deit_t_gqa_64l_4gpu.sh` — GQA baseline
- `deit_t_moda_64l_4gpu.sh` — MoDA variant

**All three are DeiT-Tiny at 64 layers** — not the standard DeiT-Tiny
depth (12 layers). The authors deliberately inflated a small ViT to 64
layers specifically to reproduce the deep-model signal-degradation
phenomenon on vision data, then compared dense/GQA/MoDA at that same
artificial depth. This is a materially important detail for the relevance
assessment below.

## Relevance to ProMoD-SR — updated read after source verification

- **Different axis from both our MoD and MoE work**: MoD (token routing,
  saves FLOPs) and MoE (width/expert routing, this session's soft-dense
  design, spends FLOPs for capacity) are both about *what gets computed
  per-token per-layer*. MoDA is about *what a layer's attention can see*
  across depth — a third, orthogonal axis in principle.
- **The causal-masking concern is resolved, in our favor**: confirmed from
  actual source that `parallel_moda`/`MoDAAttention` support
  `is_causal=False` directly, and the vision (DeiT) variant uses exactly
  that mode. A non-causal SR model is not blocked on this count — this is
  a genuine update from the first-pass version of this doc, which had
  flagged causal masking as an open uncertainty.
- **But the depth/scale mismatch got more concrete, and more concerning**:
  the authors' own vision validation didn't test MoDA on a normal-depth ViT
  — they specifically inflated DeiT-Tiny to **64 layers** (vs. the standard
  12) to reproduce the same signal-degradation phenomenon they study in
  1.5B-parameter, presumably-much-deeper LLMs, then compared dense/GQA/MoDA
  baselines all at that same inflated depth. That is, even the paper's own
  vision existence-proof required an artificially deep network to show the
  effect — they did not report results at ordinary ViT depths. ProMoD-SR is
  **24 layers**, well short of even that vision benchmark's 64, and its
  windowed attention already threads information across layers via PFT's
  own progressive focusing/key-shrinkage cascade (`pfa_values`/
  `pfa_indices` — see `pft_arch.py`'s `WindowAttention`). Both facts point
  the same direction: there's no existing evidence (from this paper or
  otherwise) that signal degradation is actually a problem at our depth,
  and PFT's existing cross-layer PFA mechanism may already be doing
  related work.
- **Engineering path, if pursued, is now concrete rather than vague**: the
  real code confirms the exact primitives needed — a depth-KV buffer with
  gradient-preserving slot writes (`_WriteDepthSlotKV`), the
  concatenate-then-softmax combination, and group-reduced depth indexing.
  This is a bigger lift than either MoD or the MoE work this session (both
  of which reused existing PFT/ProMoD primitives) — MoDA would need a new
  cross-layer buffer threaded through `forward_features`'s layer loop in
  `promod_arch.py`, not just a new per-layer module.
- **Recommended before any implementation**: run the small-scale ablation
  question already flagged in the first pass, now more clearly motivated —
  does ProMoD-light actually show measurable signal degradation across its
  24 layers (e.g. probe intermediate-layer feature similarity/effective
  rank across the `self.layers` loop in `PMDModel.forward_features`)? Given
  that MoDA's own authors needed 64 layers to demonstrate the effect on a
  ViT, the prior should lean toward "probably not at 24 layers" — this
  probe would confirm or overturn that before committing to building a
  depth-KV mechanism to fix a problem that may not exist at this scale.
- No implementation attempted — this document remains research notes only,
  per the request.
