# Gradient Accumulation in this BasicSR Fork

Report on the accumulation feature added to the BasicSR training stack
(`basicsr/train.py`, `basicsr/models/sr_model.py`, `basicsr/models/base_model.py`).
Written 2026-07-04, after the feature was debugged and validated end-to-end
(see `REPORT.md` for the investigation that hardened it).

## What it does

`accum_iters: N` accumulates gradients over N forward/backward micro-passes and
calls `optimizer.step()` once per window — emulating an N× larger batch when
GPU memory can't hold it. Effective batch = `batch_size_per_gpu × N × world_size`.

## Configuration

All keys live under `datasets: train:` in the YAML:

```yaml
datasets:
  train:
    batch_size_per_gpu: 4
    accum_iters: 4          # micro-passes per optimizer step (default 1 = off)
    use_grad_clip: true     # optional, clip at the window end (default false)
    grad_clip_norm: 1.0     #   total-norm threshold
```

## Unit semantics — read this before touching any iteration count

With accumulation on, the training loop counts **raw iterations** (one per
micro-pass), but the YAML schedule keys keep their upstream meaning of
**optimizer steps**. `train.py` converts between the two:

| YAML key | Unit | Handling with accum_iters = N |
|---|---|---|
| `total_iter` | optimizer steps | raw loop runs `total_iter × N` iterations |
| `warmup_iter` | optimizer steps | multiplied by N before the raw-counter comparison (`train.py`, "warmup_iter … convert steps -> raw") |
| scheduler `milestones` | optimizer steps | `scheduler.step()` fires only on update steps, so MultiStepLR's internal counter *is* the step count — no conversion needed |
| `val_freq`, `save_checkpoint_freq`, `print_freq` | **raw iterations** | compared against the raw counter directly — divide your intended step frequency by N if you want step-based cadence |

The logged `iter:` is the raw counter; the log line gains an `accum [k/N]` tag
showing the window position. Startup logging prints both interpretations
("Optimizer steps (YAML)" and "Raw loop iters") plus the effective batch size.

**History:** the `warmup_iter × N` conversion was once misdiagnosed as a bug
and removed, which silently cut warmup 4× short and destabilized training
(REPORT.md root cause #4). It is unit conversion — leave it in.

## Implementation

### `basicsr/train.py`
- Scales `total_iters`/`total_epochs` by `accum_iters` for the raw loop and
  fixes the ETA logger's `max_iters` accordingly.
- Converts `warmup_iter` from steps to raw iterations (see table above).
- Computes the 1-based window position each raw iteration;
  `model.update_learning_rate(...)` is called **only on update steps**, so the
  scheduler advances once per optimizer step (upstream parity).

### `basicsr/models/sr_model.py` — `optimize_parameters()`
Per raw iteration, with `window_pos = ((current_iter - 1) % accum_iters) + 1`:
1. `zero_grad()` at the **start** of each window (`window_pos == 1`).
2. Forward + loss; backward on `l_total / accum_iters` so the summed gradient
   equals a single large-batch gradient.
3. On non-update steps, the backward runs inside `net_g.no_sync()` — skips the
   DDP all-reduce (~20–30% comms saving); the reduce happens once on the
   update step.
4. On the update step (`window_pos == accum_iters`): optional
   `clip_grad_norm_` over all parameters, then `optimizer_g.step()`.
5. **EMA updates only on update steps** (`model_ema` gated by
   `is_update_step`), so `ema_decay: 0.999` keeps its per-optimizer-step
   meaning.

The 1-based windowing makes the logic correct for fresh runs, resume from any
checkpoint iteration, and `accum_iters: 1` (in which case every branch reduces
to the upstream code path exactly).

### `basicsr/models/base_model.py`
- `self.accum_iters = 1` default so non-SR models are unaffected.
- Helpers `_should_update(current_iter)` / `_loss_scale()` expose the window
  logic for other model classes. Note: `SRModel.optimize_parameters` currently
  inlines the same logic instead of calling them — use the helpers if you add
  accumulation to another model class.

## Interaction with resume / pod restarts

Launch scripts pass `--auto_resume`, which picks up the latest
`training_states/*.state`. The state stores the raw iteration counter;
because windowing is derived from `current_iter` arithmetic (not hidden
state), resuming mid-window is safe — the worst case is one partial window's
gradients discarded at the kill point.

## Validation status

- **`accum_iters: 1` path:** proven equivalent to upstream — the 300 control
  (PFT-light, batch 8×2, no accum) reproduced the published benchmark
  (38.21 dB Set5 ×2 @ 215K vs 38.10 published).
- **`accum_iters: 4` path:** ran mechanically correctly for 200K+ iterations
  across multiple runs (correct lr schedule, EMA, checkpointing, DDP).
  Caveat: those runs' *quality* was confounded by unrelated bugs (REPORT.md),
  so an apples-to-apples "accum 4 × batch 4 ≡ batch 16" quality equivalence
  run has not been done. If a result ever hinges on accumulation, run that
  control first.

## Known limitations

- Only `SRModel` (and subclasses `PFTModel`, `PMDModel`) implement the
  windowed `optimize_parameters`; GAN-style models (`srgan_model.py` etc.)
  ignore `accum_iters`.
- `val_freq`/`save_checkpoint_freq`/`print_freq` staying in raw units is a
  deliberate minimal-change choice but easy to trip over — see the unit table.
- Loss values logged per iteration are per-micro-batch (unscaled), so the
  `l_pix` trace is comparable across accum settings.
