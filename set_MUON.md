# Muon Optimizer Setup (PyTorch 2.9.1, glider HPC)

How the official `torch.optim.Muon` was set up for ProMoD-SR training, and how
to use it. Verified working 2026-07-03 (smoke test + 300-iter live DDP training
run on 2×A100). Background on why the custom implementation was replaced:
see `REPORT.md` (root cause #1).

## Why

Muon (MomentUm Orthogonalized by Newton-schulz) became an official PyTorch
optimizer in 2.9. An earlier hand-rolled implementation in this repo had
mis-calibrated update scaling that silently crippled training (weight matrices
barely moved; see git history around `eb2a2a0`/`b476bd2`). The official
implementation replaces it.

Per the PyTorch docs, Muon only optimizes **2D hidden-layer weight matrices**;
biases, norms, embeddings and conv kernels should use a standard optimizer.
Our wrapper (`basicsr/utils/muon.py`) handles this split automatically.

## Environment: `SISR29`

The regular `SISR` env (Python 3.9, torch 2.5) cannot run torch 2.9
(needs Python ≥ 3.10). A separate env exists on the shared PVC:

| | |
|---|---|
| Env path | `/mnt/pvc-shared-pvc-environment-ff3ed7c7/miniconda3/envs/SISR29` |
| Python | 3.11 |
| PyTorch | 2.9.1+cu126 |
| Why cu126 works on the 12.4 driver | CUDA 12.x minor-version compatibility — any 12.x runtime runs on a 12.x driver. (cu130 wheels do **not** work: CUDA 13 needs a driver-major upgrade.) |

### How it was built (for reproduction)

```bash
CONDA=/mnt/pvc-shared-pvc-environment-ff3ed7c7/miniconda3
export PIP_NO_CACHE_DIR=1     # PVC is ~99% full — do not cache wheels

# 1. env
$CONDA/bin/conda create -y -n SISR29 python=3.11
PY=$CONDA/envs/SISR29/bin/python

# 2. torch (cu126 index)
$PY -m pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. deps
$PY -m pip install numpy opencv-python-headless pyyaml tqdm requests lmdb scipy \
    pillow tb-nightly yapf addict future scikit-image fairscale einops timm ninja

# 4. CUDA build toolchain (pod has no system nvcc; torch headers need the
#    math-lib dev headers too — cusparse.h etc.)
$CONDA/bin/conda install -y -n SISR29 -c nvidia/label/cuda-12.6.3 \
    cuda-nvcc cuda-cudart-dev cuda-crt \
    libcusparse-dev libcublas-dev libcusolver-dev cuda-nvrtc-dev

# 5. rebuild smm_cuda against torch 2.9
cd ~/research-sisr/ProMoD-SR/ops_smm
rm -rf build
CUDA_HOME=$CONDA/envs/SISR29 $PY setup.py build_ext --inplace
```

The build produces `smm_cuda.cpython-311-*.so` **alongside** the existing
`cpython-39` one — each interpreter loads its own; the SISR env is unaffected.

Note: verifying the extension requires `import torch` first (it links against
`libc10.so` from torch): `python -c "import torch; import smm_cuda"`.

## The wrapper: `basicsr/utils/muon.py`

BasicSR expects one optimizer object. The `Muon` class splits parameters:

- `ndim == 2` → `torch.optim.Muon` (wqkv / projection / FFN matrices)
- everything else → `torch.optim.AdamW` (biases, LayerNorms, conv kernels)

and exposes the **live combined `param_groups`**, so MultiStepLR, warmup and
BasicSR's `_set_lr` mutate the real underlying groups. `state_dict()` /
`load_state_dict()` round-trip both sub-optimizers, so checkpoint resume works.
On the light model: 0.486M params in the Muon group, 0.291M in AdamW.

Running it on torch < 2.9 raises an ImportError telling you to use SISR29.

## Usage

Config (`options/train/311_ProMoD_light_SRx2_muon.yml` — mirrors 301 exactly
except the optimizer):

```yaml
train:
  optim_g:
    type: Muon
    lr: !!float 5e-4
    weight_decay: 0
    betas: [0.9, 0.99]      # AdamW group only
    # optional Muon-group knobs: momentum: 0.95, nesterov: true, ns_steps: 5
```

Launch (uses the SISR29 env):

```bash
bash scripts/train_promod_glider_x2_muon.sh                      # 311 config
bash scripts/train_promod_glider_x2_muon.sh options/train/other.yml
MASTER_PORT=4322 bash scripts/train_promod_glider_x2_muon.sh     # alongside another job
```

The training log shows two lr values per line — Muon group first, AdamW group
second; both follow the same warmup/MultiStepLR schedule.

## Verification performed

1. **Smoke test** (single process, GPU): model forward/backward, `opt.step()`,
   `zero_grad()`, `state_dict` round-trip — passed.
2. **Live DDP training** (2×A100, torchrun): 300 iters of config 311; loss
   2.95e-1 → 9.7e-2, both lr groups warming up in sync, no errors.

## Caveats

- **Do not mix with the SISR env**: config `type: Muon` under torch 2.5 raises
  an informative ImportError by design.
- Pod restarts wipe container-layer installs, but SISR29 lives on the PVC and
  survives. tmux is installed in the conda **base** env for the same reason
  (`$CONDA/bin/tmux`).
- The PVC is nearly full (~43GB free before this env; ~15GB consumed by it).
- Both launch scripts pass `--auto_resume`, so a run killed by a pod restart
  resumes from its last saved training state on relaunch.
