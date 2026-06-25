# ProMoD: Progressive Mixture-of-Depths for Efficient Single Image Super-Resolution

**ProMoD** extends [PFT (CVPR 2025)](https://arxiv.org/abs/2503.20337) with **Mixture-of-Depths (MoD)** routing, using PFT's progressive focused attention cascade as a zero-parameter importance signal to dynamically skip computation for less important tokens per layer.

> **Base**: Progressive Focused Transformer (PFT, CVPR 2025) — Wei Long, Xingyu Zhou, Leheng Zhang, Shuhang Gu  
> **Extension**: ProMoD — adds MoD routing with PFA-based importance accumulation

---

## Contents
1. [Architecture](#architecture)
2. [Environment](#environment)
3. [Data Preparation](#data-preparation)
4. [Training](#training)
5. [Testing](#testing)
6. [Gradient Accumulation Patch](#gradient-accumulation-patch)
7. [LANTA HPC Setup](#lanta-hpc-setup)
8. [Acknowledgements](#acknowledgements)
9. [Citation](#citation)

---

## Architecture

ProMoD stacks a capacity schedule on top of PFT's transformer layers. Each layer receives a capacity ratio `r ∈ [0.4, 1.0]`, and only the top-`k = ceil(r × N)` tokens (ranked by the accumulated PFA importance score) participate in attention and FFN. All other tokens skip via residual connection.

```
PFT layer i:
  importance[i] = importance[i-1] × decay + attn_score[i]   ← progressive accumulation
  active_mask   = top_k(importance[i], k=ceil(r×N))
  x = shortcut + attn(x) × active_mask
  x = x + ffn(x)  × active_mask
```

Capacity schedule (default, 24-layer ProMoD-light):
```
Layers 0-1  : r = 1.0  (warmup, full compute)
Layers 2-3  : r = 0.8
Layers 4-9  : r = 0.6
Layers 10-15: r = 0.5
Layers 16-23: r = 0.4
```

**Key files:**
- `basicsr/archs/promod_arch.py` — ProMoD architecture
- `basicsr/archs/pft_arch.py` — original PFT base
- `scripts/apply_grad_accum_patch.sh` — gradient accumulation patcher
- `scripts/preprocess_div2k.sh` — DIV2K → LMDB converter
- `scripts/setup_lanta.sh` — LANTA HPC one-shot setup
- `scripts/train_promod_light_x2.sh` — LANTA Slurm training job

---

## Environment

```bash
git clone https://github.com/windx987/ProMoD-SR.git
cd ProMoD-SR

conda create -n PFT python=3.9
conda activate PFT

pip install -r requirements.txt
python setup.py develop

# Build sparse matrix multiplication CUDA kernel (requires nvcc)
cd ops_smm
CUDA_HOME=/path/to/cuda python setup.py install
cd ..
```

---

## Data Preparation

### Training data (DIV2K)

Download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) HR and LR bicubic × 2/3/4 into `datasets/DIV2K/`, then convert to LMDB for fast I/O:

```bash
bash scripts/preprocess_div2k.sh .
```

This produces:
```
datasets/DIV2K/DIV2K_train_HR.lmdb
datasets/DIV2K/DIV2K_train_LR_bicubic_X2.lmdb
datasets/DIV2K/DIV2K_train_LR_bicubic_X3.lmdb
datasets/DIV2K/DIV2K_train_LR_bicubic_X4.lmdb
```

### Test data

Download [TestDataSR](https://drive.google.com/file/d/1_4Fy9emAcqdiBwVM6FvbJU50LCtaBoMt/view?usp=sharing) (Set5 / Set14 / BSD100 / Urban100 / Manga109) and place in `datasets/TestDataSR/`.

---

## Training

### ProMoD-light (LANTA 4 × A100)

```bash
sbatch scripts/train_promod_light_x2.sh
```

Config: `options/train/lanta_201_ProMoD_light_SRx2.yml`  
— 4 GPUs × batch 8 = effective batch 32, 500 K optimizer steps.

### ProMoD-light (single GPU with gradient accumulation)

Apply the gradient accumulation patch first (see [below](#gradient-accumulation-patch)), then:

```bash
python basicsr/train.py \
    -opt options/train/local_201_ProMoD_light_SRx2_gradaccum_test.yml \
    --launcher none
```

Config: `options/train/local_201_ProMoD_light_SRx2_gradaccum_test.yml`  
— 1 GPU × batch 2 × `accum_iters=4` = effective batch 8. `total_iter` in YAML is in **optimizer-step units**; the loop runs `total_iter × accum_iters` raw iterations automatically.

### PFT-light baseline (for comparison)

```bash
# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --use-env --nproc_per_node=4 --master_port=1145 \
    basicsr/train.py -opt options/train/101_PFT_light_SRx2_scratch.yml \
    --launcher pytorch
```

---

## Testing

```bash
# ProMoD
python basicsr/test.py -opt options/test/201_ProMoD_light_SRx2_scratch.yml

# PFT-light (baseline)
python basicsr/test.py -opt options/test/101_PFT_light_SRx2_scratch.yml
```

Download [pretrained models](https://drive.google.com/drive/folders/1ChkxVDghFWUtJydJKLp5yssrUfm0VWfg?usp=sharing) and put them in `experiments/pretrained_models/`.

---

## Gradient Accumulation Patch

`scripts/apply_grad_accum_patch.sh` patches three BasicSR files to add gradient accumulation support. It is **idempotent** (safe to re-run), creates a timestamped backup before touching any file, and validates syntax with `ast.parse` before writing.

### What gets patched

| File | Change |
|---|---|
| `basicsr/models/base_model.py` | `self.accum_iters = 1` default + `_should_update()` helper |
| `basicsr/models/sr_model.py` | New `optimize_parameters` with window logic, DDP `no_sync()`, optional grad clip |
| `basicsr/train.py` | `total_iter` / `warmup_iter` auto-scaled by `accum_iters`; LR scheduler guarded to update steps only |

### Apply

```bash
bash scripts/apply_grad_accum_patch.sh .
```

### YAML configuration

Add under `datasets.train` (not under `train:`):

```yaml
datasets:
  train:
    batch_size_per_gpu: 4
    accum_iters: 8          # effective batch = 4 × 8 × num_gpu
    use_grad_clip: false    # optional — set true + grad_clip_norm: 1.0 for stability
```

`total_iter` and `warmup_iter` under `train:` are interpreted as **optimizer steps**, identical to non-accumulation runs. The patch scales them to raw iterations internally, so configs are portable between single-GPU (with accum) and multi-GPU (without accum) runs.

### Undo

```bash
# Backup path is printed at the end of each run, e.g.:
cp .grad_accum_backup_YYYYMMDD_HHMMSS/base_model.py  basicsr/models/base_model.py
cp .grad_accum_backup_YYYYMMDD_HHMMSS/sr_model.py    basicsr/models/sr_model.py
cp .grad_accum_backup_YYYYMMDD_HHMMSS/train.py       basicsr/train.py
```

---

## LANTA HPC Setup

LANTA is NSTDA's supercomputer (Slurm, A100 nodes). Full guide: [LANTA.md](../LANTA.md)

### One-shot setup on login node

```bash
# After rsync-ing the repo and raw_data zips to LANTA:
bash scripts/setup_lanta.sh
```

Installs pip deps, runs `setup.py develop`, extracts DIV2K + TestDataSR zips, and verifies all data paths.

### Build CUDA kernel

```bash
sbatch scripts/build_smm.sh
```

### Submit training job

```bash
sbatch scripts/train_promod_light_x2.sh
```

---

## Acknowledgements

This code is built on [PFT-SR (CVPR 2025)](https://github.com/LabShuHangGU/PFT-SR), [BasicSR](https://github.com/XPixelGroup/BasicSR), and [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary.git).

---

## Citation

If you use ProMoD, please also cite the PFT paper:

```bibtex
@inproceedings{long2025progressive,
  title={Progressive Focused Transformer for Single Image Super-Resolution},
  author={Long, Wei and Zhou, Xingyu and Zhang, Leheng and Gu, Shuhang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2279--2288},
  year={2025}
}
```
