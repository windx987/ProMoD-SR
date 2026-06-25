#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 5-00:00:00
#SBATCH -A zz992004
#SBATCH -J promod_x2
#SBATCH -o /project/zz992000-zdevb/zz992004/ub086/research-sisr/ProMoD-SR/scripts/logs/promod_x2_%j.out
set -e

ml purge
ml load Miniforge3/25.3.0-3 cuda/12.6 2>/dev/null || true

export CONDA_PREFIX=/home/ub086/.conda/envs/demo
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

PROJ=/project/zz992000-zdevb/zz992004/ub086/research-sisr
PYTHON=$CONDA_PREFIX/bin/python

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "GPUs:    $CUDA_VISIBLE_DEVICES"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| GPUs:', torch.cuda.device_count())"

cd $PROJ/ProMoD-SR

torchrun \
    --nproc_per_node=4 \
    --master_port=4321 \
    basicsr/train.py \
    -opt options/train/lanta_201_ProMoD_light_SRx2.yml \
    --launcher pytorch
