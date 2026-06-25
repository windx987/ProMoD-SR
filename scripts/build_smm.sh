#!/bin/bash
#SBATCH -p gpu-devel
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A zz992004
#SBATCH -J build_smm
#SBATCH -o /project/zz992000-zdevb/zz992004/ub086/research-sisr/PFT-SR/scripts/logs/build_smm_%j.out
set -e

ml purge
ml load Miniforge3/25.3.0-3 cuda/12.6 2>/dev/null || true

PROJ=/project/zz992000-zdevb/zz992004/ub086/research-sisr
PYTHON=~/.conda/envs/demo/bin/python
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc: $(which nvcc)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

cd $PROJ/PFT-SR/ops_smm
$PYTHON setup.py install

echo "smm_cuda built successfully."
$PYTHON -c "import smm_cuda; print('smm_cuda import OK')"
