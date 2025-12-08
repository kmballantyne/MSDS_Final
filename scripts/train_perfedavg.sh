#!/bin/bash
#SBATCH --job-name=perfedavg_training
#SBATCH --output=/net/scratch/kmballantyne/msds_final/logs/%j.%N.stdout
#SBATCH --error=/net/scratch/kmballantyne/msds_final/logs/%j.%N.stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kballantyne@cs.uchicago.edu
#SBATCH --chdir=/net/scratch/kmballantyne/msds_final/scripts
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Use scratch for temp files
export TMPDIR=/net/scratch/kmballantyne/tmp

echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "TMPDIR: $TMPDIR"

# Activate your working FL environment
echo "Activating FL_Project virtual env..."

# Initialize conda from scratch location
source /net/scratch/kmballantyne/miniconda3/etc/profile.d/conda.sh
conda activate fl_project

# Sanity check: show which Python we're using
echo "Python executable:"
which python3

# GPU check/diagnostics with PyTorch
echo "Checking GPU availability and PyTorch setup..."
nvidia-smi
python - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
EOF

# Run PerFedAvg training
python -u main_perfedavg.py
