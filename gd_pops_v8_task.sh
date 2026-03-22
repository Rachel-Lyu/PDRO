#!/bin/bash
#SBATCH --job-name="v8_task_gd_pops"
#SBATCH --output=./logs/v8_task_gd_%j.out
#SBATCH --error=./logs/v8_task_gd_%j.err
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4  # Increased CPU count for CPU-only processing
#SBATCH --time=08:00:00
#SBATCH --partition=general
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv   # or your env name

# # Force PyTorch to use CPU only
# export CUDA_VISIBLE_DEVICES=""
# export PYTORCH_DEVICE="cpu"
# export VSS_USE_CPU_FALLBACK=1  # Your custom env variable if implemented
# # Set PyTorch to use all available CPUs
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SCRIPT_DIR=$(dirname "$0")/..
python3 "$SCRIPT_DIR/gd_pops_v8.py" "$@"