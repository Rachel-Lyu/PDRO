#!/bin/bash
#SBATCH --job-name="acs_expt"
#SBATCH --output=./logs/acs_expt_%j.out
#SBATCH --error=./logs/acs_expt_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

echo "Activating environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Experiment settings
SAVE_PATH="./results_v10_acs_single"
POPULATIONS=("acs")
ACS_DATA_FRACTION=0.1
M1=10
M=18
DATASET_SIZE=30000
NOISE_SCALE=0.0
CORR_STRENGTH=0.0
BUDGET=15
LEARNING_RATE=0.01 # Single learning rate
PENALTY_TYPE="Reciprocal_L1"
PENALTY_LAMBDA=0.0001
SEED=123 # Single seed

python3 gd_pops_v10.py \
  --populations ${POPULATIONS[@]} \
  --acs_data_fraction $ACS_DATA_FRACTION \
  --m1 $M1 \
  --m $M \
  --dataset_size $DATASET_SIZE \
  --noise_scale $NOISE_SCALE \
  --corr_strength $CORR_STRENGTH \
  --budget $BUDGET \
  --learning_rate $LEARNING_RATE \
  --penalty_type $PENALTY_TYPE \
  --penalty_lambda $PENALTY_LAMBDA \
  --optimizer_type adam \
  --parameterization alpha \
  --alpha_init random_5 \
  --num_epochs 2 \
  --patience 10 \
  --gradient_mode autograd \
  --objective_value_estimator if \
  --t2_estimator_type mc_plugin \
  --N_grad_samples 10 \
  --estimator_type plugin \
  --base_model_type xgb \
  --seed $SEED \
  --save_path $SAVE_PATH \
  --force_regenerate_data \
  --k_kernel 500