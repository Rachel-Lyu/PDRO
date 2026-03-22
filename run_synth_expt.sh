#!/bin/bash
#SBATCH --job-name="synth_bl5_expt"
#SBATCH --output=./logs/synth_bl5_expt_%j.out
#SBATCH --error=./logs/synth_bl5_expt_%j.err
#SBATCH --time=05:00:00 # Adjusted time based on master script
#SBATCH --mem=64G # Assuming similar memory requirements
#SBATCH --gres=gpu:1 # Assuming GPU is needed if task script uses it
#SBATCH --partition=debug # Or general, adjust as needed
#SBATCH --cpus-per-task=4 # Assuming similar CPU requirements

echo "Loading environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv   # or your env name

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Settings from gd_pops_v8_master_bl5.sh for baseline_failure_5
t2="mc_plugin"
seed=7 # Single seed
estimator="plugin"
population="baseline_failure_5" # Specific baseline failure
N_GRAD_SAMPLES=5
lr=0.05 # Single learning rate

# Adjusted SAVE_PATH for a single run
SAVE_PATH="./results_v8/${t2}/${population}/single_run_may24/"
mkdir -p "$SAVE_PATH" "./logs/" # Ensure logs dir from master script exists

echo "Running synthetic experiment for baseline_failure_5..."
bash gd_pops_v8_task.sh \
  --populations $population $population $population \
  --m1 4 \
  --m 15 \
  --dataset-size 12000 \
  --baseline-data-size 30000 \
  --noise-scale 0.1 \
  --corr-strength 0.1 \
  --num-epochs 100 \
  --budget 10 \
  --penalty-type Reciprocal_L1 \
  --penalty-lambda 0.001 \
  --learning-rate $lr \
  --optimizer-type sgd \
  --parameterization theta \
  --alpha-init random_1 \
  --patience 20 \
  --gradient-mode autograd \
  --t2-estimator-type $t2 \
  --N-grad-samples $N_GRAD_SAMPLES \
  --estimator-type $estimator \
  --base-model-type xgb \
  --objective-value-estimator if \
  --k-kernel 1000 \
  --scheduler-type CosineAnnealingLR \
  --scheduler-t-max 180 \
  --scheduler-min-lr 1e-6 \
  --seed $seed \
  --save-path "$SAVE_PATH" \
  --verbose \
  --param-freezing

echo "Synthetic experiment for baseline_failure_5 submitted/run."
