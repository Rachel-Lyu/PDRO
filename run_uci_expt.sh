#!/bin/bash
#SBATCH --job-name="uci_expt"
#SBATCH --output=./logs/uci_expt_%j.out
#SBATCH --error=./logs/uci_expt_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=debug

pop=uci
t2=mc_plugin
seed=26 # Single seed
penalty_lambda=0.0001
lr=0.001 # Single learning rate

SAVE_PATH="./results_v8/${t2}/uci_sex/tuning_single/"
mkdir -p "$SAVE_PATH"

sbatch gd_pops_v8_task.sh \
  --populations $pop $pop $pop \
  --budget 10 \
  --penalty-type Reciprocal_L1 \
  --penalty-lambda $penalty_lambda \
  --learning-rate $lr \
  --optimizer-type adam \
  --parameterization theta \
  --alpha-init random_2 \
  --num-epochs 5 \
  --patience 15 \
  --gradient-mode autograd \
  --t2-estimator-type $t2 \
  --N-grad-samples 25 \
  --estimator-type plugin \
  --base-model-type xgb \
  --objective-value-estimator if \
  --k-kernel 2000 \
  --scheduler-type CosineAnnealingLR \
  --scheduler-t-max 80 \
  --seed $seed \
  --save-path $SAVE_PATH \
  --verbose
