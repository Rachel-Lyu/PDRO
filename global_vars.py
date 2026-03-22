import math
# --- Global constants ---
CLAMP_MAX_ALPHA = 10.0 # Max value for alpha
CLAMP_MIN_ALPHA = 1e-2 # Min value for alpha (increased slightly for stability)
EPS = 1e-8       # Epsilon for division stability
FREEZE_THRESHOLD_ALPHA = 0.1 # Threshold for freezing alpha

# Corresponding thresholds/limits for theta = log(alpha)
# Ensure THETA_CLAMP_MIN is not -inf if CLAMP_MIN_ALPHA is very small
THETA_CLAMP_MAX = math.log(CLAMP_MAX_ALPHA)
THETA_CLAMP_MIN = math.log(CLAMP_MIN_ALPHA) if CLAMP_MIN_ALPHA > 0 else -float('inf')
THETA_FREEZE_THRESHOLD = math.log(FREEZE_THRESHOLD_ALPHA) if FREEZE_THRESHOLD_ALPHA > 0 else -float('inf')

N_FOLDS = 5

CPU_COUNT = 4