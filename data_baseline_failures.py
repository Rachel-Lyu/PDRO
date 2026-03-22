# data_baseline_failures.py
import numpy as np
from global_vars import EPS # Assuming EPS and N_FOLDS are in global_vars
from estimators import *
import torch

def standardize_data(X, Y):
    """Standardizes X (features) and Y (outcome). Returns standardized data and original means/stds."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    Y_mean = np.mean(Y)
    Y_std = np.std(Y)

    X_std[X_std < EPS] = EPS # Avoid division by zero
    if Y_std < EPS:
        Y_std = EPS

    X_stdized = (X - X_mean) / X_std
    Y_stdized = (Y - Y_mean) / Y_std

    return X_stdized, Y_stdized, X_mean, X_std, Y_mean, Y_std

def generate_baseline_failure_1_heterogeneous_importance(
    dataset_size: int = 15000,
    n_features: int = 20,
    noise_scale: float = 0.1,
    corr_strength: float = 0.5, # Slightly increased default
    seed: int = None
):
    """
    Scenario 1 v2: Sharper Heterogeneous Feature Importance.
    Pop A (Large, 60%): Y ~ 4*X0 + epsilon (X0 is key)
    Pop B (Medium, 30%): Y ~ 4*X1 + epsilon (X1 is key, X0 irrelevant)
    Pop C (Small & Critical, 10%): Y ~ 8*X2 + epsilon (X2 is key, X0,X1 irrelevant, higher noise, X2 correlated with noise)
    """
    if n_features < 3: # Need at least X0, X1, X2
        raise ValueError("Scenario 1 requires at least 3 features.")
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.60 * dataset_size)
    n_b = int(0.30 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A: Only X0 matters
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = 4.0 * X_pop_a[:, 0] + np.random.normal(0, noise_scale * 0.7, n_a)
    meaningful_indices_A = np.array([0])

    # Pop B: Only X1 matters
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    Y_all[idx_b_start:idx_b_end] = 4.0 * X_pop_b[:, 1] + np.random.normal(0, noise_scale * 0.7, n_b)
    meaningful_indices_B = np.array([1])

    # Pop C: Only X2 matters, higher noise, X2 correlated with other noise features (X3, X4)
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    if n_features > 3:
        X_pop_c[:, 3] = corr_strength * X_pop_c[:, 2] + (1 - corr_strength) * np.random.normal(0, 1, n_c)
    if n_features > 4:
        X_pop_c[:, 4] = corr_strength * X_pop_c[:, 2] + (1 - corr_strength) * np.random.normal(0, 1, n_c)
    Y_all[idx_c_start:] = 8.0 * X_pop_c[:, 2] + np.random.normal(0, noise_scale * 2.0, n_c) # Higher signal and noise
    meaningful_indices_C = np.array([2])

    pop_data = [
        {'pop_id': 'A', 'X_raw': X_pop_a, 'Y_raw': Y_all[:idx_a_end], 'meaningful_indices': meaningful_indices_A},
        {'pop_id': 'B', 'X_raw': X_pop_b, 'Y_raw': Y_all[idx_b_start:idx_b_end], 'meaningful_indices': meaningful_indices_B},
        {'pop_id': 'C', 'X_raw': X_pop_c, 'Y_raw': Y_all[idx_c_start:], 'meaningful_indices': meaningful_indices_C}
    ]
    return pop_data

def generate_baseline_failure_2_opposing_and_typed_effects(
    dataset_size: int = 15000,
    n_features: int = 5,
    noise_scale: float = 0.1,
    seed: int = None
):
    """
    Scenario 2 v2: Stronger Opposing Effects & Different Relationship Types.
    Pop A (45%): Y ~ 5*X0 + X1 + epsilon
    Pop B (45%): Y ~ -5*X0 + X1 + epsilon
    Pop C (10%): Y ~ 2*X0^2 + 3*X2 + epsilon (X0 has quadratic effect, X1 irrelevant, X2 unique)
    """
    if n_features < 3: # Need X0, X1, X2
        raise ValueError("Scenario 2 requires at least 3 features.")
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.45 * dataset_size)
    n_b = int(0.45 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = 5.0 * X_pop_a[:, 0] + X_pop_a[:, 1] + np.random.normal(0, noise_scale, n_a)
    meaningful_indices_A = np.array([0, 1])

    # Pop B
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    Y_all[idx_b_start:idx_b_end] = -5.0 * X_pop_b[:, 0] + X_pop_b[:, 1] + np.random.normal(0, noise_scale, n_b)
    meaningful_indices_B = np.array([0, 1])

    # Pop C: X0 is quadratic, X1 is irrelevant, X2 is new and linear
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    Y_all[idx_c_start:] = 2.0 * (X_pop_c[:, 0]**2) + 3.0 * X_pop_c[:, 2] + np.random.normal(0, noise_scale * 1.5, n_c)
    meaningful_indices_C = np.array([0, 2]) # X0 (due to X0^2) and X2

    pop_data = [
        {'pop_id': 'A', 'X_raw': X_pop_a, 'Y_raw': Y_all[:idx_a_end], 'meaningful_indices': meaningful_indices_A},
        {'pop_id': 'B', 'X_raw': X_pop_b, 'Y_raw': Y_all[idx_b_start:idx_b_end], 'meaningful_indices': meaningful_indices_B},
        {'pop_id': 'C', 'X_raw': X_pop_c, 'Y_raw': Y_all[idx_c_start:], 'meaningful_indices': meaningful_indices_C}
    ]
    return pop_data

def generate_baseline_failure_3_diverse_non_linearities(
    dataset_size: int = 15000,
    n_features: int = 5,
    noise_scale: float = 0.1,
    seed: int = None
):
    """
    Scenario 3 v2: Diverse Non-Linearities for the Same Feature.
    Pop A (40%): Y ~ 2*X0 + X1 (X0 linear)
    Pop B (40%): Y ~ 3*(X0-1)**2 + X1 (X0 cubic, shifted)
    Pop C (20%): Y ~ 4*sin(2*np.pi*X0) - X1 + X2^2 (X0 sinusoidal, X1 shared, X2 unique)
    """
    if n_features < 3: # Need X0, X1, X2
        raise ValueError("Scenario 3 requires at least 3 features.")
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.35 * dataset_size)
    n_b = int(0.35 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features) # X0 values typically in [-3, 3]
    Y_all = np.zeros(dataset_size)

    # Pop A: X0 linear
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = 2.0 * X_pop_a[:, 0] + X_pop_a[:, 1] + np.random.normal(0, noise_scale, n_a)
    meaningful_indices_A = np.array([0, 1])
    # Pop B: X0 cubic (and X1 still present)
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    Y_all[idx_b_start:idx_b_end] = 3.0 * ((X_pop_b[:, 0] - 1)**2) + X_pop_b[:, 1] + np.random.normal(0, noise_scale, n_b)
    meaningful_indices_B = np.array([0, 1])
    # Pop C: X0 sinusoidal, X1 shared, X2 unique
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    Y_all[idx_c_start:] = 4.0 * np.sin(2 * np.pi * X_pop_c[:, 0]) + X_pop_c[:, 1] + (X_pop_c[:, 2]**2) + np.random.normal(0, noise_scale, n_c)
    meaningful_indices_C = np.array([0, 1, 2])
    pop_data = [
        {'pop_id': 'A', 'X_raw': X_pop_a, 'Y_raw': Y_all[:idx_a_end], 'meaningful_indices': meaningful_indices_A},
        {'pop_id': 'B', 'X_raw': X_pop_b, 'Y_raw': Y_all[idx_b_start:idx_b_end], 'meaningful_indices': meaningful_indices_B},
        {'pop_id': 'C', 'X_raw': X_pop_c, 'Y_raw': Y_all[idx_c_start:], 'meaningful_indices': meaningful_indices_C}
    ]
    return pop_data

def generate_baseline_failure_4(n_samples=1000, n_features=50, n_pops=3, 
                                noise_scale=0.1, 
                                 seed=42):
    np.random.seed(seed)
    pop_data = []
    
    # Define features that affect the mean (signal features)
    signal_features = np.random.choice(range(n_features), size=5, replace=False)
    
    # Define features that affect the variance (uncertainty features)
    remaining = [f for f in range(n_features) if f not in signal_features]
    variance_features = np.random.choice(remaining, size=5, replace=False)
    
    for pop_idx in range(n_pops):
        X_raw = np.random.randn(n_samples, n_features)
        
        # Generate signal component
        signal = np.zeros(n_samples)
        for feat in signal_features:
            signal += X_raw[:, feat] * np.random.uniform(0.5, 1.0)
        
        # Generate heteroskedastic noise component based on variance features
        log_variance = np.zeros(n_samples)
        for feat in variance_features:
            log_variance += X_raw[:, feat] * np.random.uniform(0.3, 0.7)
        
        # Normalize log variance for stability
        log_variance = log_variance - np.mean(log_variance)
        
        # Generate heteroskedastic noise
        noise_scale = np.exp(log_variance)
        noise = noise_scale * np.random.randn(n_samples)
        
        # Final target
        Y_raw = signal + noise
        
        # Important features are both signal and variance features
        # A method that minimizes Bayes predictor variance should identify both types
        all_important = np.concatenate([signal_features, variance_features])
        
        pop_data.append({
            'pop_id': f'Pop_{pop_idx+1}',
            'X_raw': X_raw,
            'Y_raw': Y_raw,
            'meaningful_indices': all_important.tolist(),
            'signal_features': signal_features.tolist(),
            'variance_features': variance_features.tolist()
        })
    
    return pop_data

def generate_baseline_failure_5(dataset_size: int = 15000, 
                                n_features=100, noise_scale: float = 0.1, n_pops=3, seed=42):  
    np.random.seed(seed)
    pop_data = []
    n_samples = dataset_size // n_pops
    # Create population-specific important features
    pop_important_features = [
        np.random.choice(range(n_features), size=5, replace=False) 
        for _ in range(n_pops)
    ]
    
    for pop_idx in range(n_pops):
        X_raw = np.random.randn(n_samples, n_features)
        
        # Make important features have non-linear effects
        important_idx = pop_important_features[pop_idx]
        Y_raw = np.zeros(n_samples)
        
        # Add non-linear transformations of important features
        for i, feat_idx in enumerate(important_idx):
            if i % 3 == 0:  # Squared terms
                Y_raw += (X_raw[:, feat_idx]**2 - 1) * np.random.uniform(0.5, 1.5)
            elif i % 3 == 1:  # Exponential terms
                Y_raw += np.exp(X_raw[:, feat_idx] * 0.5) * 0.2 * np.random.uniform(-0.5, 0.5)
            else:  # Threshold effects
                Y_raw += (X_raw[:, feat_idx] > 0.5) * np.random.uniform(0.5, 1.5)
        
        # Create fake correlations between unimportant features
        # This will confuse Lasso which looks for linear relationships
        for i in range(0, n_features, 10):
            if i not in important_idx and i+1 < n_features and i+1 not in important_idx:
                X_raw[:, i+1] = 0.05 * X_raw[:, i] + 0.95 * np.random.randn(n_samples)
                
        # Add some noise
        Y_raw += np.random.normal(0, noise_scale, n_samples)
        
        pop_data.append({
            'pop_id': f'Pop_{pop_idx+1}',
            'X_raw': X_raw,
            'Y_raw': Y_raw,
            'meaningful_indices': important_idx.tolist()
        })
    
    return pop_data

def generate_baseline_failure_6_sharp_interactions_conditionals(
    dataset_size: int = 10000,
    n_features: int = 20, # Needs at least 7 for X0-X6
    noise_scale: float = 0.1,
    seed: int = None
):
    """
    Scenario 6 v2: Sharper Interactions & Conditional Effects.
    Pop A (50%): Y ~ 3*X0 + 2*X1 + epsilon (simple linear)
    Pop B (30%): Y ~ 5*(X2*X3) + 0.1*X0 + epsilon (Interaction is dominant for X2,X3; X0 minor)
    Pop C (20%): Y ~ 4*X4*(X4 > 0.5) - 2*X4*(X4 <= 0.5) + 0.1*X0 + epsilon (Sharp conditional for X4; X0 minor)
    """
    if n_features < 5: # Needs X0, X1, X2, X3, X4
        raise ValueError("Scenario 6 requires at least 5 features.")
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.50 * dataset_size)
    n_b = int(0.30 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = 3.0 * X_pop_a[:, 0] + 2.0 * X_pop_a[:, 1] + np.random.normal(0, noise_scale, n_a)
    meaningful_indices_A = np.array([0, 1])

    # Pop B: Interaction X2*X3 is key. X2, X3 have no main effects. X0 is minor.
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    Y_all[idx_b_start:idx_b_end] = 5.0 * (X_pop_b[:, 2] * X_pop_b[:, 3]) + \
                                 0.1 * X_pop_b[:, 0] + \
                                 np.random.normal(0, noise_scale * 1.2, n_b) # Slightly higher noise
    meaningful_indices_B = np.array([0, 2, 3]) # X0, X2, X3

    # Pop C: Sharp conditional for X4. X0 is minor.
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    y_c = np.zeros(n_c)
    y_c[X_pop_c[:, 4] > 0.5] = 4.0 * X_pop_c[X_pop_c[:, 4] > 0.5, 4]
    y_c[X_pop_c[:, 4] <= 0.5] = -2.0 * X_pop_c[X_pop_c[:, 4] <= 0.5, 4]
    Y_all[idx_c_start:] = y_c + 0.1 * X_pop_c[:, 0] + np.random.normal(0, noise_scale * 1.2, n_c)
    meaningful_indices_C = np.array([0, 4]) # X0, X4

    pop_data = [
        {'pop_id': 'A', 'X_raw': X_pop_a, 'Y_raw': Y_all[:idx_a_end], 'meaningful_indices': meaningful_indices_A},
        {'pop_id': 'B', 'X_raw': X_pop_b, 'Y_raw': Y_all[idx_b_start:idx_b_end], 'meaningful_indices': meaningful_indices_B},
        {'pop_id': 'C', 'X_raw': X_pop_c, 'Y_raw': Y_all[idx_c_start:], 'meaningful_indices': meaningful_indices_C}
    ]
    return pop_data

def generate_baseline_failure_7_disjoint_correlated_sets(dataset_size=10000, n_features=20, noise_scale=0.1, seed=None):
    """
    Cleaner test case:
    - Pop A (25%): Y strongly depends on features 0-4, weakly negative on 10-14
    - Pop B (25%): Y strongly depends on features 5-9, weakly negative on 10-14
    - Pop C (25%): Y strongly depends on features 0-9, weakly negative on 10-14
    - Pop D (25%): Y weakly depends on features 0-9, strongly positive on 10-14
    
    All other features are pure noise. This creates a clear tension where standard
    methods will likely pick features 0-9 but miss the importance of 10-14 for pop D.
    """
    # [Initialization code...]
    if n_features < 6: # Minimum for X0-X5
        raise ValueError("Scenario 7 needs at least 6 features for the distinct true sets.")
    if n_features < 16:
        print(f"Warning: Scenario 7 ideally uses n_features >= 16 for full correlation setup. Current: {n_features}. Correlations might be limited.")

    if seed is not None:
        np.random.seed(seed)

    n_a = dataset_size // 4
    n_b = dataset_size // 4
    n_c = dataset_size // 4
    n_d = dataset_size - n_a - n_b - n_c

    # X_all = np.random.randn(dataset_size, n_features)
    # Y_all = np.zeros(dataset_size)
    # for populations A, B, C; the first 8 features are higher valued than the next 8 features; the last 4 features are noise
    # for population D; the first 8 features are lower valued than the next 8 features; the last 4 features are noise
    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A: 
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    # Population A: strong 0-4, weak negative 10-14
    Y_all[:idx_a_end] = 5.0 * X_pop_a[:, 0] + 5.0 * X_pop_a[:, 1] + 5.0 * X_pop_a[:, 2] + \
                        5.0 * X_pop_a[:, 3] + 5.0 * X_pop_a[:, 4] - \
                        0.5 * X_pop_a[:, 10] - 0.5 * X_pop_a[:, 11] - 0.5 * X_pop_a[:, 12] - \
                        0.5 * X_pop_a[:, 13] - 0.5 * X_pop_a[:, 14]
    meaningful_indices_A = np.array([0, 1, 2, 3, 4])
    
    # Population B: strong 5-9, weak negative 10-14
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    Y_all[idx_b_start:idx_b_end] = 5.0 * X_pop_b[:, 5] + 5.0 * X_pop_b[:, 6] + 5.0 * X_pop_b[:, 7] + \
                                   5.0 * X_pop_b[:, 8] + 5.0 * X_pop_b[:, 9] - \
                                   0.5 * X_pop_b[:, 10] - 0.5 * X_pop_b[:, 11] - 0.5 * X_pop_b[:, 12] - \
                                   0.5 * X_pop_b[:, 13] - 0.5 * X_pop_b[:, 14]
    meaningful_indices_B = np.array([5, 6, 7, 8, 9])
    
    # Population C: strong 0-9
    idx_c_start = n_a + n_b
    idx_c_end = n_a + n_b + n_c
    X_pop_c = X_all[idx_c_start:idx_c_end, :]
    Y_all[idx_c_start:idx_c_end] = 4.0 * X_pop_c[:, 0] + 4.0 * X_pop_c[:, 1] + 4.0 * X_pop_c[:, 2] + \
                                   4.0 * X_pop_c[:, 3] + 4.0 * X_pop_c[:, 4] + 4.0 * X_pop_c[:, 5] + \
                                   4.0 * X_pop_c[:, 6] + 4.0 * X_pop_c[:, 7] + 4.0 * X_pop_c[:, 8] + \
                                   4.0 * X_pop_c[:, 9] - \
                                   0.5 * X_pop_c[:, 10] - 0.5 * X_pop_c[:, 11] - 0.5 * X_pop_c[:, 12] - \
                                   0.5 * X_pop_c[:, 13] - 0.5 * X_pop_c[:, 14]
    meaningful_indices_C = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Population D: weak 0-9, strong positive 10-14
    idx_d_start = n_a + n_b + n_c
    X_pop_d = X_all[idx_d_start:, :]
    # Population D: weak 0-9, strong positive 10-14
    Y_all[idx_d_start:] = 0.5 * X_pop_d[:, 0] + 0.5 * X_pop_d[:, 1] + 0.5 * X_pop_d[:, 2] + \
                         0.5 * X_pop_d[:, 3] + 0.5 * X_pop_d[:, 4] + 0.5 * X_pop_d[:, 5] + \
                         0.5 * X_pop_d[:, 6] + 0.5 * X_pop_d[:, 7] + 0.5 * X_pop_d[:, 8] + \
                         0.5 * X_pop_d[:, 9] + \
                         5.0 * X_pop_d[:, 10] + 5.0 * X_pop_d[:, 11] + 5.0 * X_pop_d[:, 12] + \
                         5.0 * X_pop_d[:, 13] + 5.0 * X_pop_d[:, 14]
    meaningful_indices_D = np.array([10, 11, 12, 13, 14])
    
    # Add proper noise features
    for i in range(15, n_features):
        for pop_data in [X_pop_a, X_pop_b, X_pop_c, X_pop_d]:
            pop_data[:, i] = np.random.randn(pop_data.shape[0]) * noise_scale
    # Add noise to Y
    Y_all += np.random.normal(0, noise_scale, dataset_size)
    # Add noise to Y for each population
    Y_all[:idx_a_end] += np.random.normal(0, noise_scale, n_a)
    Y_all[idx_b_start:idx_b_end] += np.random.normal(0, noise_scale, n_b)
    Y_all[idx_c_start:idx_c_end] += np.random.normal(0, noise_scale, n_c)
    Y_all[idx_d_start:] += np.random.normal(0, noise_scale, n_d)
    
    pop_data = [
        {'pop_id': 'A', 'X_raw': X_pop_a, 'Y_raw': Y_all[:idx_a_end], 'meaningful_indices': meaningful_indices_A},
        {'pop_id': 'B', 'X_raw': X_pop_b, 'Y_raw': Y_all[idx_b_start:idx_b_end], 'meaningful_indices': meaningful_indices_B},
        {'pop_id': 'C', 'X_raw': X_pop_c, 'Y_raw': Y_all[idx_c_start:idx_c_end], 'meaningful_indices': meaningful_indices_C},
        {'pop_id': 'D', 'X_raw': X_pop_d, 'Y_raw': Y_all[idx_d_start:], 'meaningful_indices': meaningful_indices_D}
    ]
    return pop_data

def generate_baseline_failure_8(
    dataset_size: int = 10000,
    n_features: int = 20, # Needs enough for distinct true features and their correlates
    noise_scale: float = 0.1,
    corr_strength: float = 0.6, # Stronger correlation
    seed: int = None
):
    """
    Scenario 8: 4 populations. 3 of them agree on all of the same 8 features, and 2 unique ones. With 8 others, they have a negative relation.
    One of the agrees weakly on the same 8 and very strongly on the 8 with which the others have a negative relation (but this one has a strong positive relation).
    Pop A (25%): Y ~ 3*X0 + 2*X1 + 4*X2 + 5*X3 + 6*X4 + 7*X5 + 8*X6^2 + 9*X7 - 3*X8 - 2*X9 - 4*X10 - 5*X11 - 6*X12 - 7*X13 - 8*X14 - 9*X15
    Pop B (25%): Y ~ 3*X0 + 2*X1 + 4*X2 + 5*X3 + 6*X4 + 7*X5 + 8*X6^2 + 9*X7 - 3*X8 - 2*X9 - 4*X10 - 5*X11 - 6*X12 - 7*X13 - 8*X14 - 9*X15
    Pop C (25%): Y ~ 3*X0 + 2*X1 + 4*X2 + 5*X3 + 6*X4 + 7*X5 + 8*X6^2 + 9*X7 - 3*X8 - 2*X9 - 4*X10 - 5*X11 - 6*X12 - 7*X13 - 8*X14 - 9*X15
    Pop D (25%): Y ~ 0.01*X0 + 0.02*X1 + 0.04*X2 + 0.05*X3 + 0.06*X4 + 0.07*X5 + 0.08*X6^2 + 0.09*X7 + 3*X8 + 2*X9 + 4*X10 + 5*X11 + 6*X12 + 7*X13 + 8*X14 + 9*X15
    Requires n_features >= 16 for this setup.
    """
    if n_features < 6: # Minimum for X0-X5
        raise ValueError("Scenario 7 needs at least 6 features for the distinct true sets.")
    if n_features < 16:
        print(f"Warning: Scenario 7 ideally uses n_features >= 16 for full correlation setup. Current: {n_features}. Correlations might be limited.")

    if seed is not None:
        np.random.seed(seed)

    n_a = dataset_size // 4
    n_b = dataset_size // 4
    n_c = dataset_size // 4
    n_d = dataset_size - n_a - n_b - n_c

    # X_all = np.random.randn(dataset_size, n_features)
    # Y_all = np.zeros(dataset_size)
    # for populations A, B, C; the first 8 features are higher valued than the next 8 features; the last 4 features are noise
    # for population D; the first 8 features are lower valued than the next 8 features; the last 4 features are noise
    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A: 
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = 3.0 * X_pop_a[:, 0] + 2.0 * X_pop_a[:, 1] + 4.0 * X_pop_a[:, 2] + 5.0 * X_pop_a[:, 3] + \
                        6.0 * X_pop_a[:, 4] + 7.0 * X_pop_a[:, 5] + \
                        8.0 * (X_pop_a[:, 6]) + 9.0 * X_pop_a[:, 7] - \
                        2.0 * X_pop_a[:, 8] - 2.0 * X_pop_a[:, 9] - \
                        2.0 * X_pop_a[:, 10] - 2.0 * X_pop_a[:, 11] - \
                        2.0 * X_pop_a[:, 12] - 2.0 * X_pop_a[:, 13] - \
                        2.0 * X_pop_a[:, 14] - 2.0 * X_pop_a[:, 15]
    meaningful_indices_A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Pop B: 
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    Y_all[idx_b_start:idx_b_end] = 2.0 * X_pop_b[:, 0] + 3.0 * X_pop_b[:, 1] + 6.0 * X_pop_b[:, 2] + 7.0 * X_pop_b[:, 3] + \
                                    4.0 * X_pop_b[:, 4] + 5.0 * X_pop_b[:, 5] + \
                                    8.0 * (X_pop_b[:, 6]) + 9.0 * X_pop_b[:, 7] - \
                                    2.0 * X_pop_b[:, 8] - 2.0 * X_pop_b[:, 9] - \
                                    2.0 * X_pop_b[:, 10] - 2.0 * X_pop_b[:, 11] - \
                                    2.0 * X_pop_b[:, 12] - 2.0 * X_pop_b[:, 13] - \
                                    2.0 * X_pop_b[:, 14] - 2.0 * X_pop_b[:, 15]
    meaningful_indices_B = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # Pop C: 
    idx_c_start = n_a + n_b
    idx_c_end = n_a + n_b + n_c
    X_pop_c = X_all[idx_c_start:idx_c_end, :]
    Y_all[idx_c_start:idx_c_end] = 6.0 * X_pop_c[:, 0] + 2.0 * X_pop_c[:, 1] + 5.0 * X_pop_c[:, 2] + 4.0 * X_pop_c[:, 3] + \
                        3.0 * X_pop_c[:, 4] + 2.0 * X_pop_c[:, 5] + \
                        8.0 * (X_pop_c[:, 6]) + 9.0 * X_pop_c[:, 7] - \
                        2.0 * X_pop_c[:, 8] - 2.0 * X_pop_c[:, 9] - \
                        2.0 * X_pop_c[:, 10] - 2.0 * X_pop_c[:, 11] - \
                        2.0 * X_pop_c[:, 12] - 2.0 * X_pop_c[:, 13] - \
                        2.0 * X_pop_c[:, 14] - 2.0 * X_pop_c[:, 15]
    meaningful_indices_C = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Pop D: 
    idx_d_start = n_a + n_b + n_c
    X_pop_d = X_all[idx_d_start:, :]
    Y_all[idx_d_start:] = 0.01 * X_pop_d[:, 0] + 0.02 * X_pop_d[:, 1] + 0.04 * X_pop_d[:, 2] + 0.05 * X_pop_d[:, 3] + 0.06 * X_pop_d[:, 4] + 0.07 * X_pop_d[:, 5] + 0.08 * (X_pop_d[:, 6]**2) + 0.09 * X_pop_d[:, 7]
    Y_all[idx_d_start:] = Y_all[idx_d_start:] + 6.0 * X_pop_d[:, 8] + 2.0 * X_pop_d[:, 9] + 6.0 * X_pop_d[:, 10] + 5.0 * X_pop_d[:, 11] + 5.0 * X_pop_d[:, 12] + 8.0 * X_pop_d[:, 13] + 5.0 * X_pop_d[:, 14] + 5.0 * X_pop_d[:, 15]
    meaningful_indices_D = np.array([8, 9, 10, 11, 12, 13, 14, 15])

    # Append random noise to the last 4 features
    for i in range(16, n_features):
        X_pop_a[:, i] = np.random.randn(n_a)
        X_pop_b[:, i] = np.random.randn(n_b)
        X_pop_c[:, i] = np.random.randn(n_c)
        X_pop_d[:, i] = np.random.randn(n_d)

    pop_data = [
        {'pop_id': 'A', 'X_raw': X_pop_a, 'Y_raw': Y_all[:idx_a_end], 'meaningful_indices': meaningful_indices_A},
        {'pop_id': 'B', 'X_raw': X_pop_b, 'Y_raw': Y_all[idx_b_start:idx_b_end], 'meaningful_indices': meaningful_indices_B},
        {'pop_id': 'C', 'X_raw': X_pop_c, 'Y_raw': Y_all[idx_c_start:idx_c_end], 'meaningful_indices': meaningful_indices_C},
        {'pop_id': 'D', 'X_raw': X_pop_d, 'Y_raw': Y_all[idx_d_start:], 'meaningful_indices': meaningful_indices_D}
    ]
    return pop_data

def generate_baseline_failure_9(
    dataset_size: int = 10000,
    n_features: int = 25,
    noise_scale: float = 0.1,
    seed: int = None
):
    """
    Scenario 9: Complex non-linear interactions that defeat DRO LASSO
    - Pop A: Primarily interaction effects (X0*X1, X2*X3)
    - Pop B: Threshold effects with different directions above/below threshold
    - Pop C: Complex conditional logic with exclusive OR (XOR) patterns
    - Pop D: Non-linear transformations (sine, exponential)
    - Added correlations between features to create misleading linear signals
    """
    if n_features < 20:
        raise ValueError("Scenario 9 requires at least 20 features")
    if seed is not None:
        np.random.seed(seed)

    n_a = dataset_size // 4
    n_b = dataset_size // 4
    n_c = dataset_size // 4
    n_d = dataset_size - n_a - n_b - n_c

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Add misleading correlations to confuse LASSO
    # Correlation structure: X17-X19 correlate with outcome but aren't causal
    for i in range(dataset_size):
        # Cross-population correlations to confuse global models
        X_all[i, 17] = 0.7 * (X_all[i, 0] + X_all[i, 1]) + 0.3 * np.random.randn()
        X_all[i, 18] = 0.7 * (X_all[i, 5] + X_all[i, 6]) + 0.3 * np.random.randn()
        X_all[i, 19] = 0.7 * (X_all[i, 10] + X_all[i, 11]) + 0.3 * np.random.randn()

    # Pop A: Pure interaction effects, no main effects
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    interaction1 = X_pop_a[:, 0] * X_pop_a[:, 1]  # Interaction term
    interaction2 = X_pop_a[:, 2] * X_pop_a[:, 3]  # Interaction term
    interaction3 = X_pop_a[:, 4] * (X_pop_a[:, 4] > 0)  # Conditional effect
    Y_all[:idx_a_end] = 5.0 * interaction1 + 3.0 * interaction2 + 4.0 * interaction3 + np.random.normal(0, noise_scale, n_a)
    meaningful_indices_A = np.array([0, 1, 2, 3, 4])

    # Pop B: Threshold effects (different directions above/below threshold)
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    threshold_effect1 = np.zeros(n_b)
    threshold_effect1[X_pop_b[:, 5] > 0.5] = 4.0 * X_pop_b[X_pop_b[:, 5] > 0.5, 5]
    threshold_effect1[X_pop_b[:, 5] <= 0.5] = -3.0 * X_pop_b[X_pop_b[:, 5] <= 0.5, 5]
    
    threshold_effect2 = np.zeros(n_b)
    threshold_effect2[X_pop_b[:, 6] > 0] = 3.0 * X_pop_b[X_pop_b[:, 6] > 0, 7]
    threshold_effect2[X_pop_b[:, 6] <= 0] = -3.0 * X_pop_b[X_pop_b[:, 6] <= 0, 8]
    
    Y_all[idx_b_start:idx_b_end] = threshold_effect1 + threshold_effect2 + 2.0 * X_pop_b[:, 9] + np.random.normal(0, noise_scale, n_b)
    meaningful_indices_B = np.array([5, 6, 7, 8, 9])

    # Pop C: Complex conditional logic (XOR-like patterns)
    idx_c_start = n_a + n_b
    idx_c_end = n_a + n_b + n_c
    X_pop_c = X_all[idx_c_start:idx_c_end, :]
    # XOR-like: (X10>0 AND X11<0) OR (X10<0 AND X11>0)
    xor_effect = ((X_pop_c[:, 10] > 0) & (X_pop_c[:, 11] < 0)) | ((X_pop_c[:, 10] < 0) & (X_pop_c[:, 11] > 0))
    xor_effect = xor_effect.astype(float) * 5.0
    
    # Another complex pattern
    complex_pattern = np.zeros(n_c)
    complex_pattern[(X_pop_c[:, 12] > 0.5) & (X_pop_c[:, 13] > 0.5)] = 4.0
    complex_pattern[(X_pop_c[:, 12] < -0.5) & (X_pop_c[:, 13] < -0.5)] = -4.0
    
    Y_all[idx_c_start:idx_c_end] = xor_effect + complex_pattern + np.random.normal(0, noise_scale, n_c)
    meaningful_indices_C = np.array([10, 11, 12, 13])

    # Pop D: Non-linear transformations
    idx_d_start = n_a + n_b + n_c
    X_pop_d = X_all[idx_d_start:, :]
    sine_effect = 5.0 * np.sin(3.0 * X_pop_d[:, 14])  # Sinusoidal effect
    exp_effect = 4.0 * np.exp(-np.abs(X_pop_d[:, 15]))  # Exponential effect
    cubic_effect = 2.0 * np.power(X_pop_d[:, 16], 3)  # Cubic effect
    
    Y_all[idx_d_start:] = sine_effect + exp_effect + cubic_effect + np.random.normal(0, noise_scale * 1.5, n_d)
    meaningful_indices_D = np.array([14, 15, 16])

    # Add heavy-tailed noise to a random subset to create outlier challenges
    outlier_indices = np.random.choice(dataset_size, size=int(dataset_size * 0.05), replace=False)
    Y_all[outlier_indices] += np.random.standard_t(df=3, size=len(outlier_indices)) * noise_scale * 3

    pop_data = [
        {'pop_id': 'A', 'X_raw': X_pop_a, 'Y_raw': Y_all[:idx_a_end], 'meaningful_indices': meaningful_indices_A},
        {'pop_id': 'B', 'X_raw': X_pop_b, 'Y_raw': Y_all[idx_b_start:idx_b_end], 'meaningful_indices': meaningful_indices_B},
        {'pop_id': 'C', 'X_raw': X_pop_c, 'Y_raw': Y_all[idx_c_start:idx_c_end], 'meaningful_indices': meaningful_indices_C},
        {'pop_id': 'D', 'X_raw': X_pop_d, 'Y_raw': Y_all[idx_d_start:], 'meaningful_indices': meaningful_indices_D}
    ]
    return pop_data

# Add other v2 scenarios here if needed, following the pattern...
# generate_baseline_failure_4...
# generate_baseline_failure_5...
# generate_baseline_failure_8...
# generate_baseline_failure_9...
# generate_baseline_failure_10...

def get_pop_data_baseline_failures( # Renamed main getter function
    pop_configs: list, # Expects dataset_type like 'baseline_failure_1'
    dataset_size: int = 15000,
    n_features: int = 20,
    noise_scale: float = 0.1,
    corr_strength: float = 0.4, # Default, individual scenarios might override
    estimator_type: str = 'plugin',
    device: str = 'cpu',
    base_model_type: str = 'rf',
    test_val_fraction=0.6, # Fraction for TEST/VAL, so (1-test_val_fraction) is for TRAIN
    seed: int = None
):
    """
    Generates data for multiple populations using the v2 baseline failure scenarios.
    These scenarios are designed to be more challenging for pooled models.
    """
    if not pop_configs:
        raise ValueError("pop_configs list cannot be empty.")
    
    # Assuming all pop_configs in a single call are for the same base scenario type (e.g., all are 'baseline_failure_1')
    # The specific 'pop_id' within pop_configs is mostly for external tracking if needed,
    # as the generation functions internally define sub-populations (A, B, C).
    baseline_type = pop_configs[0]['dataset_type'] # e.g., "baseline_failure_1"

    pop_data_generated = [] # This will be a list of dicts, one for each sub-population (A, B, C)

    if baseline_type == 'baseline_failure_1':
        pop_data_generated = generate_baseline_failure_1_heterogeneous_importance(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale,
            corr_strength=corr_strength, seed=seed
        )
    elif baseline_type == 'baseline_failure_2':
        pop_data_generated = generate_baseline_failure_2_opposing_and_typed_effects(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale, seed=seed
        )
    elif baseline_type == 'baseline_failure_3':
        pop_data_generated = generate_baseline_failure_3_diverse_non_linearities(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale, seed=seed
        )
    elif baseline_type == 'baseline_failure_4':
        pop_data_generated = generate_baseline_failure_4(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale, seed=seed
        )
    elif baseline_type == 'baseline_failure_5':
        pop_data_generated = generate_baseline_failure_5(
            dataset_size=dataset_size, n_features=n_features, seed=seed
        )
    elif baseline_type == 'baseline_failure_6':
        pop_data_generated = generate_baseline_failure_6_sharp_interactions_conditionals(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale, seed=seed
        )
    elif baseline_type == 'baseline_failure_7':
        pop_data_generated = generate_baseline_failure_7_disjoint_correlated_sets(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale,
            corr_strength=corr_strength, seed=seed
        )
    elif baseline_type == 'baseline_failure_8':
        pop_data_generated = generate_baseline_failure_8(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale,
            corr_strength=corr_strength, seed=seed
        )
    # Add other elif blocks for other v2 scenarios if you implement them
    # elif baseline_type == 'baseline_failure_4': ...
    elif baseline_type == 'baseline_failure_9':
        pop_data_generated = generate_baseline_failure_9(
            dataset_size=dataset_size, n_features=n_features, noise_scale=noise_scale, seed=seed
        )
    else:
        raise ValueError(f"Unknown V2 dataset type: {baseline_type}")

    final_pop_data_train = []
    final_pop_data_test_val = []

    # The pop_data_generated is already a list of sub-population dicts (A, B, C)
    for j, sub_pop_dict in enumerate(pop_data_generated):
        current_seed = seed + j if seed is not None else None # Seed for this sub-population's processing
        rng = np.random.default_rng(current_seed)

        X_raw_subpop = sub_pop_dict['X_raw']
        Y_raw_subpop = sub_pop_dict['Y_raw']
        
        n_total_subpop = X_raw_subpop.shape[0]
        n_train = int(n_total_subpop * (1 - test_val_fraction))

        shuffled_indices = rng.permutation(n_total_subpop)
        train_indices = shuffled_indices[:n_train]
        test_val_indices = shuffled_indices[n_train:]

        X_train_raw = X_raw_subpop[train_indices]
        Y_train_raw = Y_raw_subpop[train_indices]
        X_test_val_raw = X_raw_subpop[test_val_indices]
        Y_test_val_raw = Y_raw_subpop[test_val_indices]

        print(f"Sub-population {sub_pop_dict['pop_id']}: Precomputing E[Y|X] ({estimator_type}/{base_model_type}) for training data...")
        try:
            # N_FOLDS should be defined, e.g., from global_vars or passed as an argument
            from global_vars import N_FOLDS # Make sure N_FOLDS is accessible
            if estimator_type == "plugin":
                E_Yx_orig_np_train = plugin_estimator_conditional_mean(X_train_raw, Y_train_raw, base_model_type, n_folds=N_FOLDS, seed=current_seed)
            elif estimator_type == "if":
                E_Yx_orig_np_train = IF_estimator_conditional_mean(X_train_raw, Y_train_raw, base_model_type, n_folds=N_FOLDS, seed=current_seed)
            else:
                raise ValueError("estimator_type must be 'plugin' or 'if'")
        except Exception as e:
            print(f"ERROR: Failed to precompute E[Y|X] for sub-pop {sub_pop_dict['pop_id']}: {e}")
            continue

        X_train_std, Y_train_std, X_train_mean, X_train_sd, Y_train_mean, Y_train_sd = standardize_data(X_train_raw, Y_train_raw)
        E_Yx_std_np_train = (E_Yx_orig_np_train - Y_train_mean) / Y_train_sd
        term1_std_train = np.mean(E_Yx_std_np_train ** 2)
        print(f"Sub-population {sub_pop_dict['pop_id']} (Train): Precomputed Term1_std = {term1_std_train:.4f}")

        X_test_val_std = (X_test_val_raw - X_train_mean) / X_train_sd
        Y_test_val_std = (Y_test_val_raw - Y_train_mean) / Y_train_sd

        final_pop_data_train.append({
            'pop_id': sub_pop_dict['pop_id'], # This is 'A', 'B', etc.
            'X_std': torch.tensor(X_train_std, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_train_std, dtype=torch.float32).to(device),
            'E_Yx_std': torch.tensor(E_Yx_std_np_train, dtype=torch.float32).to(device),
            'term1_std': term1_std_train,
            'meaningful_indices': sub_pop_dict['meaningful_indices'], # Crucially, this is now specific to A, B, or C
            'X_raw': X_train_raw,
            'Y_raw': Y_train_raw
        })

        final_pop_data_test_val.append({
            'pop_id': sub_pop_dict['pop_id'],
            'X_std': torch.tensor(X_test_val_std, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_test_val_std, dtype=torch.float32).to(device),
            'X_raw': X_test_val_raw,
            'Y_raw': Y_test_val_raw,
            'meaningful_indices': sub_pop_dict['meaningful_indices']
        })
    
    # Print shapes for verification
    for i, pop_data_item in enumerate(final_pop_data_train):
        print(f"Train Sub-Pop {pop_data_item['pop_id']} ({i}): X_std {pop_data_item['X_std'].shape}, Y_std {pop_data_item['Y_std'].shape}, E_Yx_std {pop_data_item['E_Yx_std'].shape}")
    for i, pop_data_item in enumerate(final_pop_data_test_val):
        print(f"Test/Val Sub-Pop {pop_data_item['pop_id']} ({i}): X_std {pop_data_item['X_std'].shape}, Y_std {pop_data_item['Y_std'].shape}")

    return final_pop_data_train, final_pop_data_test_val