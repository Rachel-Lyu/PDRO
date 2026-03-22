# gd_pops_v9.py: Variable Selection with Optional Theta Reparameterization and Advanced T2 Estimator
# for ACS data handling
# Introduce batch gd
"""
Performs variable subset selection using gradient descent.
Version 8 introduces an optional IF-like T2 estimator for gradient computation.

Objective: Minimize L(param) = SmoothMax_pop [ T1_std - T2(param) + P(param) ]
  T1_std = E[(standardized E[Y|X])^2]
  T2(param) = E[(standardized E[Y|S(param)])^2] (estimated via various methods)
  P(param) = Penalty Term applied to alpha (derived from theta if needed)
  S(param) = X_std + sqrt(variance(param)) * epsilon

Gradient Modes:
  - 'autograd': Computes gradient via automatic differentiation.
  - 'reinforce': Computes gradient of objective (T2 and P parts) using REINFORCE for T2.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error, r2_score
import argparse
from torch.utils.data import DataLoader
import re
import time
import math
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pickle
from downstream_models import (
    SimpleMLP
)

# --- Import project modules ---
try:
    from data import generate_data_continuous, generate_data_continuous_with_corr
    from data_baseline_failures import get_pop_data_baseline_failures
    from data_acs import get_acs_pop_data
    from estimators import (
        plugin_estimator_conditional_mean,
        plugin_estimator_squared_conditional,
        IF_estimator_conditional_mean,
        IF_estimator_squared_conditional,
        estimate_conditional_keops_flexible,
        estimate_T2_mc_flexible,
        estimate_T2_kernel_IF_like_flexible, # <-- New Estimator
        estimate_gradient_reinforce_flexible
    )
    from torch.optim import lr_scheduler
    print("Successfully imported data, estimators (v7), and scheduler functions.")
    from baselines import (
        compute_population_stats,
        standardize_data,
        baseline_lasso_comparison,
        baseline_xgb_comparison,
        baseline_dro_lasso_comparison,
        baseline_dro_xgb_comparison
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure data.py, estimators.py are accessible and updated for v7.")
    exit(1)

# Assuming global_vars.py defines these appropriately
try:
    from global_vars import *
except ImportError:
    print("Warning: global_vars.py not found. Using placeholder values for gd_pops_v7.py.")
    EPS = 1e-9
    CLAMP_MIN_ALPHA = 1e-5
    CLAMP_MAX_ALPHA = 1e5
    THETA_CLAMP_MIN = math.log(CLAMP_MIN_ALPHA) if CLAMP_MIN_ALPHA > 0 else -11.5
    THETA_CLAMP_MAX = math.log(CLAMP_MAX_ALPHA) if CLAMP_MAX_ALPHA > 0 else 11.5
    N_FOLDS = 5
    FREEZE_THRESHOLD_ALPHA = 1e-4
    THETA_FREEZE_THRESHOLD = math.log(FREEZE_THRESHOLD_ALPHA) if FREEZE_THRESHOLD_ALPHA > 0 else -9.2


def standardize_data(X, Y):
    X_mean = np.mean(X, axis=0); X_std = np.std(X, axis=0)
    Y_mean = np.mean(Y); Y_std = np.std(Y)
    X_std[X_std < EPS] = EPS
    if Y_std < EPS: Y_std = EPS
    return (X - X_mean) / X_std, (Y - Y_mean) / Y_std, X_mean, X_std, Y_mean, Y_std

def compute_penalty(alpha: torch.Tensor, penalty_type: Optional[str], penalty_lambda: float, epsilon: float = EPS) -> torch.Tensor:
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
        return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype, requires_grad=alpha.requires_grad)
    pt_lower = penalty_type.lower()
    if pt_lower == "reciprocal_l1": return penalty_lambda * torch.sum(1.0 / (alpha_clamped + epsilon))
    if pt_lower == "neg_l1": return penalty_lambda * torch.sum(torch.abs(alpha_clamped))
    if pt_lower == "max_dev": return penalty_lambda * torch.sum(torch.abs(1.0 - alpha_clamped))
    if pt_lower == "quadratic_barrier": return penalty_lambda * torch.sum((alpha_clamped + epsilon) ** (-2))
    if pt_lower == "exponential": return penalty_lambda * torch.sum(torch.exp(-alpha_clamped))
    raise ValueError(f"Unknown penalty_type: {penalty_type}")

def compute_objective_value_mc(X, E_Y_given_X_std, term1_std, param, param_type, penalty_lambda=0.0, penalty_type=None, num_mc_samples=25, k_kernel=1000):
    param_val = param.detach().clone()
    if param_type == 'alpha': alpha_val = param_val.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    elif param_type == 'theta': alpha_val = torch.exp(param_val.clamp(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    else: raise ValueError("Invalid param_type")
    avg_term2_std = 0.0
    with torch.no_grad():
        noise_var = alpha_val
        for _ in range(num_mc_samples):
            S_param = X + torch.randn_like(X) * torch.sqrt(noise_var)
            E_Y_S_std = estimate_conditional_keops_flexible(X, S_param, E_Y_given_X_std, param_val, param_type, k=k_kernel)
            avg_term2_std += E_Y_S_std.pow(2).mean()
    term2_value_std = avg_term2_std / num_mc_samples
    penalty = compute_penalty(alpha_val, penalty_type, penalty_lambda)
    return (term1_std - term2_value_std + penalty).item()

def compute_objective_value_if(X, Y_std, term1_std, param, param_type, penalty_lambda=0.0, penalty_type=None, base_model_type="rf", n_folds=N_FOLDS):
    param_val = param.detach().clone()
    if param_type == 'alpha': alpha_val = param_val.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    elif param_type == 'theta': alpha_val = torch.exp(param_val.clamp(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    else: raise ValueError("Invalid param_type")
    term2_val_float = float('nan')
    with torch.no_grad():
        S_alpha_np = (X + torch.randn_like(X) * torch.sqrt(alpha_val)).cpu().numpy()
        Y_std_np = Y_std.cpu().numpy()
        try:
            term2_val_float = IF_estimator_squared_conditional(S_alpha_np, Y_std_np, estimator_type=base_model_type, n_folds=n_folds)
        except Exception as e: print(f"Warning: IF T2 calculation failed: {e}")
    if math.isnan(term2_val_float): return float('nan')
    penalty = compute_penalty(alpha_val, penalty_type, penalty_lambda)
    return (term1_std - torch.tensor(term2_val_float, device=param.device) + penalty).item()

def get_pop_data(pop_configs, m1, m, dataset_size=15000, noise_scale=0.0, corr_strength=0.0, common_meaningful_indices=None, estimator_type="plugin", device="cpu", base_model_type="rf", 
                 test_val_fraction=0.8, # Fraction of data to use for test/val
                 seed=None):
    if common_meaningful_indices is None: common_meaningful_indices = np.arange(max(1, m1 // 2))
    pop_data_list = []
    pop_data_list_test_val = []
    for i, pop_config in enumerate(pop_configs):
        rng = np.random.default_rng(seed + i)  
        pop_id = pop_config.get('pop_id', i); dataset_type = pop_config['dataset_type']
        current_seed = seed + pop_id if seed is not None else None
        gen_fn = generate_data_continuous_with_corr
        X_np, Y_np, _, meaningful_idx = gen_fn(pop_id=pop_id, m1=m1, m=m, dataset_type=dataset_type, dataset_size=dataset_size, noise_scale=noise_scale, corr_strength=corr_strength, seed=current_seed, common_meaningful_indices=common_meaningful_indices)
        # subsample the data to dataset_size
        idx = rng.permutation(X_np.shape[0])
        indices = idx[:int(X_np.shape[0] * test_val_fraction)]
        X_np_test_val = X_np[indices]
        Y_np_test_val = Y_np[indices]
        X_np = np.delete(X_np, indices, axis=0)
        Y_np = np.delete(Y_np, indices, axis=0)
        X_std_np, Y_std_np, _, _, Y_mean, Y_std_val = standardize_data(X_np, Y_np)
        # Use train mean/std and guard against zeros
        train_mean = np.mean(X_np, axis=0)
        train_std = np.std(X_np, axis=0)
        # avoid zero‐division
        train_std[train_std < EPS] = EPS
        X_std_np_test_val = (X_np_test_val - train_mean) / train_std
        Y_std_np_test_val = (Y_np_test_val - Y_mean) / Y_std_val
        try:
            if estimator_type == "plugin": E_Yx_orig_np = plugin_estimator_conditional_mean(X_np, Y_np, base_model_type, n_folds=N_FOLDS, seed=current_seed)
            elif estimator_type == "if": E_Yx_orig_np = IF_estimator_conditional_mean(X_np, Y_np, base_model_type, n_folds=N_FOLDS, seed=current_seed)
            else: raise ValueError("estimator_type must be 'plugin' or 'if'")
        except Exception as e: print(f"ERROR: E[Y|X] precomputation failed for pop {pop_id}: {e}"); continue
        E_Yx_std_np = (E_Yx_orig_np - Y_mean) / Y_std_val
        term1_std = np.mean(E_Yx_std_np ** 2)
        pop_data_list.append({
            'pop_id': pop_id, 'X_std': torch.tensor(X_std_np, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_std_np, dtype=torch.float32).to(device),
            'E_Yx_std': torch.tensor(E_Yx_std_np, dtype=torch.float32).to(device),
            'term1_std': term1_std, 'meaningful_indices': meaningful_idx.tolist(),
            'X_raw': X_np, 'Y_raw': Y_np,
        })
        # Test/Val data
        pop_data_list_test_val.append({
            'pop_id': pop_id, 'X_std': torch.tensor(X_std_np_test_val, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_std_np_test_val, dtype=torch.float32).to(device),
            'X_raw': X_np_test_val, 'Y_raw': Y_np_test_val,
        })
    
    # print the dataset shapes for all population and train/test_val data
    for i, pop_data in enumerate(pop_data_list):
        print(f"Population {i} - Train Data: X shape: {pop_data['X_std'].shape}, Y shape: {pop_data['Y_std'].shape}")
    for i, pop_data in enumerate(pop_data_list_test_val):
        print(f"Population {i} - Test/Val Data: X shape: {pop_data['X_std'].shape}, Y shape: {pop_data['Y_std'].shape}")
    return pop_data_list, pop_data_list_test_val

def get_pop_data_acs(pop_configs, m1, m, 
                     dataset_size=10000, 
                     acs_data_fraction=0.5, # Fraction of data to use for ASC
                     estimator_type="plugin", device="cpu", 
                     base_model_type="rf", seed=None):
    """
    pop configs comprises population data dicts in the same format as data_baseline_failures:
      - 'pop_id': state abbreviation
      - 'X_raw': raw feature array
      - 'Y_raw': raw label array
      - 'meaningful_indices': None (unknown for real data)

      This function will subsample the data to datase_size for ech population and return a train dictionary and a test_val dictionary.
    """
    pop_data_list_train = []
    pop_data_list_test_val = []
    for i, pop_config in enumerate(pop_configs):
        rng = np.random.default_rng(seed + i)
        pop_id = pop_config.get('pop_id', i); 
        # subsample the data to dataset_size
        idx = rng.permutation(pop_config['X_raw'].shape[0])
        indices = idx[:int(pop_config['X_raw'].shape[0] * acs_data_fraction)]
        X_np = pop_config['X_raw'][indices]
        Y_np = pop_config['Y_raw'][indices]
        X_np_test_val = np.delete(pop_config['X_raw'], indices, axis=0)
        Y_np_test_val = np.delete(pop_config['Y_raw'], indices, axis=0)
        # check for nans and infs
        if np.any(np.isnan(X_np)) or np.any(np.isnan(Y_np)):
            print(f"Warning: NaN values found in population {pop_id}. Skipping this population.")
            # impute or handle nans as needed
            continue
        X_std_np, Y_std_np, _, _, Y_mean, Y_std_val = standardize_data(X_np, Y_np)
        # Use train mean/std and guard against zeros
        train_mean = np.mean(X_np, axis=0)
        train_std = np.std(X_np, axis=0)
        # avoid zero‐division
        train_std[train_std < EPS] = EPS
        X_std_np_test_val = (X_np_test_val - train_mean) / train_std
        Y_std_np_test_val = (Y_np_test_val - Y_mean) / Y_std_val
        try:
            if estimator_type == "plugin": E_Yx_orig_np = plugin_estimator_conditional_mean(X_np, Y_np, base_model_type, n_folds=N_FOLDS, seed=seed)
            elif estimator_type == "if": E_Yx_orig_np = IF_estimator_conditional_mean(X_np, Y_np, base_model_type, n_folds=N_FOLDS, seed=seed)
            else: raise ValueError("estimator_type must be 'plugin' or 'if'")
        except Exception as e: print(f"ERROR: E[Y|X] precomputation failed for pop {pop_id}: {e}"); continue
        E_Yx_std_np = (E_Yx_orig_np - Y_mean) / Y_std_val
        term1_std = np.mean(E_Yx_std_np ** 2)
        pop_data_list_train.append({
            'pop_id': pop_id, 'X_std': torch.tensor(X_std_np, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_std_np, dtype=torch.float32).to(device),
            'E_Yx_std': torch.tensor(E_Yx_std_np, dtype=torch.float32).to(device),
            'term1_std': term1_std, 'meaningful_indices': None,
            'X_raw': X_np, 'Y_raw': Y_np,
        })
        pop_data_list_test_val.append({
            'pop_id': pop_id, 'X_std': torch.tensor(X_std_np_test_val, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_std_np_test_val, dtype=torch.float32).to(device),
            'X_raw': X_np_test_val, 'Y_raw': Y_np_test_val,
        })

    for i, pop_data in enumerate(pop_data_list_train):
        print(f"Population {i} - Train Data: X shape: {pop_data['X_std'].shape}, Y shape: {pop_data['Y_std'].shape}")
    for i, pop_data in enumerate(pop_data_list_test_val):
        print(f"Population {i} - Test/Val Data: X shape: {pop_data['X_std'].shape}, Y shape: {pop_data['Y_std'].shape}")
    return pop_data_list_train, pop_data_list_test_val

def init_parameter(m, parameterization, alpha_init_config="random_1", noise=0.1, device="cpu"):
    init_alpha_val = torch.ones(m, device=device)
    alpha_init_lower = alpha_init_config.lower()
    if alpha_init_lower.startswith("random_"):
        try: k = float(alpha_init_lower.split('_', 1)[1])
        except (ValueError, IndexError): raise ValueError(f"Invalid numeric value in alpha_init_config: {alpha_init_config}")
        init_alpha_val = k * torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
    elif alpha_init_lower == "ones" or alpha_init_lower == "random": # Consolidate
        init_alpha_val = torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
    else: raise ValueError("alpha_init_config invalid.")
    init_alpha_val.clamp_(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    if parameterization == 'alpha': param = torch.nn.Parameter(init_alpha_val)
    elif parameterization == 'theta': param = torch.nn.Parameter(torch.log(torch.clamp(init_alpha_val, min=EPS)).clamp_(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX))
    else: raise ValueError("parameterization must be 'alpha' or 'theta'")
    return param

def create_results_dataframe(results, args_dict, save_path=None):
    data = {'Run_ID': os.path.basename(os.path.dirname(save_path)) if save_path else 'unknown_run'}
    # Add all args to the data dictionary
    for k, v in args_dict.items():
        if isinstance(v, list): data[k] = '_'.join(map(str,v)) # Join lists like populations
        else: data[k] = v

    data['Budget_Actual'] = len(results.get('selected_indices', []))
    data['All_True_Indices'] = str(list(set().union(*[set(indices) for indices in results.get('meaningful_indices', [])])))
    for i, indices in enumerate(results.get('meaningful_indices', [])):
        data[f'True_Indices_Pop_{i}'] = str(indices)
    data.update({
        'Our_F1': results.get('overall_f1_score'), 'Our_Precision': results.get('overall_precision'),
        'Our_Recall': results.get('overall_recall'), 'Our_Selected_Indices': str(results.get('selected_indices')),
    })
    for stat in results.get('population_stats', []):
        data[f'Our_Pop{stat["population"]}_Coverage'] = stat['percentage']
    if results.get('baseline_results'):
        baseline = results['baseline_results']
        data.update({
            'Baseline_F1': baseline.get('f1_score'), 'Baseline_Precision': baseline.get('precision'),
            'Baseline_Recall': baseline.get('recall'), 'Baseline_Selected_Indices': str(baseline.get('selected_indices')),
        })
        for stat in baseline.get('baseline_pop_stats', []):
            data[f'Baseline_Pop{stat["population"]}_Coverage'] = stat['percentage']
    df = pd.DataFrame([data])
    if save_path:
        csv_path = os.path.join(save_path, 'results_summary_v7.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results summary saved to: {csv_path}")
        df.to_pickle(os.path.join(save_path, 'results_summary_v7.pkl'))
    return df


def run_variable_selection(
    pop_data: List[Dict], m1: int, m: int, dataset_size: int = 5000,
    acs_data_fraction: float = 0.5, # Fraction of data to use for ASC
    noise_scale: float = 0.1, corr_strength: float = 0.0,
    num_epochs: int = 100, budget: Optional[int] = None,
    penalty_type: Optional[str] = None, penalty_lambda: float = 0.0,
    learning_rate: float = 0.01, optimizer_type: str = 'adam',
    parameterization: str = 'alpha', alpha_init: str = "random_1",
    early_stopping_patience: int = 15, param_freezing: bool = True,
    smooth_minmax: float = float('inf'),
    gradient_mode: str = 'autograd', t2_estimator_type: str = 'mc_plugin', # New arg
    N_grad_samples: int = 25, use_baseline: bool = True,
    estimator_type: str = "if", base_model_type: str = "rf",
    objective_value_estimator: str = 'if', k_kernel: int = 1000,
    scheduler_type: Optional[str] = None, scheduler_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None, save_path: str = './results_v7/multi_population/',
    verbose: bool = False, implement_baseline_comparison: bool = True,
    alpha_lasso: float = None
    ) -> Dict[str, Any]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Experiment (v7.0) ---")
    print(f"Parameterization: {parameterization}, Gradient Mode: {gradient_mode}, T2 Estimator (Grad): {t2_estimator_type}")
    print(f"Using device: {device}")
    if smooth_minmax != float('inf'): print(f"SmoothMax Beta: {smooth_minmax}")
    os.makedirs(save_path, exist_ok=True)

    if seed is not None: np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available() and seed is not None: torch.cuda.manual_seed_all(seed)

    # if budget is None: budget = min(m, max(1, m1 // 2) + len(pop_configs) * (m1 - max(1, m1 // 2)))
    # else: budget = min(budget, m)

    # VARIABLE SELECTION BEGINS
    param = init_parameter(m, parameterization, alpha_init, noise=0.1, device=device)
    optimizer_class = optim.Adam if optimizer_type.lower() == 'adam' else optim.SGD
    optimizer_kwargs = {'lr': learning_rate}
    if optimizer_type.lower() == 'sgd': optimizer_kwargs.update({'momentum':0.9, 'nesterov':True})
    optimizer = optimizer_class([param], **optimizer_kwargs)

    scheduler = None
    if scheduler_type and scheduler_kwargs:
        scheduler_class = getattr(lr_scheduler, scheduler_type, None)
        if scheduler_class:
            try: scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            except Exception as e: print(f"Warning: Failed to init scheduler '{scheduler_type}': {e}")
        else: print(f"Warning: Scheduler '{scheduler_type}' not found.")

    param_history, objective_history, gradient_history = [param.detach().cpu().numpy().copy()], [], []
    lr_history = [optimizer.param_groups[0]['lr']]
    best_objective_val, early_stopping_counter = float('inf'), 0
    best_param = param.detach().cpu().numpy().copy()
    stopped_epoch = num_epochs
    total_start_time = time.time()

    print(f"\n--- Starting Optimization (v7: {gradient_mode} mode, {parameterization} param, T2 est: {t2_estimator_type}) ---")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        optimizer.zero_grad()

        with torch.no_grad():
            current_alpha = param.data.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA) if parameterization == 'alpha' \
                            else torch.exp(param.data.clamp(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)

        population_objectives_for_grad = []
        param.requires_grad_(True)

        for pop in pop_data:
            X_std, E_Yx_std, term1_std_val = pop['X_std'], pop['E_Yx_std'], pop['term1_std']
            
            if gradient_mode == 'autograd':
                if t2_estimator_type == 'mc_plugin':
                    term2_value = estimate_T2_mc_flexible(X_std_torch=X_std, E_Yx_std_torch=E_Yx_std, param_torch=param,
                        param_type=parameterization, n_mc_samples=N_grad_samples, k_kernel=k_kernel)
                elif t2_estimator_type == 'kernel_if_like':
                    Y_std = pop['Y_std']
                    term2_value = estimate_T2_kernel_IF_like_flexible(
                        X_std_torch=X_std,
                        Y_std_torch=Y_std,
                        E_Yx_std_torch=E_Yx_std,
                        param_torch=param,
                        param_type=parameterization,
                        n_mc_samples=N_grad_samples,
                        k_kernel=k_kernel
                    )
                else:
                    raise ValueError(f"Unknown t2_estimator_type: {t2_estimator_type} for autograd mode.")
                
                alpha_for_penalty = param if parameterization == 'alpha' else torch.exp(param)
                penalty_val = compute_penalty(alpha_for_penalty, penalty_type, penalty_lambda)
                pop_obj_grad = term1_std_val - term2_value + penalty_val
                population_objectives_for_grad.append(pop_obj_grad)

            # REINFORCE mode will calculate its own T2 and P gradient internally if this loop structure is kept for it.
            # For simplicity, if REINFORCE is selected, it will handle the entire objective gradient for the winning pop.

        if gradient_mode == 'autograd':
            objectives_tensor = torch.stack(population_objectives_for_grad)
            if torch.isfinite(torch.tensor(smooth_minmax)) and smooth_minmax > 0:
                beta = smooth_minmax
                with torch.no_grad(): M = torch.max(beta * objectives_tensor)
                logsumexp_val = M + torch.log(torch.sum(torch.exp(beta * objectives_tensor - M)) + EPS)
                robust_objective_for_grad = (1.0 / beta) * logsumexp_val
            else:
                robust_objective_for_grad, _ = torch.max(objectives_tensor, dim=0)
        
        # --- Calculate Objective VALUE (Hard Max) for Tracking/Stopping ---
        population_objective_values = []
        current_param_val_detached = param.detach().clone()
        for pop in pop_data:
            obj_val_fn = compute_objective_value_if if objective_value_estimator == 'if' else compute_objective_value_mc
            args_val = (pop['X_std'], pop['E_Yx_std'] if objective_value_estimator == 'mc' else pop['Y_std'], pop['term1_std'],
                        current_alpha if objective_value_estimator == 'if' else current_param_val_detached, # Pass alpha for IF, param for MC
                        'alpha' if objective_value_estimator == 'if' else parameterization, # IF expects alpha, MC expects original param type
                        penalty_lambda, penalty_type)
            kwargs_val = {'base_model_type': base_model_type} if objective_value_estimator == 'if' else {'num_mc_samples': N_grad_samples, 'k_kernel': k_kernel}
            obj_val = obj_val_fn(*args_val, **kwargs_val)
            population_objective_values.append(obj_val)

        valid_obj_values = [v for v in population_objective_values if not math.isnan(v)]
        if not valid_obj_values: print(f"ERROR: All pop objective values NaN at epoch {epoch}. Stopping."); stopped_epoch=epoch; break
        current_robust_objective_value = max(valid_obj_values)
        winning_pop_idx_track = population_objective_values.index(current_robust_objective_value)
        objective_history.append(current_robust_objective_value)

        # --- Calculate Gradient ---
        if gradient_mode == 'autograd':
            robust_objective_for_grad.backward()
            total_gradient = param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            param.requires_grad_(False)
        elif gradient_mode == 'reinforce':
            winning_pop_data = pop_data[winning_pop_idx_track] # Use same winning pop for grad and tracking
            total_gradient = estimate_gradient_reinforce_flexible(
                winning_pop_data['X_std'], winning_pop_data['E_Yx_std'], param, parameterization,
                N_grad_samples, k_kernel, penalty_type, penalty_lambda, use_baseline,
                t2_estimator_type_for_reward=t2_estimator_type, # Pass T2 type for reward calc
                Y_std_torch=winning_pop_data['Y_std'] if t2_estimator_type == 'kernel_if_like' else None
            )
            if param.grad is not None: param.grad.zero_()
            param.grad = total_gradient # Assign gradient manually
        else:
            raise ValueError("gradient_mode must be 'autograd' or 'reinforce'")
        
        gradient_history.append(total_gradient.detach().cpu().numpy().copy() if total_gradient is not None else np.zeros(m))

        if param_freezing and param.grad is not None:
            with torch.no_grad():
                freeze_thresh = FREEZE_THRESHOLD_ALPHA if parameterization == 'alpha' else THETA_FREEZE_THRESHOLD
                frozen_mask = param.data < freeze_thresh if parameterization == 'alpha' else param.data < freeze_thresh # theta is negative for small alpha
                param.grad[frozen_mask] = 0.0
                # Also clear optimizer state for frozen params
                for group in optimizer.param_groups:
                    if group['params'][0] is param:
                        for p_state_key, p_state_val in optimizer.state[param].items():
                            if isinstance(p_state_val, torch.Tensor) and p_state_val.shape == param.shape:
                                p_state_val[frozen_mask] = 0.0
                        break
        
        if param.grad is not None:
            torch.nn.utils.clip_grad_norm_([param], max_norm=10.0)
        
        optimizer.step()

        with torch.no_grad():
            if parameterization == 'alpha': param.data.clamp_(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
            else: param.data.clamp_(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau): scheduler.step(current_robust_objective_value)
            else: scheduler.step()
        lr_history.append(optimizer.param_groups[0]['lr'])
        param_history.append(param.detach().cpu().numpy().copy())
        
        if verbose or (epoch % 10 == 0):
             print(f"Epoch {epoch}/{num_epochs} | Obj: {current_robust_objective_value:.4f} | LR: {current_lr:.6f} | WinPop: {pop_data[winning_pop_idx_track]['pop_id']}")

        if current_robust_objective_value < best_objective_val - EPS : # Minimizing
            best_objective_val = current_robust_objective_value
            best_param = param.detach().cpu().numpy().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if epoch > 40 and early_stopping_counter >= early_stopping_patience and abs(current_robust_objective_value - best_objective_val) < 0.01 * abs(best_objective_val):
            print(f"Early stopping at epoch {epoch}."); stopped_epoch = epoch; break
        if epoch<30:
            early_stopping_counter = 0 # Reset early stopping counter for first 30 epochs
            # at least train 40 epochs before checking for early stopping
            
    total_time = time.time() - total_start_time
    print(f"\nOptimization finished in {total_time:.2f}s. Best obj: {best_objective_val:.4f}")

    final_alpha_np = np.exp(best_param) if parameterization == 'theta' else best_param
    final_alpha_np = np.clip(final_alpha_np, CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA)
    selected_indices = np.argsort(final_alpha_np)[:budget] # Smallest alpha selected
    print(f"Selected indices: {selected_indices}")

    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
    
        # --- Save Diagnostics Plot (if verbose) ---
    if verbose:
        try:
            # collect meaningful lists (may be None)
            meaningful_indices_list = [pop.get('meaningful_indices') for pop in pop_data]
            plt.figure(figsize=(18, 6))
            # Objective
            plt.subplot(1, 3, 1)
            plt.plot(objective_history, label="Robust Objective L")
            plt.xlabel("Epoch"); plt.ylabel("Objective Value"); plt.title("Objective vs Epoch"); plt.grid(True)
            
            # Gradients
            plt.subplot(1, 3, 2)
            valid_gradients = [g for g in gradient_history if g is not None]
            grad_norms = [np.linalg.norm(g) for g in valid_gradients]
            if grad_norms: plt.plot(grad_norms, label="Total Gradient Norm")
            plt.xlabel("Epoch"); plt.ylabel("Gradient Norm"); plt.title("Gradient Norm vs Epoch"); plt.yscale('log'); plt.grid(True)
            
            # Parameters (alpha or theta)
            plt.subplot(1, 3, 3)
            param_hist_np = np.array(param_history)
            indices_to_plot = selected_indices
            # ASCII-only
            param_label = 'alpha' if parameterization == 'alpha' else 'theta'
            for i in indices_to_plot:
                label = f'{param_label}_{i}'
                # only mark "(True)" if we actually have lists of meaningful indices
                if meaningful_indices_list and all(isinstance(mi, (list, tuple)) for mi in meaningful_indices_list):
                    if any(i in mi for mi in meaningful_indices_list if mi):
                        label += " (True)"
                plt.plot(param_hist_np[:, i], label=label)
            plt.xlabel("Epoch"); plt.ylabel(f"{param_label} Value")
            plt.title(f"{param_label} Trajectories (Top {len(indices_to_plot)})")
            if parameterization == 'alpha':
                plt.yscale('log')
            plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)

            plt.tight_layout()
            diag_path = os.path.join(save_path, f"optimization_diagnostics_v6_{parameterization}.png")
            plt.savefig(diag_path); plt.close()
            print(f"Diagnostic plots saved to {diag_path}")
        except Exception as e: print(f"Warning: Failed to generate diagnostic plots: {e}")

    # Baseline Comparison - all baselines
    if implement_baseline_comparison:
        # BASELINE LASSO
        baseline_lasso_results = baseline_lasso_comparison(
            pop_data=pop_data, budget=budget, alpha_lasso=alpha_lasso
        )
        if baseline_lasso_results: 
            print(f"Baseline Lasso Results: {baseline_lasso_results}")
            if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
                print("Warning: meaningful_indices_list contains None. Skipping population stats computation.")
                baseline_pop_stats, baseline_overall_stats = None, None
            else:
                baseline_pop_stats, baseline_overall_stats = compute_population_stats(
                    baseline_lasso_results['selected_indices'], meaningful_indices_list
                )
            baseline_lasso_results.update({
                'baseline_pop_stats': baseline_pop_stats,
                'baseline_overall_stats': baseline_overall_stats
            })
        else:
            print("No valid results from Lasso baseline comparison.")
        # BASELINE DRO LASSO
        baseline_dro_lasso_results = baseline_dro_lasso_comparison(
            pop_data=pop_data, budget=budget, alpha_lasso=alpha_lasso
        )
        if baseline_dro_lasso_results:
            print(f"Baseline DRO Lasso Results: {baseline_dro_lasso_results}")
            if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
                print("Warning: meaningful_indices_list contains None. Skipping population stats computation.")
                baseline_pop_stats, baseline_overall_stats = None, None
            else:
                baseline_pop_stats, baseline_overall_stats = compute_population_stats(
                    baseline_dro_lasso_results['selected_indices'], meaningful_indices_list
                )
            baseline_dro_lasso_results.update({
                'baseline_pop_stats': baseline_pop_stats,
                'baseline_overall_stats': baseline_overall_stats
            })
        # BASELINE XGB
        baseline_xgb_results = baseline_xgb_comparison(
            pop_data=pop_data, budget=budget
        )
        if baseline_xgb_results:
            print(f"Baseline XGB Results: {baseline_xgb_results}")
            if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
                print("Warning: meaningful_indices_list contains None. Skipping population stats computation.")
                baseline_pop_stats, baseline_overall_stats = None, None
            else:
                baseline_pop_stats, baseline_overall_stats = compute_population_stats(
                    baseline_xgb_results['selected_indices'], meaningful_indices_list
                )
            baseline_xgb_results.update({
                'baseline_pop_stats': baseline_pop_stats,
                'baseline_overall_stats': baseline_overall_stats
            })
        else:
            print("No valid results from XGB baseline comparison.")
        # BASELINE DRO XGB
        baseline_dro_xgb_results = baseline_dro_xgb_comparison(
            pop_data=pop_data, budget=budget)
        if baseline_dro_xgb_results:
            print(f"Baseline DRO XGB Results: {baseline_dro_xgb_results}")
            if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
                print("Warning: meaningful_indices_list contains None. Skipping population stats computation.")
                baseline_pop_stats, baseline_overall_stats = None, None
            else:
                baseline_pop_stats, baseline_overall_stats = compute_population_stats(
                    baseline_dro_xgb_results['selected_indices'], meaningful_indices_list
                )
            baseline_dro_xgb_results.update({
                'baseline_pop_stats': baseline_pop_stats,
                'baseline_overall_stats': baseline_overall_stats
            })
    our_method_results = {
        'selected_indices': selected_indices, 'final_alpha': final_alpha_np[selected_indices].tolist(),
        'all_indices_ranked': np.argsort(final_alpha_np).tolist(), 
    }
    results = {
        'parameterization': parameterization, 't2_estimator_type': t2_estimator_type,
        'final_objective': best_objective_val if not math.isnan(best_objective_val) else None,
        'best_param': best_param.tolist(), 'final_alpha': final_alpha_np.tolist(),
        'selected_indices': selected_indices.tolist(),
        'selected_alphas': final_alpha_np[selected_indices].tolist(),
        'objective_history': [o if not math.isnan(o) else None for o in objective_history],
        'param_history': [p.tolist() for p in param_history], 'lr_history': lr_history,
        'populations': [pop['pop_id'] for pop in pop_data],
        'meaningful_indices': meaningful_indices_list,
        'total_time_seconds': total_time, 'stopped_epoch': stopped_epoch,
        'our_method_results': our_method_results,
        'baseline_lasso_results': baseline_lasso_results if implement_baseline_comparison else None,
        'baseline_dro_lasso_results': baseline_dro_lasso_results if implement_baseline_comparison else None,
        'baseline_xgb_results': baseline_xgb_results if implement_baseline_comparison else None,
        'baseline_dro_xgb_results': baseline_dro_xgb_results if implement_baseline_comparison else None,
        # 'baseline_results': baseline_results,
        # 'our_method_based_ranking': np.argsort(final_alpha_np).tolist(), # asceding order
        # 'baseline_method_based_ranking': np.argsort(-np.abs(baseline_results['baseline_coeffs'])).tolist() if baseline_results else None
    }
    return results

def run_experiment_multi_population(
    pop_configs: List[Dict], m1: int, m: int, dataset_size: int = 5000,
    baseline_data_size: int = 15000,
    acs_data_fraction: float = 0.5, # Fraction of data to use for ASC
    noise_scale: float = 0.1, corr_strength: float = 0.0,
    num_epochs: int = 100, budget: Optional[int] = None,
    penalty_type: Optional[str] = None, penalty_lambda: float = 0.0,
    learning_rate: float = 0.01, optimizer_type: str = 'adam',
    parameterization: str = 'alpha', alpha_init: str = "random_1",
    early_stopping_patience: int = 15, param_freezing: bool = True,
    smooth_minmax: float = float('inf'),
    gradient_mode: str = 'autograd', t2_estimator_type: str = 'mc_plugin', # New arg
    N_grad_samples: int = 25, use_baseline: bool = True,
    estimator_type: str = "if", base_model_type: str = "rf",
    objective_value_estimator: str = 'if', k_kernel: int = 1000,
    scheduler_type: Optional[str] = None, scheduler_kwargs: Optional[Dict] = None,
    seed: Optional[int] = None, save_path: str = './results_v7/multi_population/',
    verbose: bool = False, implement_baseline_comparison: bool = True,
    alpha_lasso: float = None
    ) -> Dict[str, Any]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Experiment (v7.0) ---")
    print(f"Parameterization: {parameterization}, Gradient Mode: {gradient_mode}, T2 Estimator (Grad): {t2_estimator_type}")
    print(f"Using device: {device}")
    if smooth_minmax != float('inf'): print(f"SmoothMax Beta: {smooth_minmax}")
    os.makedirs(save_path, exist_ok=True)

    if seed is not None: np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available() and seed is not None: torch.cuda.manual_seed_all(seed)

    if budget is None: budget = min(m, max(1, m1 // 2) + len(pop_configs) * (m1 - max(1, m1 // 2)))
    else: budget = min(budget, m)
    
    if any('baseline' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
        # Use baseline data generation function
        dataset_size = baseline_data_size
        pop_data, pop_data_test_val = get_pop_data_baseline_failures(
            pop_configs=pop_configs, dataset_size=baseline_data_size,
            n_features=m, 
            noise_scale=noise_scale, corr_strength=corr_strength,
            estimator_type=estimator_type, device=device,
            base_model_type=base_model_type, seed=seed
        )
    elif any('asc' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
        # Use ACS data generation function
        pop_data = get_acs_pop_data()
        pop_data, pop_data_test_val = get_pop_data_acs(
            pop_configs=pop_data, m1=m1, m=m, dataset_size=dataset_size,
            acs_data_fraction=acs_data_fraction,
            estimator_type=estimator_type, device=device,
            base_model_type=base_model_type, seed=seed
        )
        print(f"ACS data generated with {len(pop_data)} populations.")
        print(f"Setting the dim to number of features in the ACS data: {len(pop_data[0]['X_raw'][0])}")
        m=len(pop_data[0]['X_raw'][0])
        print(f"ACS data generated with {len(pop_data)} populations.")
    else:
        pop_data, pop_data_test_val = get_pop_data(
            pop_configs=pop_configs, m1=m1, m=m, dataset_size=dataset_size,
            noise_scale=noise_scale, corr_strength=corr_strength,
            estimator_type=estimator_type, device=device,
            base_model_type=base_model_type, seed=seed
        )
    if not pop_data: return {'error': 'Data generation failed'}
    
    # take the union of all the meaningful indices across populations; budget is the size of this union
    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
    if all(isinstance(indices, (list, tuple)) for indices in meaningful_indices_list if indices is not None):
        all_meaningful_indices = set()
        for indices in meaningful_indices_list:
            if indices is not None:
                all_meaningful_indices.update(indices)
        all_meaningful_indices = list(all_meaningful_indices)
        budget = len(all_meaningful_indices)
        print(f"Budget set to the size of the union of all meaningful indices: {budget}")
    # VARIABLE SELECTION BEGINS
    variable_selection_results = run_variable_selection(
        pop_data=pop_data, m1=m1, m=m, dataset_size=dataset_size,
        acs_data_fraction=acs_data_fraction, noise_scale=noise_scale,
        corr_strength=corr_strength, num_epochs=num_epochs,
        budget=budget, penalty_type=penalty_type, penalty_lambda=penalty_lambda,
        learning_rate=learning_rate, optimizer_type=optimizer_type,
        parameterization=parameterization, alpha_init=alpha_init,
        early_stopping_patience=early_stopping_patience,
        param_freezing=param_freezing, smooth_minmax=smooth_minmax,
        gradient_mode=gradient_mode, t2_estimator_type=t2_estimator_type,
        N_grad_samples=N_grad_samples, use_baseline=use_baseline,
        estimator_type=estimator_type, base_model_type=base_model_type,
        objective_value_estimator=objective_value_estimator, k_kernel=k_kernel,
        scheduler_type=scheduler_type, scheduler_kwargs=scheduler_kwargs,
        seed=seed, save_path=save_path, verbose=verbose,
        implement_baseline_comparison=implement_baseline_comparison,
        alpha_lasso=alpha_lasso
    )
    if variable_selection_results.get('error'):
        print(f"Variable selection failed: {variable_selection_results['error']}")
        return variable_selection_results
    
    # Check if the data is ASC
    # Fit a regression model each on the selected variables by each method, baseline and our method, for each population
    # compare accuracy of the two methods

    # split the test_val data into train and test; sample indices
    # Randomly sample train and test indices once and use them consistently
    train_test_split_ratio = 0.8  # 80% train, 20% test
    train_test_indices = {}
    for pop in pop_data_test_val:
        num_samples = pop['X_std'].shape[0]
        train_idx = np.random.choice(num_samples, size=int(num_samples * train_test_split_ratio), replace=False)
        test_idx = np.setdiff1d(np.arange(num_samples), train_idx)
        train_test_indices[pop['pop_id']] = (train_idx, test_idx)

    # Baseline selection
    # LASSO
    baseline_lasso_top_indices = variable_selection_results['baseline_lasso_results']['selected_indices'][:budget]
    baseline_lasso_results = []
    for pop in pop_data_test_val:
        train_idx, test_idx = train_test_indices[pop['pop_id']]
        # slice and move to CPU/NumPy for sklearn
        if isinstance(pop['X_std'], torch.Tensor): 
            X_train_pop = pop['X_std'][train_idx][:, baseline_lasso_top_indices].detach().cpu().numpy()
            Y_train_pop = pop['Y_std'][train_idx].detach().cpu().numpy()
            X_test_pop  = pop['X_std'][test_idx][:, baseline_lasso_top_indices].detach().cpu().numpy()
            Y_test_pop  = pop['Y_std'][test_idx].detach().cpu().numpy()
        else:
            X_train_pop = pop['X_std'][train_idx][:, baseline_lasso_top_indices]
            Y_train_pop = pop['Y_std'][train_idx]
            X_test_pop  = pop['X_std'][test_idx][:, baseline_lasso_top_indices]
            Y_test_pop  = pop['Y_std'][test_idx]
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X_train_pop, Y_train_pop)
        Y_pred = model.predict(X_test_pop)
        mse = mean_squared_error(Y_test_pop, Y_pred)
        r2 = r2_score(Y_test_pop, Y_pred)
        baseline_lasso_results.append({
            'pop_id': pop['pop_id'],
            'mse': mse,
            'r2': r2,
            'selected_indices': baseline_lasso_top_indices,
            'source' : 'baseline_lasso'
        })
    # DRO LASSO
    baseline_dro_lasso_top_indices = variable_selection_results['baseline_dro_lasso_results']['selected_indices'][:budget]
    baseline_dro_lasso_results = []
    for pop in pop_data_test_val:
        train_idx, test_idx = train_test_indices[pop['pop_id']]
        # slice and move to CPU/NumPy for sklearn
        if isinstance(pop['X_std'], torch.Tensor):
            X_train_pop = pop['X_std'][train_idx][:, baseline_dro_lasso_top_indices].detach().cpu().numpy()
            Y_train_pop = pop['Y_std'][train_idx].detach().cpu().numpy()
            X_test_pop  = pop['X_std'][test_idx][:, baseline_dro_lasso_top_indices].detach().cpu().numpy()
            Y_test_pop  = pop['Y_std'][test_idx].detach().cpu().numpy()
        else:
            X_train_pop = pop['X_std'][train_idx][:, baseline_dro_lasso_top_indices]
            Y_train_pop = pop['Y_std'][train_idx]
            X_test_pop  = pop['X_std'][test_idx][:, baseline_dro_lasso_top_indices]
            Y_test_pop  = pop['Y_std'][test_idx]
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X_train_pop, Y_train_pop)
        Y_pred = model.predict(X_test_pop)
        mse = mean_squared_error(Y_test_pop, Y_pred)
        r2 = r2_score(Y_test_pop, Y_pred)
        baseline_dro_lasso_results.append({
            'pop_id': pop['pop_id'],
            'mse': mse,
            'r2': r2,
            'selected_indices': baseline_dro_lasso_top_indices,
            'source' : 'baseline_dro_lasso'
        })

    # XGB
    baseline_xgb_top_indices = variable_selection_results['baseline_xgb_results']['selected_indices'][:budget]
    baseline_xgb_results = []
    for pop in pop_data_test_val:
        train_idx, test_idx = train_test_indices[pop['pop_id']]
        # slice and move to CPU/NumPy for sklearn
        if isinstance(pop['X_std'], torch.Tensor):
            X_train_pop = pop['X_std'][train_idx][:, baseline_xgb_top_indices].detach().cpu().numpy()
            Y_train_pop = pop['Y_std'][train_idx].detach().cpu().numpy()
            X_test_pop  = pop['X_std'][test_idx][:, baseline_xgb_top_indices].detach().cpu().numpy()
            Y_test_pop  = pop['Y_std'][test_idx].detach().cpu().numpy()
        else:
            X_train_pop = pop['X_std'][train_idx][:, baseline_xgb_top_indices]
            Y_train_pop = pop['Y_std'][train_idx]
            X_test_pop  = pop['X_std'][test_idx][:, baseline_xgb_top_indices]
            Y_test_pop  = pop['Y_std'][test_idx]
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X_train_pop, Y_train_pop)
        Y_pred = model.predict(X_test_pop)
        mse = mean_squared_error(Y_test_pop, Y_pred)
        r2 = r2_score(Y_test_pop, Y_pred)
        baseline_xgb_results.append({
            'pop_id': pop['pop_id'],
            'mse': mse,
            'r2': r2,
            'selected_indices': baseline_xgb_top_indices,
            'source' : 'baseline_xgb'
        })

    # DRO XGB
    baseline_dro_xgb_top_indices = variable_selection_results['baseline_dro_xgb_results']['selected_indices'][:budget]
    baseline_dro_xgb_results = []
    for pop in pop_data_test_val:
        train_idx, test_idx = train_test_indices[pop['pop_id']]
        # slice and move to CPU/NumPy for sklearn
        if isinstance(pop['X_std'], torch.Tensor):
            X_train_pop = pop['X_std'][train_idx][:, baseline_dro_xgb_top_indices].detach().cpu().numpy()
            Y_train_pop = pop['Y_std'][train_idx].detach().cpu().numpy()
            X_test_pop  = pop['X_std'][test_idx][:, baseline_dro_xgb_top_indices].detach().cpu().numpy()
            Y_test_pop  = pop['Y_std'][test_idx].detach().cpu().numpy()
        else:
            X_train_pop = pop['X_std'][train_idx][:, baseline_dro_xgb_top_indices]
            Y_train_pop = pop['Y_std'][train_idx]
            X_test_pop  = pop['X_std'][test_idx][:, baseline_dro_xgb_top_indices]
            Y_test_pop  = pop['Y_std'][test_idx]
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X_train_pop, Y_train_pop)
        Y_pred = model.predict(X_test_pop)
        mse = mean_squared_error(Y_test_pop, Y_pred)
        r2 = r2_score(Y_test_pop, Y_pred)
        baseline_dro_xgb_results.append({
            'pop_id': pop['pop_id'],
            'mse': mse,
            'r2': r2,
            'selected_indices': baseline_dro_xgb_top_indices,
        'source' : 'baseline_dro_xgb'
        })

    # Our selection
    our_top_indices = variable_selection_results['our_method_results']['selected_indices'][:budget]
    our_results = []
    for pop in pop_data_test_val:
        train_idx, test_idx = train_test_indices[pop['pop_id']]
        if isinstance(pop['X_std'], torch.Tensor):
            X_train_pop = pop['X_std'][train_idx][:, our_top_indices].detach().cpu().numpy()
            Y_train_pop = pop['Y_std'][train_idx].detach().cpu().numpy()
            X_test_pop  = pop['X_std'][test_idx][:, our_top_indices].detach().cpu().numpy()
            Y_test_pop  = pop['Y_std'][test_idx].detach().cpu().numpy()
        else:
            X_train_pop = pop['X_std'][train_idx][:, our_top_indices]
            Y_train_pop = pop['Y_std'][train_idx]
            X_test_pop  = pop['X_std'][test_idx][:, our_top_indices]
            Y_test_pop  = pop['Y_std'][test_idx]
        model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(X_train_pop, Y_train_pop)
        Y_pred = model.predict(X_test_pop)
        mse = mean_squared_error(Y_test_pop, Y_pred)
        r2 = r2_score(Y_test_pop, Y_pred)
        our_results.append({
            'pop_id': pop['pop_id'],
            'mse': mse,
            'r2': r2,
            'selected_indices': our_top_indices,
            'source' : 'our_method'
        })

    # Concatenate the results for results_comparison.csv (downstream model performance: MSE, R2)
    all_downstream_eval_results = []
    # These are the lists of dicts from RandomForestRegressor evaluations done earlier in the script
    all_downstream_eval_results.extend(baseline_lasso_results) 
    all_downstream_eval_results.extend(baseline_dro_lasso_results)
    all_downstream_eval_results.extend(baseline_xgb_results)
    all_downstream_eval_results.extend(baseline_dro_xgb_results)
    all_downstream_eval_results.extend(our_results) # 'our_results' is your method's downstream eval list

    results_comparison_df = pd.DataFrame(all_downstream_eval_results)

    if 'pop_id' in results_comparison_df.columns:
        results_comparison_df = results_comparison_df.rename(columns={'pop_id': 'population'})

    results_path = os.path.join(save_path, 'results_comparison.csv')
    results_comparison_df.to_csv(results_path, index=False)
    print(f"Downstream model performance comparison saved to: {results_path}")


    # This section is for variable selection quality (coverage, P/R/F1 of selected features)
    if not any('asc' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
        print("Data is not ASC. Aggregating variable selection quality results...")
        
        vsr = variable_selection_results # Main results dict from run_variable_selection

        # --- Our Method's Variable Selection Stats ---
        meaningful_indices_list = vsr.get('meaningful_indices', [[] for _ in pop_data]) # Ensure default

        # Calculate per-population coverage and overall coverage stats for "our method"
        our_pop_coverage_stats, our_overall_coverage_stats = compute_population_stats(
            vsr['selected_indices'],
            meaningful_indices_list
        )
        
        df_our = pd.DataFrame(our_pop_coverage_stats) # columns: population, selected_relevant_count, total_relevant, percentage
        if not df_our.empty:
            df_our = df_our.set_index('population').rename(columns={
                'selected_relevant_count': 'our_selected_relevant',
                'total_relevant': 'our_total_relevant',
                'percentage': 'our_coverage_pct'
            })
        else:
            df_our = pd.DataFrame(columns=['population', 'our_selected_relevant', 'our_total_relevant', 'our_coverage_pct']).set_index('population')

        # Calculate overall Precision, Recall, F1 for "our method"
        # (Based on the union of true features across all populations)
        sel_set_our = set(vsr['selected_indices'])
        # Ensure meaningful_indices_list elements are actual lists before union
        true_sets_our = [set(mi) for mi in meaningful_indices_list if isinstance(mi, list)]
        true_set_union_our = set.union(*true_sets_our) if true_sets_our else set()
        
        intersect_our = len(sel_set_our & true_set_union_our)
        our_overall_precision = intersect_our / len(sel_set_our) if sel_set_our else 0.0
        our_overall_recall = intersect_our / len(true_set_union_our) if true_set_union_our else 0.0
        our_overall_f1 = (2 * our_overall_precision * our_overall_recall / 
                           (our_overall_precision + our_overall_recall)) if (our_overall_precision + our_overall_recall) > 0 else 0.0

        # --- Baseline Method's Variable Selection Stats ---
        # Choose which baseline to use for this summary comparison.
        # The chosen key must exist in 'vsr' and contain results from baselines.py
        chosen_baseline_key_for_summary = 'baseline_lasso_results' # EXAMPLE: Can be 'baseline_xgb_results', etc.
        
        baseline_selection_summary_data = vsr.get(chosen_baseline_key_for_summary)
        if baseline_selection_summary_data is None:
            print(f"Warning: Chosen baseline '{chosen_baseline_key_for_summary}' not found in results. Baseline stats will be empty.")
            baseline_selection_summary_data = {} # Avoid errors later

        # Per-population coverage stats for the chosen baseline
        # These come directly from the 'baseline_pop_stats' key within the baseline's results dict
        baseline_pop_coverage_stats = baseline_selection_summary_data.get('baseline_pop_stats', [])
        
        df_base = pd.DataFrame(baseline_pop_coverage_stats) # columns: population, selected_relevant_count, total_relevant, percentage
        if not df_base.empty:
            df_base = df_base.set_index('population').rename(columns={
                'selected_relevant_count': 'base_selected_relevant',
                'total_relevant': 'base_total_relevant',
                'percentage': 'base_coverage_pct'
            })
        else:
            df_base = pd.DataFrame(columns=['population', 'base_selected_relevant', 'base_total_relevant', 'base_coverage_pct']).set_index('population')

        # Overall Precision, Recall, F1 for the chosen baseline
        # These come directly from the top-level keys within the baseline's results dict
        base_overall_precision = baseline_selection_summary_data.get('precision', 0.0)
        base_overall_recall = baseline_selection_summary_data.get('recall', 0.0)
        base_overall_f1 = baseline_selection_summary_data.get('f1_score', 0.0)
        
        # Overall coverage stats for the chosen baseline
        # This comes from 'baseline_overall_stats' key within the baseline's results dict
        base_overall_coverage_stats = baseline_selection_summary_data.get('baseline_overall_stats', {})

        # --- Merge for per-population variable_selection_per_population.csv ---
        summary_df_var_selection = df_our.join(df_base, how='outer').reset_index()
        summary_csv_path = os.path.join(save_path, 'variable_selection_per_population.csv')
        summary_df_var_selection.to_csv(summary_csv_path, index=False)
        print(f"Per-population variable selection quality summary saved to: {summary_csv_path}")

        # --- Overall stats for variable_selection_overall_stats.json ---
        overall_stats_summary = {
            'our_method': {
                'precision': our_overall_precision,
                'recall': our_overall_recall,
                'f1_score': our_overall_f1,
                'median_coverage_percentage': our_overall_coverage_stats.get('median_percentage'),
                'min_coverage_details': our_overall_coverage_stats.get('min_population_details'),
                'max_coverage_details': our_overall_coverage_stats.get('max_population_details'),
            },
            'chosen_baseline_for_summary': {
                'baseline_name': chosen_baseline_key_for_summary,
                'precision': base_overall_precision,
                'recall': base_overall_recall,
                'f1_score': base_overall_f1,
                'median_coverage_percentage': base_overall_coverage_stats.get('median_percentage'),
                'min_coverage_details': base_overall_coverage_stats.get('min_population_details'),
                'max_coverage_details': base_overall_coverage_stats.get('max_population_details'),
            }
        }

        overall_json_path = os.path.join(save_path, 'variable_selection_overall_stats.json')
        with open(overall_json_path, 'w') as f:
            json.dump(convert_numpy_to_python(overall_stats_summary), f, indent=2) # Assuming convert_numpy_to_python is available
        print(f"Overall variable selection quality stats saved to: {overall_json_path}")

    # Save the main variable selection results dictionary
    variable_selection_results_path = os.path.join(save_path, 'variable_selection_results.pkl')
    vsr = variable_selection_results  # Define vsr before using it
    with open(variable_selection_results_path, 'wb') as f:
        pickle.dump(vsr, f) # vsr is variable_selection_results
    print(f"Full variable selection results dictionary saved to: {variable_selection_results_path}")
    
    return vsr # which is variable_selection_results

def convert_numpy_to_python(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (torch.Tensor)): return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict): return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_to_python(item) for item in obj]
    if isinstance(obj, set): return [convert_numpy_to_python(item) for item in obj] # Convert set to list
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return "NaN" # Represent NaN/Inf as string
    return obj

def get_latest_run_number(save_path: str) -> int:
    if not os.path.exists(save_path): os.makedirs(save_path); return 0
    existing = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d)) and d.startswith('run_')]
    run_nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    return max(run_nums) + 1 if run_nums else 0

# =============================================================================
# Arg Parsing and Main Execution
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-population VSS (v7.2: Theta Option)')
    # Data args
    parser.add_argument('--m1', type=int, default=4)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--dataset-size', type=int, default=5000)
    parser.add_argument('--baseline-data-size', type=int, default=15000)
    parser.add_argument('--acs-data-fraction', type=float, default=0.1)
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    # Optimization args
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--budget', type=int, default=None, help="Selection budget (default: calculated)")
    parser.add_argument('--penalty-type', type=str, default='Reciprocal_L1', choices=['Reciprocal_L1', 'Neg_L1', 'Max_Dev', 'Quadratic_Barrier', 'Exponential', 'None'])
    parser.add_argument('--penalty-lambda', type=float, default=0.001)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--optimizer-type', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--parameterization', type=str, default='alpha', choices=['alpha', 'theta'], help='Parameter to optimize') # New
    parser.add_argument('--alpha-init', type=str, default='random_1', help='Config for initial alpha value (used to derive initial theta if needed)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--param-freezing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--smooth-minmax', type=float, default=float('inf'), help='Beta param for SmoothMax objective (inf for hard max)')
    # Gradient args
    parser.add_argument('--gradient-mode', type=str, default='autograd', choices=['autograd', 'reinforce'])
    parser.add_argument('--t2-estimator-type', type=str, default='kernel_if_like', choices=['mc_plugin', 'kernel_if_like'], help='T2 estimator type for gradient')
    parser.add_argument('--N-grad-samples', type=int, default=25, help='MC samples for grad')
    parser.add_argument('--use-baseline', action=argparse.BooleanOptionalAction, default=True, help='Baseline for REINFORCE')
    # Estimator args
    parser.add_argument('--estimator-type', type=str, default='if', choices=['plugin', 'if'], help='For T1/E[Y|X] precomputation')
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr', 'xgb'])
    parser.add_argument('--objective-value-estimator', type=str, default='if', choices=['if', 'mc'], help='For tracking objective value')
    parser.add_argument('--k-kernel', type=int, default=500, help='k for kernel estimators') # Added k_kernel
    # Scheduler args
    parser.add_argument('--scheduler-type', type=str, default=None, choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--scheduler-step-size', type=int, default=30)
    parser.add_argument('--scheduler-gamma', type=float, default=0.1)
    parser.add_argument('--scheduler-milestones', type=int, nargs='+', default=[50, 80])
    parser.add_argument('--scheduler-t-max', type=int, default=-1)
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6)
    parser.add_argument('--scheduler-patience', type=int, default=10)
    # Other args
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-path', type=str, default='./results_v7/multi_population/')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--lasso-alpha', type=float, default=0.01, 
                   help='Regularization strength for Lasso baseline')

    return parser.parse_args()

def main():
    args = parse_args()
    if args.populations[0].lower().startswith('baseline_failure'):
        args.budget = 3
        base_save_path = os.path.join(args.save_path, f'{args.populations[0]}/')
    elif args.populations[0].lower().startswith('linear'):
        base_save_path = os.path.join(args.save_path, 'linear_regression/')
    elif args.populations[0].lower().startswith('sinusoidal'):  
        base_save_path = os.path.join(args.save_path, 'sinusoidal_regression/')
    elif args.populations[0].lower().startswith('cubic_regression'):
        base_save_path = os.path.join(args.save_path, 'cubic_regression/')
    else:
        base_save_path = args.save_path
    os.makedirs(base_save_path, exist_ok=True)
    run_no = get_latest_run_number(base_save_path)
    save_path = os.path.join(base_save_path, f'run_{run_no}/')
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved in: {save_path}")

    pop_configs = [{'pop_id': i, 'dataset_type': dt} for i, dt in enumerate(args.populations)]

    # Prepare scheduler kwargs
    scheduler_kwargs = {}
    if args.scheduler_type == 'StepLR': scheduler_kwargs = {'step_size': args.scheduler_step_size, 'gamma': args.scheduler_gamma}
    elif args.scheduler_type == 'MultiStepLR': scheduler_kwargs = {'milestones': args.scheduler_milestones, 'gamma': args.scheduler_gamma}
    elif args.scheduler_type == 'ExponentialLR': scheduler_kwargs = {'gamma': args.scheduler_gamma}
    elif args.scheduler_type == 'CosineAnnealingLR': scheduler_kwargs = {'T_max': args.num_epochs if args.scheduler_t_max < 0 else args.scheduler_t_max, 'eta_min': args.scheduler_min_lr}
    elif args.scheduler_type == 'ReduceLROnPlateau': scheduler_kwargs = {'factor': args.scheduler_gamma, 'patience': args.scheduler_patience, 'min_lr': args.scheduler_min_lr}

    # Save experiment parameters
    experiment_params = vars(args).copy()
    experiment_params['pop_configs'] = pop_configs
    experiment_params['script_version'] = 'v7.2_theta_option' # Updated version name
    experiment_params['final_save_path'] = save_path
    print("\n--- Running Experiment (v7.2 + Theta Option) ---")
    print(json.dumps(convert_numpy_to_python(experiment_params), indent=2))
    with open(os.path.join(save_path, 'params_v7.json'), 'w') as f:
        json.dump(convert_numpy_to_python(experiment_params), f, indent=2)

    # Run the experiment
    results = run_experiment_multi_population(
        pop_configs=pop_configs, m1=args.m1, m=args.m, dataset_size=args.dataset_size,
        baseline_data_size=args.baseline_data_size,
        acs_data_fraction=args.acs_data_fraction,
        noise_scale=args.noise_scale, corr_strength=args.corr_strength, num_epochs=args.num_epochs,
        budget=args.budget, penalty_type=args.penalty_type, penalty_lambda=args.penalty_lambda,
        learning_rate=args.learning_rate, optimizer_type=args.optimizer_type,
        parameterization=args.parameterization, # Pass parameterization choice
        alpha_init=args.alpha_init,
        early_stopping_patience=args.patience, param_freezing=args.param_freezing,
        smooth_minmax=args.smooth_minmax,
        gradient_mode=args.gradient_mode, 
        t2_estimator_type=args.t2_estimator_type,
        N_grad_samples=args.N_grad_samples, use_baseline=args.use_baseline,
        estimator_type=args.estimator_type, base_model_type=args.base_model_type,
        objective_value_estimator=args.objective_value_estimator,
        k_kernel=args.k_kernel, # Pass k_kernel
        scheduler_type=args.scheduler_type, scheduler_kwargs=scheduler_kwargs,
        seed=args.seed, save_path=save_path, verbose=args.verbose
    )

    # --- Post-processing and Saving ---
    print("\n--- Experiment Finished ---")
    if results.get('error'):
        print(f"Experiment failed: {results['error']}")
        return

if __name__ == '__main__':
    main()