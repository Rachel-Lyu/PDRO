import os
import pickle
import torch
torch.set_num_threads(4)
import numpy as np
import time
import math
from typing import List, Dict, Any, Optional

class VariableSelector:
    def __init__(self, checkpoint_manager=None):
        """
        Initialize the variable selector with an optional checkpoint manager.
        
        Args:
            checkpoint_manager: Object that handles saving and loading checkpoints
        """
        self.checkpoint_manager = checkpoint_manager
        
    def _init_parameter(self, m, parameterization, alpha_init, noise=0.1, device="cpu"):
        """Initialize parameter based on parameterization"""
        from global_vars import CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA, THETA_CLAMP_MIN, THETA_CLAMP_MAX, EPS
        
        init_alpha_val = torch.ones(m, device=device)
        alpha_init_lower = alpha_init.lower()
        
        if alpha_init_lower.startswith("random_"):
            try:
                k = float(alpha_init_lower.split('_', 1)[1])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid numeric value in alpha_init: {alpha_init}")
            init_alpha_val = k * torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
        elif alpha_init_lower == "ones" or alpha_init_lower == "random":
            init_alpha_val = torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
        elif alpha_init_lower == "adaptive":
            # This would require pooled data from populations for variance calculation
            # Will be handled in the run_selection method
            pass
        else:
            raise ValueError("alpha_init must be 'random_X', 'ones', 'random', or 'adaptive'")
        
        init_alpha_val.clamp_(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        
        if parameterization == 'alpha':
            param = torch.nn.Parameter(init_alpha_val)
        elif parameterization == 'theta':
            param = torch.nn.Parameter(torch.log(torch.clamp(init_alpha_val, min=EPS)).clamp_(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX))
        else:
            raise ValueError("parameterization must be 'alpha' or 'theta'")
        
        return param

    def _setup_optimizer(self, param, optimizer_type, learning_rate):
        """Setup the optimizer based on type"""
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam([param], lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD([param], lr=learning_rate, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
        
        return optimizer

    def _setup_scheduler(self, optimizer, scheduler_type, scheduler_kwargs):
        """Setup learning rate scheduler if specified"""
        if not scheduler_type or not scheduler_kwargs:
            return None
        
        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type, None)
            if scheduler_class:
                return scheduler_class(optimizer, **scheduler_kwargs)
        except Exception as e:
            print(f"Warning: Failed to initialize scheduler '{scheduler_type}': {e}")
        
        return None

    def _compute_penalty(self, alpha, penalty_type, penalty_lambda):
        """Compute the penalty term for the objective function"""
        from global_vars import CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA, EPS
        
        alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        
        if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
            return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype, requires_grad=alpha.requires_grad)
        
        pt_lower = penalty_type.lower()
        if pt_lower == "reciprocal_l1":
            return penalty_lambda * torch.sum(1.0 / (alpha_clamped + EPS))
        if pt_lower == "neg_l1":
            return penalty_lambda * torch.sum(torch.abs(alpha_clamped))
        if pt_lower == "max_dev":
            return penalty_lambda * torch.sum(torch.abs(1.0 - alpha_clamped))
        if pt_lower == "quadratic_barrier":
            return penalty_lambda * torch.sum((alpha_clamped + EPS) ** (-2))
        if pt_lower == "exponential":
            return penalty_lambda * torch.sum(torch.exp(-alpha_clamped))
        
        raise ValueError(f"Unknown penalty_type: {penalty_type}")

    def run_selection(self, pop_data, m1, m, budget, params):
        """Run our variable selection method with checkpointing"""
        # Check if already completed
        if self.checkpoint_manager and self.checkpoint_manager.is_our_method_complete():
            print("Our method already completed. Loading results...")
            results = self.checkpoint_manager.load_checkpoint('our_method')
            if results:
                return results
            else:
                print("Could not load checkpoint. Re-running our method.")
                
        # Extract parameters from params dict
        save_path = params.get('save_path', './results/')
        parameterization = params.get('parameterization', 'alpha')
        num_epochs = params.get('num_epochs', 100)
        penalty_type = params.get('penalty_type', None)
        penalty_lambda = params.get('penalty_lambda', 0.0)
        early_stopping_patience = params.get('early_stopping_patience', 15)
        param_freezing = params.get('param_freezing', True)
        t2_estimator_type = params.get('t2_estimator_type', 'mc_plugin')
        gradient_mode = params.get('gradient_mode', 'autograd')
        N_grad_samples = params.get('N_grad_samples', 25)
        smooth_minmax = params.get('smooth_minmax', float('inf'))
        use_baseline = params.get('use_baseline', True)
        k_kernel = params.get('k_kernel', 1000)
        objective_value_estimator = params.get('objective_value_estimator', 'if')
        base_model_type = params.get('base_model_type', 'rf')
        
        print(f"Starting our variable selection method...")
        start_time = time.time()
        
        # Initialize parameters, optimizer, etc.
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        if params.get('alpha_init') == 'adaptive':
            # Combine data from all populations for adaptive initialization
            X_combined = np.concatenate([pop['X_std'].cpu().numpy() for pop in pop_data])
            feature_vars = np.var(X_combined, axis=0)
            normalized_vars = feature_vars / np.mean(feature_vars)
            alpha_init_val = 2.0 * normalized_vars  # Scale factor of 2.0
            param = torch.nn.Parameter(torch.tensor(alpha_init_val, dtype=torch.float32, device=device))
        else:
            param = self._init_parameter(m, parameterization, params.get('alpha_init', 'random_1'), device=device)
        
        # Setup optimizer
        optimizer_type = params.get('optimizer_type', 'adam')
        learning_rate = params.get('learning_rate', 0.01)
        optimizer = self._setup_optimizer(param, optimizer_type, learning_rate)
        
        # Setup scheduler if specified
        scheduler_type = params.get('scheduler_type')
        scheduler_kwargs = params.get('scheduler_kwargs', {})
        scheduler = self._setup_scheduler(optimizer, scheduler_type, scheduler_kwargs)
        
        from global_vars import EPS, FREEZE_THRESHOLD_ALPHA, THETA_FREEZE_THRESHOLD
        
        # Optimization loop with intermediate checkpoints
        param_history = [param.detach().cpu().numpy().copy()]
        objective_history = []
        best_objective_val = float('inf')
        best_param = param.detach().cpu().numpy().copy()
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()
            
            with torch.no_grad():
                current_alpha = param.data.clamp(min=params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                            max=params.get('CLAMP_MAX_ALPHA', 1e5)) if parameterization == 'alpha' \
                                else torch.exp(param.data).clamp(min=params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                                                max=params.get('CLAMP_MAX_ALPHA', 1e5))
            
            population_objectives_for_grad = []
            param.requires_grad_(True)
            
            # Calculate objective for each population
            for pop in pop_data:
                X_std, E_Yx_std, term1_std_val = pop['X_std'], pop['E_Yx_std'], pop['term1_std']
                
                if gradient_mode == 'autograd':
                    if t2_estimator_type == 'mc_plugin':
                        # Import this here to avoid circular imports
                        from estimators import estimate_T2_mc_flexible
                        term2_value = estimate_T2_mc_flexible(
                            X_std_torch=X_std, 
                            E_Yx_std_torch=E_Yx_std, 
                            param_torch=param,
                            param_type=parameterization, 
                            n_mc_samples=N_grad_samples, 
                            k_kernel=k_kernel
                        )
                    elif t2_estimator_type == 'kernel_if_like':
                        from estimators import estimate_T2_kernel_IF_like_flexible
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
                    penalty_val = self._compute_penalty(alpha_for_penalty, penalty_type, penalty_lambda)
                    pop_obj_grad = term1_std_val - term2_value + penalty_val
                    population_objectives_for_grad.append(pop_obj_grad)
                
                elif gradient_mode == 'reinforce':
                    # We'll handle REINFORCE gradient calculation after identifying the winning population
                    pass
            
            # Calculate objective value for tracking/stopping
            population_objective_values = []
            current_param_val_detached = param.detach().clone()
            
            for pop in pop_data:
                if objective_value_estimator == 'if':
                    from estimators import IF_estimator_squared_conditional
                    
                    with torch.no_grad():
                        # Get the device of X_std
                        x_device = pop['X_std'].device
                        
                        # Move current_alpha to the same device as X_std
                        current_alpha_device = current_alpha.to(x_device)
                        
                        # Now all tensors are on the same device
                        noise = torch.randn_like(pop['X_std'])
                        S_alpha = pop['X_std'] + noise * torch.sqrt(current_alpha_device)
                        
                        # Convert to CPU only after all operations are complete
                        S_alpha_np = S_alpha.cpu().numpy()
                        Y_std_np = pop['Y_std'].cpu().numpy()
                        
                        try:
                            term2_val_float = IF_estimator_squared_conditional(
                                S_alpha_np, Y_std_np, 
                                estimator_type=base_model_type, 
                                n_folds=params.get('N_FOLDS', 5)
                            )
                        except Exception as e:
                            print(f"Warning: IF T2 calculation failed: {e}")
                            term2_val_float = float('nan')
                        
                        if not math.isnan(term2_val_float):
                            penalty = self._compute_penalty(current_alpha, penalty_type, penalty_lambda).item()
                            obj_val = pop['term1_std'] - term2_val_float + penalty
                            population_objective_values.append(obj_val)
                        else:
                            population_objective_values.append(float('nan'))
                else:
                    # MC estimator logic - would need to implement this based on gd_pops_v8.py
                    population_objective_values.append(float('nan'))
            
            # Calculate robust objective for gradient
            valid_obj_values = [v for v in population_objective_values if not math.isnan(v)]
            if not valid_obj_values:
                print(f"ERROR: All population objective values are NaN at epoch {epoch}. Stopping.")
                break
                
            current_robust_objective_value = max(valid_obj_values)
            winning_pop_idx = population_objective_values.index(current_robust_objective_value)
            objective_history.append(current_robust_objective_value)
            
            # Calculate gradient and update parameters
            if gradient_mode == 'autograd':
                objectives_tensor = torch.stack(population_objectives_for_grad)
                
                if torch.isfinite(torch.tensor(smooth_minmax)) and smooth_minmax > 0:
                    beta = smooth_minmax
                    with torch.no_grad():
                        M = torch.max(beta * objectives_tensor)
                    logsumexp_val = M + torch.log(torch.sum(torch.exp(beta * objectives_tensor - M)) + EPS)
                    robust_objective_for_grad = (1.0 / beta) * logsumexp_val
                else:
                    robust_objective_for_grad, _ = torch.max(objectives_tensor, dim=0)
                    
                robust_objective_for_grad.backward()
                total_gradient = param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            
            elif gradient_mode == 'reinforce':
                from estimators import estimate_gradient_reinforce_flexible
                
                winning_pop_data = pop_data[winning_pop_idx]
                total_gradient = estimate_gradient_reinforce_flexible(
                    winning_pop_data['X_std'], 
                    winning_pop_data['E_Yx_std'], 
                    param, 
                    parameterization,
                    N_grad_samples, 
                    k_kernel, 
                    penalty_type, 
                    penalty_lambda, 
                    use_baseline,
                    t2_estimator_type_for_reward=t2_estimator_type,
                    Y_std_torch=winning_pop_data['Y_std'] if t2_estimator_type == 'kernel_if_like' else None
                )
                
                if param.grad is not None:
                    param.grad.zero_()
                param.grad = total_gradient
            
            # Apply parameter freezing if enabled
            if param_freezing and param.grad is not None:
                with torch.no_grad():
                    freeze_thresh = FREEZE_THRESHOLD_ALPHA if parameterization == 'alpha' else THETA_FREEZE_THRESHOLD
                    frozen_mask = param.data < freeze_thresh
                    param.grad[frozen_mask] = 0.0
                    
                    # Also clear optimizer state for frozen params
                    for group in optimizer.param_groups:
                        if group['params'][0] is param:
                            for p_state_key, p_state_val in optimizer.state[param].items():
                                if isinstance(p_state_val, torch.Tensor) and p_state_val.shape == param.shape:
                                    p_state_val[frozen_mask] = 0.0
                            break
            
            # Gradient clipping for stability
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_([param], max_norm=5.0)
            
            # Optimizer step
            optimizer.step()
            
            # Clamp parameter values
            with torch.no_grad():
                if parameterization == 'alpha':
                    param.data.clamp_(min=params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                    max=params.get('CLAMP_MAX_ALPHA', 1e5))
                else:
                    param.data.clamp_(min=params.get('THETA_CLAMP_MIN', -11.5), 
                                    max=params.get('THETA_CLAMP_MAX', 11.5))
            
            # Update scheduler if used
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_robust_objective_value)
                else:
                    scheduler.step()
            
            # Save parameter history
            param_history.append(param.detach().cpu().numpy().copy())
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs} | Obj: {current_robust_objective_value:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"Win Pop: {pop_data[winning_pop_idx]['pop_id']}")
            
            # Early stopping check
            if current_robust_objective_value < best_objective_val - EPS:
                best_objective_val = current_robust_objective_value
                best_param = param.detach().cpu().numpy().copy()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            # Reset early stopping counter for first 40 epochs to allow more exploration
            if epoch < 40:
                early_stopping_counter = 0
            elif early_stopping_counter >= early_stopping_patience and epoch > 40:
                if abs(current_robust_objective_value - best_objective_val) < 0.01 * abs(best_objective_val):
                    print(f"Early stopping at epoch {epoch}.")
                    break
            
            # Save intermediate checkpoint every N epochs
            if epoch % 10 == 0 and self.checkpoint_manager:
                intermediate_state = {
                    'epoch': epoch,
                    'param': param.detach().cpu().numpy().copy(),
                    'best_objective_val': best_objective_val,
                    'objective_history': objective_history,
                    'param_history': param_history,
                    'best_param': best_param,
                    'early_stopping_counter': early_stopping_counter
                }
                self.checkpoint_manager.save_checkpoint(
                    'our_method_intermediate', 
                    intermediate_state,
                    f"our_method_checkpoint_epoch{epoch}.pkl"
                )
        
        # Final results
        final_alpha_np = np.exp(best_param) if parameterization == 'theta' else best_param
        final_alpha_np = np.clip(final_alpha_np, params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                params.get('CLAMP_MAX_ALPHA', 1e5))
        selected_indices = np.argsort(final_alpha_np)[:budget]
        
        results = {
            'selected_indices': selected_indices.tolist(),
            'final_alpha': final_alpha_np.tolist(),
            'all_indices_ranked': np.argsort(final_alpha_np).tolist(),
            'best_objective_val': best_objective_val,
            'objective_history': objective_history,
            'param_history': [p.tolist() for p in param_history],
            'total_time_seconds': time.time() - start_time,
            'parameterization': parameterization,
            't2_estimator_type': t2_estimator_type,
            'gradient_mode': gradient_mode,
            'stopping_epoch': epoch
        }
        
        # Save final results
        if self.checkpoint_manager:
            results_path = self.checkpoint_manager.save_checkpoint('our_method', results)
            self.checkpoint_manager.mark_our_method_complete(results_path)
        
        return results
import os
import pickle
import torch
import numpy as np
import time
import math
from typing import List, Dict, Any, Optional

class VariableSelector:
    def __init__(self, checkpoint_manager=None):
        """
        Initialize the variable selector with an optional checkpoint manager.
        
        Args:
            checkpoint_manager: Object that handles saving and loading checkpoints
        """
        self.checkpoint_manager = checkpoint_manager
        
    def _init_parameter(self, m, parameterization, alpha_init, noise=0.1, device="cpu"):
        """Initialize parameter based on parameterization"""
        from global_vars import CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA, THETA_CLAMP_MIN, THETA_CLAMP_MAX, EPS
        
        init_alpha_val = torch.ones(m, device=device)
        alpha_init_lower = alpha_init.lower()
        
        if alpha_init_lower.startswith("random_"):
            try:
                k = float(alpha_init_lower.split('_', 1)[1])
            except (ValueError, IndexError):
                raise ValueError(f"Invalid numeric value in alpha_init: {alpha_init}")
            init_alpha_val = k * torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
        elif alpha_init_lower == "ones" or alpha_init_lower == "random":
            init_alpha_val = torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
        elif alpha_init_lower == "adaptive":
            # This would require pooled data from populations for variance calculation
            # Will be handled in the run_selection method
            pass
        else:
            raise ValueError("alpha_init must be 'random_X', 'ones', 'random', or 'adaptive'")
        
        init_alpha_val.clamp_(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        
        if parameterization == 'alpha':
            param = torch.nn.Parameter(init_alpha_val)
        elif parameterization == 'theta':
            param = torch.nn.Parameter(torch.log(torch.clamp(init_alpha_val, min=EPS)).clamp_(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX))
        else:
            raise ValueError("parameterization must be 'alpha' or 'theta'")
        
        return param

    def _setup_optimizer(self, param, optimizer_type, learning_rate):
        """Setup the optimizer based on type"""
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam([param], lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD([param], lr=learning_rate, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unsupported optimizer_type: {optimizer_type}")
        
        return optimizer

    def _setup_scheduler(self, optimizer, scheduler_type, scheduler_kwargs):
        """Setup learning rate scheduler if specified"""
        if not scheduler_type or not scheduler_kwargs:
            return None
        
        try:
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type, None)
            if scheduler_class:
                return scheduler_class(optimizer, **scheduler_kwargs)
        except Exception as e:
            print(f"Warning: Failed to initialize scheduler '{scheduler_type}': {e}")
        
        return None

    def _compute_penalty(self, alpha, penalty_type, penalty_lambda):
        """Compute the penalty term for the objective function"""
        from global_vars import CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA, EPS
        
        alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        
        if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
            return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype, requires_grad=alpha.requires_grad)
        
        pt_lower = penalty_type.lower()
        if pt_lower == "reciprocal_l1":
            return penalty_lambda * torch.sum(1.0 / (alpha_clamped + EPS))
        if pt_lower == "neg_l1":
            return penalty_lambda * torch.sum(torch.abs(alpha_clamped))
        if pt_lower == "max_dev":
            return penalty_lambda * torch.sum(torch.abs(1.0 - alpha_clamped))
        if pt_lower == "quadratic_barrier":
            return penalty_lambda * torch.sum((alpha_clamped + EPS) ** (-2))
        if pt_lower == "exponential":
            return penalty_lambda * torch.sum(torch.exp(-alpha_clamped))
        
        raise ValueError(f"Unknown penalty_type: {penalty_type}")

    def run_selection(self, pop_data, m1, m, budget, params):
        """Run our variable selection method with checkpointing"""
        # Check if already completed
        if self.checkpoint_manager and self.checkpoint_manager.is_our_method_complete():
            print("Our method already completed. Loading results...")
            results = self.checkpoint_manager.load_checkpoint('our_method')
            if results:
                return results
            else:
                print("Could not load checkpoint. Re-running our method.")
                
        # Extract parameters from params dict
        save_path = params.get('save_path', './results/')
        parameterization = params.get('parameterization', 'alpha')
        num_epochs = params.get('num_epochs', 100)
        penalty_type = params.get('penalty_type', None)
        penalty_lambda = params.get('penalty_lambda', 0.0)
        early_stopping_patience = params.get('early_stopping_patience', 15)
        param_freezing = params.get('param_freezing', True)
        t2_estimator_type = params.get('t2_estimator_type', 'mc_plugin')
        gradient_mode = params.get('gradient_mode', 'autograd')
        N_grad_samples = params.get('N_grad_samples', 25)
        smooth_minmax = params.get('smooth_minmax', float('inf'))
        use_baseline = params.get('use_baseline', True)
        k_kernel = params.get('k_kernel', 1000)
        objective_value_estimator = params.get('objective_value_estimator', 'if')
        base_model_type = params.get('base_model_type', 'rf')
        
        print(f"Starting our variable selection method...")
        start_time = time.time()
        
        # Initialize parameters, optimizer, etc.
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        if params.get('alpha_init') == 'adaptive':
            # Combine data from all populations for adaptive initialization
            X_combined = np.concatenate([pop['X_std'].cpu().numpy() for pop in pop_data])
            feature_vars = np.var(X_combined, axis=0)
            normalized_vars = feature_vars / np.mean(feature_vars)
            alpha_init_val = 2.0 * normalized_vars  # Scale factor of 2.0
            param = torch.nn.Parameter(torch.tensor(alpha_init_val, dtype=torch.float32, device=device))
        else:
            param = self._init_parameter(m, parameterization, params.get('alpha_init', 'random_1'), device=device)
        
        # Setup optimizer
        optimizer_type = params.get('optimizer_type', 'adam')
        learning_rate = params.get('learning_rate', 0.01)
        optimizer = self._setup_optimizer(param, optimizer_type, learning_rate)
        
        # Setup scheduler if specified
        scheduler_type = params.get('scheduler_type')
        scheduler_kwargs = params.get('scheduler_kwargs', {})
        scheduler = self._setup_scheduler(optimizer, scheduler_type, scheduler_kwargs)
        
        from global_vars import EPS, FREEZE_THRESHOLD_ALPHA, THETA_FREEZE_THRESHOLD
        
        # Optimization loop with intermediate checkpoints
        param_history = [param.detach().cpu().numpy().copy()]
        objective_history = []
        best_objective_val = float('inf')
        best_param = param.detach().cpu().numpy().copy()
        early_stopping_counter = 0
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()
            
            with torch.no_grad():
                current_alpha = param.data.clamp(min=params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                            max=params.get('CLAMP_MAX_ALPHA', 1e5)) if parameterization == 'alpha' \
                                else torch.exp(param.data).clamp(min=params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                                                max=params.get('CLAMP_MAX_ALPHA', 1e5))
            
            population_objectives_for_grad = []
            param.requires_grad_(True)
            
            # Calculate objective for each population
            for pop in pop_data:
                X_std, E_Yx_std, term1_std_val = pop['X_std'], pop['E_Yx_std'], pop['term1_std']
                
                if gradient_mode == 'autograd':
                    if t2_estimator_type == 'mc_plugin':
                        # Import this here to avoid circular imports
                        from estimators import estimate_T2_mc_flexible
                        term2_value = estimate_T2_mc_flexible(
                            X_std_torch=X_std, 
                            E_Yx_std_torch=E_Yx_std, 
                            param_torch=param,
                            param_type=parameterization, 
                            n_mc_samples=N_grad_samples, 
                            k_kernel=k_kernel
                        )
                    elif t2_estimator_type == 'kernel_if_like':
                        from estimators import estimate_T2_kernel_IF_like_flexible
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
                    penalty_val = self._compute_penalty(alpha_for_penalty, penalty_type, penalty_lambda)
                    pop_obj_grad = term1_std_val - term2_value + penalty_val
                    population_objectives_for_grad.append(pop_obj_grad)
                
                elif gradient_mode == 'reinforce':
                    # We'll handle REINFORCE gradient calculation after identifying the winning population
                    pass
            
            # Calculate objective value for tracking/stopping
            population_objective_values = []
            current_param_val_detached = param.detach().clone()
            
            for pop in pop_data:
                if objective_value_estimator == 'if':
                    from estimators import IF_estimator_squared_conditional
                    
                    with torch.no_grad():
                        # Get the device of X_std
                        x_device = pop['X_std'].device
                        
                        # Move current_alpha to the same device as X_std
                        current_alpha_device = current_alpha.to(x_device)
                        
                        # Now all tensors are on the same device
                        noise = torch.randn_like(pop['X_std'])
                        S_alpha = pop['X_std'] + noise * torch.sqrt(current_alpha_device)
                        
                        # Convert to CPU only after all operations are complete
                        S_alpha_np = S_alpha.cpu().numpy()
                        Y_std_np = pop['Y_std'].cpu().numpy()
                        
                        try:
                            term2_val_float = IF_estimator_squared_conditional(
                                S_alpha_np, Y_std_np, 
                                estimator_type=base_model_type, 
                                n_folds=params.get('N_FOLDS', 5)
                            )
                        except Exception as e:
                            print(f"Warning: IF T2 calculation failed: {e}")
                            term2_val_float = float('nan')
                        
                        if not math.isnan(term2_val_float):
                            penalty = self._compute_penalty(current_alpha, penalty_type, penalty_lambda).item()
                            obj_val = pop['term1_std'] - term2_val_float + penalty
                            population_objective_values.append(obj_val)
                        else:
                            population_objective_values.append(float('nan'))
                else:
                    # MC estimator logic - would need to implement this based on gd_pops_v8.py
                    population_objective_values.append(float('nan'))
            
            # Calculate robust objective for gradient
            valid_obj_values = [v for v in population_objective_values if not math.isnan(v)]
            if not valid_obj_values:
                print(f"ERROR: All population objective values are NaN at epoch {epoch}. Stopping.")
                break
                
            current_robust_objective_value = max(valid_obj_values)
            winning_pop_idx = population_objective_values.index(current_robust_objective_value)
            objective_history.append(current_robust_objective_value)
            
            # Calculate gradient and update parameters
            if gradient_mode == 'autograd':
                objectives_tensor = torch.stack(population_objectives_for_grad)
                
                if torch.isfinite(torch.tensor(smooth_minmax)) and smooth_minmax > 0:
                    beta = smooth_minmax
                    with torch.no_grad():
                        M = torch.max(beta * objectives_tensor)
                    logsumexp_val = M + torch.log(torch.sum(torch.exp(beta * objectives_tensor - M)) + EPS)
                    robust_objective_for_grad = (1.0 / beta) * logsumexp_val
                else:
                    robust_objective_for_grad, _ = torch.max(objectives_tensor, dim=0)
                    
                robust_objective_for_grad.backward()
                total_gradient = param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            
            elif gradient_mode == 'reinforce':
                from estimators import estimate_gradient_reinforce_flexible
                
                winning_pop_data = pop_data[winning_pop_idx]
                total_gradient = estimate_gradient_reinforce_flexible(
                    winning_pop_data['X_std'], 
                    winning_pop_data['E_Yx_std'], 
                    param, 
                    parameterization,
                    N_grad_samples, 
                    k_kernel, 
                    penalty_type, 
                    penalty_lambda, 
                    use_baseline,
                    t2_estimator_type_for_reward=t2_estimator_type,
                    Y_std_torch=winning_pop_data['Y_std'] if t2_estimator_type == 'kernel_if_like' else None
                )
                
                if param.grad is not None:
                    param.grad.zero_()
                param.grad = total_gradient
            
            # Apply parameter freezing if enabled
            if param_freezing and param.grad is not None:
                with torch.no_grad():
                    freeze_thresh = FREEZE_THRESHOLD_ALPHA if parameterization == 'alpha' else THETA_FREEZE_THRESHOLD
                    frozen_mask = param.data < freeze_thresh
                    param.grad[frozen_mask] = 0.0
                    
                    # Also clear optimizer state for frozen params
                    for group in optimizer.param_groups:
                        if group['params'][0] is param:
                            for p_state_key, p_state_val in optimizer.state[param].items():
                                if isinstance(p_state_val, torch.Tensor) and p_state_val.shape == param.shape:
                                    p_state_val[frozen_mask] = 0.0
                            break
            
            # Gradient clipping for stability
            if param.grad is not None:
                torch.nn.utils.clip_grad_norm_([param], max_norm=5.0)
            
            # Optimizer step
            optimizer.step()
            
            # Clamp parameter values
            with torch.no_grad():
                if parameterization == 'alpha':
                    param.data.clamp_(min=params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                    max=params.get('CLAMP_MAX_ALPHA', 1e5))
                else:
                    param.data.clamp_(min=params.get('THETA_CLAMP_MIN', -11.5), 
                                    max=params.get('THETA_CLAMP_MAX', 11.5))
            
            # Update scheduler if used
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(current_robust_objective_value)
                else:
                    scheduler.step()
            
            # Save parameter history
            param_history.append(param.detach().cpu().numpy().copy())
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs} | Obj: {current_robust_objective_value:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"Win Pop: {pop_data[winning_pop_idx]['pop_id']}")
            
            # Early stopping check
            if current_robust_objective_value < best_objective_val - EPS:
                best_objective_val = current_robust_objective_value
                best_param = param.detach().cpu().numpy().copy()
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            # Reset early stopping counter for first 40 epochs to allow more exploration
            if epoch < 40:
                early_stopping_counter = 0
            elif early_stopping_counter >= early_stopping_patience and epoch > 40:
                if abs(current_robust_objective_value - best_objective_val) < 0.01 * abs(best_objective_val):
                    print(f"Early stopping at epoch {epoch}.")
                    break
            
            # Save intermediate checkpoint every N epochs
            if epoch % 10 == 0 and self.checkpoint_manager:
                intermediate_state = {
                    'epoch': epoch,
                    'param': param.detach().cpu().numpy().copy(),
                    'best_objective_val': best_objective_val,
                    'objective_history': objective_history,
                    'param_history': param_history,
                    'best_param': best_param,
                    'early_stopping_counter': early_stopping_counter
                }
                self.checkpoint_manager.save_checkpoint(
                    'our_method_intermediate', 
                    intermediate_state,
                    f"our_method_checkpoint_epoch{epoch}.pkl"
                )
        
        # Final results
        final_alpha_np = np.exp(best_param) if parameterization == 'theta' else best_param
        final_alpha_np = np.clip(final_alpha_np, params.get('CLAMP_MIN_ALPHA', 1e-5), 
                                params.get('CLAMP_MAX_ALPHA', 1e5))
        selected_indices = np.argsort(final_alpha_np)[:budget]
        
        results = {
            'selected_indices': selected_indices.tolist(),
            'final_alpha': final_alpha_np.tolist(),
            'all_indices_ranked': np.argsort(final_alpha_np).tolist(),
            'best_objective_val': best_objective_val,
            'objective_history': objective_history,
            'param_history': [p.tolist() for p in param_history],
            'total_time_seconds': time.time() - start_time,
            'parameterization': parameterization,
            't2_estimator_type': t2_estimator_type,
            'gradient_mode': gradient_mode,
            'stopping_epoch': epoch
        }
        
        # Save final results
        if self.checkpoint_manager:
            results_path = self.checkpoint_manager.save_checkpoint('our_method', results)
            self.checkpoint_manager.mark_our_method_complete(results_path)
        
        return results