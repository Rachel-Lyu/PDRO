# estimators.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree
from sklearn.base import clone # Import clone
import torch # Set number of threads for PyTorch
from torch import Tensor
import matplotlib.pyplot as plt
import xgboost as xgb # Import XGBoost
from typing import Optional
import math

from global_vars import *
torch.set_num_threads(CPU_COUNT)

def compute_penalty(alpha: torch.Tensor, # Input is always alpha
                    penalty_type: Optional[str],
                    penalty_lambda: float,
                    epsilon: float = EPS) -> torch.Tensor:
    """
    Compute a penalty term P(alpha) designed to encourage large alpha values.
    We minimize L = (T1 - T2) + P(alpha).
    """
    # Clamp alpha within the function for calculation, ensuring gradients flow
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)

    if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
        return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype, requires_grad=alpha.requires_grad)

    penalty_type_lower = penalty_type.lower()

    if penalty_type_lower == "reciprocal_l1":
        return penalty_lambda * torch.sum(1.0 / (alpha_clamped + epsilon))
    elif penalty_type_lower == "neg_l1":
        print("Warning: Using Neg_L1 penalty encourages small alpha.")
        return penalty_lambda * torch.sum(torch.abs(alpha_clamped))
    elif penalty_type_lower == "max_dev":
        target_val = torch.tensor(1.0, device=alpha.device) # Target alpha=1
        return penalty_lambda * torch.sum(torch.abs(target_val - alpha_clamped))
    elif penalty_type_lower == "quadratic_barrier":
        return penalty_lambda * torch.sum((alpha_clamped + epsilon) ** (-2))
    elif penalty_type_lower == "exponential":
        return penalty_lambda * torch.sum(torch.exp(-alpha_clamped))
    else:
        raise ValueError("Unknown penalty_type: " + str(penalty_type))


# =============================================================================
# K-fold based estimators for conditional means and squared functionals
# =============================================================================

def plugin_estimator_conditional_mean(X, Y, estimator_type="rf", n_folds=N_FOLDS,
                                      seed=42):
    """
    Compute out-of-fold plugin predictions for E[Y|X] using K-fold CV.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    n_samples = X.shape[0]
    out_preds = np.zeros(n_samples)

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1) # Example parameters
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base) # Use clone for fresh model
        model.fit(X, Y)
        return model.predict(X)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            out_preds[test_idx] = model.predict(X_test)
        return out_preds

def plugin_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS,
                                         seed=42):
    """
    Compute the plugin estimator for E[E[Y|X]^2] using K-fold CV.
    Returns a scalar computed out-of-fold.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    n_samples = X.shape[0]

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base)
        model.fit(X, Y)
        mu_X = model.predict(X)
        return np.mean(mu_X ** 2)
    else:
        mu_X_all = np.zeros(n_samples)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            mu_X_all[test_idx] = model.predict(X_test)
        return np.mean(mu_X_all ** 2)


def IF_estimator_conditional_mean(X, Y, estimator_type="rf",
                                  n_folds=5,
                                  k_neighbors_factor=0.1, # k as a fraction of n_samples
                                  min_k_neighbors=10,     # Minimum k value
                                  bandwidth_factor=0.1,
                                  seed=42):  # Factor for bandwidth heuristic
    """
    Computes the Influence Function (IF) based estimator for the conditional mean E[Y|X].

    Uses K-Fold cross-validation to mitigate bias from using the same data for
    model fitting and residual calculation. Adjusts k dynamically based on fold size.
    Supports 'rf', 'krr', and 'xgb' estimator types.

    Args:
        X (np.ndarray): Input features (n_samples, n_features).
        Y (np.ndarray): Outcome variable (n_samples,).
        estimator_type (str, optional): Base model type ('rf', 'krr', 'xgb'). Defaults to "rf".
        n_folds (int, optional): Number of folds for cross-validation.
                                 Set to <= 1 to disable CV. Defaults to 5.
        k_neighbors_factor (float, optional): Factor to determine default k
                                              (k = n_samples * factor). Defaults to 0.1.
        min_k_neighbors (int, optional): Minimum value for k neighbors. Defaults to 10.
        bandwidth_factor (float, optional): Factor for bandwidth heuristic
                                           (bw = factor * sqrt(n_features)). Defaults to 0.1.


    Returns:
        np.ndarray: Out-of-fold predictions for E[Y|X] (n_samples,).
                    Returns plugin predictions if CV is disabled or fails.
    """
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    bandwidth = bandwidth_factor * np.sqrt(n_features)
    if bandwidth < EPS:
        print(f"Warning: Calculated bandwidth is very small ({bandwidth}). Setting to EPS.")
        bandwidth = EPS
    default_k = max(min_k_neighbors, int(n_samples * k_neighbors_factor))

    # Define model base outside the loop/if conditions
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    # --- Case 1: No Cross-Validation (n_folds <= 1) ---
    if n_folds <= 1:
        print("Warning: Running IF estimator without cross-validation (n_folds <= 1).")
        k_actual = min(default_k, n_samples - 1 if n_samples > 1 else 1)
        if k_actual < 1:
             print("Error: Cannot perform k-NN correction with k < 1.")
             try:
                 model_fallback = clone(model_base)
                 model_fallback.fit(X, Y)
                 return model_fallback.predict(X)
             except Exception as e_plugin:
                 print(f"Error during fallback plugin calculation: {e_plugin}")
                 return np.full(n_samples, np.nan)

        # Fit base model
        model = None # Initialize
        try:
            model = clone(model_base)
            model.fit(X, Y)
            mu_X = model.predict(X)
            residuals = Y - mu_X
        except Exception as e_fit:
            print(f"Error fitting base model (no CV): {e_fit}")
            return np.full(n_samples, np.nan)

        # Perform k-NN correction
        try:
            scale = 1.0 / bandwidth
            tree = BallTree(X * scale, leaf_size=40)
            dist, ind = tree.query(X * scale, k=k_actual)
            W = np.exp(-0.5 * (dist**2))
            W_sum = W.sum(axis=1, keepdims=True)
            W_sum = np.where(W_sum < EPS, EPS, W_sum)
            W /= W_sum
            corrections = np.sum(W * residuals[ind], axis=1)
            out_preds = mu_X + corrections
        except Exception as e_corr:
             print(f"Error during k-NN correction calculation (no CV): {e_corr}")
             out_preds = mu_X # Fallback to plugin
    # --- Case 2: K-Fold Cross-Validation ---
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_num = 0
        for train_idx, test_idx in kf.split(X):
            fold_num += 1
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]
            n_train = X_train.shape[0]

            if n_train < max(2, min_k_neighbors):
                 print(f"Warning: Training fold {fold_num} size ({n_train}) too small. Skipping IF correction, using fallback.")
                 try:
                      model_fold = clone(model_base)
                      model_fold.fit(X_train, Y_train)
                      out_preds[test_idx] = model_fold.predict(X_test)
                 except Exception as e_fallback:
                      print(f"  Error during fallback plugin calculation for fold {fold_num}: {e_fallback}")
                      out_preds[test_idx] = np.nan
                 continue

            k_fold = max(1, min(default_k, n_train - 1))
            model = None
            try:
                model = clone(model_base) # Use clone for fresh model per fold
                model.fit(X_train, Y_train)
                mu_test = model.predict(X_test)
                mu_train = model.predict(X_train)
                residuals_train = Y_train - mu_train

                scale = 1.0 / bandwidth
                tree = BallTree(X_train * scale, leaf_size=40)
                dist, ind = tree.query(X_test * scale, k=k_fold)
                W = np.exp(-0.5 * (dist**2))
                W_sum = W.sum(axis=1, keepdims=True)
                W_sum = np.where(W_sum < EPS, EPS, W_sum)
                W /= W_sum
                corrections = np.sum(W * residuals_train[ind], axis=1)
                out_preds[test_idx] = mu_test + corrections

            except ValueError as ve:
                 if "k must be less than or equal to the number of training points" in str(ve) or "k exceeds number of points" in str(ve):
                     print(f"Error during k-NN query in fold {fold_num} (k={k_fold}, n_train={n_train}): {ve}")
                     if model is not None:
                         try: out_preds[test_idx] = model.predict(X_test)
                         except: out_preds[test_idx] = np.nan
                     else: out_preds[test_idx] = np.nan
                 else:
                     print(f"ValueError during fold {fold_num} processing: {ve}")
                     out_preds[test_idx] = np.nan
            except Exception as e:
                 print(f"Error during fold {fold_num} processing: {e}")
                 if model is not None:
                     try: out_preds[test_idx] = model.predict(X_test)
                     except: out_preds[test_idx] = np.nan
                 else:
                     out_preds[test_idx] = np.nan

    return out_preds

def IF_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS, seed=42):
    """
    Compute the IF-based estimator for E[E[Y|X]^2] using K-fold CV.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    if isinstance(X, Tensor): X = X.detach().cpu().numpy()
    if isinstance(Y, Tensor): Y = Y.detach().cpu().numpy()

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base)
        model.fit(X, Y)
        mu_X = model.predict(X)
        plugin_estimate = np.mean(mu_X ** 2)
        residuals = Y - mu_X
        correction_term = 2 * np.mean(residuals * mu_X)
        return plugin_estimate + correction_term
    else:
        plugin_terms = []
        correction_terms = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            mu_X_test = model.predict(X_test)
            plugin_terms.append(np.mean(mu_X_test ** 2))
            residuals_test = Y[test_idx] - mu_X_test
            correction_terms.append(2 * np.mean(residuals_test * mu_X_test))

        # Handle potential NaNs if some folds failed? For now, assume they succeed.
        plugin_estimate = np.mean(plugin_terms) if plugin_terms else np.nan
        correction_term = np.mean(correction_terms) if correction_terms else np.nan

        if np.isnan(plugin_estimate) or np.isnan(correction_term):
            return np.nan
        else:
            return plugin_estimate + correction_term

# =============================================================================
# Kernel reweighting function (unchanged)
# =============================================================================
def estimate_conditional_expectation_knn(
        X_ref: torch.Tensor,       # Reference features [n_ref, d]
        S_query: torch.Tensor,     # Query features S(alpha) [n_query, d]
        E_Y_X_ref: torch.Tensor,   # Reference E[Y|X] values [n_ref]
        alpha: torch.Tensor,       # Noise parameters [d]
        k: int = 1000,             # Number of neighbors
        clamp_min: float = 1e-5,   # Min value for alpha and squared distances
        clamp_max_dist: float = 1e6 # Max value for squared distances
        ) -> torch.Tensor:
    """
    Differentiable kNN kernel-weighted estimate of E[Y|S_query] using references.
      W_ij ~ exp( -(1/2)* || (S_i - X_j) / sqrt(alpha) ||^2 )
      E[Y|S_i] = sum_{j in kNN(S_i)} W_ij * E_Y_X_ref[j].

    Ensures tensors are on the same device as S_query.
    """
    device = S_query.device
    n_ref = X_ref.shape[0]
    n_query = S_query.shape[0]

    # Move reference data to the correct device if necessary
    X_ref = X_ref.to(device)
    E_Y_X_ref = E_Y_X_ref.to(device)
    alpha = alpha.to(device) # Ensure alpha is also on the right device

    # 1) Clamp alpha and compute inverse sqrt variance
    alpha_safe = torch.clamp(alpha, min=clamp_min)       # (d,)
    inv_sqrt_alpha = torch.rsqrt(alpha_safe)             # (d,) -> 1/sqrt(alpha)

    # 2) Scale features into Mahalanobis space based on alpha
    Xs_scaled = X_ref * inv_sqrt_alpha                   # (n_ref, d)
    Ss_scaled = S_query * inv_sqrt_alpha                 # (n_query, d)

    # 3) Compute pairwise squared distances in scaled space
    #    D2_ij = || Ss_scaled_i - Xs_scaled_j ||^2
    #    Using cdist is generally efficient and stable
    if n_ref > 10000:
        # Compute squared distances in mini-batches to reduce peak memory
        batch_size = 50
        D2_chunks = []
        for i in range(0, n_query, batch_size):
            chunk = Ss_scaled[i:i+batch_size]
            D2_chunks.append(torch.cdist(chunk, Xs_scaled, p=2).pow(2))
        D2 = torch.cat(D2_chunks, dim=0)
    else:
        D2 = torch.cdist(Ss_scaled, Xs_scaled, p=2).pow(2)     # (n_query, n_ref)

    # Clamp distances to avoid potential numerical issues (optional but safe)
    D2 = torch.clamp(D2, min=clamp_min, max=clamp_max_dist) # (n_query, n_ref)

    # 4) Find k-nearest neighbors for each query point S_i based on scaled distance
    actual_k = min(k, n_ref)
    if actual_k < 1:
        print(f"Warning: actual_k={actual_k} < 1 in kernel estimation. Returning mean.")
        # Return mean of reference E[Y|X] as fallback
        return torch.full((n_query,), E_Y_X_ref.mean(), device=device, dtype=S_query.dtype)

    # topk finds the k smallest distances and their indices
    # Use torch.no_grad() for idx finding if not backpropping through indices (usually safe)
    with torch.no_grad():
         # D2_knn: distances to k nearest neighbors (n_query, k)
         # knn_indices: indices of these neighbors in X_ref (n_query, k)
        D2_knn, knn_indices = torch.topk(D2, actual_k, dim=1, largest=False)

    # Important: Re-select distances using indices *within* the computation graph
    # if gradients through D2 are needed for alpha (which they are).
    # Gather the distances corresponding to the selected indices.
    # This ensures the gradient path for D2 -> alpha is maintained.
    D2_knn_grad = D2.gather(1, knn_indices) # (n_query, k)

    # 5) Calculate weights using softmax over the k neighbors
    #    logW = -0.5 * D2_knn (use the version with grad)
    logW = -0.5 * D2_knn_grad                            # (n_query, k)
    W = torch.softmax(logW, dim=1)                       # (n_query, k), rows sum to 1

    # 6) Gather the E[Y|X] values for the k neighbors
    #    knn_indices shape: (n_query, k)
    #    E_Y_X_ref shape: (n_ref,) -> Need to index E_Y_X_ref using knn_indices
    #    Use gather or direct indexing
    E_Y_X_neighbors = E_Y_X_ref[knn_indices]             # (n_query, k)

    # 7) Compute weighted average
    E_Y_S_estimate = (W * E_Y_X_neighbors).sum(dim=1)    # (n_query,)

    return E_Y_S_estimate


def estimate_conditional_kernel_oof(
    X_batch: torch.Tensor,
    S_batch: torch.Tensor,
    E_Y_X: torch.Tensor,
    alpha: torch.Tensor,
    n_folds: int = 5,
    clamp_min: float = 1e-4,
    clamp_max: float = 1e6,
    k: int = 100,
    seed: int = 42
) -> torch.Tensor:
    """
    Out-of-fold kNN-kernel estimates for E[Y|S].
    """
    n_test = S_batch.size(0)
    oof = torch.zeros(n_test, device=X_batch.device)

    # Ensure k is not larger than the smallest possible training fold size
    min_train_size = max(1, int(X_batch.shape[0] * (1 - 1/n_folds)) if n_folds > 1 else X_batch.shape[0])
    actual_k = min(k, min_train_size)
    if actual_k < 1:
        print("Warning: k adjusted to 0 in estimate_conditional_kernel_oof. Returning zeros.")
        return oof # Or handle differently

    if n_folds <= 1:
        oof = estimate_conditional_expectation_knn(
            X_batch, S_batch, E_Y_X, alpha, k=actual_k, clamp_min=clamp_min, clamp_max=clamp_max
        )
        return oof
    else:
        # Note: This OOF implementation for the kernel estimator is slightly different
        # from the plugin/IF OOF. Here, for each test fold of S_batch, it uses the
        # *entire* X_batch and E_Y_X as the "training" set for the kernel weighting.
        # This might be intended, but differs from typical CV where the model/reference
        # data is also split. If true OOF is needed, X_batch and E_Y_X should also be split.
        # Assuming current implementation is intended:
        kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for _, te_idx in kf.split(S_batch): # We only need test indices for S_batch
            oof[te_idx] = estimate_conditional_expectation_knn(
                X_batch, S_batch[te_idx], E_Y_X, alpha, k=actual_k, clamp_min=clamp_min, clamp_max=clamp_max
            )
        return oof

def estimate_conditional_keops(
    X: Tensor,           # (n_train, d) - Reference features
    S: Tensor,           # (n_test, d)  - Query features
    E_Y_X: Tensor,       # (n_train,) - Reference values
    alpha: Tensor,       # (d,)       - Noise parameters
    k: int = 1000,       # Number of neighbors to use
    clamp_min: float = 1e-4, # Min clamping for alpha variance
    # clamp_max_dist: float = 1e6 # Optional: Max clamping for squared distances
) -> Tensor:
    """
    Estimates E[ E_Y_X | S=s ] using PyTorch k-NN kernel weighting
    on variance-scaled features. Maintains differentiability w.r.t. alpha.

    Args:
        X: Reference features tensor.
        S: Query features tensor.
        E_Y_X: Reference values tensor (E[Y|X]).
        alpha: Noise variance parameters tensor.
        k: Number of nearest neighbors to consider.
        clamp_min: Minimum value for variance clamping.
        # clamp_max_dist: Optional max value for squared distances.

    Returns:
        Tensor: Estimated values E[Y|S] for each query point in S.
    """
    device = S.device
    n_train = X.shape[0]
    n_test = S.shape[0]
    d = X.shape[1]

    # --- Input Validation ---
    if E_Y_X.ndim != 1 or E_Y_X.shape[0] != n_train:
        raise ValueError(f"E_Y_X must be a 1D tensor of shape ({n_train},), but got shape {E_Y_X.shape}")
    if alpha.shape != (d,):
         raise ValueError(f"alpha must be a 1D tensor of shape ({d},), but got shape {alpha.shape}")
    if X.device != device or E_Y_X.device != device or alpha.device != device:
        # For simplicity in this example, move all to S's device.
        # In practice, ensure inputs are on the desired device beforehand.
        X = X.to(device)
        E_Y_X = E_Y_X.to(device)
        alpha = alpha.to(device)
        # print("Warning: Moving input tensors to device:", device)


    # --- 1. Scaling ---
    # Clamp alpha (variance) and compute inverse square root
    alpha_clamped = torch.clamp(alpha, min=clamp_min)       # (d,)
    inv_sqrt_var_t = torch.rsqrt(alpha_clamped)             # (d,) -> 1/sqrt(variance)

    # Scale features into Mahalanobis-like space
    # Ensure broadcasting works correctly: (n, d) * (d,) -> (n, d)
    Xs_scaled = X * inv_sqrt_var_t                   # (n_train, d)
    Ss_scaled = S * inv_sqrt_var_t                   # (n_test, d)

    # --- 2. Pairwise Distances ---
    # Calculate squared Euclidean distances in the scaled space
    # This operation is differentiable w.r.t. Ss_scaled, Xs_scaled, and thus alpha
    if n_train > 10000:
        # Compute squared distances in mini-batches to reduce peak memory
        batch_size = 50
        D2_chunks = []
        for i in range(0, n_test, batch_size):
            chunk = Ss_scaled[i:i+batch_size]
            D2_chunks.append(torch.cdist(chunk, Xs_scaled, p=2).pow(2))
        D2 = torch.cat(D2_chunks, dim=0) # (n_test, n_train)
    else:
        D2 = torch.cdist(Ss_scaled, Xs_scaled, p=2).pow(2)     # (n_test, n_train)

    # Optional: Clamp distances for numerical stability
    # D2 = torch.clamp(D2, min=clamp_min, max=clamp_max_dist)

    # --- 3. Find K-Nearest Neighbors ---
    # Determine the actual k to use (cannot exceed n_train)
    actual_k = min(k, n_train)
    if actual_k < 1:
        print(f"Warning: actual_k={actual_k} < 1. Returning mean E[Y|X].")
        # Fallback: return the mean of reference values
        return torch.full((n_test,), E_Y_X.mean(), device=device, dtype=S.dtype)

    # Find the indices and *squared distances* of the k nearest neighbors
    # `topk` returns (values, indices). values are the smallest squared distances here.
    # Importantly, the returned distances (D2_knn) retain gradient history.
    D2_knn, knn_indices = torch.topk(D2, actual_k, dim=1, largest=False) # (n_test, k)

    # --- 4. Calculate Weights ---
    # Gaussian kernel weights based on the squared distances of the k neighbors
    logW = -0.5 * D2_knn                            # (n_test, k)
    # Normalize weights using softmax across the k neighbors for each test point
    W = torch.softmax(logW, dim=1)                  # (n_test, k), rows sum to 1

    # --- 5. Weighted Sum ---
    # Gather the E[Y|X] values for the k neighbors using the indices
    # E_Y_X[knn_indices] uses advanced indexing and works correctly
    E_Y_X_neighbors = E_Y_X[knn_indices]             # (n_test, k)

    # Compute the weighted average
    E_Y_S_estimate = (W * E_Y_X_neighbors).sum(dim=1)    # (n_test,)

    return E_Y_S_estimate

def chunked_pairwise_distance(X: torch.Tensor, S: torch.Tensor, chunk_size: int = 50) -> torch.Tensor:
    """
    Calculates pairwise distances between X and S in a memory-efficient way using chunks.
    Preserves gradient flow while reducing peak memory usage.
    
    Args:
        X: First set of points (n_train, d)
        S: Second set of points (n_test, d)
        chunk_size: Size of chunks to process at a time
        
    Returns:
        Pairwise squared distances (n_test, n_train)
    """
    n_test, d = S.shape
    n_train = X.shape[0]
    device = S.device
    
    # Pre-allocate the full distance matrix
    D2 = torch.zeros((n_test, n_train), device=device)
    
    # Process in chunks to reduce memory usage
    for i in range(0, n_test, chunk_size):
        end_idx = min(i + chunk_size, n_test)
        chunk = S[i:end_idx]
        
        # Calculate distances for this chunk
        try:
            D2[i:end_idx] = torch.cdist(chunk, X, p=2).pow(2)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                # Fallback to even smaller chunks if necessary
                for j in range(i, end_idx):
                    D2[j:j+1] = torch.cdist(S[j:j+1], X, p=2).pow(2)
            else:
                raise e
                
    return D2

def estimate_conditional_keops_flexible_optimized(
    X: torch.Tensor,           # (n_train, d) - Reference features
    S: torch.Tensor,           # (n_test, d)  - Query features
    E_Y_X: torch.Tensor,       # (n_train,) - Reference values
    param: torch.Tensor,       # (d,)       - Either alpha or theta
    param_type: str = 'alpha', # 'alpha' or 'theta'
    k: int = 1000,
    clamp_min_alpha: float = 1e-4, 
    max_batch_size: int = 500  # Max batch size to control memory usage
) -> torch.Tensor:
    """
    Memory-efficient GPU implementation that avoids OOM errors
    by processing in smaller batches while staying on GPU.
    """
    device = param.device
    n_train, d = X.shape
    n_test = S.shape[0]
    
    # Scale features according to parameter type
    if param_type == 'alpha':
        alpha_clamped = torch.clamp(param, min=clamp_min_alpha)
        inv_sqrt_var_t = torch.rsqrt(alpha_clamped)
    elif param_type == 'theta':
        inv_sqrt_var_t = torch.exp(-param / 2.0)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")
    
    # Scale reference points (only once)
    Xs_scaled = X * inv_sqrt_var_t
    
    # Create output tensor
    results = torch.zeros(n_test, dtype=S.dtype, device=device)
    
    # Automatically determine batch size based on data dimensions
    # Adjust the divisor based on your GPU memory
    min_batch_size = 1
    suggested_batch_size = max(min_batch_size, min(max_batch_size, int(1e9 / (n_train * d * 4))))
    batch_size = suggested_batch_size
    
    # Process in batches to manage memory
    for i in range(0, n_test, batch_size):
        end_i = min(i + batch_size, n_test)
        batch_S = S[i:end_i]
        
        # Scale query points for this batch
        batch_S_scaled = batch_S * inv_sqrt_var_t
        
        # Compute distances efficiently in chunks if reference set is large
        if n_train > 10000:
            inner_batch_size = min(5000, n_train)
            D2 = torch.zeros((end_i - i, n_train), device=device)
            
            for j in range(0, n_train, inner_batch_size):
                end_j = min(j + inner_batch_size, n_train)
                # Compute and store partial distance matrix
                D2[:, j:end_j] = torch.cdist(batch_S_scaled, Xs_scaled[j:end_j], p=2).pow(2)
                
                # Free memory explicitly
                torch.cuda.empty_cache()
        else:
            # If reference set is small enough, compute in one go
            D2 = torch.cdist(batch_S_scaled, Xs_scaled, p=2).pow(2)
        
        # Find k nearest neighbors
        k_actual = min(k, n_train)
        dists, indices = torch.topk(D2, k_actual, largest=False, dim=1)
        
        # Free the distance matrix explicitly
        del D2
        torch.cuda.empty_cache()
        
        # Compute weights
        weights = torch.exp(-0.5 * dists)
        weights_sum = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / torch.clamp(weights_sum, min=1e-8)
        
        # Get values and compute weighted sum
        values = E_Y_X[indices]
        results[i:end_i] = torch.sum(weights * values, dim=1)
        
        # Free batch memory
        del batch_S_scaled, dists, indices, weights, values
        torch.cuda.empty_cache()
        
    return results

def estimate_conditional_keops_flexible(
    X: torch.Tensor,           # (n_train, d) - Reference features
    S: torch.Tensor,           # (n_test, d)  - Query features
    E_Y_X: torch.Tensor,       # (n_train,) - Reference values
    param: torch.Tensor,       # (d,)       - Either alpha or theta
    param_type: str = 'alpha', # 'alpha' or 'theta'
    k: int = 1000,
    clamp_min_alpha: float = 1e-4, # Min clamping for alpha variance
    chunk_size: int = 50       # Size of chunks for distance calculation
) -> torch.Tensor:
    """
    Estimates E[ E_Y_X | S=s ] using PyTorch k-NN kernel weighting.
    Uses chunked distance calculation to reduce memory usage while preserving gradients.
    
    Args:
        X: Reference features tensor
        S: Query features tensor
        E_Y_X: Reference values tensor (E[Y|X])
        param: Noise variance parameters tensor (alpha or theta)
        param_type: 'alpha' or 'theta'
        k: Number of nearest neighbors to consider
        clamp_min_alpha: Minimum value for alpha clamping
        chunk_size: Size of chunks for distance calculation
        
    Returns:
        Tensor: Estimated values E[Y|S] for each query point in S
    """
    # Store original device
    device = param.device
    
    # Get tensor dimensions
    n_train, d = X.shape
    n_test = S.shape[0]
    
    # --- Input Validation ---
    if E_Y_X.ndim != 1 or E_Y_X.shape[0] != n_train:
        raise ValueError(f"E_Y_X shape mismatch: expected ({n_train},), got {E_Y_X.shape}")
    if param.shape != (d,):
         raise ValueError(f"Parameter shape mismatch: expected ({d},), got {param.shape}")
    
    # Ensure inputs are on the same device
    X, E_Y_X = X.to(device), E_Y_X.to(device)
    
    # --- 1. Scaling based on param_type ---
    if param_type == 'alpha':
        alpha_clamped = torch.clamp(param, min=clamp_min_alpha)
        inv_sqrt_var_t = torch.rsqrt(alpha_clamped) # 1/sqrt(alpha)
    elif param_type == 'theta':
        inv_sqrt_var_t = torch.exp(-param / 2.0) # 1/sqrt(exp(theta))
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    # Apply scaling - this preserves gradients
    Xs_scaled = X * inv_sqrt_var_t
    Ss_scaled = S * inv_sqrt_var_t
    
    # Actual k to use (cannot exceed n_train)
    actual_k = min(k, n_train)
    if actual_k < 1:
        print(f"Warning: actual_k={actual_k} < 1. Returning mean E[Y|X].")
        return torch.full((n_test,), E_Y_X.mean(), device=device, dtype=S.dtype)
    
    # --- 2. Memory-efficient distance calculation ---
    try:
        # Try using chunked distance calculation
        D2 = chunked_pairwise_distance(Xs_scaled, Ss_scaled, chunk_size=chunk_size)
        
        # --- 3. Find K-Nearest Neighbors ---
        dists, inds = torch.topk(D2, actual_k, largest=False, dim=1)
        
        # --- 4. Calculate Weights ---
        W = torch.exp(-0.5 * dists)
        W = W / (torch.sum(W, dim=1, keepdim=True) + 1e-8)
        
        # --- 5. Weighted Sum ---
        E_Y_X_neighbors = E_Y_X[inds]
        E_Y_S_estimate = torch.sum(W * E_Y_X_neighbors, dim=1)
        
        return E_Y_S_estimate
        
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print(f"Warning: CUDA OOM with chunk size {chunk_size}. Processing one sample at a time.")
            
            # Process one sample at a time as a last resort
            results = []
            for i in range(n_test):
                # Calculate distances for one sample
                sample = Ss_scaled[i:i+1]
                try:
                    # Try on current device
                    D2_i = torch.cdist(sample, Xs_scaled, p=2).pow(2)
                    dists_i, inds_i = torch.topk(D2_i, actual_k, largest=False, dim=1)
                    W_i = torch.exp(-0.5 * dists_i)
                    W_i = W_i / (torch.sum(W_i, dim=1, keepdim=True) + 1e-8)
                    E_Y_X_neighbors_i = E_Y_X[inds_i]
                    result_i = torch.sum(W_i * E_Y_X_neighbors_i, dim=1)
                    results.append(result_i)
                    
                except RuntimeError:
                    # Last resort: move to CPU while preserving gradient flow
                    cpu_sample = sample.cpu()
                    cpu_Xs = Xs_scaled.cpu()
                    cpu_E_Y_X = E_Y_X.cpu()
                    
                    D2_i = torch.cdist(cpu_sample, cpu_Xs, p=2).pow(2)
                    dists_i, inds_i = torch.topk(D2_i, actual_k, largest=False, dim=1)
                    W_i = torch.exp(-0.5 * dists_i)
                    W_i = W_i / (torch.sum(W_i, dim=1, keepdim=True) + 1e-8)
                    E_Y_X_neighbors_i = cpu_E_Y_X[inds_i]
                    result_i = torch.sum(W_i * E_Y_X_neighbors_i, dim=1)
                    results.append(result_i.to(device))  # Back to original device
            
            # Combine results while preserving gradient flow
            return torch.cat(results, dim=0)
        else:
            # Re-raise if not OOM error
            raise e
        
# Changed on May 10, 25, when keops kept giving memory errors
# def estimate_conditional_keops_flexible(
#     X: torch.Tensor,           # (n_train, d) - Reference features
#     S: torch.Tensor,           # (n_test, d)  - Query features
#     E_Y_X: torch.Tensor,       # (n_train,) - Reference values
#     param: torch.Tensor,       # (d,)       - Either alpha or theta
#     param_type: str = 'alpha', # 'alpha' or 'theta'
#     k: int = 1000,
#     clamp_min_alpha: float = 1e-4, # Min clamping for alpha variance
#     use_knn: bool = True # Use k-NN for estimation
# ) -> torch.Tensor:
#     """
#     Estimates E[ E_Y_X | S=s ] using PyTorch k-NN kernel weighting.
#     Handles scaling based on either alpha or theta = log(alpha).
#     Maintains differentiability w.r.t. param.
#     """
#     device = S.device
#     n_train, d = X.shape
#     n_test = S.shape[0]

#     # --- Input Validation ---
#     if E_Y_X.ndim != 1 or E_Y_X.shape[0] != n_train:
#         raise ValueError(f"E_Y_X shape mismatch: expected ({n_train},), got {E_Y_X.shape}")
#     if param.shape != (d,):
#          raise ValueError(f"Parameter shape mismatch: expected ({d},), got {param.shape}")
#     X, E_Y_X, param = X.to(device), E_Y_X.to(device), param.to(device)

#     # --- 1. Scaling based on param_type ---
#     if param_type == 'alpha':
#         alpha_clamped = torch.clamp(param, min=clamp_min_alpha)
#         inv_sqrt_var_t = torch.rsqrt(alpha_clamped) # 1/sqrt(alpha)
#     elif param_type == 'theta':
#         inv_sqrt_var_t = torch.exp(-param / 2.0) # 1/sqrt(exp(theta))
#     else:
#         raise ValueError("param_type must be 'alpha' or 'theta'")

#     Xs_scaled = X * inv_sqrt_var_t
#     Ss_scaled = S * inv_sqrt_var_t

#     # --- 2. Pairwise Distances ---
#     # if too large, use batches
#     if n_train > 5000:
#         batch_size = 50
#         results = []
#         actual_k = min(k, n_train)
#         for i in range(0, n_test, batch_size):
#             chunk = Ss_scaled[i:i+batch_size]                  # (bs, d)
#             # 1) distances for this chunk only
#             D2_chunk = torch.cdist(chunk, Xs_scaled, p=2).pow(2)  # (bs, n_train)
#             # 2) find neighbors
#             dists, inds = D2_chunk.topk(actual_k, largest=False, dim=1)
#             # 3) weights & normalize
#             W = torch.exp(-0.5 * dists)
#             W = W / (W.sum(dim=1, keepdim=True) + 1e-8)
#             # 4) gather predictions and combine
#             neigh_preds = E_Y_X[inds]                     # (bs, k)
#             chunk_mu = (W * neigh_preds).sum(dim=1)       # (bs,)
#             results.append(chunk_mu)
#         return torch.cat(results, dim=0)  
#     else:
#         D2 = torch.cdist(Ss_scaled, Xs_scaled, p=2).pow(2) # (n_test, n_train)

#     # --- 3. Find K-Nearest Neighbors ---
#     actual_k = min(k, n_train)
#     if actual_k < 1:
#         print(f"Warning: actual_k={actual_k} < 1. Returning mean E[Y|X].")
#         return torch.full((n_test,), E_Y_X.mean(), device=device, dtype=S.dtype)

#     D2_knn, knn_indices = torch.topk(D2, actual_k, dim=1, largest=False) # (n_test, k)

#     # --- 4. Calculate Weights ---
#     logW = -0.5 * D2_knn                            # (n_test, k)
#     W = torch.softmax(logW, dim=1)                  # (n_test, k)

#     # --- 5. Weighted Sum ---
#     E_Y_X_neighbors = E_Y_X[knn_indices]             # (n_test, k)
#     E_Y_S_estimate = (W * E_Y_X_neighbors).sum(dim=1)    # (n_test,)

#     return E_Y_S_estimate

# def estimate_T2_mc_flexible(X_std_torch: torch.Tensor,
#                             E_Yx_std_torch: torch.Tensor, # Standardized E[Y|X]
#                             param_torch: torch.Tensor,    # Alpha or Theta parameter tensor
#                             param_type: str,              # 'alpha' or 'theta'
#                             n_mc_samples: int,
#                             k_kernel: int) -> torch.Tensor:
#     """
#     Estimates T2 = E[E[Y_std|S]^2] using Monte Carlo sampling and the flexible kernel estimator.
#     Differentiable w.r.t. param_torch.
#     """
#     avg_term2_std = 0.0
#     # Clamp alpha for noise generation, but pass original param to kernel
#     if param_type == 'alpha':
#         # Clamp the parameter directly if it's alpha
#         alpha_clamped = param_torch.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
#         noise_scale = torch.sqrt(alpha_clamped)
#         param_for_kernel = param_torch # Pass original for grad
#     elif param_type == 'theta':
#         # Calculate alpha, clamp it for noise, but pass original theta to kernel
#         with torch.no_grad(): # Avoid tracking grad through exp just for noise
#              alpha_for_noise = torch.exp(param_torch).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
#         noise_scale = torch.sqrt(alpha_for_noise)
#         param_for_kernel = param_torch # Pass original theta for grad
#     else:
#         raise ValueError("param_type must be 'alpha' or 'theta'")

#     for _ in range(n_mc_samples):
#         epsilon_k = torch.randn_like(X_std_torch)
#         S_param_k = X_std_torch + epsilon_k * noise_scale # Noise depends on param type

#         # Estimate E[Y_std|S] using kernel estimator, passing the original param
#         E_Y_S_std_k = estimate_conditional_keops_flexible(
#             X_std_torch, S_param_k, E_Yx_std_torch, param_for_kernel, param_type, k=k_kernel
#         )
#         term2_sample_std_k = E_Y_S_std_k.pow(2).mean()
#         avg_term2_std += term2_sample_std_k

#     return avg_term2_std / n_mc_samples

def estimate_T2_mc_flexible(
    X_std_torch: torch.Tensor,
    E_Yx_std_torch: torch.Tensor,
    param_torch: torch.Tensor,
    param_type: str,
    n_mc_samples: int,
    k_kernel: int,
    chunk_size: int = 50      # Size of chunks for distance calculation
) -> torch.Tensor:
    """
    Estimates T2 = E[E[Y_std|S]^2] using Monte Carlo sampling and memory-efficient kernel estimator.
    Differentiable w.r.t. param_torch.
    
    Args:
        X_std_torch: Standardized features (n, d)
        E_Yx_std_torch: Standardized conditional expectations E[Y|X] (n,)
        param_torch: Parameter tensor (alpha or theta) (d,)
        param_type: 'alpha' or 'theta'
        n_mc_samples: Number of Monte Carlo samples
        k_kernel: Number of neighbors for kernel estimation
        chunk_size: Size of chunks for distance calculation
        
    Returns:
        Scalar estimate of T2 = E[E[Y_std|S]^2]
    """
    device = param_torch.device
    avg_term2_std = torch.zeros(1, device=device, dtype=torch.float32)

    # Move input tensors to GPU instead of moving param to CPU
    X_std_torch = X_std_torch.to(device)
    E_Yx_std_torch = E_Yx_std_torch.to(device)

    # # print the devices of all tensors
    # print(f"X_std_torch device: {X_std_torch.device}")
    # print(f"E_Yx_std_torch device: {E_Yx_std_torch.device}")
    # print(f"param_torch device: {param_torch.device}")

    
    # Determine noise scale based on param type
    if param_type == 'alpha':
        alpha_clamped = param_torch.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_clamped)
        param_for_kernel = param_torch
    elif param_type == 'theta':
        alpha_derived = torch.exp(param_torch)
        alpha_clamped = alpha_derived.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_clamped)
        param_for_kernel = param_torch
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")
    
    # Use dynamic MC sample count based on data size
    n_train = X_std_torch.shape[0]
    actual_mc_samples = min(n_mc_samples, 5 if n_train > 10000 else n_mc_samples)
    
    # Monte Carlo estimation
    for i in range(actual_mc_samples):
        epsilon_k = torch.randn_like(X_std_torch)
        S_param_k = X_std_torch + epsilon_k * noise_scale
        
        try:
            # Use memory-efficient kernel estimator
            E_Y_S_std_k = estimate_conditional_keops_flexible(
            X_std_torch, S_param_k, E_Yx_std_torch, 
            param_for_kernel, param_type, k=k_kernel,
            chunk_size=chunk_size
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Warning: CUDA out of memory error detected. Switching to optimized estimator.")
                E_Y_S_std_k = estimate_conditional_keops_flexible_optimized(
                    X_std_torch, S_param_k, E_Yx_std_torch,
                    param_for_kernel, param_type, k=k_kernel,
                    max_batch_size=500
                )
            else:
                raise e
        
        term2_sample_std_k = E_Y_S_std_k.pow(2).mean()
        avg_term2_std += term2_sample_std_k
    
    return avg_term2_std / actual_mc_samples

def estimate_T2_kernel_IF_like_flexible(
    X_std_torch: torch.Tensor,
    Y_std_torch: torch.Tensor,
    E_Yx_std_torch: torch.Tensor,
    param_torch: torch.Tensor,
    param_type: str,
    n_mc_samples: int,
    k_kernel: int,
    chunk_size: int = 50      # Size of chunks for distance calculation
) -> torch.Tensor:
    """
    Estimates T2 using IF-like form: E[2*Y*mu_S - mu_S^2] with memory-efficient kernel estimator.
    Differentiable w.r.t. param_torch through all paths.
    
    Args:
        X_std_torch: Standardized features (n, d)
        Y_std_torch: Standardized outcomes (n,)
        E_Yx_std_torch: Standardized conditional expectations E[Y|X] (n,)
        param_torch: Parameter tensor (alpha or theta) (d,)
        param_type: 'alpha' or 'theta'
        n_mc_samples: Number of Monte Carlo samples
        k_kernel: Number of neighbors for kernel estimation
        chunk_size: Size of chunks for distance calculation
        
    Returns:
        Scalar estimate of T2 using IF-like form
    """
    device = param_torch.device
    avg_term2_if_like = torch.tensor(0.0, device=device, dtype=param_torch.dtype)
    
    # Determine noise scale based on param type
    if param_type == 'alpha':
        alpha_for_noise = param_torch.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    elif param_type == 'theta':
        alpha_derived = torch.exp(param_torch)
        alpha_for_noise = alpha_derived.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")
    
    noise_scale = torch.sqrt(alpha_for_noise)
    
    # Use dynamic MC sample count based on data size
    n_train = X_std_torch.shape[0]
    actual_mc_samples = min(n_mc_samples, 5 if n_train > 10000 else n_mc_samples)
    
    # Monte Carlo estimation
    for i in range(actual_mc_samples):
        epsilon_k = torch.randn_like(X_std_torch)
        S_param_k = X_std_torch + epsilon_k * noise_scale
        
        # Use memory-efficient kernel estimator
        try:
            # Use memory-efficient kernel estimator
            mu_S_hat_k = estimate_conditional_keops_flexible(
                X_std_torch, S_param_k, E_Yx_std_torch,
                param_torch, param_type, k=k_kernel,
                chunk_size=chunk_size
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Warning: CUDA out of memory error detected. Switching to optimized estimator.")
                mu_S_hat_k = estimate_conditional_keops_flexible_optimized(
                    X_std_torch, S_param_k, E_Yx_std_torch,
                    param_torch, param_type, k=k_kernel,
                    max_batch_size=500
                )
            else:
                raise e
        
        term2_sample_if_like_k = (2 * Y_std_torch * mu_S_hat_k - mu_S_hat_k.pow(2)).mean()
        avg_term2_if_like += term2_sample_if_like_k
    
    return avg_term2_if_like / actual_mc_samples

def estimate_E_Y_S_kernel_flexible(X_std_torch: torch.Tensor,
                                E_Yx_std_torch: torch.Tensor,
                                param_torch: torch.Tensor, # Detached alpha or theta
                                param_type: str,
                                n_mc_samples_S: int = 1,
                                k_kernel: int = 1000) -> torch.Tensor:
    """
    Estimates E[Y_std|S] value using the flexible kernel method.
    Averages over n_mc_samples_S realizations of S. Uses detached param.
    """
    avg_E_Y_S = torch.zeros(X_std_torch.shape[0], device=X_std_torch.device, dtype=X_std_torch.dtype)
    param_val = param_torch.detach().clone() # Use detached value

    if param_type == 'alpha':
        alpha_clamped = param_val.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_clamped)
    elif param_type == 'theta':
        # Clamp the resulting alpha for noise stability
        alpha_for_noise = torch.exp(param_val).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
        noise_scale = torch.sqrt(alpha_for_noise)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    with torch.no_grad():
        for _ in range(n_mc_samples_S):
            epsilon_k = torch.randn_like(X_std_torch)
            S_param_k = X_std_torch + epsilon_k * noise_scale
            # Pass detached param for value estimation
        try:
            E_Y_S_std_k = estimate_conditional_keops_flexible(
                X_std_torch, S_param_k, E_Yx_std_torch, param_val, param_type, k=k_kernel
            )
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Warning: CUDA out of memory error detected. Switching to optimized estimator.")
                E_Y_S_std_k = estimate_conditional_keops_flexible_optimized(
                    X_std_torch, S_param_k, E_Yx_std_torch, param_val, param_type, k=k_kernel,
                    max_batch_size=500
                )
            else:
                raise e
        avg_E_Y_S += E_Y_S_std_k

    return avg_E_Y_S / n_mc_samples_S


# def estimate_T2_kernel_IF_like_flexible(
#     X_std_torch: torch.Tensor,
#     Y_std_torch: torch.Tensor,
#     E_Yx_std_torch: torch.Tensor,
#     param_torch: torch.Tensor,
#     param_type: str,
#     n_mc_samples: int,
#     k_kernel: int) -> torch.Tensor:
#     """
#     Estimates T2 = E[(E[Y_std|S])^2] using an IF-like form: E[2*Y_std*mu_S - mu_S^2],
#     where mu_S = E[Y_std|S] is estimated via the flexible kernel method.
#     Differentiable w.r.t. param_torch through all paths.
#     """
#     avg_term2_if_like = torch.tensor(0.0, device=param_torch.device, dtype=param_torch.dtype)
#     param_for_kernel_call = param_torch # param being optimized (alpha or theta)

#     # Determine noise_scale based on param_torch, keeping it on the computation graph
#     if param_type == 'alpha':
#         alpha_for_noise = param_torch.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
#     elif param_type == 'theta':
#         alpha_derived = torch.exp(param_torch) # alpha = exp(theta), on graph
#         alpha_for_noise = alpha_derived.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
#     else:
#         raise ValueError("param_type must be 'alpha' or 'theta'")
#     noise_scale = torch.sqrt(alpha_for_noise) # This is on the graph wrt param_torch

#     for _ in range(n_mc_samples):
#         epsilon_k = torch.randn_like(X_std_torch)
#         S_param_k = X_std_torch + epsilon_k * noise_scale # S_param_k is on graph wrt param_torch

#         mu_S_hat_k = estimate_conditional_keops_flexible(
#             X_std_torch, S_param_k, E_Yx_std_torch,
#             param_for_kernel_call, param_type, k=k_kernel
#         )
#         term2_sample_if_like_k = (2 * Y_std_torch * mu_S_hat_k - mu_S_hat_k.pow(2)).mean()
#         avg_term2_if_like += term2_sample_if_like_k
#     return avg_term2_if_like / n_mc_samples

# --- Gradient Estimator Functions ---
def analytical_gradient_penalty_alpha(alpha_torch: torch.Tensor,
                                      penalty_type: Optional[str],
                                      penalty_lambda: float) -> torch.Tensor:
    """Calculates analytical gradient dP/d(alpha) using autograd."""
    if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
        return torch.zeros_like(alpha_torch)

    alpha_torch_param = alpha_torch.clone().detach().requires_grad_(True)
    penalty_val = compute_penalty(alpha_torch_param, penalty_type, penalty_lambda)

    if penalty_val.requires_grad:
        try:
            grad_P = torch.autograd.grad(penalty_val, alpha_torch_param, retain_graph=False)[0]
            return grad_P
        except RuntimeError as e:
            print(f"Warning: Autograd failed for penalty {penalty_type}. Error: {e}")
            return torch.full_like(alpha_torch, float('nan')) # Indicate failure
    else:
        return torch.zeros_like(alpha_torch)
    


def estimate_gradient_autograd_flexible(X_std_torch: torch.Tensor,
                                        E_Yx_std_torch: torch.Tensor,
                                        param_torch: torch.Tensor, # Alpha or Theta (requires_grad=True)
                                        param_type: str,
                                        n_mc_samples: int,
                                        k_kernel: int,
                                        penalty_type: Optional[str],
                                        penalty_lambda: float) -> torch.Tensor:
    """
    Estimates total gradient dL/d(param) using Autograd through MC estimate of T2.
    L = T1_est - T2_est + P
    dL/dparam = -dT2/dparam + dP/dparam
    """
    if not param_torch.requires_grad:
        param_torch.requires_grad_(True)

    # --- Estimate T2 using MC + Flexible Kernel ---
    # This term needs to be differentiable w.r.t. param_torch
    term2_est = estimate_T2_mc_flexible(
        X_std_torch, E_Yx_std_torch, param_torch, param_type, n_mc_samples, k_kernel
    )

    # --- Calculate Penalty ---
    # Compute penalty based on alpha, deriving alpha from theta if needed
    if param_type == 'alpha':
        alpha_for_penalty = param_torch
    elif param_type == 'theta':
        # Need alpha = exp(theta) for penalty calculation, maintain grad graph
        alpha_for_penalty = torch.exp(param_torch)
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    penalty_value = compute_penalty(alpha_for_penalty, penalty_type, penalty_lambda)

    # --- Total Objective (for Autograd, ignoring constant T1) ---
    # L = -T2 + P (we minimize this, gradient is -dT2/dparam + dP/dparam)
    # Or L = T2 + P (if minimizing this, gradient is dT2/dparam + dP/dparam)
    # Let's assume we minimize L = T2 + P based on gd_pops_v6 logic
    objective_L = term2_est + penalty_value

    # --- Compute Gradient ---
    grad_L = torch.autograd.grad(objective_L, param_torch, retain_graph=False)[0]

    return grad_L

def estimate_gradient_reinforce_flexible(X_std_torch: torch.Tensor,
                                         E_Yx_std_torch: torch.Tensor,
                                         param_torch: torch.Tensor, # Alpha or Theta (NO grad needed initially)
                                         param_type: str,
                                         n_grad_samples: int,
                                         k_kernel: int,
                                         penalty_type: Optional[str],
                                         penalty_lambda: float,
                                         use_baseline: bool) -> torch.Tensor:
    """
    Estimates total gradient dL/d(param) using REINFORCE for T2 part.
    L = T2 + P => dL/dparam = dT2/dparam + dP/dparam
    """
    device = param_torch.device
    m = param_torch.shape[0]
    # Use detached version for internal calculations not needing grad from param
    param_detached = param_torch.detach().clone()

    # --- REINFORCE Gradient for Term 2 ---
    grad_term2_accum = torch.zeros_like(param_torch)

    if param_type == 'alpha':
        alpha_clamped_detached = param_detached.clamp(CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA)
        noise_var = alpha_clamped_detached # Variance for noise generation
    elif param_type == 'theta':
        # Calculate alpha from theta for noise generation, clamp alpha
        alpha_detached = torch.exp(param_detached)
        alpha_clamped_detached = alpha_detached.clamp(CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA)
        noise_var = alpha_clamped_detached
    else:
        raise ValueError("param_type must be 'alpha' or 'theta'")

    for _ in range(n_grad_samples):
        with torch.no_grad(): # Sample noise and estimate reward g(S)^2
            epsilon_k = torch.randn_like(X_std_torch)
            S_param_k = X_std_torch + epsilon_k * torch.sqrt(noise_var)
            # Estimate g(S) = E[Y_std|S] using kernel, pass detached param
            g_hat_S_k = estimate_conditional_keops_flexible(
                X_std_torch, S_param_k, E_Yx_std_torch, param_detached, param_type, k=k_kernel
            )
            g_hat_S_k_squared = g_hat_S_k.pow(2) # Reward

        baseline = g_hat_S_k_squared.mean() if use_baseline else 0.0

        # Calculate score function: grad_param log p(S|param)
        if param_type == 'alpha':
            # Score for alpha: (epsilon^2 - 1) / (2 * alpha)
            score_term = (epsilon_k.pow(2) - 1.0) / (2.0 * alpha_clamped_detached + EPS)
        elif param_type == 'theta':
            # Score for theta: [(S-X)^2 - alpha] / (2 * alpha) = [(epsilon*sqrt(alpha))^2 - alpha]/(2*alpha)
            # score_term = (epsilon_k.pow(2) * alpha_clamped_detached - alpha_clamped_detached) / (2.0 * alpha_clamped_detached + EPS) # Simplify?
            # Original derivation: ((S-X)^2 - exp(theta)) / (2 * exp(theta))
            score_term = (epsilon_k.pow(2) * noise_var - noise_var) / (2.0 * noise_var + EPS)
            # score_term = (epsilon_k.pow(2) - 1.0) / 2.0 # Simplified version if noise_var cancels if > 0

        # Gradient estimate for E[g(S)^2] for this sample
        term_to_average = (g_hat_S_k_squared - baseline).unsqueeze(1) * score_term
        grad_term2_accum += term_to_average.mean(dim=0) # Average over batch N

    # Final REINFORCE gradient estimate for T2
    grad_term2_reinforce = grad_term2_accum / n_grad_samples

    # --- Penalty Gradient (using Autograd and Chain Rule if needed) ---
    alpha_torch_param = None
    if param_type == 'alpha':
        alpha_torch_param = param_torch.clone().detach().requires_grad_(True)
        grad_penalty_torch = analytical_gradient_penalty_alpha(alpha_torch_param, penalty_type, penalty_lambda)
    elif param_type == 'theta':
        theta_torch_param = param_torch.clone().detach().requires_grad_(True)
        alpha_from_theta = torch.exp(theta_torch_param)
        # Get grad w.r.t alpha first
        grad_penalty_alpha = analytical_gradient_penalty_alpha(alpha_from_theta, penalty_type, penalty_lambda)
        # Apply chain rule: dP/dtheta = dP/dalpha * dalpha/dtheta = dP/dalpha * alpha
        grad_penalty_torch = grad_penalty_alpha * alpha_from_theta # Element-wise product
    else:
        grad_penalty_torch = torch.zeros_like(param_torch)

    if torch.isnan(grad_penalty_torch).any():
         print(f"Warning: NaN detected in penalty gradient calculation for {param_type}.")
         grad_penalty_torch.nan_to_num_(0.0) # Replace NaN with 0 for safety

    # --- Total Gradient: dL/dparam = dT2/dparam + dP/dparam ---
    total_gradient = grad_term2_reinforce + grad_penalty_torch

    return total_gradient


def test_estimator(seeds, alpha_lists, X, Y, save_path=None):
    """
    Compares different estimators for the objective E[E[Y|X]^2] - E[E[Y|S]^2].

    Estimators for Term 2 (E[E[Y|S]^2]):
    - Plugin: plugin_estimator_squared_conditional(S, Y)
    - IF: IF_estimator_squared_conditional(S, Y)
    - IF-Plugin: Kernel(Plugin E[Y|X]) -> mean square
    - IF-IF: Kernel(IF E[Y|X]) -> mean square

    Term 1 (E[E[Y|X]^2]) is estimated using IF.

    Args:
        seeds (list): List of random seeds.
        alpha_lists (list): List of alpha vectors (or scalars if uniform noise).
                            Each element corresponds to one setting of alphas.
        X (Tensor or ndarray): Features.
        Y (Tensor or ndarray): Outcomes.
        save_path (str, optional): Path to save the comparison plot. Defaults to None.
    """
    # Sort the list of alphas by the maximum alpha value in each list/scalar
    alpha_lists = sorted(alpha_lists, key=lambda x: np.max(x) if isinstance(x, (np.ndarray, list)) else x)
    n = X.shape[0]  # Sample size
    eps_err = 1e-9 # Epsilon for division by zero in error calculation

    print(f"Running with seeds: {seeds}")
    print(f"Number of alpha settings: {len(alpha_lists)}")
    print(f"Sample size: {n}")

    # Storage: indexed by alpha setting index
    stats = {
        i: {
            'term_2_wrt_if': {'if_plugin': [], 'if_if': []}, # Compare Kernel Term2 estimates to IF Term2
            'objective_wrt_if': {'if_plugin': [], 'if_if': []} # Compare Obj(Kernel T2) to Obj(IF T2)
            # Add comparisons to Plugin if desired
            # 'term_2_wrt_plugin': {'if': [], 'if_plugin': [], 'if_if': []},
            # 'objective_wrt_plugin': {'if': [], 'if_plugin': [], 'if_if': []},
        }
        for i in range(len(alpha_lists))
    }

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx+1}/{len(seeds)} ({seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed) # Also seed torch

        # Ensure data are NumPy arrays for sklearn estimators
        if isinstance(X, Tensor): X_np = X.detach().cpu().numpy()
        else: X_np = np.array(X) # Ensure it's numpy
        if isinstance(Y, Tensor): Y_np = Y.detach().cpu().numpy()
        else: Y_np = np.array(Y) # Ensure it's numpy

        # --- Estimate Term 1 (using IF, assumed more stable/accurate) ---
        # This is constant for all alphas within a seed run
        print("    Estimating Term 1 (IF)...")
        if1 = IF_estimator_squared_conditional(X_np, Y_np, "rf", n_folds=N_FOLDS)
        if np.isnan(if1):
             print("    Term 1 (IF) calculation failed. Skipping seed.")
             continue

        for i, alpha_setting in enumerate(alpha_lists):
            # Handle both scalar alpha and vector alpha
            if isinstance(alpha_setting, (np.ndarray, list)):
                alpha = np.array(alpha_setting)
                alpha_max_str = f"{np.max(alpha):.4f}"
                alpha_min_str = f"{np.min(alpha):.4f}"
            else: # Assume scalar
                alpha = np.full(X_np.shape[1], alpha_setting) # Create vector
                alpha_max_str = f"{alpha_setting:.4f}"
                alpha_min_str = alpha_max_str

            print(f"\tAlpha setting {i}: Max={alpha_max_str}, Min={alpha_min_str}")

            # --- Generate S (Noisy Features) ---
            alpha_safe = np.maximum(alpha, 1e-12) # Ensure positivity
            noise = np.random.multivariate_normal(
                    mean=np.zeros(X_np.shape[1]),
                    cov=np.diag(alpha_safe),
                    size=n)
            S_np = X_np + noise

            # --- Convert relevant data to Tensors for Kernel estimator ---
            X_t = torch.from_numpy(X_np).float()
            S_t = torch.from_numpy(S_np).float()
            alpha_t = torch.from_numpy(alpha).float()

            # --- Estimate Term 2 variants ---
            print("      Estimating Term 2 variants...")
            try:
                # T2 Plugin: plugin_estimator_squared_conditional(S, Y)
                p2 = plugin_estimator_squared_conditional(S_np, Y_np, "rf", n_folds=N_FOLDS)

                # T2 IF: IF_estimator_squared_conditional(S, Y)
                if2 = IF_estimator_squared_conditional(S_np, Y_np, "rf", n_folds=N_FOLDS)

                # T2 IF-IF: Kernel(IF E[Y|X]) -> mean square
                E_Y_X_if = IF_estimator_conditional_mean(X_np, Y_np, "rf", n_folds=N_FOLDS)
                E_Y_X_if_t = torch.from_numpy(E_Y_X_if).float().to(X_t.device) # Move to same device
                E_Y_S_if = estimate_conditional_keops(X_t.to(E_Y_X_if_t.device), S_t.to(E_Y_X_if_t.device), E_Y_X_if_t, alpha_t.to(E_Y_X_if_t.device)).cpu().numpy()
                if2k = np.mean(E_Y_S_if**2)

                # T2 IF-Plugin: Kernel(Plugin E[Y|X]) -> mean square
                E_Y_X_plugin = plugin_estimator_conditional_mean(X_np, Y_np, "rf", n_folds=N_FOLDS)
                E_Y_X_plugin_t = torch.from_numpy(E_Y_X_plugin).float().to(X_t.device) # Move to same device
                E_Y_S_plugin = estimate_conditional_keops(X_t.to(E_Y_X_plugin_t.device), S_t.to(E_Y_X_plugin_t.device), E_Y_X_plugin_t, alpha_t.to(E_Y_X_plugin_t.device)).cpu().numpy()
                if2k_plugin = np.mean(E_Y_S_plugin**2)

            except Exception as e:
                 print(f"      Error during Term 2 estimation for alpha setting {i}: {e}")
                 # Store NaNs or skip this alpha setting for this seed
                 if i in stats: # Check if index exists
                     stats[i]['term_2_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['term_2_wrt_if']['if_if'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_if'].append(np.nan)
                 continue # Skip to next alpha

            # --- Calculate Objectives ---
            # Ensure Term 2 estimates are valid numbers before calculating objectives
            if np.isnan(if2) or np.isnan(if2k) or np.isnan(if2k_plugin):
                 print(f"      Skipping objective calculation due to NaN in Term 2 estimates.")
                 if i in stats:
                     stats[i]['term_2_wrt_if']['if_plugin'].append(np.nan if np.isnan(if2k_plugin) else abs(if2k_plugin - if2) / (abs(if2) + eps_err) * 100)
                     stats[i]['term_2_wrt_if']['if_if'].append(np.nan if np.isnan(if2k) else abs(if2k - if2) / (abs(if2) + eps_err) * 100)
                     stats[i]['objective_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_if'].append(np.nan)
                 continue

            ifobj = if1 - if2
            ifobjk = if1 - if2k
            ifobjk_plugin = if1 - if2k_plugin

            # --- Store Percentage Errors ---
            if2_denom = abs(if2) + eps_err
            ifobj_denom = abs(ifobj) + eps_err

            stats[i]['term_2_wrt_if']['if_plugin'].append(abs(if2k_plugin - if2) / if2_denom * 100)
            stats[i]['term_2_wrt_if']['if_if'].append(abs(if2k - if2) / if2_denom * 100)
            stats[i]['objective_wrt_if']['if_plugin'].append(abs(ifobjk_plugin - ifobj) / ifobj_denom * 100)
            stats[i]['objective_wrt_if']['if_if'].append(abs(ifobjk - ifobj) / ifobj_denom * 100)

            print(f"\t\tIF Obj: {ifobj:.4f}, IF-Plugin Obj: {ifobjk_plugin:.4f}, IF-IF Obj: {ifobjk:.4f}")
            print(f"\t\tIF T2: {if2:.4f}, IF-Plugin T2: {if2k_plugin:.4f}, IF-IF T2: {if2k:.4f}")


    # --- Aggregation and Plotting ---
    print("\n--- Aggregating and Plotting Results ---")
    # Use simple integer indices 1..N on the X‑axis
    x_positions = np.arange(1, len(alpha_lists) + 1)

    def mean_std(metric, method):
        # Handle cases where an alpha_max might not have results if script interrupted
        # Also handle potential NaNs stored during runs
        data_for_alphas = [stats[a][metric][method] for a in range(len(alpha_lists)) if a in stats and method in stats[a][metric]]

        if not data_for_alphas:
             print(f"Warning: No data found for metric='{metric}', method='{method}'. Skipping.")
             return np.full(len(alpha_lists), np.nan), np.full(len(alpha_lists), np.nan)

        # Calculate mean/std ignoring NaNs and handling potentially ragged lists if seeds failed
        means = []
        stds = []
        for alpha_data in data_for_alphas:
            valid_data = [d for d in alpha_data if not np.isnan(d)]
            if not valid_data:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(np.mean(valid_data))
                stds.append(np.std(valid_data))

        # Pad with NaNs if some alpha settings were skipped entirely
        if len(means) < len(alpha_lists):
             padded_means = np.full(len(alpha_lists), np.nan)
             padded_stds = np.full(len(alpha_lists), np.nan)
             # This assumes data_for_alphas corresponds to the first len(means) indices
             padded_means[:len(means)] = means
             padded_stds[:len(stds)] = stds
             return padded_means, padded_stds
        else:
             return np.array(means), np.array(stds)


    # Plotting setup
    metrics = ['term_2_wrt_if', 'objective_wrt_if'] # Focus on comparisons wrt IF
    method_map = {
        'term_2_wrt_if': ['if_plugin', 'if_if'],
        'objective_wrt_if': ['if_plugin', 'if_if']
    }
    styles = {
        'if_plugin':  {'fmt': '^--', 'label': 'Kernel(Plugin E[Y|X])'},
        'if_if':      {'fmt': 'x--', 'label': 'Kernel(IF E[Y|X])'}
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1: axes = [axes] # Handle single metric case

    for ax, metric in zip(axes, metrics):
        for method in method_map[metric]:
            m, s = mean_std(metric, method)
            if np.isnan(m).all(): continue # Skip plotting if no data

            st = styles[method]
            label = f"{st['label']} (vs IF)" # Clarify baseline is IF

            ax.errorbar(x_positions, m, yerr=s, fmt=st['fmt'], capsize=5, label=label, alpha=0.8)

        title_parts = metric.replace('_wrt_if', ' Error wrt IF').replace('_', ' ').title()
        ax.set_title(title_parts)
        ax.set_ylabel('Percentage Error (%)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_yscale('log')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in x_positions])

    axes[-1].set_xlabel('Index of Alpha Setting')
    fig.suptitle('Comparison of Kernel-based Estimators vs IF Estimator', fontsize=16)

    # Create subtitle with alpha setting details (showing max value)
    alpha_max_values = [f"{np.max(a):.2f}" if isinstance(a, (np.ndarray, list)) else f"{a:.2f}" for a in alpha_lists]
    subtitle = 'Alpha Setting Index -> Max Alpha Value:\n' + ', '.join([f"{i+1}:{val}" for i, val in enumerate(alpha_max_values)])
    fig.text(0.5, 0.01, subtitle, ha='center', va='bottom', fontsize=8) # Smaller font for subtitle

    fig.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust rect for subtitle
    plot_filename = save_path or 'kernel_vs_if_comparison.png'
    fig.savefig(plot_filename, dpi=300)
    print(f"\nPlot saved to {plot_filename}")
    plt.close(fig)
