# baselines.py
import os
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb 
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import mean_squared_error
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

# UTILITIES
def standardize_data(X, Y, classification=False):
    X_mean = np.mean(X, axis=0); X_std = np.std(X, axis=0)
    X_std[X_std < EPS] = EPS
    
    if classification:
        # Don't standardize Y for classification tasks
        return (X - X_mean) / X_std, Y, X_mean, X_std, None, None
    else:
        # For regression, standardize Y as before
        Y_mean = np.mean(Y); Y_std = np.std(Y)
        if Y_std < EPS: Y_std = EPS
        return (X - X_mean) / X_std, (Y - Y_mean) / Y_std, X_mean, X_std, Y_mean, Y_std

def compute_population_stats(selected_indices: List[int],
                             meaningful_indices_list: List[List[int]]) -> Tuple[List[Dict], Dict]:
    """Compute population-wise statistics for selected variables."""
    pop_stats = []
    percentages = []
    selected_set = set(selected_indices)

    # check if meaningful_indices_list is None
    if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
        print(f"Since meaningful_indices_list is None, using prediction loss for selection.")
        return pop_stats, {
            'min_percentage': None, 'max_percentage': None,
            'median_percentage': None, 'min_population_details': None,
            'max_population_details': None
        }
    for i, meaningful in enumerate(meaningful_indices_list):
        meaningful_set = set(meaningful)
        common = selected_set.intersection(meaningful_set)
        count = len(common)
        total = len(meaningful_set)
        percentage = (count / total * 100) if total > 0 else 0.0
        percentages.append(percentage)
        pop_stats.append({
            'population': i, 'selected_relevant_count': count,
            'total_relevant': total, 'percentage': percentage
        })

    min_perc = min(percentages) if percentages else 0.0
    max_perc = max(percentages) if percentages else 0.0
    median_perc = float(np.median(percentages)) if percentages else 0.0
    min_pop_idx = np.argmin(percentages) if percentages else -1
    max_pop_idx = np.argmax(percentages) if percentages else -1

    overall_stats = {
        'min_percentage': min_perc, 'max_percentage': max_perc,
        'median_percentage': median_perc,
        'min_population_details': pop_stats[min_pop_idx] if min_pop_idx != -1 else None,
        'max_population_details': pop_stats[max_pop_idx] if max_pop_idx != -1 else None,
    }
    return pop_stats, overall_stats
# VANILLA POOLED BASELINES
def baseline_lasso_comparison(
    pop_data: List[Dict[str, Any]],
    budget: int,
    alpha_lasso: Optional[float] = None,
    lasso_alphas_to_try: Optional[List[float]] = None,
    classification: bool = False,  # New parameter
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run Lasso/LogisticRegression baseline comparison on pooled pop_data.
    For classification tasks, uses LogisticRegression with L1 regularization.
    
    Returns the best baseline_results dict with keys:
      - alpha_value, selected_indices, baseline_coeffs,
        baseline_pop_stats, baseline_overall_stats,
        precision, recall, f1_score
      - For classification, also includes: accuracy
    """
    if seed is not None:
        np.random.seed(seed)
    # pool raw X, Y
    X_pooled = np.vstack([pop['X_raw'] for pop in pop_data])
    Y_pooled = np.hstack([pop['Y_raw'] for pop in pop_data])
    X_std, Y_std, _, _, _, _ = standardize_data(X_pooled, Y_pooled, classification=classification)

    # determine values to try
    if lasso_alphas_to_try is None:
        lasso_alphas_to_try = [alpha_lasso] if alpha_lasso is not None else [0.001, 0.01]

    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
    best_lasso_f1 = -1.0
    best_results = {}
    best_metric = float('inf')  # Lower is better for MSE and log loss

    for current_alpha in lasso_alphas_to_try:
        model = Lasso(alpha=current_alpha, fit_intercept=False, max_iter=10000, tol=1e-4)
        model.fit(X_std, Y_std)
        
        # Get coefficients and selected features
        if classification:
            coeffs = model.coef_[0]  # For binary classification
        else:
            coeffs = model.coef_
            
        selected_idx = np.argsort(np.abs(coeffs))[-budget:]
        
        # Calculate prediction metric based on task type
        if classification:
            # Use log loss for classification
            y_pred_proba = model.predict_proba(X_std)
            prediction_metric = log_loss(Y_std, y_pred_proba)
            accuracy = accuracy_score(Y_std, model.predict(X_std))
        else:
            # Use MSE for regression
            prediction_metric = mean_squared_error(Y_std, model.predict(X_std))
            accuracy = None
        
        if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
            print(f"Since meaningful_indices_list is None, using prediction loss for selection.")
            if prediction_metric < best_metric:
                best_metric = prediction_metric
                best_results = {
                    'alpha_value': current_alpha,
                    'selected_indices': selected_idx.tolist(),
                    'all_indices_ranked': np.argsort(-np.abs(coeffs)).tolist(),
                    'baseline_coeffs': coeffs[selected_idx].tolist(),
                    'baseline_pop_stats': None,
                    'baseline_overall_stats': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None,
                    'accuracy': accuracy  # Only meaningful for classification
                }
        else:
            sel_set = set(selected_idx)
            true_set = set.union(*(set(mi) for mi in meaningful_indices_list))
            intersect = len(sel_set & true_set)
            prec = intersect / len(sel_set) if sel_set else 0.0
            rec = intersect / len(true_set) if true_set else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

            if prediction_metric < best_metric:
                best_metric = prediction_metric
                best_lasso_f1 = f1
                pop_stats, overall_stats = compute_population_stats(
                    selected_idx.tolist(), meaningful_indices_list
                )
                best_results = {
                    'alpha_value': current_alpha,
                    'selected_indices': selected_idx.tolist(),
                    'all_indices_ranked': np.argsort(-np.abs(coeffs)).tolist(),
                    'baseline_coeffs': coeffs[selected_idx].tolist(),
                    'baseline_pop_stats': pop_stats,
                    'baseline_overall_stats': overall_stats,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1,
                    'accuracy': accuracy  # Only meaningful for classification
                }

    return best_results

def baseline_xgb_comparison(pop_data: List[Dict[str, Any]],
                            budget: int,
                            classification: bool = False,
                            seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Run XGBoost baseline comparison on pooled pop_data.
    Returns the best baseline_results dict with keys:
      - alpha_value, selected_indices, baseline_coeffs,
        baseline_pop_stats, baseline_overall_stats,
        precision, recall, f1_score
      - For classification, also includes: accuracy
    """
    if seed is not None:
        np.random.seed(seed)
    # pool raw X, Y
    X_pooled = np.vstack([pop['X_raw'] for pop in pop_data])
    Y_pooled = np.hstack([pop['Y_raw'] for pop in pop_data])
    X_std, Y_std, _, _, _, _ = standardize_data(X_pooled, Y_pooled, classification=classification)
    
    # XGBoost model
    if classification:
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=seed)
    else:
        model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=seed)
    
    model.fit(X_std, Y_std)
    
    # Get feature importances
    importances = model.feature_importances_
    # Select top features
    selected_idx = np.argsort(importances)[-budget:]

    # Calculate prediction metrics
    if classification:
        y_pred = model.predict(X_std)
        y_pred_proba = model.predict_proba(X_std)[:, 1] if len(model.classes_) == 2 else None
        prediction_metric = log_loss(Y_std, y_pred_proba) if y_pred_proba is not None else None
        accuracy = accuracy_score(Y_std, y_pred)
    else:
        prediction_metric = mean_squared_error(Y_std, model.predict(X_std))
        accuracy = None

    if pop_data[0]['meaningful_indices'] is None or any(pop['meaningful_indices'] is None for pop in pop_data):
        print(f"Since meaningful_indices_list is None, using prediction loss for selection.")
        best_results = {
            'alpha_value': None,
            'selected_indices': selected_idx.tolist(),
            'all_indices_ranked': np.argsort(-importances).tolist(),
            'baseline_coeffs': importances[selected_idx].tolist(),
            'baseline_pop_stats': None,
            'baseline_overall_stats': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'accuracy': accuracy,  # Added for classification
            'prediction_metric': prediction_metric
        }
    else:
        meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
        sel_set = set(selected_idx)
        true_set = set.union(*(set(mi) for mi in meaningful_indices_list))
        intersect = len(sel_set & true_set)
        prec = intersect / len(sel_set) if sel_set else 0.0
        rec = intersect / len(true_set) if true_set else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

        pop_stats, overall_stats = compute_population_stats(
            selected_idx.tolist(), meaningful_indices_list
        )
        best_results = {
            'alpha_value': None,
            'selected_indices': selected_idx.tolist(),
            'all_indices_ranked': np.argsort(-importances).tolist(),
            'baseline_coeffs': importances[selected_idx].tolist(),
            'baseline_pop_stats': pop_stats,
            'baseline_overall_stats': overall_stats,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'accuracy': accuracy,  # Added for classification
            'prediction_metric': prediction_metric
        }
    return best_results

def baseline_dro_lasso_comparison(
    pop_data: List[Dict[str, Any]],
    budget: int,
    alpha_lasso: Optional[float] = None,
    lasso_alphas_to_try: Optional[List[float]] = None,
    classification: bool = False,
    max_iter: int = 100,
    tol: float = 1e-4,
    eta: float = 0.1,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run DRO Lasso/LogisticRegression comparison on pop_data.
    Uses a min-max approach focusing on worst-case performance across populations.
    For classification, uses LogisticRegression with L1 penalty.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Determine values to try
    if lasso_alphas_to_try is None:
        lasso_alphas_to_try = [alpha_lasso] if alpha_lasso is not None else [0.001, 0.01]
    
    # Get population data
    population_data = []
    for pop in pop_data:
        X_std, Y_std, _, _, _, _ = standardize_data(pop['X_raw'], pop['Y_raw'], classification=classification)
        population_data.append((X_std, Y_std))
    
    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
    best_lasso_f1 = -1.0
    best_prediction_max_loss = float('inf')
    best_results = {}
    
    for current_alpha in lasso_alphas_to_try:
        # Initialize uniform weights for each population
        pop_weights = np.ones(len(population_data)) / len(population_data)
        
        # Initialize model
        model = Lasso(alpha=current_alpha, fit_intercept=False, max_iter=10000, tol=tol)
        
        # DRO iterations
        for _ in range(max_iter):
            if classification:
                # For classification, don't weight the target values
                # Instead, create a combined dataset with sample weights
                X_all = []
                Y_all = []
                sample_weights = []
                
                for (X, Y), w in zip(population_data, pop_weights):
                    X_all.append(X)
                    Y_all.append(Y)
                    # Same weight for all samples in this population
                    sample_weights.extend([w] * len(Y))
                
                X_all = np.vstack(X_all)
                Y_all = np.hstack(Y_all)
                sample_weights = np.array(sample_weights)
                
                # Fit with sample_weight parameter
                model.fit(X_all, Y_all, sample_weight=sample_weights)
            else:
                # For regression, weight both X and Y directly
                X_weighted = np.vstack([w * X for (X, _), w in zip(population_data, pop_weights)])
                Y_weighted = np.hstack([w * Y for (_, Y), w in zip(population_data, pop_weights)])
                model.fit(X_weighted, Y_weighted)
            
            # Calculate losses for each population
            population_losses = []
            for X, Y in population_data:
                if classification:
                    # Log loss for classification
                    pred_proba = model.predict_proba(X)
                    loss = log_loss(Y, pred_proba)
                else:
                    # MSE loss for regression
                    pred = model.predict(X)
                    loss = np.mean((Y - pred) ** 2)
                population_losses.append(loss)
            
            # Update weights based on losses (exponentiated gradient)
            updated_weights = pop_weights * np.exp(eta * np.array(population_losses))
            # Normalize
            pop_weights = updated_weights / updated_weights.sum()
            
            # Check for convergence
            if np.max(np.abs(updated_weights - pop_weights)) < tol:
                break
        
        # Get final model and coefficients
        if classification:
            coeffs = model.coef_[0]  # Binary classification
        else:
            coeffs = model.coef_
        selected_idx = np.argsort(np.abs(coeffs))[-budget:]
        
        # Calculate maximum loss across populations
        if classification:
            max_loss = max(log_loss(Y, model.predict_proba(X)) for X, Y in population_data)
            # Also get accuracy for reporting
            accuracies = [accuracy_score(Y, model.predict(X)) for X, Y in population_data]
            min_accuracy = min(accuracies)
        else:
            max_loss = max(np.mean((Y - model.predict(X)) ** 2) for X, Y in population_data)
            min_accuracy = None
        
        if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
            # Use max loss if meaningful indices aren't available
            if max_loss < best_prediction_max_loss:
                best_prediction_max_loss = max_loss
                best_results = {
                    'alpha_value': current_alpha,
                    'selected_indices': selected_idx.tolist(),
                    'all_indices_ranked': np.argsort(-np.abs(coeffs)).tolist(),
                    'baseline_coeffs': coeffs[selected_idx].tolist(),
                    'baseline_pop_stats': None,
                    'baseline_overall_stats': None,
                    'precision': None,
                    'recall': None,
                    'f1_score': None,
                    'accuracy': min_accuracy,  # Only meaningful for classification
                    'max_population_loss': max_loss,
                    'final_pop_weights': pop_weights.tolist()
                }
        else:
            # Calculate F1 score if meaningful indices are available
            sel_set = set(selected_idx)
            true_set = set.union(*(set(mi) for mi in meaningful_indices_list))
            intersect = len(sel_set & true_set)
            prec = intersect / len(sel_set) if sel_set else 0.0
            rec = intersect / len(true_set) if true_set else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            if max_loss < best_prediction_max_loss:
                best_prediction_max_loss = max_loss
                best_lasso_f1 = f1
                pop_stats, overall_stats = compute_population_stats(
                    selected_idx.tolist(), meaningful_indices_list
                )
                best_results = {
                    'alpha_value': current_alpha,
                    'selected_indices': selected_idx.tolist(),
                    'all_indices_ranked': np.argsort(-np.abs(coeffs)).tolist(),
                    'baseline_coeffs': coeffs[selected_idx].tolist(),
                    'baseline_pop_stats': pop_stats,
                    'baseline_overall_stats': overall_stats,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1,
                    'accuracy': min_accuracy,  # Only meaningful for classification
                    'max_population_loss': max_loss,
                    'final_pop_weights': pop_weights.tolist()
                }
    
    return best_results


def baseline_dro_xgb_comparison(
    pop_data: List[Dict[str, Any]],
    budget: int,
    classification: bool = False,
    max_iter: int = 20,
    eta: float = 0.1,  # Step size for weight updates
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run DRO XGBoost comparison on pop_data.
    Uses a min-max approach focusing on worst-case performance across populations.
    
    Returns the best baseline_results dict with keys:
      - selected_indices, baseline_coeffs, baseline_pop_stats,
        baseline_overall_stats, precision, recall, f1_score
      - For classification, also includes: accuracy
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get population data
    population_data = []
    for pop in pop_data:
        X_std, Y_std, _, _, _, _ = standardize_data(pop['X_raw'], pop['Y_raw'], classification=classification)
        population_data.append((X_std, Y_std))
    
    # Initialize uniform weights for each population
    pop_weights = np.ones(len(population_data)) / len(population_data)
    
    # Initialize model
    if classification:
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=seed)
    else:
        model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=seed)
    
    # DRO iterations
    for iteration in range(max_iter):
        # Create sample weights matrix for XGBoost
        # XGBoost expects sample weights per observation, not per population
        sample_weights = []
        X_all = []
        Y_all = []
        
        for idx, ((X, Y), weight) in enumerate(zip(population_data, pop_weights)):
            X_all.append(X)
            Y_all.append(Y)
            sample_weights.extend([weight] * len(Y))  # Same weight for all samples in population
        
        X_all = np.vstack(X_all)
        Y_all = np.hstack(Y_all)
        sample_weights = np.array(sample_weights)
        
        # Fit model with sample weights
        model.fit(X_all, Y_all, sample_weight=sample_weights)
        
        # Calculate losses for each population
        population_losses = []
        for X, Y in population_data:
            if classification:
                # For classification, use log loss with predict_proba
                y_pred_proba = model.predict_proba(X)
                # Handle binary and multiclass cases
                if len(model.classes_) == 2:
                    # Binary classification - get positive class probability
                    y_pred_proba = y_pred_proba[:, 1]
                    # Convert to 2D array for log_loss
                    y_pred_proba_2d = np.vstack((1-y_pred_proba, y_pred_proba)).T
                    loss = log_loss(Y, y_pred_proba_2d)
                else:
                    # Multiclass classification
                    loss = log_loss(Y, y_pred_proba)
            else:
                # MSE loss for regression
                pred = model.predict(X)
                loss = np.mean((Y - pred) ** 2)
            
            population_losses.append(loss)
        
        # Update weights based on losses (exponentiated gradient)
        updated_weights = pop_weights * np.exp(eta * np.array(population_losses))
        # Normalize
        pop_weights = updated_weights / updated_weights.sum()
        
        print(f"DRO XGBoost Iteration {iteration+1}: Population weights = {pop_weights}")
    
    # Get feature importances from final model
    importances = model.feature_importances_
    selected_idx = np.argsort(importances)[-budget:]
    
    # Calculate max loss and min accuracy across populations for reporting
    if classification:
        max_loss = max(
            log_loss(Y, model.predict_proba(X)) for X, Y in population_data
        )
        # Also calculate accuracies for reporting
        accuracies = [accuracy_score(Y, model.predict(X)) for X, Y in population_data]
        min_accuracy = min(accuracies)
    else:
        max_loss = max(
            np.mean((Y - model.predict(X)) ** 2) for X, Y in population_data
        )
        min_accuracy = None
    
    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
    
    if meaningful_indices_list is None or any(mi is None for mi in meaningful_indices_list):
        best_results = {
            'selected_indices': selected_idx.tolist(),
            'all_indices_ranked': np.argsort(-importances).tolist(),
            'baseline_coeffs': importances[selected_idx].tolist(),
            'baseline_pop_stats': None,
            'baseline_overall_stats': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'accuracy': min_accuracy,  # Added for classification
            'max_population_loss': max_loss,
            'final_pop_weights': pop_weights.tolist()
        }
    else:
        # Calculate F1 score
        sel_set = set(selected_idx)
        true_set = set.union(*(set(mi) for mi in meaningful_indices_list))
        intersect = len(sel_set & true_set)
        prec = intersect / len(sel_set) if sel_set else 0.0
        rec = intersect / len(true_set) if true_set else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        pop_stats, overall_stats = compute_population_stats(
            selected_idx.tolist(), meaningful_indices_list
        )
        best_results = {
            'selected_indices': selected_idx.tolist(),
            'baseline_coeffs': importances[selected_idx].tolist(),
            'all_indices_ranked': np.argsort(-importances).tolist(),
            'baseline_pop_stats': pop_stats,
            'baseline_overall_stats': overall_stats,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'accuracy': min_accuracy,  # Added for classification
            'max_population_loss': max_loss,
            'final_pop_weights': pop_weights.tolist()
        }
    
    return best_results