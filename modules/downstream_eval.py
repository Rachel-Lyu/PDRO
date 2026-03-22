import os
import pickle
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns

class DownstreamEvaluator:
    def __init__(self, checkpoint_manager=None):
        self.checkpoint_manager = checkpoint_manager
    
    def evaluate_selections(self, pop_data_test_val, our_method_results, baseline_results, 
                           budget, seed, is_classification=False, save_path=None):
        """Evaluate all selection methods on downstream tasks"""
        # Check if already completed
        if self.checkpoint_manager and self.checkpoint_manager.is_downstream_eval_complete():
            print("Downstream evaluation already completed. Loading results...")
            eval_results = self.checkpoint_manager.load_checkpoint('downstream_eval')
            if eval_results:
                return eval_results
            else:
                print("Could not load checkpoint. Re-running evaluation.")
        
        # Split data into train/val
        train_test_indices = self._create_train_test_split(pop_data_test_val, seed, train_frac=0.8)
        
        # Create budgets to evaluate (allow checking multiple budgets)
        budgets = [3, budget, 7, 10]
        if budget > 5:
            budgets.append(budget // 2)
        else:
            budgets.append(min(budget * 2, len(pop_data_test_val[0]['X_std'][0])))
        
        all_results = {}
        for b in budgets:
            results_for_budget = self._evaluate_for_budget(
            pop_data_test_val, train_test_indices, 
            our_method_results, baseline_results,
            b, seed, is_classification
            )
            all_results[b] = results_for_budget
            
            # Save comparison CSV for this budget
            if save_path:
                results_df = pd.DataFrame(results_for_budget['all_results'])
                csv_path = os.path.join(save_path, f'results_comparison_budget_{b}.csv')
                results_df.to_csv(csv_path, index=False)
                print(f"Saved downstream evaluation results to: {csv_path}")
                
                # Generate and save performance plots
                self._plot_performance_comparison(results_df, b, is_classification, save_path)
        
        # Save all evaluation results
        if self.checkpoint_manager:
            results_path = self.checkpoint_manager.save_checkpoint('downstream_eval', all_results)
            self.checkpoint_manager.mark_downstream_eval_complete(results_path)
        
        return all_results
    
    def _create_train_test_split(self, pop_data_test_val, seed, train_frac=0.8):
        """Create train/test splits for each population"""
        np.random.seed(seed)
        train_test_indices = {}
        
        for pop in pop_data_test_val:
            num_samples = pop['X_std'].shape[0]
            train_size = int(num_samples * train_frac)
            all_indices = np.arange(num_samples)
            np.random.shuffle(all_indices)
            
            train_idx = all_indices[:train_size]
            test_idx = all_indices[train_size:]
            
            train_test_indices[pop['pop_id']] = (train_idx, test_idx)
        
        return train_test_indices
    
    def _evaluate_for_budget(self, pop_data_test_val, train_test_indices, 
                           our_method_results, baseline_results, budget, 
                           seed, is_classification):
        """Evaluate all methods for a specific budget"""
        all_downstream_results = []
        
        # Evaluate our method
        our_indices = our_method_results['selected_indices'][:budget]
        our_results = self._evaluate_method(
            pop_data_test_val, train_test_indices, our_indices,
            'our_method', seed, is_classification
        )
        all_downstream_results.extend(our_results)
        
        # Evaluate baseline methods
        baseline_methods = [
            ('baseline_lasso_results', 'baseline_lasso'),
            ('baseline_dro_lasso_results', 'baseline_dro_lasso'),
            ('baseline_xgb_results', 'baseline_xgb'),
            ('baseline_dro_xgb_results', 'baseline_dro_xgb')
        ]
        
        for result_key, method_name in baseline_methods:
            if result_key in baseline_results and baseline_results[result_key]:
                baseline_indices = baseline_results[result_key]['selected_indices'][:budget]
                baseline_results_list = self._evaluate_method(
                    pop_data_test_val, train_test_indices, baseline_indices,
                    method_name, seed, is_classification
                )
                all_downstream_results.extend(baseline_results_list)
        
        return {
            'budget': budget,
            'all_results': all_downstream_results
        }
    
    def _evaluate_method(self, pop_data_test_val, train_test_indices, 
                        selected_indices, method_name, seed, is_classification):
        """Evaluate a specific method with selected indices"""
        results = []
        
        for pop in pop_data_test_val:
            train_idx, test_idx = train_test_indices[pop['pop_id']]
            
            # Get data for selected features
            if isinstance(pop['X_std'], torch.Tensor):
                X_train = pop['X_std'][train_idx][:, selected_indices].detach().cpu().numpy()
                Y_train = pop['Y_std'][train_idx].detach().cpu().numpy()
                X_test = pop['X_std'][test_idx][:, selected_indices].detach().cpu().numpy()
                Y_test = pop['Y_std'][test_idx].detach().cpu().numpy()
            else:
                X_train = pop['X_std'][train_idx][:, selected_indices]
                Y_train = pop['Y_std'][train_idx]
                X_test = pop['X_std'][test_idx][:, selected_indices]
                Y_test = pop['Y_std'][test_idx]
            
            # For classification tasks, ensure Y is binary
            if is_classification:
                Y_train = Y_train.astype(int)
                Y_test = Y_test.astype(int)
                
                # Train and evaluate model
                model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                Y_pred_proba = model.predict_proba(X_test)
                
                # Record classification metrics
                result = {
                    'population': pop['pop_id'],
                    'downstream_accuracy': accuracy_score(Y_test, Y_pred),
                    'logloss': log_loss(Y_test, Y_pred_proba),
                    'selected_indices': selected_indices,
                    'source': method_name
                }
            else:
                # Regression task
                model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                
                # Record regression metrics
                result = {
                    'population': pop['pop_id'],
                    'mse': mean_squared_error(Y_test, Y_pred),
                    'r2': r2_score(Y_test, Y_pred),
                    'selected_indices': selected_indices,
                    'source': method_name
                }
            
            results.append(result)
        
        return results
    
    def _plot_performance_comparison(self, results_df, budget, is_classification, save_path):
        """Create performance comparison plots across methods and populations"""
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure directory if it doesn't exist
        fig_dir = os.path.join(save_path, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Define metric based on task type
        if is_classification:
            primary_metric = 'downstream_accuracy'
            secondary_metric = 'logloss'
            primary_label = 'Accuracy'
            secondary_label = 'Log Loss (lower is better)'
        else:
            primary_metric = 'r2'
            secondary_metric = 'mse'
            primary_label = 'R² Score'
            secondary_label = 'Mean Squared Error (lower is better)'
        
        # Get unique populations and methods
        populations = results_df['population'].unique()
        methods = results_df['source'].unique()
        
        # 1. Create bar chart comparing primary metric across methods for each population
        plt.figure(figsize=(12, 6))
        
        # Reshape data for seaborn
        plot_df = results_df.pivot(index='population', columns='source', values=primary_metric)
        
        ax = sns.barplot(data=plot_df)
        plt.title(f'Comparison of {primary_label} by Method (Budget={budget})')
        plt.ylabel(primary_label)
        plt.xlabel('Population')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels on bars
        for i, container in enumerate(ax.containers):
            ax.bar_label(container, fmt='%.3f')
        
        plt.savefig(os.path.join(fig_dir, f'budget_{budget}_{primary_metric}_comparison.png'), dpi=300)
        plt.close()
        
        # 2. Create bar chart for secondary metric
        plt.figure(figsize=(12, 6))
        
        # Reshape data for seaborn
        plot_df = results_df.pivot(index='population', columns='source', values=secondary_metric)
        
        ax = sns.barplot(data=plot_df)
        plt.title(f'Comparison of {secondary_label} (Budget={budget})')
        plt.ylabel(secondary_label)
        plt.xlabel('Population')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels on bars
        for i, container in enumerate(ax.containers):
            ax.bar_label(container, fmt='%.3f')
        
        plt.savefig(os.path.join(fig_dir, f'budget_{budget}_{secondary_metric}_comparison.png'), dpi=300)
        plt.close()
        
        # 3. Create a heatmap of performance across populations and methods
        plt.figure(figsize=(10, 8))
        
        # Create heatmap for primary metric
        heatmap_data = results_df.pivot(index='population', columns='source', values=primary_metric)
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'{primary_label} Heatmap (Budget={budget})')
        plt.tight_layout()
        
        plt.savefig(os.path.join(fig_dir, f'budget_{budget}_{primary_metric}_heatmap.png'), dpi=300)
        plt.close()
    
    def generate_summary_report(self, all_results, is_classification=False, save_path=None):
        """Generate a comprehensive summary report of all evaluation results"""
        if not all_results:
            print("No results to summarize")
            return None
        
        summary_data = []
        
        # Define metric based on task type
        if is_classification:
            primary_metric = 'downstream_accuracy'
            secondary_metric = 'logloss'
        else:
            primary_metric = 'r2'
            secondary_metric = 'mse'
        
        # Process results for each budget
        for budget, results_for_budget in all_results.items():
            results_df = pd.DataFrame(results_for_budget['all_results'])
            
            # Get all methods
            methods = results_df['source'].unique()
            
            # Aggregate metrics for each method
            for method in methods:
                method_results = results_df[results_df['source'] == method]
                
                # Calculate mean, min, max, std for primary metric
                primary_mean = method_results[primary_metric].mean()
                primary_min = method_results[primary_metric].min()
                primary_max = method_results[primary_metric].max()
                primary_std = method_results[primary_metric].std()
                
                # Calculate mean, min, max, std for secondary metric
                secondary_mean = method_results[secondary_metric].mean()
                secondary_min = method_results[secondary_metric].min()
                secondary_max = method_results[secondary_metric].max()
                secondary_std = method_results[secondary_metric].std()
                
                # Save summary stats for this method and budget
                summary_data.append({
                    'budget': budget,
                    'method': method,
                    f'{primary_metric}_mean': primary_mean,
                    f'{primary_metric}_min': primary_min,
                    f'{primary_metric}_max': primary_max,
                    f'{primary_metric}_std': primary_std,
                    f'{secondary_metric}_mean': secondary_mean,
                    f'{secondary_metric}_min': secondary_min,
                    f'{secondary_metric}_max': secondary_max,
                    f'{secondary_metric}_std': secondary_std
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary report
        if save_path:
            summary_path = os.path.join(save_path, 'summary_report.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary report to: {summary_path}")
            
            # Create performance comparison plots for the summary
            self._plot_summary_comparison(summary_df, is_classification, save_path)
        
        return summary_df
    
    def _plot_summary_comparison(self, summary_df, is_classification, save_path):
        """Create summary performance comparison plots"""
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure directory if it doesn't exist
        fig_dir = os.path.join(save_path, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # Define metric based on task type
        if is_classification:
            primary_metric = 'downstream_accuracy'
            secondary_metric = 'logloss'
            primary_label = 'Accuracy'
            secondary_label = 'Log Loss (lower is better)'
        else:
            primary_metric = 'r2'
            secondary_metric = 'mse'
            primary_label = 'R² Score'
            secondary_label = 'Mean Squared Error (lower is better)'
        
        # Plot primary metric means by budget and method
        plt.figure(figsize=(12, 6))
        
        # Reshape data for plotting
        pivot_df = summary_df.pivot(index='method', columns='budget', values=f'{primary_metric}_mean')
        
        ax = pivot_df.plot(kind='bar', figsize=(12, 6))
        plt.title(f'Mean {primary_label} by Method and Budget')
        plt.ylabel(f'Mean {primary_label}')
        plt.xlabel('Method')
        plt.legend(title='Budget')
        plt.tight_layout()
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        
        plt.savefig(os.path.join(fig_dir, f'summary_{primary_metric}_by_budget.png'), dpi=300)
        plt.close()
        
        # Plot secondary metric means by budget and method
        plt.figure(figsize=(12, 6))
        
        # Reshape data for plotting
        pivot_df = summary_df.pivot(index='method', columns='budget', values=f'{secondary_metric}_mean')
        
        ax = pivot_df.plot(kind='bar', figsize=(12, 6))
        plt.title(f'Mean {secondary_label} by Method and Budget')
        plt.ylabel(f'Mean {secondary_label}')
        plt.xlabel('Method')
        plt.legend(title='Budget')
        plt.tight_layout()
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        
        plt.savefig(os.path.join(fig_dir, f'summary_{secondary_metric}_by_budget.png'), dpi=300)
        plt.close()
        
        # Create performance variability plots (min, max, std)
        plt.figure(figsize=(14, 8))
        
        # Filter for a specific budget (use the max budget for summary)
        max_budget = summary_df['budget'].max()
        budget_data = summary_df[summary_df['budget'] == max_budget]
        
        # Create error bars showing min, max, mean for primary metric
        methods = budget_data['method'].values
        means = budget_data[f'{primary_metric}_mean'].values
        mins = budget_data[f'{primary_metric}_min'].values
        maxs = budget_data[f'{primary_metric}_max'].values
        
        # Calculate errors for error bars
        yerr_min = means - mins
        yerr_max = maxs - means
        
        plt.errorbar(methods, means, yerr=[yerr_min, yerr_max], fmt='o', capsize=5, 
                    markersize=8, linewidth=2, elinewidth=1, label=f'{primary_label}')
        
        plt.title(f'{primary_label} Range by Method (Budget={max_budget})')
        plt.ylabel(primary_label)
        plt.xlabel('Method')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(fig_dir, f'summary_{primary_metric}_range.png'), dpi=300)
        plt.close()