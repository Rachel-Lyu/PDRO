import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.linear_model import Lasso
import xgboost as xgb

class BaselineMethods:
    def __init__(self, checkpoint_manager=None):
        self.checkpoint_manager = checkpoint_manager
    
    def run_all_baselines(self, pop_data, budget, params, is_classification=False):
        """Run all baseline methods with checkpointing"""
        # Check if already completed
        if self.checkpoint_manager and self.checkpoint_manager.is_baselines_complete():
            print("Baselines already completed. Loading results...")
            baseline_results = self.checkpoint_manager.load_checkpoint('baselines')
            if baseline_results:
                return baseline_results
            else:
                print("Could not load checkpoint. Re-running baselines.")
        
        all_baseline_results = {}
        
        # Run each baseline method
        lasso_results = self.baseline_lasso(pop_data, budget, params.get('alpha_lasso'), 
                                          params.get('seed'), is_classification)
        dro_lasso_results = self.baseline_dro_lasso(pop_data, budget, params.get('alpha_lasso'),
                                                  params.get('seed'), is_classification)
        xgb_results = self.baseline_xgb(pop_data, budget, params.get('seed'), is_classification)
        dro_xgb_results = self.baseline_dro_xgb(pop_data, budget, params.get('seed'), is_classification)
        
        all_baseline_results = {
            'baseline_lasso_results': lasso_results,
            'baseline_dro_lasso_results': dro_lasso_results,
            'baseline_xgb_results': xgb_results,
            'baseline_dro_xgb_results': dro_xgb_results
        }
        
        # Save baseline results
        if self.checkpoint_manager:
            results_paths = self.checkpoint_manager.save_checkpoint('baselines', all_baseline_results)
            self.checkpoint_manager.mark_baselines_complete(results_paths)
        
        return all_baseline_results
    
    def baseline_lasso(self, pop_data, budget, alpha_lasso, seed, is_classification=False):
        """Lasso baseline method"""
        from baselines import baseline_lasso_comparison
        
        print(f"Running Lasso baseline with budget {budget}, alpha={alpha_lasso}, classification={is_classification}")
        
        try:
            lasso_results = baseline_lasso_comparison(
                pop_data=pop_data,
                budget=budget,
                alpha_lasso=alpha_lasso,
                seed=seed,
                classification=is_classification
            )
            
            print(f"Lasso baseline completed successfully")
            return lasso_results
        except Exception as e:
            print(f"Error in Lasso baseline: {str(e)}")
            return None

    def baseline_dro_lasso(self, pop_data, budget, alpha_lasso, seed, is_classification=False):
        """DRO Lasso baseline method"""
        from baselines import baseline_dro_lasso_comparison
        
        print(f"Running DRO Lasso baseline with budget {budget}, alpha={alpha_lasso}, classification={is_classification}")
        
        try:
            dro_lasso_results = baseline_dro_lasso_comparison(
                pop_data=pop_data,
                budget=budget,
                alpha_lasso=alpha_lasso,
                seed=seed,
                classification=is_classification,
                max_iter=100,
                tol=1e-4,
                eta=0.1
            )
            
            print(f"DRO Lasso baseline completed successfully")
            return dro_lasso_results
        except Exception as e:
            print(f"Error in DRO Lasso baseline: {str(e)}")
            return None

    def baseline_xgb(self, pop_data, budget, seed, is_classification=False):
        """XGBoost baseline method"""
        from baselines import baseline_xgb_comparison
        
        print(f"Running XGBoost baseline with budget {budget}, classification={is_classification}")
        
        try:
            xgb_results = baseline_xgb_comparison(
                pop_data=pop_data,
                budget=budget,
                classification=is_classification,
                seed=seed
            )
            
            print(f"XGBoost baseline completed successfully")
            return xgb_results
        except Exception as e:
            print(f"Error in XGBoost baseline: {str(e)}")
            return None

    def baseline_dro_xgb(self, pop_data, budget, seed, is_classification=False):
        """DRO XGBoost baseline method"""
        from baselines import baseline_dro_xgb_comparison
        
        print(f"Running DRO XGBoost baseline with budget {budget}, classification={is_classification}")
        
        try:
            dro_xgb_results = baseline_dro_xgb_comparison(
                pop_data=pop_data,
                budget=budget,
                classification=is_classification,
                max_iter=20,
                eta=0.1,
                seed=seed
            )
            
            print(f"DRO XGBoost baseline completed successfully")
            return dro_xgb_results
        except Exception as e:
            print(f"Error in DRO XGBoost baseline: {str(e)}")
            return None