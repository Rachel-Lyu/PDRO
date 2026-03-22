import os
import json
import pickle
from typing import Dict, Any, Optional

class CheckpointManager:
    def __init__(self, save_path, run_id=None):
        self.save_path = save_path
        self.run_id = run_id
        os.makedirs(save_path, exist_ok=True)
        
        # Track which stages are completed
        self.status_file = os.path.join(save_path, 'checkpoint_status.json')
        self.status = self._load_status()
    
    def _load_status(self):
        """Load checkpoint status or create if not exists"""
        if os.path.exists(self.status_file):
            with open(self.status_file, 'r') as f:
                return json.load(f)
        else:
            # Default status - nothing completed
            status = {
                'data_loaded': False,
                'our_method_complete': False,
                'baselines_complete': False,
                'downstream_eval_complete': False,
                'run_params': {},
                'stage_paths': {}
            }
            self._save_status(status)
            return status
    
    def _save_status(self, status=None):
        """Save current checkpoint status"""
        if status is None:
            status = self.status
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def mark_data_loaded(self, data_path):
        """Mark data loading as complete"""
        self.status['data_loaded'] = True
        self.status['stage_paths']['data'] = data_path
        self._save_status()
    
    def mark_our_method_complete(self, results_path):
        """Mark our method as complete with results path"""
        self.status['our_method_complete'] = True
        self.status['stage_paths']['our_method'] = results_path
        self._save_status()
    
    def mark_baselines_complete(self, results_paths):
        """Mark baseline methods as complete with results paths"""
        self.status['baselines_complete'] = True
        self.status['stage_paths']['baselines'] = results_paths
        self._save_status()
    
    def mark_downstream_eval_complete(self, results_path):
        """Mark downstream evaluation as complete"""
        self.status['downstream_eval_complete'] = True
        self.status['stage_paths']['downstream_eval'] = results_path
        self._save_status()
    
    def store_run_params(self, params):
        """Store run parameters for reproducibility"""
        self.status['run_params'] = params
        self._save_status()
    
    def is_data_loaded(self):
        """Check if data is already loaded"""
        return self.status['data_loaded']
    
    def is_our_method_complete(self):
        """Check if our method is already complete"""
        return self.status['our_method_complete']
    
    def is_baselines_complete(self):
        """Check if baselines are already complete"""
        return self.status['baselines_complete']
    
    def is_downstream_eval_complete(self):
        """Check if downstream evaluation is already complete"""
        return self.status['downstream_eval_complete']
    
    def get_data_path(self):
        """Get path to saved data"""
        return self.status['stage_paths'].get('data')
    
    def get_our_method_results_path(self):
        """Get path to our method results"""
        return self.status['stage_paths'].get('our_method')
    
    def get_baselines_results_paths(self):
        """Get paths to baseline results"""
        return self.status['stage_paths'].get('baselines')
    
    def get_downstream_eval_results_path(self):
        """Get path to downstream evaluation results"""
        return self.status['stage_paths'].get('downstream_eval')
    
    def save_checkpoint(self, stage, data, filename=None):
        """Save a checkpoint for any stage"""
        if filename is None:
            filename = f"{stage}_checkpoint.pkl"
        
        checkpoint_path = os.path.join(self.save_path, filename)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Update status
        self.status['stage_paths'][stage] = checkpoint_path
        self._save_status()
        
        return checkpoint_path
    
    def load_checkpoint(self, stage):
        """Load a checkpoint for a stage"""
        checkpoint_path = self.status['stage_paths'].get(stage)
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_completion_status(self):
        """Get overall completion status"""
        return {
            'data_loaded': self.status['data_loaded'],
            'our_method_complete': self.status['our_method_complete'],
            'baselines_complete': self.status['baselines_complete'],
            'downstream_eval_complete': self.status['downstream_eval_complete']
        }