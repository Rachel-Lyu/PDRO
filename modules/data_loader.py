# data_loader.py
import os
import pickle
import numpy as np
import torch
torch.set_num_threads(4)
from typing import List, Dict, Tuple, Optional, Any

# Import your existing data handling functions
# from data_uci import get_uci_pop_data
# from data_acs import get_acs_pop_data
# ...

class DataManager:
    def __init__(self, cache_dir="./data_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_dataset_cache_path(self, pop_configs, seed, dataset_size, noise_scale=0.1, corr_strength=0.0):
        """Generate a unique cache path for the dataset configuration"""
        pop_str = "_".join([p['dataset_type'] for p in pop_configs])
        cache_name = f"data_{pop_str}_seed{seed}_size{dataset_size}_noise{noise_scale}_corr{corr_strength}.pkl"
        return os.path.join(self.cache_dir, cache_name)
    
    def load_or_generate_data(self, pop_configs, m1, m, dataset_size, 
                              noise_scale=0.1, corr_strength=0.0, 
                              estimator_type="if", device="cpu", 
                              base_model_type="rf", seed=None,
                              acs_data_fraction=0.5,
                              force_regenerate=False,
                              uci_populations=None,
                              exclude_population_features=True,
                              is_classification=False):
        """Load cached dataset or generate a new one"""
        cache_path = self.get_dataset_cache_path(pop_configs, seed, dataset_size, noise_scale, corr_strength)
        
        # Try to load from cache if not forcing regeneration
        if not force_regenerate and os.path.exists(cache_path):
            print(f"Loading dataset from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check if cached data needs subsampling
                if cached_data['dataset_size'] >= dataset_size:
                    # Subsample if needed
                    if cached_data['dataset_size'] > dataset_size:
                        pop_data, pop_data_test_val = self._subsample_data(
                            cached_data['pop_data'], 
                            cached_data['pop_data_test_val'],
                            dataset_size,
                            seed
                        )
                    else:
                        pop_data = cached_data['pop_data']
                        pop_data_test_val = cached_data['pop_data_test_val']
                    
                    return pop_data, pop_data_test_val
                else:
                    print(f"Cached dataset too small ({cached_data['dataset_size']} < {dataset_size}). Regenerating.")
            except Exception as e:
                print(f"Error loading cached data: {e}. Regenerating.")
        
        # Generate new dataset
        print(f"Generating new dataset with seed {seed}...")
        
        # Determine data type and call appropriate generation function
        if any('baseline' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
            pop_data, pop_data_test_val = self._generate_baseline_data(
                pop_configs, dataset_size, m, noise_scale, corr_strength,
                estimator_type, device, base_model_type, seed
            )
        elif any('acs' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
            pop_data, pop_data_test_val = self._generate_acs_data(
                pop_configs, m1, m, dataset_size, acs_data_fraction,
                estimator_type, device, base_model_type, seed
            )
        elif any('uci' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
            pop_data, pop_data_test_val = self._generate_uci_data(
                pop_configs, m1, m, dataset_size, seed, 
                estimator_type, device, base_model_type,
                uci_populations=uci_populations,
                exclude_population_features=exclude_population_features,  # Pass parameter
                force_regenerate=force_regenerate
            )
        else:
            pop_data, pop_data_test_val = self._generate_synthetic_data(
                pop_configs, m1, m, dataset_size, noise_scale, corr_strength, 
                estimator_type, device, base_model_type, seed
            )
            
        # Cache the generated data
        cache_data = {
            'pop_data': pop_data,
            'pop_data_test_val': pop_data_test_val,
            'dataset_size': dataset_size,
            'seed': seed,
            'm1': m1,
            'm': m,
            'noise_scale': noise_scale,
            'corr_strength': corr_strength
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Dataset cached to: {cache_path}")
        return pop_data, pop_data_test_val
    
    def _subsample_data(self, pop_data, pop_data_test_val, target_size, seed):
        """Subsample existing data to target size while maintaining population proportions"""
        np.random.seed(seed if seed is not None else 42)
        
        # Calculate current total size and ratio to target
        current_sizes = [len(pop['X_std']) for pop in pop_data]
        total_current = sum(current_sizes)
        ratio = target_size / total_current
        
        new_pop_data = []
        new_pop_data_test_val = []
        
        # Subsample each population
        for i, (pop, pop_test_val) in enumerate(zip(pop_data, pop_data_test_val)):
            # Calculate target size for this population (maintaining proportions)
            pop_target_size = max(1, int(len(pop['X_std']) * ratio))
            
            # Sample indices without replacement
            indices = np.random.choice(
                len(pop['X_std']), 
                size=pop_target_size, 
                replace=False
            )
            
            # Create new population dict with subsampled data
            new_pop = {k: v for k, v in pop.items()}
            for key in ['X_std', 'Y_std', 'X_raw', 'Y_raw']:
                if key in pop and isinstance(pop[key], (torch.Tensor, np.ndarray)):
                    new_pop[key] = pop[key][indices]
            
            if 'E_Yx_std' in pop and isinstance(pop['E_Yx_std'], (torch.Tensor, np.ndarray)):
                new_pop['E_Yx_std'] = pop['E_Yx_std'][indices]
            
            # Keep test/val data as is
            new_pop_data_test_val.append(pop_test_val)
            new_pop_data.append(new_pop)
        
        return new_pop_data, new_pop_data_test_val
    
    # Implementation of the various data generation methods
    def _generate_baseline_data(self, pop_configs, dataset_size, m, noise_scale, corr_strength,
                               estimator_type, device, base_model_type, seed):
        # Implementation using get_pop_data_baseline_failures 
        from data_baseline_failures import get_pop_data_baseline_failures
        return get_pop_data_baseline_failures(
            pop_configs=pop_configs, dataset_size=dataset_size,
            n_features=m, noise_scale=noise_scale, corr_strength=corr_strength,
            estimator_type=estimator_type, device=device,
            base_model_type=base_model_type, seed=seed
        )
    
    def _generate_acs_data(self, pop_configs, m1, m, dataset_size, acs_data_fraction,
                      estimator_type, device, base_model_type, seed):
        """Generate data from ACS dataset split by states"""
        from data_acs import get_pop_data_acs
        
        # Extract states if specified in pop_configs
        states = None
        if pop_configs and isinstance(pop_configs, list) and len(pop_configs) > 0:
            if 'states' in pop_configs[0]:
                states = pop_configs[0]['states']
        
        # Use the get_pop_data_acs function which already exists in data_acs.py
        return get_pop_data_acs(
            states=states,  # Use specified states or default (CA, NY, FL)
            year=2018,
            target="PINCP",
            root_dir="/data/user_data/mswaroop/Subset-Selection-Code/folktables_data_storage",
            seed=seed,
            estimator_type=estimator_type,
            device=device,
            base_model_type=base_model_type,
            acs_data_fraction=acs_data_fraction,
        )
    
    def _generate_uci_data(self, pop_configs, m1, m, dataset_size, seed, 
                     estimator_type, device, base_model_type, uci_populations=None,
                        exclude_population_features=True,
                     force_regenerate=True):
        """Implement UCI data generation with appropriate population groups"""
        from data_uci import get_uci_pop_data
        
        # Use specified populations  
        pop_groups = uci_populations if uci_populations else ["Male", "Female"]
        
        print(f"Processing UCI data with population groups: {pop_groups}")
        
        
        return self._process_uci_data(
            pop_groups, seed, estimator_type, device, base_model_type, 
            is_classification=True,
            exclude_population_features=exclude_population_features
        )
    
    def _process_uci_data(self, pop_groups, seed, estimator_type, device, base_model_type, 
                         is_classification=True, exclude_population_features=True):
        """Process UCI data with the selected population groups"""
        from data_uci import get_uci_pop_data
        from estimators import plugin_estimator_conditional_mean, IF_estimator_conditional_mean
        from global_vars import N_FOLDS, EPS
        
        # Get population data
        pop_data_raw = get_uci_pop_data(
            populations=pop_groups,
            subsample=True,
            subsample_fraction=0.2,
            target="income_binary",
            categorical_encoding='onehot',
            seed=seed,
            force_regenerate=True,
            exclude_population_features=exclude_population_features  # Pass through
        )
        
        # Prepare train and test/val data
        pop_data = []
        pop_data_test_val = []
        
        for pop_idx, pop in enumerate(pop_data_raw):
            # Use 60% for training, 40% for test/val
            n_samples = pop['X_raw'].shape[0]
            n_train = int(n_samples * 0.6)
            
            # Create random split with seed
            rng = np.random.RandomState(seed + pop_idx if seed is not None else None)
            indices = rng.permutation(n_samples)
            train_indices = indices[:n_train]
            test_val_indices = indices[n_train:]
            
            # Split data
            X_train = pop['X_raw'][train_indices]
            Y_train = pop['Y_raw'][train_indices]
            X_test_val = pop['X_raw'][test_val_indices]
            Y_test_val = pop['Y_raw'][test_val_indices]
            
            # Standardize data
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0)
            X_std[X_std < EPS] = EPS  # Avoid division by zero
            
            Y_mean = np.mean(Y_train)
            Y_std_val = np.std(Y_train)
            if Y_std_val < EPS:
                Y_std_val = EPS
            
            X_std_train = (X_train - X_mean) / X_std
            Y_std_train = (Y_train - Y_mean) / Y_std_val
            X_std_test_val = (X_test_val - X_mean) / X_std
            Y_std_test_val = (Y_test_val - Y_mean) / Y_std_val

            if is_classification:
                # Keep original labels for classification
                Y_std_train = Y_train
                Y_std_test_val = Y_test_val
                Y_mean = 0
                Y_std_val = 1
            else:
                # Standardize Y for regression
                Y_mean = np.mean(Y_train)
                Y_std_val = np.std(Y_train)
                if Y_std_val < EPS:
                    Y_std_val = EPS
                Y_std_train = (Y_train - Y_mean) / Y_std_val
                Y_std_test_val = (Y_test_val - Y_mean) / Y_std_val
        
            
            # Estimate conditional expectation
            try:
                if estimator_type == "plugin":
                    E_Yx_orig_np = plugin_estimator_conditional_mean(
                        X_train, Y_train, base_model_type, n_folds=N_FOLDS, seed=seed
                    )
                elif estimator_type == "if":
                    E_Yx_orig_np = IF_estimator_conditional_mean(
                        X_train, Y_train, base_model_type, n_folds=N_FOLDS, seed=seed
                    )
                else:
                    raise ValueError(f"Unknown estimator_type: {estimator_type}")
                    
                E_Yx_std_np = (E_Yx_orig_np - Y_mean) / Y_std_val
                term1_std = np.mean(E_Yx_std_np ** 2)
            except Exception as e:
                print(f"Error estimating conditional expectation for population {pop['pop_id']}: {e}")
                continue
            
            # Build population data dict
            pop_data.append({
                'pop_id': pop['pop_id'],
                'X_std': torch.tensor(X_std_train, dtype=torch.float32).to(device),
                'Y_std': torch.tensor(Y_std_train, dtype=torch.float32).to(device),
                'E_Yx_std': torch.tensor(E_Yx_std_np, dtype=torch.float32).to(device),
                'term1_std': term1_std,
                'meaningful_indices': None,  # Unknown for real data
                'X_raw': X_train,
                'Y_raw': Y_train
            })
            
            # Build test/val data dict
            pop_data_test_val.append({
                'pop_id': pop['pop_id'],
                'X_std': torch.tensor(X_std_test_val, dtype=torch.float32).to(device),
                'Y_std': torch.tensor(Y_std_test_val, dtype=torch.float32).to(device),
                'X_raw': X_test_val,
                'Y_raw': Y_test_val
            })
        
        return pop_data, pop_data_test_val

    def _generate_synthetic_data(self, pop_configs, m1, m, dataset_size, noise_scale, corr_strength,
                            estimator_type, device, base_model_type, seed):
        """Generate synthetic data using the existing data generation functions"""
        from data import generate_data_continuous_with_corr
        from estimators import plugin_estimator_conditional_mean, IF_estimator_conditional_mean
        from global_vars import N_FOLDS, EPS
        
        # Common meaningful indices for populations to share
        common_meaningful_indices = np.arange(max(1, m1 // 2))
        
        pop_data_train = []
        pop_data_test_val = []
        
        for i, pop_config in enumerate(pop_configs):
            # Use different seeds for each population
            pop_seed = seed + i if seed is not None else None
            rng = np.random.RandomState(pop_seed)
            
            pop_id = pop_config.get('pop_id', i)
            dataset_type = pop_config['dataset_type']
            
            # Generate data with correlation structure
            X_np, Y_np, _, meaningful_idx = generate_data_continuous_with_corr(
                pop_id=pop_id,
                m1=m1,
                m=m,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                noise_scale=noise_scale,
                corr_strength=corr_strength,
                seed=pop_seed,
                common_meaningful_indices=common_meaningful_indices
            )
            
            # Split into train and test/val
            test_val_fraction = 0.4
            n_total = X_np.shape[0]
            indices = rng.permutation(n_total)
            n_train = int(n_total * (1 - test_val_fraction))
            
            train_indices = indices[:n_train]
            test_val_indices = indices[n_train:]
            
            X_train = X_np[train_indices]
            Y_train = Y_np[train_indices]
            X_test_val = X_np[test_val_indices]
            Y_test_val = Y_np[test_val_indices]
            
            # Standardize data
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0)
            X_std[X_std < EPS] = EPS  # Avoid division by zero
            
            Y_mean = np.mean(Y_train)
            Y_std_val = np.std(Y_train)
            if Y_std_val < EPS:
                Y_std_val = EPS
            
            X_std_train = (X_train - X_mean) / X_std
            Y_std_train = (Y_train - Y_mean) / Y_std_val
            X_std_test_val = (X_test_val - X_mean) / X_std
            Y_std_test_val = (Y_test_val - Y_mean) / Y_std_val
            
            # Estimate conditional expectation
            try:
                if estimator_type == "plugin":
                    E_Yx_orig_np = plugin_estimator_conditional_mean(
                        X_train, Y_train, base_model_type, n_folds=N_FOLDS, seed=pop_seed
                    )
                elif estimator_type == "if":
                    E_Yx_orig_np = IF_estimator_conditional_mean(
                        X_train, Y_train, base_model_type, n_folds=N_FOLDS, seed=pop_seed
                    )
                else:
                    raise ValueError(f"Unknown estimator_type: {estimator_type}")
                    
                E_Yx_std_np = (E_Yx_orig_np - Y_mean) / Y_std_val
                term1_std = np.mean(E_Yx_std_np ** 2)
            except Exception as e:
                print(f"Error estimating conditional expectation for population {pop_id}: {e}")
                continue
            
            # Build population data dict
            pop_data_train.append({
                'pop_id': pop_id,
                'X_std': torch.tensor(X_std_train, dtype=torch.float32).to(device),
                'Y_std': torch.tensor(Y_std_train, dtype=torch.float32).to(device),
                'E_Yx_std': torch.tensor(E_Yx_std_np, dtype=torch.float32).to(device),
                'term1_std': term1_std,
                'meaningful_indices': meaningful_idx.tolist(),
                'X_raw': X_train,
                'Y_raw': Y_train
            })
            
            # Build test/val data dict
            pop_data_test_val.append({
                'pop_id': pop_id,
                'X_std': torch.tensor(X_std_test_val, dtype=torch.float32).to(device),
                'Y_std': torch.tensor(Y_std_test_val, dtype=torch.float32).to(device),
                'X_raw': X_test_val,
                'Y_raw': Y_test_val,
                'meaningful_indices': meaningful_idx.tolist()
            })
        
        return pop_data_train, pop_data_test_val