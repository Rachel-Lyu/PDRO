#!/usr/bin/env python3
"""
data_acs.py

Loads ACS person-level data via Folktables, splits into state-based populations,
provides preprocessing (imputation & scaling), plotting utilities for
covariate distributions and covariate-target relationships, formatted
population dicts, and supports specifying ACS survey year.

Usage:
  # As a module:
  from data_acs import (
      generate_data_acs,
      preprocess_data,
      plot_feature_histograms,
      plot_feature_target_scatter,
      get_acs_pop_data
  )

  # As a script (with plotting):
  python data_acs.py --year 2018 --plot --save_dir output_data --plot_dir output_plots

"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from folktables import ACSDataSource
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
import torch

# FIPS codes for the states of interest
STATE_FIPS = {"CA": 6, "NY": 36, "FL": 12, "TX": 48, "IL": 17}


def generate_data_acs(
    states=["CA", "NY", "FL"],
    year=2018,
    survey="person",
    horizon="1-Year",
    root_dir="data",
    feature_cols=None,
    target="PINCP",    # default to the raw personal-income column
    save_dir=None,
):
    """
    Load ACS person data for a given year, filter to specified states,
    and split into X/Y arrays.

    Returns:
      X_all: np.ndarray, pooled features
      Y_all: np.ndarray, pooled labels
      Xs: list of np.ndarray, per-state features
      Ys: list of np.ndarray, per-state labels
      feature_cols: list of str, names of features
      states: list of str, state abbreviations
    """
    # ----------------------------------------------------------------
    # If save_dir is set and contains all pickles, just load & return them
    if save_dir:
        expected = [
            "X_all.pkl", "Y_all.pkl",
            "Xs.pkl",    "Ys.pkl",
            "feature_cols.pkl", "states.pkl"
        ]
        if all(os.path.isfile(os.path.join(save_dir, fn)) for fn in expected):
            X_all_path = os.path.join(save_dir, "X_all.pkl")
            Y_all_path = os.path.join(save_dir, "Y_all.pkl")
            Xs_path    = os.path.join(save_dir, "Xs.pkl")
            Ys_path    = os.path.join(save_dir, "Ys.pkl")
            feats_path = os.path.join(save_dir, "feature_cols.pkl")
            sts_path   = os.path.join(save_dir, "states.pkl")

            with open(X_all_path,    "rb") as f: X_all      = pickle.load(f)
            with open(Y_all_path,    "rb") as f: Y_all      = pickle.load(f)
            with open(Xs_path,       "rb") as f: Xs         = pickle.load(f)
            with open(Ys_path,       "rb") as f: Ys         = pickle.load(f)
            with open(feats_path,    "rb") as f: feature_cols = pickle.load(f)
            with open(sts_path,      "rb") as f: states      = pickle.load(f)

            print(f"Loaded cached ACS data from {save_dir}")
            return X_all, Y_all, Xs, Ys, feature_cols, states
    # ----------------------------------------------------------------

    # folktables ACSDataSource now requires survey_year, horizon, survey in ctor
    ds = ACSDataSource(survey_year=year, horizon=horizon, survey=survey, root_dir=root_dir)
    # get_data now only takes states parameter
    df = ds.get_data(states=None, download=True)  # Get all states, then filter

    # figure out which column holds the state FIPS code
    # it might be named "state" (new folktables), "ST", "STFIP", or already "STATEFIP"
    fips_candidates = ["STATEFIP", "state", "STFIP", "ST", "STATE"]
    found = [c for c in fips_candidates if c in df.columns]
    if not found:
        print("ERROR: no FIPS column found – df.columns =", list(df.columns))
        print("All columns in dataset:", list(df.columns))
        raise KeyError("Cannot locate any of " + str(fips_candidates))
    
    # rename the first matched one to our canonical name
    fips_col = found[0]
    if (fips_col != "STATEFIP"):
        df.rename(columns={fips_col: "STATEFIP"}, inplace=True)
        print(f"Found state FIPS code in column '{fips_col}', renamed to 'STATEFIP'")
    
    # Check if STATEFIP is numeric or string and convert if needed
    if df["STATEFIP"].dtype != 'int64':
        try:
            df["STATEFIP"] = df["STATEFIP"].astype(int)
            print("Converted STATEFIP to integer type")
        except ValueError:
            print("WARNING: Could not convert STATEFIP to integer. Current values:", df["STATEFIP"].unique())

    # Filter to chosen states
    keep = [STATE_FIPS[s] for s in states]
    df = df[df["STATEFIP"].isin(keep)].copy()
    
    if df.empty:
        print(f"WARNING: No data found for states {states}. Available STATEFIP values: {df['STATEFIP'].unique()}")
        raise ValueError(f"No data found for states {states}")

    # Determine feature columns if not provided
    if feature_cols is None:
        drop = {target, "person_weight", "STATEFIP", "STATE", "domain", "geoid", "PUMA"}
        feature_cols = [c for c in df.columns if c not in drop]

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    if len(non_numeric_cols) > 0:
        print("Warning: non-numeric columns in feature_cols:", non_numeric_cols)
        # Optionally drop non-numeric columns
        # df.drop(columns=non_numeric_cols, inplace=True)
    # print number of features; numeric vs non-numeric
    print(f"Number of features: {len(feature_cols)} (numeric: {len(numeric_cols)}, non-numeric: {len(non_numeric_cols)})")
    feature_cols = [c for c in feature_cols if c in numeric_cols]

    # Check if target column exists
    if target not in df.columns:
        print(f"ERROR: Target column '{target}' not found in data. Available columns:", list(df.columns))
        raise KeyError(f"Target column '{target}' not found")

    # Build pooled arrays
    X_all = df[feature_cols].to_numpy(dtype=float)
    Y_all = df[target].to_numpy()

    # Drop any rows where the target is NaN
    non_nan_mask = ~np.isnan(Y_all)
    if not non_nan_mask.all():
        print(f"Dropping {np.count_nonzero(~non_nan_mask)} rows with NaN target")
    X_all = X_all[non_nan_mask]
    Y_all = Y_all[non_nan_mask]

    # Split by state
    Xs, Ys = [], []
    for s in states:
        # look up the numeric FIPS code for this abbreviation
        fcode = STATE_FIPS[s]
        sub = df[df["STATEFIP"] == fcode]
        if sub.empty:
            print(f"WARNING: No data found for state {s} (FIPS {fcode})")
            Xs.append(np.array([]))
            Ys.append(np.array([]))
        else:
            Xs.append(sub[feature_cols].to_numpy(dtype=float))
            Ys.append(sub[target].to_numpy())

    # Optionally save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "X_all.pkl"), "wb") as f:
            pickle.dump(X_all, f)
        with open(os.path.join(save_dir, "Y_all.pkl"), "wb") as f:
            pickle.dump(Y_all, f)
        with open(os.path.join(save_dir, "Xs.pkl"), "wb") as f:
            pickle.dump(Xs, f)
        with open(os.path.join(save_dir, "Ys.pkl"), "wb") as f:
            pickle.dump(Ys, f)
        with open(os.path.join(save_dir, "feature_cols.pkl"), "wb") as f:
            pickle.dump(feature_cols, f)
        with open(os.path.join(save_dir, "states.pkl"), "wb") as f:
            pickle.dump(states, f)

    return X_all, Y_all, Xs, Ys, feature_cols, states

def preprocess_acs_features(X_raw, categorical_cols=None):
    """Apply ACS-specific feature preprocessing"""
    # Identify categorical columns if not specified
    if categorical_cols is None:
        categorical_cols = []
        for i in range(X_raw.shape[1]):
            if len(np.unique(X_raw[:, i])) < 10:  # Threshold for categorical
                categorical_cols.append(i)
    
    # Create preprocessing pipeline
    numeric_cols = [i for i in range(X_raw.shape[1]) if i not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X_raw)
    
    return X_processed, preprocessor

def preprocess_data(X_all, Xs, impute_strategy="mean", scale=True):
    """
    Impute missing values and optionally standard-scale features.

    Returns:
      X_all_proc: np.ndarray
      Xs_proc: list of np.ndarray
      imputer: fitted SimpleImputer
      scaler: fitted StandardScaler or None
    """
    imputer = SimpleImputer(strategy=impute_strategy)
    X_all_imp = imputer.fit_transform(X_all)

    if scale:
        scaler = StandardScaler()
        X_all_proc = scaler.fit_transform(X_all_imp)
        Xs_proc = [scaler.transform(imputer.transform(X)) for X in Xs if X.size > 0]
    else:
        scaler = None
        X_all_proc = X_all_imp
        Xs_proc = [imputer.transform(X) for X in Xs if X.size > 0]

    return X_all_proc, Xs_proc, imputer, scaler


def plot_feature_histograms(
    Xs,
    feature_cols,
    states,
    features_to_plot=None,
    bins=30,
    figsize=(12, 4),
    save_path=None,
):
    """
    Plot side-by-side histograms of specified covariates by state.
    """
    if features_to_plot is None:
        features_to_plot = feature_cols

    for feat in features_to_plot:
        idx = feature_cols.index(feat)
        fig, axes = plt.subplots(1, len(states), figsize=figsize, sharey=True)
        
        # Handle case when there's only one state
        if len(states) == 1:
            axes = [axes]
            
        for ax, X, st in zip(axes, Xs, states):
            if X.size > 0:
                ax.hist(X[:, idx], bins=bins, density=True, alpha=0.7)
                ax.set_title(f"{feat} in {st}")
                ax.set_xlabel(feat)
                ax.set_ylabel("Density")
            else:
                ax.text(0.5, 0.5, f"No data for {st}", 
                        horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"hist_{feat}.png"), dpi=300)
        # plt.show()


def plot_feature_target_scatter(
    Xs,
    Ys,
    feature_cols,
    states,
    features_to_plot=None,
    figsize=(12, 4),
    save_path=None,
):
    """
    Plot side-by-side scatter plots of covariate vs. target by state.
    """
    if features_to_plot is None:
        features_to_plot = feature_cols

    for feat in features_to_plot:
        idx = feature_cols.index(feat)
        fig, axes = plt.subplots(1, len(states), figsize=figsize, sharey=True)
        
        # Handle case when there's only one state
        if len(states) == 1:
            axes = [axes]
            
        for ax, X, y, st in zip(axes, Xs, Ys, states):
            if X.size > 0 and y.size > 0:
                ax.scatter(X[:, idx], y, s=1, alpha=0.3)
                ax.set_title(f"{feat} vs target in {st}")
                ax.set_xlabel(feat)
                ax.set_ylabel("Target")
            else:
                ax.text(0.5, 0.5, f"No data for {st}", 
                        horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"scatter_{feat}.png"), dpi=300)
        # plt.show() 


def get_acs_pop_data(
    states=None,
    year=2018,
    survey="person",
    horizon="1-Year",
    root_dir="data",
    feature_cols=None,
    subsample=True,
    target="PINCP",  # default to the raw personal-income column
    acs_source='./data_acs',
):
    """
    Return population data dicts in the same format as data_baseline_failures:
      - 'pop_id': state abbreviation
      - 'X_raw': raw feature array
      - 'Y_raw': raw label array
      - 'meaningful_indices': None (unknown for real data)
    """
    st_list = states or list(STATE_FIPS.keys())
    try:
        X_all, Y_all, Xs, Ys, feats, sts = generate_data_acs(
            states=st_list,
            year=year,
            survey=survey,
            horizon=horizon,
            root_dir=root_dir,
            feature_cols=feature_cols,
            target=target,
            save_dir='./data_acs'
        )
    except Exception as e:
        print(f"Error generating data: {e}")
        raise
        
    # For any feature, if percentage of missing values is > 0.5, drop it
    missing_threshold = 0.5
    missing_mask = np.mean(np.isnan(X_all), axis=0) > missing_threshold
    
    if np.any(missing_mask):
        print(f"Dropping features with > {missing_threshold * 100}% missing values")
        X_all = X_all[:, ~missing_mask]
        
        # Fix: handle the case where feature_cols is None
        if feature_cols is not None:
            feature_cols = [f for f, m in zip(feature_cols, missing_mask) if not m]
        
        for i, st in enumerate(sts):
            if Xs[i].size > 0:  # Only process non-empty arrays
                Xs[i] = Xs[i][:, ~missing_mask]
    
    # Check if any state has empty data
    empty_states = [st for st, X, y in zip(sts, Xs, Ys) if X.size == 0 or y.size == 0]
    if empty_states:
        print(f"WARNING: The following states have empty data: {empty_states}")
        # Optionally raise an error or handle it as needed
        # raise ValueError(f"Empty data for states: {empty_states}")

    # Create non-empty data indices for further processing
    valid_indices = [i for i, (X, y) in enumerate(zip(Xs, Ys)) if X.size > 0 and y.size > 0]
    
    # remaining nans are median imputed
    imputer = SimpleImputer(strategy="median")
    if X_all.size > 0:
        X_all = imputer.fit_transform(X_all)
    
        # Only process non-empty datasets
        for i in valid_indices:
            Xs[i] = imputer.transform(Xs[i])
    
    # Handle target missing values separately
    for i in valid_indices:
        # Drop samples where target is NaN
        mask = ~np.isnan(Ys[i])
        Xs[i] = Xs[i][mask]
        Ys[i] = Ys[i][mask]
        if Xs[i].size == 0 or Ys[i].size == 0:
            print(f"WARNING: No data left for state {sts[i]} after filtering")
            Xs[i] = np.array([])
            Ys[i] = np.array([])
            valid_indices.remove(i)  # Remove from valid indices
    
    pop_data = []
    for i, s in enumerate(sts):
        if i in valid_indices and Xs[i].size > 0 and Ys[i].size > 0:
            if subsample:
                # Randomly sample 10% of the data for each state
                n_samples = Xs[i].shape[0]
                sample_size = max(1, int(n_samples * 0.1))
                indices = np.random.choice(n_samples, sample_size, replace=False)
                Xs[i] = Xs[i][indices]
                Ys[i] = Ys[i][indices]
            pop_data.append({
                'pop_id': s,
                'X_raw': Xs[i],
                'Y_raw': Ys[i],
                'meaningful_indices': None
            })
        else:
            print(f"Skipping state {s} due to empty data")

    # print the final dataset shapes and sizes
    for i, s in enumerate(sts):
        if i in valid_indices and Xs[i].size > 0 and Ys[i].size > 0:
            print(f"State {s}: X shape {Xs[i].shape}, Y shape {Ys[i].shape}")
        else:
            print(f"State {s}: No data available")
    
    return pop_data

def get_pop_data_acs(states=None, 
                     year=2018, 
                     target="PINCP",
                     root_dir="/data/user_data/mswaroop/Subset-Selection-Code/folktables_data_storage",
                     seed=None,
                     estimator_type="if",
                     device="cpu",
                     base_model_type="rf",
                     acs_data_fraction=0.1,
                     importance_file="/data/user_data/mswaroop/Subset-Selection-Code/acs_analysis_output/acs_feature_importances.csv"):
    """
    Process ACS data for population-based subset selection with specific data cleaning steps.
    
    Returns:
        tuple: (pop_data, pop_data_test_val) formatted for the DataLoader
    """
    # Use default states if not provided
    if states is None:
        states = ["CA", "NY", "FL"]  # Use first 3 states
    
    # Load feature importances
    try:
        importance_df = pd.read_csv(importance_file)
        # Remove PWGTP variables
        filtered_importance = importance_df[~importance_df['feature'].str.startswith('PWGTP')]
        # Get top 20 remaining features
        top_features = filtered_importance['feature'].tolist()[:20]
        print(f"Selected top 20 features: {top_features}")
    except Exception as e:
        print(f"Error loading feature importances: {e}")
        top_features = None
    
    # Load the ACS data
    try:
        X_all, Y_all, Xs, Ys, feature_cols, loaded_states = generate_data_acs(
            states=states,
            year=year,
            target=target,
            root_dir=root_dir
        )
    except Exception as e:
        print(f"Error loading ACS data: {e}")
        raise
    
    print(f"Loaded ACS data for states: {loaded_states}")
    print(f"Initial data shape: X={X_all.shape}, Y={Y_all.shape}")
    
    # Create a DataFrame for easier processing
    df = pd.DataFrame(X_all, columns=feature_cols)
    
    # 1. Remove columns with > 10% NaN values
    nan_percentage = df.isna().mean()
    cols_to_drop = nan_percentage[nan_percentage > 0.1].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >10% NaN values")
        df = df.drop(columns=cols_to_drop)
    
    # 2. Remove categorical variables with >3 categories
    cat_cols = []
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        if len(unique_values) > 3 and len(unique_values) < 10:  # Categorical with >3 categories
            cat_cols.append(col)
    
    if cat_cols:
        print(f"Dropping {len(cat_cols)} categorical columns with >3 categories")
        df = df.drop(columns=cat_cols)
    
    # 3. Remove PWGTP variables
    pwgtp_cols = [col for col in df.columns if col.startswith('PWGTP')]
    if pwgtp_cols:
        print(f"Dropping {len(pwgtp_cols)} PWGTP columns")
        df = df.drop(columns=pwgtp_cols)
    
    # 4. Filter to top features if available
    if top_features:
        # Only keep columns that exist in our filtered dataframe
        valid_top_features = [f for f in top_features if f in df.columns]
        if valid_top_features:
            print(f"Keeping top {len(valid_top_features)} important features")
            df = df[valid_top_features]
        else:
            print("Warning: None of the top features remain after filtering")
    
    # Get the final feature list
    final_features = df.columns.tolist()
    print(f"Final feature set ({len(final_features)} columns): {final_features}")
    
    # 5. Median impute remaining NaN values
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns
    )
    
    # Process each state's data
    state_dfs = []
    for i, state in enumerate(loaded_states):
        if len(Xs[i]) > 0:  # Skip empty states
            # Extract features for this state
            state_df = pd.DataFrame(Xs[i], columns=feature_cols)
            
            # Apply the same filtering
            state_df = state_df[final_features]
            
            # Apply imputation
            state_df_imputed = pd.DataFrame(
                imputer.transform(state_df),
                columns=final_features
            )
            # ACS data fraction
            if acs_data_fraction < 1.0:
                n_samples = state_df_imputed.shape[0]
                sample_size = max(1, int(n_samples * acs_data_fraction))
                indices = np.random.choice(n_samples, sample_size, replace=False)
                state_df_imputed = state_df_imputed.iloc[indices]
                Ys[i] = Ys[i][indices]
                print(f"Subsampling {sample_size} samples for state {state}")
            else:
                print(f"Using all {state_df_imputed.shape[0]} samples for state {state}")
            
            state_dfs.append({
                'state': state,
                'X': state_df_imputed.to_numpy(),
                'Y': Ys[i]
            })
    
    # --- NEW: drop any samples with NaN target here ---
    for sd in state_dfs:
        mask = ~np.isnan(sd['Y'])
        if not mask.all():
            n_drop = np.count_nonzero(~mask)
            print(f"Dropping {n_drop} samples with NaN target in state {sd['state']}")
        sd['X'] = sd['X'][mask]
        sd['Y'] = sd['Y'][mask]
    
    # Process into training and test data
    from estimators import plugin_estimator_conditional_mean, IF_estimator_conditional_mean
    from global_vars import N_FOLDS, EPS
    
    pop_data = []
    pop_data_test_val = []
    
    for state_data in state_dfs:
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Split into train (60%) and test/val (40%)
        n_samples = len(state_data['Y'])
        indices = np.random.permutation(n_samples)
        split_idx = int(n_samples * 0.6)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train = state_data['X'][train_indices]
        Y_train = state_data['Y'][train_indices]
        X_test = state_data['X'][test_indices]
        Y_test = state_data['Y'][test_indices]
        
        # Standardize
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_std[X_std < EPS] = EPS  # Avoid division by zero
        
        Y_mean = np.mean(Y_train)
        Y_std_val = np.std(Y_train)
        if Y_std_val < EPS:
            Y_std_val = EPS
        
        X_std_train = (X_train - X_mean) / X_std
        Y_std_train = (Y_train - Y_mean) / Y_std_val
        X_std_test = (X_test - X_mean) / X_std
        Y_std_test = (Y_test - Y_mean) / Y_std_val
        
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
            print(f"Error estimating conditional expectation for state {state_data['state']}: {e}")
            continue
        
        # Build population data dicts
        pop_data.append({
            'pop_id': state_data['state'],
            'X_std': torch.tensor(X_std_train, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_std_train, dtype=torch.float32).to(device),
            'E_Yx_std': torch.tensor(E_Yx_std_np, dtype=torch.float32).to(device),
            'term1_std': term1_std,
            'meaningful_indices': None,  # Unknown for real data
            'X_raw': X_train,
            'Y_raw': Y_train,
            'feature_names': final_features  # Include feature names
        })
        
        # Build test/val data dict
        pop_data_test_val.append({
            'pop_id': state_data['state'],
            'X_std': torch.tensor(X_std_test, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_std_test, dtype=torch.float32).to(device),
            'X_raw': X_test,
            'Y_raw': Y_test
        })
    
    # number of instances in test/val data vs training data
    for i, state_data in enumerate(pop_data_test_val):
        print("Test/Val data:")
        print(f"State {state_data['pop_id']}: X_std shape {state_data['X_std'].shape}, Y_std shape {state_data['Y_std'].shape}")
        print("Training data:")
        print(f"State {pop_data[i]['pop_id']}: X_std shape {pop_data[i]['X_std'].shape}, Y_std shape {pop_data[i]['Y_std'].shape}")
    return pop_data, pop_data_test_val

def main():
    p = argparse.ArgumentParser(
        description="Generate, preprocess, and plot ACS data by state"
    )
    p.add_argument("--states",
        nargs="+", default=["CA", "NY", "FL"],
        help="State abbreviations to include",
    )
    p.add_argument("--year",
        type=int, default=2018,
        help="ACS survey year (e.g. 2018)",
    )
    p.add_argument("--survey",
        choices=["person", "household"], default="person",
        help="Folktables survey",
    )
    p.add_argument("--horizon",
        choices=["1-Year", "5-Year"], default="1-Year",
        help="Folktables horizon",
    )
    p.add_argument("--root_dir", default="data",
        help="Folder for Folktables cache/download",
    )
    p.add_argument("--target", default="PINCP",
        help="Name of the label column",
    )
    p.add_argument("--save_dir", default="./data_acs",
        help="Directory to pickle out X/Y arrays",
    )
    p.add_argument("--impute_strategy",
        choices=["mean", "median", "most_frequent"], default="mean",
        help="Imputation strategy",
    )
    p.add_argument("--no_scale",
        action="store_true",
        help="Disable standard scaling",
    )
    p.add_argument("--plot",
        action="store_true",
        help="Generate histogram and scatter plots",
    )
    p.add_argument("--features",
        nargs="+",
        help="Subset of covariate names to plot",
    )
    p.add_argument("--plot_dir", default='./data_acs_plots',
        help="Directory to save plot images",
    )
    p.add_argument("--debug",
        action="store_true",
        help="Show debug information",
    )
    args = p.parse_args()

    try:
        X_all, Y_all, Xs, Ys, feature_cols, states = generate_data_acs(
            states=args.states,
            year=args.year,
            survey=args.survey,
            horizon=args.horizon,
            root_dir=args.root_dir,
            feature_cols=None,
            target=args.target,
            save_dir=args.save_dir,
        )
        print(f"Loaded pooled X_all {X_all.shape}, Y_all {Y_all.shape}")
        for st, X, y in zip(states, Xs, Ys):
            print(f"  {st}: X {X.shape}, Y {y.shape}")

        X_all_proc, Xs_proc, imputer, scaler = preprocess_data(
            X_all,
            Xs,
            impute_strategy=args.impute_strategy,
            scale=not args.no_scale,
        )
        print("Preprocessing complete.")

        if args.plot:
            # randomly select 20 features to plot
            if args.features:
                feature_cols = args.features
            else:
                feature_cols = feature_cols[:20]
            print(f"Plotting features: {feature_cols}")
            feats = feature_cols[:20]
            plot_feature_histograms(
                Xs_proc, feature_cols, states, features_to_plot=feats, save_path=args.plot_dir
            )
            plot_feature_target_scatter(
                Xs_proc, Ys, feature_cols, states, features_to_plot=feats, save_path=args.plot_dir
            )
    except Exception as e:
        print(f"ERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()