#!/usr/bin/env python3
"""
data_uci_adult.py

Loads UCI Adult Income dataset, splits into demographic-based populations,
provides preprocessing (imputation & scaling), plotting utilities for
covariate distributions and covariate-target relationships, and formatted
population dicts.

Usage:
  # As a module:
  from data_uci_adult import (
      generate_data_uci,
      preprocess_data,
      plot_feature_histograms,
      plot_feature_target_scatter,
      get_uci_pop_data
  )

  # As a script (with plotting):
  python data_uci_adult.py --plot --save_dir output_data --plot_dir output_plots
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import argparse
from urllib.request import urlretrieve

# Population groups (demographic categories)
POPULATION_GROUPS = {
    "Female": {"sex": "Female"},
    "Male": {"sex": "Male"},
    "WhiteMale": {"race": "White", "sex": "Male"},
    "NonWhiteMale": {"race": "Other", "sex": "Male"},
    "WhiteFemale": {"race": "White", "sex": "Female"},
    "NonWhiteFemale": {"race": "Other", "sex": "Female"},
    "Young": {"age_group": "Young"},
    "Middle": {"age_group": "Middle"},
    "Senior": {"age_group": "Senior"},
}


def download_adult_dataset(cache_dir="./data_uci"):
    """
    Download UCI Adult dataset if it doesn't exist locally.
    
    Returns:
        Tuple of (train_path, test_path)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    train_path = os.path.join(cache_dir, "adult.data")
    test_path = os.path.join(cache_dir, "adult.test")
    
    if not os.path.exists(train_path):
        print(f"Downloading training data to {train_path}")
        urlretrieve(train_url, train_path)
    
    if not os.path.exists(test_path):
        print(f"Downloading test data to {test_path}")
        urlretrieve(test_url, test_path)
    
    return train_path, test_path


def load_adult_dataset(train_path, test_path):
    """
    Load the UCI Adult dataset from local files.
    
    Returns:
        DataFrame containing the full dataset
    """
    # Column names for the dataset
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race',
        'sex', 'capital_gain', 'capital_loss', 'hours_per_week',
        'native_country', 'income'
    ]
    
    # Load training data
    df_train = pd.read_csv(train_path, header=None, names=columns, 
                          sep=r',\s*', engine='python', na_values='?')
    
    # Load test data (test file has a header line to skip)
    df_test = pd.read_csv(test_path, header=None, names=columns, 
                         sep=r',\s*', engine='python', skiprows=1, na_values='?')
    
    # Remove the dot from income labels in test data
    df_test['income'] = df_test['income'].str.rstrip('.')
    
    # Combine the datasets
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # Clean up whitespace in string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df


def prepare_adult_dataset(df):
    """
    Prepare the Adult dataset for modeling by creating age groups,
    simplifying race categories, and transforming the income target.
    
    Returns:
        Processed DataFrame
    """
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 30, 50, 100], 
                            labels=['Young', 'Middle', 'Senior'])
    
    # Simplify race (White vs Other)
    df['race'] = df['race'].apply(lambda x: 'White' if x == 'White' else 'Other')
    
    # Convert income to binary numeric target
    # df['income_binary'] = (df['income'] == '>50K').astype(float)
    df['income_binary'] = df['income']

    return df


def generate_data_uci(
    populations=list(POPULATION_GROUPS.keys()),
    feature_cols=None,
    target="income_binary",
    save_dir=None,
    categorical_encoding=None,  # None, 'onehot', or 'label'
    force_regenerate=False,
):
    """
    Load UCI Adult data, filter to specified populations,
    and split into X/Y arrays.

    Returns:
      X_all: np.ndarray, pooled features
      Y_all: np.ndarray, pooled labels
      Xs: list of np.ndarray, per-population features
      Ys: list of np.ndarray, per-population labels
      feature_cols: list of str, names of features
      populations: list of str, population names
    """
    # ----------------------------------------------------------------
    # If save_dir is set and contains all pickles, just load & return them
    if save_dir and not force_regenerate:
        # Use population names in cache key
        pop_key = "_".join(sorted(populations))
        pop_cache_dir = os.path.join(save_dir, f"pop_{pop_key}")
        os.makedirs(pop_cache_dir, exist_ok=True)
        
        expected = [
            "X_all.pkl", "Y_all.pkl",
            "Xs.pkl",    "Ys.pkl",
            "feature_cols.pkl", "populations.pkl"
        ]
        
        if all(os.path.isfile(os.path.join(pop_cache_dir, fn)) for fn in expected):
            X_all_path = os.path.join(pop_cache_dir, "X_all.pkl")
            Y_all_path = os.path.join(pop_cache_dir, "Y_all.pkl")
            Xs_path    = os.path.join(pop_cache_dir, "Xs.pkl")
            Ys_path    = os.path.join(pop_cache_dir, "Ys.pkl")
            feats_path = os.path.join(pop_cache_dir, "feature_cols.pkl")
            pops_path  = os.path.join(pop_cache_dir, "populations.pkl")

            with open(X_all_path,    "rb") as f: X_all         = pickle.load(f)
            with open(Y_all_path,    "rb") as f: Y_all         = pickle.load(f)
            with open(Xs_path,       "rb") as f: Xs            = pickle.load(f)
            with open(Ys_path,       "rb") as f: Ys            = pickle.load(f)
            with open(feats_path,    "rb") as f: feature_cols  = pickle.load(f)
            with open(pops_path,     "rb") as f: populations   = pickle.load(f)

            print(f"Loaded cached UCI Adult data from {pop_cache_dir}")
            return X_all, Y_all, Xs, Ys, feature_cols, populations
    # ----------------------------------------------------------------

    # Download and load the data
    train_path, test_path = download_adult_dataset()
    df = load_adult_dataset(train_path, test_path)
    df = prepare_adult_dataset(df)
    
    # Fix: Convert income_binary to numeric type
    if target == "income_binary":
        df['income_binary'] = (df['income'] == '>50K').astype(float)
    
    # Check if target column exists
    if target not in df.columns:
        print(f"ERROR: Target column '{target}' not found in data. Available columns:", list(df.columns))
        raise KeyError(f"Target column '{target}' not found")
    
    # Determine feature columns if not provided
    if feature_cols is None:
        # Default features that work well with the Adult dataset
        feature_cols = [
            'age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week',
            'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex'
        ]
    
    # Separate numerical and categorical features
    num_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"Number of features: {len(feature_cols)} (numeric: {len(num_features)}, categorical: {len(cat_features)})")
    
    # Handle categorical features based on encoding choice
    if categorical_encoding == 'onehot':
        # One-hot encode categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(df[cat_features])
        
        # Create dataframe with encoded features
        encoded_df = pd.DataFrame(
            encoded_cats, 
            columns=encoder.get_feature_names_out(cat_features)
        )
        
        # Combine with numerical features
        processed_df = pd.concat([df[num_features], encoded_df], axis=1)
        X_all = processed_df.to_numpy(dtype=float)
        
        # Update feature columns to include encoded names
        feature_cols = num_features + list(encoder.get_feature_names_out(cat_features))
    elif categorical_encoding == 'label':
        # Label encode categorical features
        for col in cat_features:
            df[col + '_encoded'] = df[col].astype('category').cat.codes
        
        # Use encoded features instead of original categorical ones
        encoded_cat_features = [col + '_encoded' for col in cat_features]
        X_all = df[num_features + encoded_cat_features].to_numpy(dtype=float)
        
        # Update feature columns to include encoded names
        feature_cols = num_features + encoded_cat_features
    else:
        # For simplicity in this example, we'll use only numerical features if no encoding is specified
        print("No categorical encoding specified, using only numerical features.")
        feature_cols = num_features
        X_all = df[feature_cols].to_numpy(dtype=float)
    
    # Get target values
    Y_all = df[target].to_numpy()
    
    # Drop any rows where the target is NaN
    non_nan_mask = ~pd.isnull(Y_all)
    if not non_nan_mask.all():
        print(f"Dropping {np.count_nonzero(~non_nan_mask)} rows with NaN target")
    X_all = X_all[non_nan_mask]
    Y_all = Y_all[non_nan_mask]
    
    # Split by population
    Xs, Ys = [], []
    for pop_name in populations:
        # Skip if population definition not found
        if pop_name not in POPULATION_GROUPS:
            print(f"WARNING: Population '{pop_name}' not defined, skipping")
            Xs.append(np.array([]))
            Ys.append(np.array([]))
            continue
        
        # Create mask for this population based on criteria
        criteria = POPULATION_GROUPS[pop_name]
        mask = pd.Series(True, index=df.index)
        for column, value in criteria.items():
            mask = mask & (df[column] == value)
        
        pop_df = df[mask]
        
        if pop_df.empty:
            print(f"WARNING: No data found for population {pop_name}")
            Xs.append(np.array([]))
            Ys.append(np.array([]))
        else:
            # Extract features based on encoding choice
            if categorical_encoding == 'onehot':
                # Get encoded features for this population
                pop_cats = encoder.transform(pop_df[cat_features])
                pop_cats_df = pd.DataFrame(
                    pop_cats, 
                    columns=encoder.get_feature_names_out(cat_features),
                    index=pop_df.index
                )
                pop_features = pd.concat([pop_df[num_features], pop_cats_df], axis=1)
                Xs.append(pop_features.to_numpy(dtype=float))
            elif categorical_encoding == 'label':
                # Use encoded categorical features
                Xs.append(pop_df[num_features + encoded_cat_features].to_numpy(dtype=float))
            else:
                # Only numerical features
                Xs.append(pop_df[feature_cols].to_numpy(dtype=float))
            
            # Extract target
            Ys.append(pop_df[target].to_numpy())

    # Optionally save
    if save_dir:
        # Create population-specific cache directory
        pop_key = "_".join(sorted(populations))
        pop_cache_dir = os.path.join(save_dir, f"pop_{pop_key}")
        os.makedirs(pop_cache_dir, exist_ok=True)
        
        with open(os.path.join(pop_cache_dir, "X_all.pkl"), "wb") as f:
            pickle.dump(X_all, f)
        with open(os.path.join(pop_cache_dir, "Y_all.pkl"), "wb") as f:
            pickle.dump(Y_all, f)
        with open(os.path.join(pop_cache_dir, "Xs.pkl"), "wb") as f:
            pickle.dump(Xs, f)
        with open(os.path.join(pop_cache_dir, "Ys.pkl"), "wb") as f:
            pickle.dump(Ys, f)
        with open(os.path.join(pop_cache_dir, "feature_cols.pkl"), "wb") as f:
            pickle.dump(feature_cols, f)
        with open(os.path.join(pop_cache_dir, "populations.pkl"), "wb") as f:
            pickle.dump(populations, f)
    
    return X_all, Y_all, Xs, Ys, feature_cols, populations


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
    populations,
    features_to_plot=None,
    bins=30,
    figsize=(12, 4),
    save_path=None,
):
    """
    Plot side-by-side histograms of specified covariates by population.
    """
    if features_to_plot is None:
        features_to_plot = feature_cols

    for feat in features_to_plot:
        if feat not in feature_cols:
            print(f"Warning: Feature '{feat}' not found in feature_cols, skipping")
            continue
            
        idx = feature_cols.index(feat)
        fig, axes = plt.subplots(1, len(populations), figsize=figsize, sharey=True)
        
        # Handle case when there's only one population
        if len(populations) == 1:
            axes = [axes]
            
        for ax, X, pop in zip(axes, Xs, populations):
            if X.size > 0:
                ax.hist(X[:, idx], bins=bins, density=True, alpha=0.7)
                ax.set_title(f"{feat} in {pop}")
                ax.set_xlabel(feat)
                ax.set_ylabel("Density")
            else:
                ax.text(0.5, 0.5, f"No data for {pop}", 
                        horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"hist_{feat}.png"), dpi=300)
        plt.close()


def plot_feature_target_scatter(
    Xs,
    Ys,
    feature_cols,
    populations,
    features_to_plot=None,
    figsize=(12, 4),
    save_path=None,
):
    """
    Plot side-by-side scatter plots of covariate vs. target by population.
    """
    if features_to_plot is None:
        features_to_plot = feature_cols

    for feat in features_to_plot:
        if feat not in feature_cols:
            print(f"Warning: Feature '{feat}' not found in feature_cols, skipping")
            continue
            
        idx = feature_cols.index(feat)
        fig, axes = plt.subplots(1, len(populations), figsize=figsize, sharey=True)
        
        # Handle case when there's only one population
        if len(populations) == 1:
            axes = [axes]
            
        for ax, X, y, pop in zip(axes, Xs, Ys, populations):
            if X.size > 0 and y.size > 0:
                # For binary targets, jitter to visualize density
                if np.all(np.logical_or(y == 0, y == 1)):
                    jitter = 0.05 * np.random.randn(len(y))
                    ax.scatter(X[:, idx], y + jitter, s=1, alpha=0.3)
                else:
                    ax.scatter(X[:, idx], y, s=1, alpha=0.3)
                ax.set_title(f"{feat} vs target in {pop}")
                ax.set_xlabel(feat)
                ax.set_ylabel("Target")
            else:
                ax.text(0.5, 0.5, f"No data for {pop}", 
                        horizontalalignment='center', verticalalignment='center')
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"scatter_{feat}.png"), dpi=300)
        plt.close()


def get_uci_pop_data(
    populations=None,
    feature_cols=None,
    subsample=True,
    subsample_fraction=0.1,
    target="income_binary",
    seed=42,
    save_dir='./data_uci',
    categorical_encoding=None,
    force_regenerate=False,
    exclude_population_features=True,
):
    """
    Return population data dicts for the specified populations
    """
    pop_list = populations or list(POPULATION_GROUPS.keys())
    print(f"Using explicitly provided UCI populations: {pop_list}")
    np.random.seed(seed)
    
    population_defining_features = set()
    if exclude_population_features:
        for pop_name in pop_list:
            if pop_name in POPULATION_GROUPS:
                # Add the criteria columns to the set of population-defining features
                population_defining_features.update(POPULATION_GROUPS[pop_name].keys())
        
        print(f"Will exclude these population-defining features: {population_defining_features}")

    custom_feature_cols = None
    if exclude_population_features and feature_cols is None:
        # Start with default features
        custom_feature_cols = [
            'age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week',
            'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex'
        ]
        # Remove population-defining features
        custom_feature_cols = [f for f in custom_feature_cols 
                              if f not in population_defining_features]
        print(f"Using custom feature columns (excluded population features): {custom_feature_cols}")

    try:
        X_all, Y_all, Xs, Ys, feats, pops = generate_data_uci(
            populations=pop_list,
            feature_cols=custom_feature_cols,
            target=target,
            save_dir=save_dir,
            categorical_encoding=categorical_encoding,
            force_regenerate=force_regenerate  # Pass this parameter through
        )
    except Exception as e:
        print(f"Error generating data: {e}")
        raise
    
    # Further filter one-hot encoded features if needed
    if exclude_population_features and categorical_encoding == 'onehot':
        # Find feature indexes to keep (exclude one-hot encoded population features)
        feat_indices_to_keep = []
        for i, feat in enumerate(feats):
            # Check if this is a one-hot encoded population feature
            is_pop_feature = False
            for pop_feat in population_defining_features:
                if feat.startswith(f"{pop_feat}_"):
                    is_pop_feature = True
                    break
            
            if not is_pop_feature:
                feat_indices_to_keep.append(i)
        
        # Filter features by index
        if len(feat_indices_to_keep) < len(feats):
            print(f"Filtering out {len(feats) - len(feat_indices_to_keep)} one-hot encoded population features")
            feats = [feats[i] for i in feat_indices_to_keep]
            X_all = X_all[:, feat_indices_to_keep]
            Xs = [X[:, feat_indices_to_keep] if X.size > 0 else X for X in Xs]
            
    # Check if any population has empty data
    empty_pops = [pop for pop, X, y in zip(pops, Xs, Ys) if X.size == 0 or y.size == 0]
    if empty_pops:
        print(f"WARNING: The following populations have empty data: {empty_pops}")

    # Create non-empty data indices for further processing
    valid_indices = [i for i, (X, y) in enumerate(zip(Xs, Ys)) if X.size > 0 and y.size > 0]
    
    # Impute missing values if any
    for i in valid_indices:
        # Check if there are any NaNs
        if np.isnan(Xs[i]).any():
            # Use median imputation for simplicity
            imputer = SimpleImputer(strategy="median")
            Xs[i] = imputer.fit_transform(Xs[i])
    
    # Handle target missing values
    for i in valid_indices:
        # Drop samples where target is NaN
        mask = ~np.isnan(Ys[i])
        if not mask.all():
            print(f"Removing {(~mask).sum()} samples with NaN target from population {pops[i]}")
            Xs[i] = Xs[i][mask]
            Ys[i] = Ys[i][mask]
        
        if Xs[i].size == 0 or Ys[i].size == 0:
            print(f"WARNING: No data left for population {pops[i]} after filtering")
            Xs[i] = np.array([])
            Ys[i] = np.array([])
            if i in valid_indices:
                valid_indices.remove(i)  # Remove from valid indices
    
    pop_data = []
    for i, pop in enumerate(pops):
        if i in valid_indices and Xs[i].size > 0 and Ys[i].size > 0:
            if subsample:
                # Randomly sample a percentage of the data for each population
                n_samples = Xs[i].shape[0]
                sample_size = max(1, int(n_samples * subsample_fraction))
                indices = np.random.choice(n_samples, sample_size, replace=False)
                Xs[i] = Xs[i][indices]
                Ys[i] = Ys[i][indices]
            
            # Add to population data
            pop_data.append({
                'pop_id': pop,
                'X_raw': Xs[i],
                'Y_raw': Ys[i],
                'meaningful_indices': None  # Unknown for real data
            })
        else:
            print(f"Skipping population {pop} due to empty data")

    # Print final dataset shapes and sizes
    for i, pop in enumerate(pops):
        if i in valid_indices and Xs[i].size > 0 and Ys[i].size > 0:
            print(f"Population {pop}: X shape {Xs[i].shape}, Y shape {Ys[i].shape}")
        else:
            print(f"Population {pop}: No data available")
    
    return pop_data


def main():
    p = argparse.ArgumentParser(
        description="Generate, preprocess, and plot UCI Adult data by population group"
    )
    p.add_argument("--populations",
        nargs="+", default=list(POPULATION_GROUPS.keys()),
        help="Population groups to include",
    )
    p.add_argument("--target", default="income_binary",
        help="Name of the target column",
    )
    p.add_argument("--categorical_encoding",
        choices=[None, 'onehot', 'label'], default='onehot',
        help="How to encode categorical features",
    )
    p.add_argument("--save_dir", default="./data_uci",
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
    p.add_argument("--plot_dir", default='./data_uci_plots',
        help="Directory to save plot images",
    )
    p.add_argument("--subsample_fraction", 
        type=float, default=0.1,
        help="Fraction of data to use for each population (default 0.1)",
    )
    p.add_argument("--debug",
        action="store_true",
        help="Show debug information",
    )
    
    args = p.parse_args()

    try:
        # Generate data
        X_all, Y_all, Xs, Ys, feature_cols, populations = generate_data_uci(
            populations=args.populations,
            feature_cols=None,  # Default features
            target=args.target,
            save_dir=args.save_dir,
            categorical_encoding=args.categorical_encoding,
        )
        
        print(f"Loaded pooled X_all {X_all.shape}, Y_all {Y_all.shape}")
        for pop, X, y in zip(populations, Xs, Ys):
            if X.size > 0 and y.size > 0:
                print(f"  {pop}: X {X.shape}, Y {y.shape}")
            else:
                print(f"  {pop}: No data")

        # Preprocess data
        X_all_proc, Xs_proc, imputer, scaler = preprocess_data(
            X_all,
            Xs,
            impute_strategy=args.impute_strategy,
            scale=not args.no_scale,
        )
        print("Preprocessing complete.")

        # Generate plots if requested
        if args.plot:
            # Use specified features or all features
            features_to_plot = args.features or feature_cols
            
            # Limit to 20 features for plotting
            if len(features_to_plot) > 20:
                print(f"Limiting to first 20 features for plotting")
                features_to_plot = features_to_plot[:20]
                
            print(f"Plotting features: {features_to_plot}")
            
            # Only include populations with data
            valid_indices = [i for i, X in enumerate(Xs_proc) if X.size > 0]
            valid_pops = [populations[i] for i in valid_indices]
            valid_Xs = [Xs_proc[i] for i in valid_indices]
            valid_Ys = [Ys[i] for i in valid_indices]
            
            # Generate histograms
            plot_feature_histograms(
                valid_Xs, feature_cols, valid_pops, 
                features_to_plot=features_to_plot, 
                save_path=args.plot_dir
            )
            
            # Generate scatter plots
            plot_feature_target_scatter(
                valid_Xs, valid_Ys, feature_cols, valid_pops, 
                features_to_plot=features_to_plot, 
                save_path=args.plot_dir
            )
            
            print(f"Plots saved to {args.plot_dir}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()