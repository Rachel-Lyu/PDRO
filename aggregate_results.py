# aggregate_results.py
import os
import pandas as pd
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_csv_files(directory, run_range, csv_filename, 
                        exclude_columns=None,
                        run_numbers=None,
                        suffix=None,
                        file_paths=None):
                        suffix=None,
                        file_paths=None):
    """
    Aggregate CSV files across run directories and compute averages
    """
    if run_range is None:
        if run_numbers is None:
            # raise ValueError("Either run_range or run_numbers must be provided")
            # raise ValueError("Either run_range or run_numbers must be provided")
        # Use provided run numbers
            if file_paths:
                files_to_read = file_paths
            else:
                # your existing run‐number / glob logic
                files_to_read = []
                for run_num in run_numbers:
                    pattern = os.path.join(directory, f"**/run_{run_num}", csv_filename)
                    files_to_read += glob.glob(pattern, recursive=True)
            if file_paths:
                files_to_read = file_paths
            else:
                # your existing run‐number / glob logic
                files_to_read = []
                for run_num in run_numbers:
                    pattern = os.path.join(directory, f"**/run_{run_num}", csv_filename)
                    files_to_read += glob.glob(pattern, recursive=True)
        else:
            # Use provided run numbers
            print(f"Using provided run numbers: {run_numbers}")
            files_to_read = []
            for run_num in run_numbers:
                pattern = os.path.join(directory, f"**/run_{run_num}", csv_filename)
                files_to_read += glob.glob(pattern, recursive=True)
    else:
        # Parse run range
        start_run, end_run = map(int, run_range.split('-'))
        run_numbers = range(start_run, end_run + 1)



    all_dfs = []
        
    for file_path in files_to_read:
        print(f"Reading {file_path}")
        df = pd.read_csv(file_path)
        
        # Exclude specified columns
        if exclude_columns:
            df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        all_dfs.append(df)
        files_to_read = []
        for run_num in run_numbers:
            pattern = os.path.join(directory, f"**/run_{run_num}", csv_filename)
            files_to_read += glob.glob(pattern, recursive=True)
    print(f"reading files: {files_to_read}")

    all_dfs = []
        
    for file_path in files_to_read:
        print(f"Reading {file_path}")
        df = pd.read_csv(file_path)
        
        # Exclude specified columns
        if exclude_columns:
            df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        all_dfs.append(df)
    
    if not all_dfs:
        print(f"No {csv_filename} files found in the specified run range")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Determine grouping columns based on the file type
    group_cols = []
    if csv_filename.startswith("results_comparison"):
        # In results_comparison.csv, "population" and "source" are categorical
        if "source" in combined_df.columns and "population" in combined_df.columns:
            group_cols = ["source", "population"]
        elif "source" in combined_df.columns:
            group_cols = ["source"]
        elif "population" in combined_df.columns:
            group_cols = ["population"]
    elif csv_filename == "variable_selection_per_population.csv":
        if "population" in combined_df.columns:
            group_cols = ["population"]
    
    # Identify numeric columns for averaging
    numeric_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
    # Filter out any grouping columns from numeric columns if they happen to be numeric
    numeric_cols = [col for col in numeric_cols if col not in group_cols]
    
    # If we have grouping columns, group by them and average numeric columns
    if group_cols and numeric_cols:
        # 1) Compute mean + std
        mean_df = combined_df.groupby(group_cols, as_index=False)[numeric_cols].mean()
        std_df  = combined_df.groupby(group_cols, as_index=False)[numeric_cols].std()
        # rename std cols → "<col>_std"
        std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric_cols})
        # merge mean + std
        aggregated_df = pd.merge(mean_df, std_df, on=group_cols, how='left')
        # 2) carry through any other non-numeric cols (first value)
        other_cols = [c for c in combined_df.columns
                      if c not in numeric_cols + group_cols]
        if other_cols:
            first_vals = combined_df.groupby(group_cols)[other_cols].first().reset_index()
            aggregated_df = pd.merge(aggregated_df, first_vals, on=group_cols, how='left')
        # 1) Compute mean + std
        mean_df = combined_df.groupby(group_cols, as_index=False)[numeric_cols].mean()
        std_df  = combined_df.groupby(group_cols, as_index=False)[numeric_cols].std()
        # rename std cols → "<col>_std"
        std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric_cols})
        # merge mean + std
        aggregated_df = pd.merge(mean_df, std_df, on=group_cols, how='left')
        # 2) carry through any other non-numeric cols (first value)
        other_cols = [c for c in combined_df.columns
                      if c not in numeric_cols + group_cols]
        if other_cols:
            first_vals = combined_df.groupby(group_cols)[other_cols].first().reset_index()
            aggregated_df = pd.merge(aggregated_df, first_vals, on=group_cols, how='left')
    else:
        # No valid grouping or no numeric columns to average
        if numeric_cols:
            # Calculate means only for numeric columns
            numeric_means = combined_df[numeric_cols].mean().to_frame().T
            
            # For non-numeric columns, take the first value
            non_numeric_cols = [col for col in combined_df.columns if col not in numeric_cols]
            if non_numeric_cols:
                first_values = combined_df[non_numeric_cols].iloc[0:1]
                aggregated_df = pd.concat([first_values.reset_index(drop=True), 
                                          numeric_means.reset_index(drop=True)], axis=1)
            else:
                aggregated_df = numeric_means
        else:
            # No numeric columns, just return the first row
            aggregated_df = combined_df.iloc[0:1].copy()
    
    return aggregated_df



    all_dfs = []
        
    for file_path in files_to_read:
        print(f"Reading {file_path}")
        df = pd.read_csv(file_path)
        
        # Exclude specified columns
        if exclude_columns:
            df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
        
        all_dfs.append(df)
    
    if not all_dfs:
        print(f"No {csv_filename} files found in the specified run range")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Determine grouping columns based on the file type
    group_cols = []
    if csv_filename.startswith("results_comparison"):
        # In results_comparison.csv, "population" and "source" are categorical
        if "source" in combined_df.columns and "population" in combined_df.columns:
            group_cols = ["source", "population"]
        elif "source" in combined_df.columns:
            group_cols = ["source"]
        elif "population" in combined_df.columns:
            group_cols = ["population"]
    elif csv_filename == "variable_selection_per_population.csv":
        if "population" in combined_df.columns:
            group_cols = ["population"]
    
    # Identify numeric columns for averaging
    numeric_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
    # Filter out any grouping columns from numeric columns if they happen to be numeric
    numeric_cols = [col for col in numeric_cols if col not in group_cols]
    
    # If we have grouping columns, group by them and average numeric columns
    if group_cols and numeric_cols:
        # 1) Compute mean + std
        mean_df = combined_df.groupby(group_cols, as_index=False)[numeric_cols].mean()
        std_df  = combined_df.groupby(group_cols, as_index=False)[numeric_cols].std()
        # rename std cols → "<col>_std"
        std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric_cols})
        # merge mean + std
        aggregated_df = pd.merge(mean_df, std_df, on=group_cols, how='left')
        # 2) carry through any other non-numeric cols (first value)
        other_cols = [c for c in combined_df.columns
                      if c not in numeric_cols + group_cols]
        if other_cols:
            first_vals = combined_df.groupby(group_cols)[other_cols].first().reset_index()
            aggregated_df = pd.merge(aggregated_df, first_vals, on=group_cols, how='left')
    else:
        # No valid grouping or no numeric columns to average
        if numeric_cols:
            # Calculate means only for numeric columns
            numeric_means = combined_df[numeric_cols].mean().to_frame().T
            
            # For non-numeric columns, take the first value
            non_numeric_cols = [col for col in combined_df.columns if col not in numeric_cols]
            if non_numeric_cols:
                first_values = combined_df[non_numeric_cols].iloc[0:1]
                aggregated_df = pd.concat([first_values.reset_index(drop=True), 
                                          numeric_means.reset_index(drop=True)], axis=1)
            else:
                aggregated_df = numeric_means
        else:
            # No numeric columns, just return the first row
            aggregated_df = combined_df.iloc[0:1].copy()
    
    return aggregated_df

def dataframe_to_latex(df, caption, label):
    """
    Convert DataFrame to LaTeX table code
    """
    # Start LaTeX table
    latex_code = "\\begin{table}[htbp]\n\\centering\n"
    latex_code += f"\\caption{{{caption}}}\n"
    latex_code += f"\\label{{{label}}}\n"
    
    # Create table format: first column left-aligned, others right-aligned
    n_cols = len(df.columns)
    col_format = "l" + "r" * (n_cols - 1)
    latex_code += f"\\begin{{tabular}}{{{col_format}}}\n"
    latex_code += "\\toprule\n"
    
    # Format column headers
    headers = []
    for col in df.columns:
        # Replace underscores with spaces, capitalize words
        beautified = " ".join(word.capitalize() for word in col.replace("_", " ").split())
        headers.append(beautified)
    
    latex_code += " & ".join(headers) + " \\\\\n"
    latex_code += "\\midrule\n"
    
    # Format table data rows
    for _, row in df.iterrows():
        row_values = []
        for col, val in zip(df.columns, row):
            if isinstance(val, (int, float)):
                if "percentage" in col.lower() or "pct" in col.lower() or "coverage" in col.lower():
                    # Format percentages with 2 decimal places
                    row_values.append(f"{val:.2f}\\%")
                elif "mse" in col.lower() or "error" in col.lower():
                    # Scientific notation for error metrics
                    row_values.append(f"{val:.4e}")
                elif "downstream_accuracy" in col.lower():
                    # Format accuracy with 2 decimal places
                    row_values.append(f"{val:.2f}")
                elif "downstream_accuracy" in col.lower():
                    # Format accuracy with 2 decimal places
                    row_values.append(f"{val:.2f}")
                elif val == int(val):  # Check if float is effectively an integer
                    row_values.append(f"{int(val)}")
                else:
                    # Default float format
                    row_values.append(f"{val:.4f}")
            else:
                row_values.append(str(val))
        latex_code += " & ".join(row_values) + " \\\\\n"
    
    # End LaTeX table
    latex_code += "\\bottomrule\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\end{table}"
    
    return latex_code

def create_bar_plots(df, output_dir, file_prefix):
    """
    Create refined bar plots with:
    - Bars grouped by population (each cluster shows all methods for one population)
    - Improved number formatting
    - Legend at bottom center
    - Pastel color palette
    - Reduced whitespace
    """
    if 'population' not in df.columns or 'source' not in df.columns:
        print("Required columns 'population' and 'source' not found")
        return
    
    # Get metrics to plot
    metric_cols = ['mse', 'r2']  # Focus on these key metrics
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    if not available_metrics:
        print("No metrics found to plot")
        return
    
    # Set a professional plotting style with minimalist aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Pastel color palette with enough contrast to be distinguishable
    colors = ['#457b9d', '#e76f51', '#f4a261', '#2a9d8f', '#e9c46a']
    
    # Get unique populations and methods
    populations = sorted(df['population'].unique())
    methods = sorted(df['source'].unique())
    
    # Create a plot for each metric
    for metric in available_metrics:
        # Create one subplot per metric
        fig, ax = plt.subplots(figsize=(max(12, len(populations) * 2.5), 6))
        
        # Set up the bar width and positions
        bar_width = 0.15  # Narrower bars that are closer together
        pop_spacing = 1.0  # Space between population groups
        index = np.arange(len(populations)) * pop_spacing  # Wider spacing between population groups
        
        # Track min/max values for y-axis limits
        min_val = float('inf')
        max_val = float('-inf')
        
        # Plot bars for each method within each population group
        bar_handles = []  # To store handles for legend
        
        for i, method in enumerate(methods):
            # Prepare data across all populations for this method
            values = []
            positions = []
            
            for j, pop in enumerate(populations):
                # Get data for this population and method
                pop_data = df[(df['population'] == pop) & (df['source'] == method)]
                
                if not pop_data.empty:
                    val = pop_data[metric].values[0]
                    values.append(val)
                    min_val = min(min_val, val)
                    max_val = max(max_val, val)
                else:
                    values.append(0)
                
                # Calculate position within the population group
                positions.append(index[j] + (i - len(methods)/2 + 0.5) * bar_width)
            
            # Create bars with pastel colors for this method across all populations
            bars = ax.bar(positions, values, bar_width * 0.95, 
                         color=colors[i % len(colors)], 
                         label=clean_method_name(method),  # Will define this function
                         edgecolor='white', linewidth=0.5,
                         alpha=0.85)
            
            bar_handles.append(bars[0])  # Store first bar for legend
            
            # Add value labels on top of bars - IMPROVED NUMBER FORMATTING
            for pos, val in zip(positions, values):
                # Improved number formatting based on magnitude
                if metric == 'mse':
                    if val < 0.001:
                        label = f"{val:.2e}"
                    elif val < 0.01:
                        label = f"{val:.4f}"
                    elif val < 0.1:
                        label = f"{val:.3f}"
                    elif val < 1:
                        label = f"{val:.2f}"
                    else:
                        label = f"{val:.1f}"
                else:
                    # For R², use consistent 3 decimal places
                    label = f"{val:.3f}"
                
                # Position labels above the bars
                y_pos = val + (max_val - min_val) * 0.02
                ax.text(pos, y_pos, label, ha='center', va='bottom', 
                        fontsize=9, color='black', fontweight='bold')
        
        # Set x-ticks at population group centers
        ax.set_xticks(index)
        ax.set_xticklabels([f'{pop}' for pop in populations], fontsize=11)
        ax.set_xticklabels([f'{pop}' for pop in populations], fontsize=11)
        
        # Set title and labels
        if metric == 'mse':
            title = 'Mean Squared Error by Method'
        elif metric == 'r2':
            title = 'R² Score by Method'
        elif metric == 'downstream_accuracy' or metric == 'accuracy' or metric == 'accuracy_score':
            title = 'Downstream Accuracy by Method'
        else:
            title = 'Metric by Method'
        if metric == 'mse':
            title = 'Mean Squared Error by Method'
        elif metric == 'r2':
            title = 'R² Score by Method'
        elif metric == 'downstream_accuracy' or metric == 'accuracy' or metric == 'accuracy_score':
            title = 'Downstream Accuracy by Method'
        else:
            title = 'Metric by Method'
        ax.set_title(title, fontsize=14, pad=10, fontweight='bold')
        
        # Set appropriate y-axis label
        y_label = 'Mean Squared Error' if metric == 'mse' else 'R² Score' 
        ax.set_ylabel(y_label, fontsize=12)
        
        # Set y-limits with reduced whitespace
        y_padding = (max_val - min_val) * 0.15  # Reduced padding
        
        if metric == 'mse':
            # For MSE, if we have very small values, use log scale
            if max_val / (min_val + 1e-10) > 100:
                # ax.set_yscale('log')
                ax.set_yscale('linear')
                ax.set_ylim(min_val * 0.8, max_val * 1.2)
            else:
                ax.set_ylim(max(0, min_val - y_padding), max_val + y_padding)
        else:
            # For R², maintain a reasonable range
            ax.set_ylim(max(-0.1, min_val - y_padding), min(1.0, max_val + y_padding))
        
        # # Add a "better direction" indicator
        # if metric == 'mse':
        #     direction_text = "↓ Lower is better"
        # else:
        #     direction_text = "↑ Higher is better"
        
        # # Add the indicator in a subtle box in the upper left
        # text_box = ax.text(0.02, 0.98, direction_text, 
        #                  transform=ax.transAxes,
        #                  fontsize=10, verticalalignment='top',
        #                  bbox=dict(boxstyle='round,pad=0.3', 
        #                           facecolor='white', alpha=0.8,
        #                           edgecolor='lightgrey'))
        
        # Add light grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Adjust layout with reduced margins
        plt.tight_layout(pad=1.0)  # Reduced padding
        
        # Clean method names for the legend
        clean_names = []
        for method in methods:
            if method == 'our_method':
                clean_names.append('Our Method')
            elif method.startswith('baseline_'):
                # Convert snake_case to Title Case without the baseline_ prefix
                name = ' '.join(word.capitalize() for word in method.replace('baseline_', '').split('_'))
                clean_names.append(name)
            else:
                clean_names.append(method)
        
        # MOVE LEGEND TO BOTTOM CENTER
        # Create legend with the stored handles
        legend = ax.legend(
            handles=bar_handles,
            labels=clean_names,
            title="Method",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),  # Positions legend below the plot
            ncol=min(len(methods), 3),  # Up to 3 methods per row to avoid spreading too wide
            frameon=True,
            framealpha=0.8,
            fontsize=10
        )
        
        # Add a bit more padding at the bottom for the legend
        plt.subplots_adjust(bottom=0.2)
        
        # Save the figure with extra bottom margin to accommodate the legend
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}_refined.pdf"), 
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}_refined.pdf"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Refined bar plots saved to {output_dir}")

# Helper function to clean method names
def clean_method_name(method):
    if method == 'our_method':
        return 'Our Method'
    elif method.startswith('baseline_'):
        # Convert snake_case to Title Case without the baseline_ prefix
        return ' '.join(word.capitalize() for word in method.replace('baseline_', '').split('_'))
    else:
        return method

def create_horizontal_comparison_plots(df, output_dir, file_prefix):
    """
    Create horizontal bar charts that make comparison easier
    with elegant styling and cleaner layout
    """
    if 'population' not in df.columns or 'source' not in df.columns:
        print("Required columns not found")
        return
    
    metric_cols = ['mse', 'r2']
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    if not available_metrics:
        print("No metrics found to plot")
        return
    
    # Use a clean style
    plt.style.use('ggplot')
    
    # Set a harmonious color palette (more pastel)
    method_colors = {
        'our_method': '#6a994e',  # Soft green for our method
        'baseline_dro_lasso': '#bc6c25',  # Soft orange
        'baseline_dro_xgb': '#dda15e',  # Soft yellow
        'baseline_lasso': '#606c38',  # Olive green
        'baseline_xgb': '#283618'   # Dark green
    }
    
    # Get unique methods and populations
    methods = sorted(df['source'].unique())
    populations = sorted(df['population'].unique())
    
    # Clean method names
    method_names = {}
    for method in methods:
        if method == 'our_method':
            method_names[method] = 'Our Method'
        elif method.startswith('baseline_'):
            name = ' '.join(word.capitalize() for word in method.replace('baseline_', '').split('_'))
            method_names[method] = name
        else:
            method_names[method] = method
    
    # For each metric, create a separate plot
    for metric in available_metrics:
        # Create figure with one row per population
        fig, axes = plt.subplots(len(populations), 1, 
                                figsize=(10, 3 * len(populations)),
                                sharex=True)
        
        if len(populations) == 1:
            axes = [axes]
        
        # Determine metric formatting
        if metric == 'mse':
            format_func = lambda x: f"{x:.2e}" if x < 0.01 else f"{x:.3f}"
            metric_label = "Mean Squared Error"
            is_lower_better = True
        else:
            format_func = lambda x: f"{x:.3f}"
            metric_label = "R² Score"
            is_lower_better = False
        
        # Track global min/max for consistent scaling
        all_values = []
        for pop in populations:
            pop_data = df[df['population'] == pop]
            for method in methods:
                method_data = pop_data[pop_data['source'] == method]
                if not method_data.empty:
                    all_values.append(method_data[metric].values[0])
        
        global_min = min(all_values)
        global_max = max(all_values)
        
        # Plot each population
        for i, pop in enumerate(populations):
            ax = axes[i]
            
            # Filter data for this population
            pop_data = df[df['population'] == pop]
            
            # Create sorted method-value pairs
            method_values = []
            for method in methods:
                method_data = pop_data[pop_data['source'] == method]
                if not method_data.empty:
                    method_values.append((method, method_data[metric].values[0]))
            
            # Sort methods by value (ascending for MSE, descending for R²)
            method_values.sort(key=lambda x: x[1], reverse=not is_lower_better)
            
            # Extract sorted methods and values
            sorted_methods, sorted_values = zip(*method_values)
            
            # Get positions
            y_pos = range(len(sorted_methods))
            
            # Create horizontal bars
            bars = ax.barh(y_pos, sorted_values, 
                          color=[method_colors.get(m, '#999999') for m in sorted_methods],
                          height=0.6, alpha=0.8, edgecolor='white', linewidth=0.5)
            
            # Add method labels on y-axis
            ax.set_yticks(y_pos)
            ax.set_yticklabels([method_names[m] for m in sorted_methods], fontsize=10)
            
            # Add value labels inside/outside bars
            for j, (bar, val) in enumerate(zip(bars, sorted_values)):
                # Determine text color and position based on bar width
                bar_width = bar.get_width()
                max_width = ax.get_xlim()[1]
                
                if bar_width / max_width < 0.25:  # If bar is short
                    text_color = 'black'
                    x_pos = bar_width + (global_max - global_min) * 0.01
                    ha = 'left'
                else:
                    text_color = 'white'
                    x_pos = bar_width - (global_max - global_min) * 0.02
                    ha = 'right'
                
                ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                       format_func(val), va='center', ha=ha,
                       color=text_color, fontweight='bold', fontsize=9)
            
            # Highlight the best method
            if sorted_methods:
                best_idx = 0 if is_lower_better else -1
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(1.5)
            
            # Add a title for each population
            ax.set_title(f'{pop}', fontsize=12, pad=5)
            ax.set_title(f'{pop}', fontsize=12, pad=5)
            
            # # Add indicator for which direction is better
            # direction = "← Lower is better" if is_lower_better else "→ Higher is better"
            # ax.text(0.98, 0.05, direction, transform=ax.transAxes,
            #        ha='right', va='bottom', fontsize=9,
            #        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
            #                 alpha=0.8, edgecolor='lightgrey'))
            
            # Clean up the axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            
            # Add subtle grid lines just for x-axis
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            # First plot only: add x-label
            if i == len(populations) - 1:
                ax.set_xlabel(metric_label, fontsize=11)
        
        # Set consistent x limits with minimal padding
        x_range = global_max - global_min
        x_min = max(0, global_min - x_range * 0.05) if metric == 'mse' else global_min - x_range * 0.05
        x_max = global_max + x_range * 0.1
        
        for ax in axes:
            ax.set_xlim(x_min, x_max)
        
        # Add overall title
        fig.suptitle(f'{metric_label} Comparison', fontsize=14, y=0.98, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.subplots_adjust(hspace=0.3)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}_horizontal.pdf"), 
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}_horizontal.pdf"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Horizontal comparison plots saved to {output_dir}")

def create_multi_series_horizontal_bars_2(df, output_dir, file_prefix):
    """
    Create horizontal multi-series bar charts for all metrics side by side,
    with a single shared legend at the bottom center.
    """
    if 'population' not in df.columns or 'source' not in df.columns:
        print("Required columns 'population' and 'source' not found")
        return

    # decide which metrics to plot
    if 'downstream_accuracy' in df.columns or 'logloss' in df.columns:
        metrics = [m for m in ['downstream_accuracy', 'logloss'] if m in df.columns]
    else:
        metrics = [m for m in ['mse', 'r2'] if m in df.columns]
    if not metrics:
        print("No metrics found to plot")
        return

    plt.style.use('ggplot')
    methods     = sorted(df['source'].unique())
    populations = sorted(df['population'].unique(), reverse=True)

    # same colours as requested
    method_colors = {
        'our_method':        '#4e79a7',
        'baseline_lasso':    '#59a14f',
        'baseline_dro_lasso':'#f28e2b',
        'baseline_xgb':      '#4e4e4e',
        'baseline_dro_xgb':  '#edc948'
    }

    # clean names for legend
    method_names = {}
    for m in methods:
        if m == 'our_method':
            method_names[m] = 'Our Method'
        elif m.startswith('baseline_'):
            name = ' '.join(w.capitalize() for w in m.replace('baseline_', '').split('_'))
            method_names[m] = name
        else:
            method_names[m] = m

    # layout
    n_metrics   = len(metrics)
    fig, axes   = plt.subplots(1, n_metrics,
                               figsize=(6 * n_metrics, max(6, len(populations)*1.2 + 1)),
                               sharey=True)
    if n_metrics == 1:
        axes = [axes]

    fig.subplots_adjust(wspace=0.1)   # <- try values like 0.1 or 0.15 too
    # shared y positions and bar offsets
    y_pos       = np.arange(len(populations))
    bar_height  = 0.7 / len(methods)
    offsets     = {m: (i - len(methods)/2 + 0.5) * bar_height
                   for i, m in enumerate(methods)}

    # gather legend handles
    handles, labels = [], []

    for ax, metric in zip(axes, metrics):
        std_col = f"{metric}_std"
        for m in methods:
            vals, errs, ys = [], [], []
            for i, pop in enumerate(populations):
                sel = df[(df['population']==pop) & (df['source']==m)]
                if not sel.empty:
                    v = float(sel[metric].iloc[0])
                    e = float(sel[std_col].iloc[0]) if std_col in sel.columns else 0.0
                else:
                    v, e = 0.0, 0.0
                vals.append(v); errs.append(e)
                ys.append(y_pos[i] + offsets[m])

            bar = ax.barh(ys, vals,
                          height=bar_height * 0.9,
                          color=method_colors.get(m, '#999999'),
                          xerr=errs if std_col in df.columns else None,
                          error_kw=dict(ecolor='gray', capsize=3),
                          label=method_names[m],
                          alpha=0.8, edgecolor='white', linewidth=0.5)
            # only collect one handle per method
            if ax is axes[0]:
                handles.append(bar[0])
                labels.append(method_names[m])

        # formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(populations, fontsize=10)
        title_map = {'mse':'Mean Squared Error', 'r2':'R2 Score',
                     'logloss':'Log Loss', 'downstream_accuracy':'Accuracy'}
        ax.set_title(title_map.get(metric, metric), fontsize=12, pad=8)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(title_map.get(metric, metric))

    # shared legend at bottom center
    fig.legend(handles, labels,
               loc='lower center',
               ncol=min(len(methods), 3),
               frameon=True, framealpha=0.8)
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)

    out_path = os.path.join(output_dir, f"{file_prefix}_multi_series.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Multi-series bar plots saved to {out_path}")

def create_multi_series_horizontal_bars_2(df, output_dir, file_prefix):
    """
    Create horizontal multi-series bar charts for all metrics side by side,
    with a single shared legend at the bottom center.
    """
    if 'population' not in df.columns or 'source' not in df.columns:
        print("Required columns 'population' and 'source' not found")
        return

    # decide which metrics to plot
    if 'downstream_accuracy' in df.columns or 'logloss' in df.columns:
        metrics = [m for m in ['downstream_accuracy', 'logloss'] if m in df.columns]
    else:
        metrics = [m for m in ['mse', 'r2'] if m in df.columns]
    if not metrics:
        print("No metrics found to plot")
        return

    plt.style.use('ggplot')
    methods     = sorted(df['source'].unique())
    populations = sorted(df['population'].unique(), reverse=True)

    # same colours as requested
    method_colors = {
        'our_method':        '#4e79a7',
        'baseline_lasso':    '#59a14f',
        'baseline_dro_lasso':'#f28e2b',
        'baseline_xgb':      '#4e4e4e',
        'baseline_dro_xgb':  '#edc948'
    }

    # clean names for legend
    method_names = {}
    for m in methods:
        if m == 'our_method':
            method_names[m] = 'Our Method'
        elif m.startswith('baseline_'):
            name = ' '.join(w.capitalize() for w in m.replace('baseline_', '').split('_'))
            method_names[m] = name
        else:
            method_names[m] = m

    # layout
    n_metrics   = len(metrics)
    fig, axes   = plt.subplots(1, n_metrics,
                               figsize=(6 * n_metrics, max(6, len(populations)*1.2 + 1)),
                               sharey=True)
    if n_metrics == 1:
        axes = [axes]

    fig.subplots_adjust(wspace=0.1)   # <- try values like 0.1 or 0.15 too
    # shared y positions and bar offsets
    y_pos       = np.arange(len(populations))
    bar_height  = 0.7 / len(methods)
    offsets     = {m: (i - len(methods)/2 + 0.5) * bar_height
                   for i, m in enumerate(methods)}

    # gather legend handles
    handles, labels = [], []

    for ax, metric in zip(axes, metrics):
        std_col = f"{metric}_std"
        for m in methods:
            vals, errs, ys = [], [], []
            for i, pop in enumerate(populations):
                sel = df[(df['population']==pop) & (df['source']==m)]
                if not sel.empty:
                    v = float(sel[metric].iloc[0])
                    e = float(sel[std_col].iloc[0]) if std_col in sel.columns else 0.0
                else:
                    v, e = 0.0, 0.0
                vals.append(v); errs.append(e)
                ys.append(y_pos[i] + offsets[m])

            bar = ax.barh(ys, vals,
                          height=bar_height * 0.9,
                          color=method_colors.get(m, '#999999'),
                          xerr=errs if std_col in df.columns else None,
                          error_kw=dict(ecolor='gray', capsize=3),
                          label=method_names[m],
                          alpha=0.8, edgecolor='white', linewidth=0.5)
            # only collect one handle per method
            if ax is axes[0]:
                handles.append(bar[0])
                labels.append(method_names[m])

        # formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(populations, fontsize=10)
        title_map = {'mse':'Mean Squared Error', 'r2':'R2 Score',
                     'logloss':'Log Loss', 'downstream_accuracy':'Accuracy'}
        ax.set_title(title_map.get(metric, metric), fontsize=12, pad=8)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(title_map.get(metric, metric))

    # shared legend at bottom center
    fig.legend(handles, labels,
               loc='lower center',
               ncol=min(len(methods), 3),
               frameon=True, framealpha=0.8)
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)

    out_path = os.path.join(output_dir, f"{file_prefix}_multi_series.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Multi-series bar plots saved to {out_path}")

def create_multi_series_horizontal_bars(df, output_dir, file_prefix):
    """
    Create horizontal multi-series bar charts with:
    - Each population on its own row
    - Methods represented by different colored bars
    - Clean, minimal design with good readability
    """
    if 'population' not in df.columns or 'source' not in df.columns:
        print("Required columns 'population' and 'source' not found")
        return
    
    # Get metrics to plot
    # the metrics are either mse and r2 or downstream_accuracy

    if 'downstream_accuracy' in df.columns or 'logloss' in df.columns:
        metric_cols = ['downstream_accuracy', 'logloss']
    else:
        metric_cols = ['mse', 'r2']  # Focus on these key metrics
    # the metrics are either mse and r2 or downstream_accuracy

    if 'downstream_accuracy' in df.columns or 'logloss' in df.columns:
        metric_cols = ['downstream_accuracy', 'logloss']
    else:
        metric_cols = ['mse', 'r2']  # Focus on these key metrics
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    if not available_metrics:
        print("No metrics found to plot")
        return
    
    # Set a clean, minimal style
    plt.style.use('ggplot')
    
    # Vibrant color palette for methods (distinct colors)
    # if the metrics are accuracy and logloss, instead of lasso we have 
    # if the metrics are accuracy and logloss, instead of lasso we have 
    method_colors = {
        'our_method': '#4e79a7',           # Blue
        'baseline_lasso': '#59a14f',        # Green
        'baseline_dro_lasso': '#f28e2b',    # Orange
        'baseline_xgb': '#4e4e4e',          # Dark Gray/Black
        'baseline_dro_xgb': '#edc948'       # Yellow
    }
    
    # Get unique methods and populations
    methods = sorted(df['source'].unique())
    # Reverse population order so top row is Population A
    populations = sorted(df['population'].unique(), reverse=True)
    
    # Create clean method names for legend
    method_names = {}
    for method in methods:
        if method == 'our_method':
            method_names[method] = 'Our Method'
        elif method.startswith('baseline_'):
            # Convert snake_case to Title Case without the baseline_ prefix
            name = ' '.join(word.capitalize() for word in method.replace('baseline_', '').split('_'))
            method_names[method] = name
        else:
            method_names[method] = method
    
    # For each metric, create a separate plot
    for metric in available_metrics:
        # Create figure
        fig, ax = plt.subplots(figsize=(7, max(6, len(populations) * 1.2 + 2)))
        
        # Determine metric formatting and title
        if metric == 'mse':
            format_func = lambda x: f"{x:.2e}" if x < 0.01 else f"{x:.3f}"
            metric_label = "Mean Squared Error"
            is_lower_better = True
        elif metric == 'logloss':
            format_func = lambda x: f"{x:.3f}"
            metric_label = "Log Loss"
            is_lower_better = True
        elif metric == 'downstream_accuracy':
            format_func = lambda x: f"{x:.3f}" 
            metric_label = "Accuracy"
            is_lower_better = False
        elif metric == 'logloss':
            format_func = lambda x: f"{x:.3f}"
            metric_label = "Log Loss"
            is_lower_better = True
        elif metric == 'downstream_accuracy':
            format_func = lambda x: f"{x:.3f}" 
            metric_label = "Accuracy"
            is_lower_better = False
        else:
            format_func = lambda x: f"{x:.3f}"
            metric_label = "R² Score"
            is_lower_better = False
        
        # Configure y positions for populations
        y_positions = np.arange(len(populations))
        
        # Track max value for setting proper x range
        max_value = 0
        
        # Method index to bar y-position mapping
        method_y_offsets = {}
        bar_height = 0.7 / len(methods)
        
        std_col = f"{metric}_std"
        std_col = f"{metric}_std"
        for i, method in enumerate(methods):
            # Calculate offset for this method's bars (vertically within each population's row)
            offset = i - len(methods) / 2 + 0.5
            method_y_offsets[method] = offset * bar_height
            
            # Collect values for this method across all populations
            values = []
            labels = []
            y_coords = []
            
            for j, pop in enumerate(populations):
                # Get data for this population and method
                pop_data = df[(df['population'] == pop) & (df['source'] == method)]
                
                if not pop_data.empty:
                    val = pop_data[metric].values[0]
                    values.append(val)
                    labels.append(format_func(val))
                    y_coords.append(y_positions[j] + method_y_offsets[method])
                    max_value = max(max_value, val)
                else:
                    values.append(0)
                    labels.append("N/A")
                    y_coords.append(y_positions[j] + method_y_offsets[method])
            
            # Plot horizontal bars for this method across all populations
            if std_col in df.columns:
                yerrs = []
                for j, pop in enumerate(populations):
                    pop_data = df[(df['population']==pop) & (df['source']==method)]
                    yerrs.append(float(pop_data[std_col].values[0]) if not pop_data.empty else 0.0)
                bars = ax.barh(
                    y_coords, values, height=bar_height * 0.9,
                    color=method_colors.get(method, '#999999'),
                    edgecolor='white',
                    linewidth=0.5,
                    label=method_names[method],
                    alpha=0.9,
                    xerr=yerrs,
                    error_kw=dict(ecolor='gray', capsize=3)
                )
            else:
                bars = ax.barh(
                    y_coords, values, height=bar_height * 0.9,
                    color=method_colors.get(method, '#999999'),
                    edgecolor='white',
                    linewidth=0.5,
                    label=method_names[method],
                    alpha=0.9
                )
            if std_col in df.columns:
                yerrs = []
                for j, pop in enumerate(populations):
                    pop_data = df[(df['population']==pop) & (df['source']==method)]
                    yerrs.append(float(pop_data[std_col].values[0]) if not pop_data.empty else 0.0)
                bars = ax.barh(
                    y_coords, values, height=bar_height * 0.9,
                    color=method_colors.get(method, '#999999'),
                    edgecolor='white',
                    linewidth=0.5,
                    label=method_names[method],
                    alpha=0.9,
                    xerr=yerrs,
                    error_kw=dict(ecolor='gray', capsize=3)
                )
            else:
                bars = ax.barh(
                    y_coords, values, height=bar_height * 0.9,
                    color=method_colors.get(method, '#999999'),
                    edgecolor='white',
                    linewidth=0.5,
                    label=method_names[method],
                    alpha=0.9
                )
            
            # Add value labels
            for bar, label, value in zip(bars, labels, values):
                # Only add label if there's a significant value
                if value > max_value * 0.01:
                    # Position label inside or outside bar based on length
                    if value > max_value * 0.25:
                        # Inside bar with white text
                        x_pos = value - max_value * 0.02
                        text_color = 'white'
                        ha = 'right'
                    else:
                        # Outside bar with black text
                        x_pos = value + max_value * 0.02
                        text_color = 'black'
                        ha = 'left'
                    
                    ax.text(
                        x_pos, bar.get_y() + bar.get_height()/2,
                        label, va='center', ha=ha,
                        fontsize=9, fontweight='bold',
                        color=text_color
                    )
        
        # Set y-ticks at population positions
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f'{pop}' for pop in populations], fontsize=10)
        ax.set_yticklabels([f'{pop}' for pop in populations], fontsize=10)
        
        # Set x-axis limits with a bit of padding
        ax.set_xlim(0, max_value * 1.15)
        
        # Set title and labels
        title = f'{metric_label} by Method' if metric == 'mse' else f'{metric_label} by Method'
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Value', fontsize=12, labelpad=10)
        ax.set_ylabel('Population', fontsize=12, labelpad=10)
        
        # Add gridlines on the x-axis only
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)  # Put gridlines behind bars
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # # Add a better vs worse indicator text
        # if metric == 'mse':
        #     indicator = "← Lower is better"
        # else:
        #     indicator = "→ Higher is better"
        
        # fig.text(0.98, 0.02, indicator, 
        #         ha='right', va='bottom', 
        #         fontsize=10, fontweight='bold',
        #         bbox=dict(facecolor='white', alpha=0.8, 
        #                   boxstyle='round,pad=0.3', 
        #                   edgecolor='lightgrey'))
        
        # Add legend at the bottom
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(methods), 3),
            frameon=True,
            fontsize=10,
            title='Method'
        )
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Make room for the legend
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}_multi_series.pdf"), 
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}_multi_series.pdf"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Multi-series horizontal bar plots saved to {output_dir}")




def main():
    parser = argparse.ArgumentParser(description="Aggregate CSV results across multiple runs")
    parser.add_argument("directory", help="Base directory containing run directories")
    parser.add_argument("--run_range", default=None,
                        help="Range of run numbers to include (e.g., '1-5')")
    parser.add_argument("--run_numbers", "-r", nargs='+', type=int, default=None,
                        help="Specific run numbers to include (overrides run_range)")
    parser.add_argument("--output", "-o", default="aggregated_results", 
                        help="Output directory for aggregated results")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--suffix", default="", help="Suffix for csv file (eg if we values results_comparison_5.csv)")
    parser.add_argument("--file_paths", default=None, nargs='+', 
                        help="Specific file paths to include (overrides run_range and run_numbers)")
    parser.add_argument("--file_paths", default=None, nargs='+', 
                        help="Specific file paths to include (overrides run_range and run_numbers)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Process results_comparison.csv
    results_comp_df = aggregate_csv_files(
        args.directory, 
        args.run_range, 
        f"results_comparison{args.suffix}.csv",
        exclude_columns=["selected_indices"],
        run_numbers=args.run_numbers,
        suffix= args.suffix,
        file_paths=args.file_paths
        suffix= args.suffix,
        file_paths=args.file_paths
    )
    
    if results_comp_df is not None:
        results_comp_output = os.path.join(args.output, "aggregated_results_comparison.csv")
        results_comp_df.to_csv(results_comp_output, index=False)
        print(f"Saved aggregated results comparison to {results_comp_output}")
        
        # Generate LaTeX table
        latex_code = dataframe_to_latex(
            results_comp_df, 
            "Average Performance Metrics Across Runs", 
            "tab:results-comparison"
        )
        latex_output = os.path.join(args.output, "results_comparison_latex.txt")
        with open(latex_output, "w") as f:
            f.write(latex_code)
        print(f"Saved LaTeX table to {latex_output}")
        
        # Generate bar plots
        if not args.no_plots:
            create_bar_plots(results_comp_df, args.output, "results_comparison")

        # Generate horizontal comparison plots
        # create_horizontal_comparison_plots(results_comp_df, args.output, "results_comparison")
        # create_horizontal_comparison_plots(results_comp_df, args.output, "results_comparison")
        # Generate multi-series horizontal bar plots
        create_multi_series_horizontal_bars(results_comp_df, args.output, "results_comparison")
        create_multi_series_horizontal_bars_2(results_comp_df, args.output, "results_comparison")
        create_multi_series_horizontal_bars_2(results_comp_df, args.output, "results_comparison")
    
    # # Process variable_selection_per_population.csv
    # var_select_df = aggregate_csv_files(
    #     args.directory, 
    #     args.run_range, 
    #     "variable_selection_per_population.csv",
    #     run_numbers=args.run_numbers,
    #     suffix= args.suffix,
    # )
    # # Process variable_selection_per_population.csv
    # var_select_df = aggregate_csv_files(
    #     args.directory, 
    #     args.run_range, 
    #     "variable_selection_per_population.csv",
    #     run_numbers=args.run_numbers,
    #     suffix= args.suffix,
    # )
    
    # if var_select_df is not None:
    #     var_select_output = os.path.join(args.output, "aggregated_variable_selection.csv")
    #     var_select_df.to_csv(var_select_output, index=False)
    #     print(f"Saved aggregated variable selection to {var_select_output}")
    # if var_select_df is not None:
    #     var_select_output = os.path.join(args.output, "aggregated_variable_selection.csv")
    #     var_select_df.to_csv(var_select_output, index=False)
    #     print(f"Saved aggregated variable selection to {var_select_output}")
        
    #     # Generate LaTeX table
    #     latex_code = dataframe_to_latex(
    #         var_select_df, 
    #         "Average Variable Selection Performance by Population", 
    #         "tab:variable-selection"
    #     )
    #     latex_output = os.path.join(args.output, "variable_selection_latex.txt")
    #     with open(latex_output, "w") as f:
    #         f.write(latex_code)
    #     print(f"Saved LaTeX table to {latex_output}")
    #     # Generate LaTeX table
    #     latex_code = dataframe_to_latex(
    #         var_select_df, 
    #         "Average Variable Selection Performance by Population", 
    #         "tab:variable-selection"
    #     )
    #     latex_output = os.path.join(args.output, "variable_selection_latex.txt")
    #     with open(latex_output, "w") as f:
    #         f.write(latex_code)
    #     print(f"Saved LaTeX table to {latex_output}")
        
    #     # Generate bar plots for variable selection metrics
    #     if not args.no_plots and 'population' in var_select_df.columns:
    #         create_bar_plots(var_select_df, args.output, "variable_selection")
    #     # Generate bar plots for variable selection metrics
    #     if not args.no_plots and 'population' in var_select_df.columns:
    #         create_bar_plots(var_select_df, args.output, "variable_selection")

if __name__ == "__main__":
    main()