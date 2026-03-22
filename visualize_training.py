import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_checkpoint(checkpoint_path):
    """Load a checkpoint file containing our method results"""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)

def plot_training_metrics(results, save_dir):
    """Plot training metrics from our method results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    objective_history = results.get('objective_history', [])
    param_history = results.get('param_history', [])
    
    # Convert to numpy arrays for easier handling
    epochs = np.arange(len(objective_history))
    
    # 1. Plot objective vs epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, objective_history, 'b-', linewidth=2)
    plt.title('Objective Value vs Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Objective Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'objective_vs_epoch.png'), dpi=300)
    plt.close()
    
    # 2. Plot parameter (alpha) trajectories
    if param_history and len(param_history) > 0:
        param_array = np.array(param_history)
        
        # Check if lengths match - if param history is longer, align with epochs
        if len(param_array) > len(epochs):
            print(f"Note: Parameter history ({len(param_array)}) longer than epochs ({len(epochs)})")
            print("Aligning parameter history with epochs (discarding initial values)")
            # If param history is one longer (initial + per epoch), use all but first entry
            if len(param_array) == len(epochs) + 1:
                param_array = param_array[1:]  # Skip initial parameters
            else:
                # Otherwise trim to match epoch length
                param_array = param_array[-len(epochs):]
        
        plt.figure(figsize=(12, 8))
        
        # Plot up to 10 most variable params for clarity
        if param_array.shape[1] > 10:
            # Calculate variance of each parameter
            param_vars = np.var(param_array, axis=0)
            # Get indices of 10 most variable parameters
            top_indices = np.argsort(param_vars)[-10:]
            # Plot these parameters
            for idx in top_indices:
                plt.plot(epochs, param_array[:, idx], label=f'Param {idx}')
        else:
            # Plot all parameters if fewer than 10
            for i in range(param_array.shape[1]):
                plt.plot(epochs, param_array[:, i], label=f'Param {i}')
        
        plt.title('Parameter Values vs Epoch', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Parameter Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'params_vs_epoch.png'), dpi=300)
        plt.close()
    
    # 3. Calculate and plot gradient magnitudes (if available)
    if 'gradient_history' in results and results['gradient_history']:
        gradient_history = results['gradient_history']
        
        # Handle length mismatch - gradients often have one fewer entry than epochs/params
        grad_epochs = epochs
        if len(gradient_history) < len(epochs):
            grad_epochs = epochs[:len(gradient_history)]
        elif len(gradient_history) > len(epochs):
            gradient_history = gradient_history[:len(epochs)]
            
        gradient_norms = [np.linalg.norm(g) for g in gradient_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(grad_epochs, gradient_norms, 'r-', linewidth=2)
        plt.title('Gradient Norm vs Epoch', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Gradient Norm', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gradient_vs_epoch.png'), dpi=300)
        plt.close()

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
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    if not available_metrics:
        print("No metrics found to plot")
        return
    
    # Set a clean, minimal style
    plt.style.use('ggplot')
    
    # Vibrant color palette for methods (distinct colors)
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
        ax.set_yticklabels([f'Population {pop}' for pop in populations], fontsize=10)
        
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
        plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}_multi_series.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Multi-series horizontal bar plots saved to {output_dir}")



def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--save_dir', type=str, default=None, 
                       help='Directory to save plots (defaults to checkpoint directory)')
    
    args = parser.parse_args()
    
    # Default save directory is same as checkpoint directory
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.checkpoint_path)
    
    # Load results
    try:
        results = load_checkpoint(args.checkpoint_path)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Plot metrics
    plot_training_metrics(results, args.save_dir)
    print(f"Plots saved to {args.save_dir}")

if __name__ == "__main__":
    main()