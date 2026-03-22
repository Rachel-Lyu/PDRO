import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import os

def plot_density(estimator, data, plot_type='pdf', heatmap=False):
    """
    Plots the PDF or CDF of a random variable using a density estimator.
    
    Parameters:
    - estimator: A density estimator object with a `fit` method and a `score_samples` method.
    - data: The data to fit the estimator to.
    - plot_type: 'pdf' or 'cdf' to specify the type of plot.
    - heatmap: Boolean, if True and data is 2D, plot a heatmap.
    """
    estimator.fit(data)
    if data.shape[1] == 1:
        x = np.linspace(np.min(data), np.max(data), 1000)
        y = estimator.score_samples(x[:, np.newaxis])
        
        if plot_type == 'cdf':
            y = np.cumsum(np.exp(y)) * (x[1] - x[0])
        
        plt.plot(x, y)
        plt.title(f'{plot_type.upper()} of the random variable')
        plt.xlabel('Value')
        plt.ylabel(plot_type.upper())
        plt.show()
    
    elif data.shape[1] == 2 and heatmap:
        x = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
        y = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 100)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(np.exp(estimator.score_samples(positions.T)), X.shape)
        
        plt.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[x.min(), x.max(), y.min(), y.max()])
        plt.title(f'{plot_type.upper()} Heatmap of the random variable')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.show()
    
    else:
        fig, axes = plt.subplots(data.shape[1], 1, figsize=(8, 6))
        for i in range(data.shape[1]):
            x = np.linspace(np.min(data[:, i]), np.max(data[:, i]), 1000)
            y = estimator.score_samples(x[:, np.newaxis])
            
            if plot_type == 'cdf':
                y = np.cumsum(np.exp(y)) * (x[1] - x[0])
            
            axes[i].plot(x, y)
            axes[i].set_title(f'{plot_type.upper()} of dimension {i+1}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel(plot_type.upper())
        
        plt.tight_layout()
        plt.show()



def compute_correlation_matrix(X):
    """
    Computes the correlation matrix of the feature matrix X.
    
    Parameters:
    - X: numpy array of shape (n_samples, n_features)
    
    Returns:
    - corr_matrix: Correlation matrix of features (n_features, n_features)
    """
    corr_matrix = np.corrcoef(X.T)  # Compute correlation matrix of features
    return corr_matrix


def get_top_m1_indices(alpha, m1):
    """
    Extracts the indices of the top m1 features based on the learned alpha values.
    
    Parameters:
    - alpha: The learned alpha values (numpy array or tensor)
    - m1: The number of real features
    
    Returns:
    - top_m1_indices: Indices of the top m1 features
    """
    alpha_values = alpha.detach().numpy()  # Convert alpha to numpy if it's a tensor
    top_m1_indices = np.argsort(np.abs(alpha_values), axis=0)[-m1:]  # Indices of top m1 features
    return top_m1_indices


def plot_correlation_matrix(corr_matrix, save_path=None):
    """
    Plots the correlation matrix of all features.
    
    Parameters:
    - corr_matrix: Correlation matrix to plot
    - save_path: Optional file path to save the plot (if None, it will display the plot)
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title("Correlation Matrix of All Features")
    plt.xticks(np.arange(corr_matrix.shape[0]), labels=np.arange(1, corr_matrix.shape[0] + 1))
    plt.yticks(np.arange(corr_matrix.shape[0]), labels=np.arange(1, corr_matrix.shape[0] + 1))
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_alpha_correlation_with_real(alpha, top_m1_indices, corr_matrix, m1, save_path=None):
    """
    Plots the correlation of alpha-selected features with the real features.
    
    Parameters:
    - alpha: The learned alpha values
    - top_m1_indices: Indices of the top m1 selected features
    - corr_matrix: Correlation matrix of features
    - m1: Number of real features
    - save_path: Optional file path to save the plot (if None, it will display the plot)
    """
    alpha_corr_with_real = np.zeros(m1)
    max_correlation_idx = np.zeros(m1)

    for i, idx in enumerate(top_m1_indices):
        # Calculate correlation of each alpha-selected feature with the real features
        alpha_corr_with_real[i] = np.max(np.abs(corr_matrix[idx, :m1]))
        max_correlation_idx[i] = np.argmax(np.abs(corr_matrix[idx, :m1]))

    plt.figure(figsize=(10, 6))
    plt.bar(range(m1), alpha_corr_with_real)
    for i, v in enumerate(max_correlation_idx):
        plt.text(i, alpha_corr_with_real[i], f'{int(v)}', ha='center', va='bottom')
    plt.xlabel("Top m1 Selected Features (from alpha)")
    plt.ylabel("Correlation with Real Features")
    plt.title("Correlation of alpha-Selected Features with Real Features")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def analyze_alpha_correlation(X, alpha, m1, save_path=None):
    """
    Analyzes the correlation of alpha-selected features with the real features.
    Computes the correlation matrix and visualizes the results.
    
    Parameters:
    - X: Feature matrix (n_samples, n_features)
    - alpha: The learned alpha values (torch tensor or numpy array)
    - m1: Number of real features (to compare with)
    - save_path: Optional file path to save the plots (if None, it will display the plot)
    """
    # Step 1: Compute the correlation matrix for all features
    corr_matrix = compute_correlation_matrix(X)
    
    # Step 2: Get the top m1 feature indices based on alpha values
    top_m1_indices = get_top_m1_indices(alpha, m1)

    # Step 3: Plot and optionally save the correlation matrix of all features
    plot_correlation_matrix(corr_matrix, save_path=f"{save_path}_correlation_matrix.png" if save_path else None)
    
    # Step 4: Plot and optionally save the correlation of alpha-selected features with the real features
    plot_alpha_correlation_with_real(alpha, top_m1_indices, corr_matrix, m1, save_path=f"{save_path}_alpha_real_correlation.png" if save_path else None)

def plot_success_rate(csv_path='aggregated_results.csv'):
    # Read the results
    df = pd.read_csv(csv_path)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot success rate vs dataset size (using log scale for x-axis)
    plt.semilogx(df['dataset_size'], df['true_alpha_success_rate'], 
                 marker='o', linewidth=2, markersize=8)
    
    # Customize the plot
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Dataset Size', fontsize=12)
    plt.ylabel('True Alpha Success Rate', fontsize=12)
    plt.title('Success Rate vs Dataset Size', fontsize=14)
    
    # Add value annotations above each point
    for x, y in zip(df['dataset_size'], df['true_alpha_success_rate']):
        plt.annotate(f'{y:.2f}', 
                    (x, y), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('success_rate_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_vs_dataset_size(df, save_path=None):
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # For each dataset size, get best, worst, and average performance
    dataset_sizes = sorted(df['dataset_size'].unique())
    
    # Arrays to store metrics
    true_alpha_best = []
    true_alpha_worst = []
    true_alpha_avg = []
    exclusive_alpha_best = []
    exclusive_alpha_worst = []
    exclusive_alpha_avg = []
    
    for size in dataset_sizes:
        size_data = df[df['dataset_size'] == size]
        
        # True alpha success rate
        true_rates = size_data['true_alpha_success_rate']
        true_alpha_best.append(np.max(true_rates))
        true_alpha_worst.append(np.min(true_rates))
        true_alpha_avg.append(np.mean(true_rates))
        
        # Exclusive alpha success rate
        exclusive_rates = size_data['exclusive_alpha_success_rate']
        exclusive_alpha_best.append(np.max(exclusive_rates))
        exclusive_alpha_worst.append(np.min(exclusive_rates))
        exclusive_alpha_avg.append(np.mean(exclusive_rates))
    
    # Plot True Alpha Success Rate
    ax1.semilogx(dataset_sizes, true_alpha_best, 'g-', marker='o', label='Best')
    ax1.semilogx(dataset_sizes, true_alpha_avg, 'b-', marker='s', label='Average')
    ax1.semilogx(dataset_sizes, true_alpha_worst, 'r-', marker='v', label='Worst')
    
    ax1.set_xlabel('Dataset Size (log scale)')
    ax1.set_ylabel('True Alpha Success Rate')
    ax1.set_title('True Alpha Success Rate vs Dataset Size')
    ax1.grid(True, alpha=0.2)
    ax1.legend()
    
    # Plot Exclusive Alpha Success Rate
    ax2.semilogx(dataset_sizes, exclusive_alpha_best, 'g-', marker='o', label='Best')
    ax2.semilogx(dataset_sizes, exclusive_alpha_avg, 'b-', marker='s', label='Average')
    ax2.semilogx(dataset_sizes, exclusive_alpha_worst, 'r-', marker='v', label='Worst')
    
    ax2.set_xlabel('Dataset Size (log scale)')
    ax2.set_ylabel('Exclusive Alpha Success Rate')
    ax2.set_title('Exclusive Alpha Success Rate vs Dataset Size')
    ax2.grid(True, alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('performance_vs_dataset_size.png', dpi=300, bbox_inches='tight')
    # plt.show()

def plot_variable_importance(results_list, dataset_sizes, optimizer_type, save_path='./results/gradient_descent_diagnostics/'):
    """
    Plot the evolution of both meaningful and non-meaningful variables across dataset sizes.
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing experiment results
    dataset_sizes : list
        List of dataset sizes used in experiments
    optimizer_type : str
        Type of optimizer used ('adam' or 'sgd')
    save_path : str
        Base path for saving the plot
    """
    meaningful_weights = []
    meaningful_stds = []
    other_weights = []
    other_stds = []
    
    for result in results_list:
        meaningful_indices = result['true_variable_index']
        final_alpha = result['final_alpha']
        
        # Create mask for non-meaningful indices
        all_indices = np.arange(len(final_alpha))
        other_indices = np.setdiff1d(all_indices, meaningful_indices)
        
        # Get weights for meaningful and other variables
        meaningful_vars = final_alpha[meaningful_indices]
        other_vars = final_alpha[other_indices]
        
        # Store means and stds for both groups
        meaningful_weights.append(np.mean(meaningful_vars))
        meaningful_stds.append(np.std(meaningful_vars))
        other_weights.append(np.mean(other_vars))
        other_stds.append(np.std(other_vars))

    plt.figure(figsize=(8, 6))
    
    # Plot meaningful variables
    plt.errorbar(dataset_sizes, meaningful_weights, yerr=meaningful_stds, 
                fmt='bo-', capsize=5, label='Meaningful Variables (Mean ± Std)', 
                zorder=2)
                
    # Plot other variables with transparency
    plt.errorbar(dataset_sizes, other_weights, yerr=other_stds, 
                fmt='o-', capsize=5, color='gray', alpha=0.3, 
                label='Other Variables (Mean ± Std)', 
                zorder=1)
    
    plt.xscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Average Noise (α)')
    plt.grid(True)
    plt.title(f'Average Noise of Variables vs Dataset Size\n({len(meaningful_indices)} Meaningful Variables)')
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path + f'{optimizer_type}/'), exist_ok=True)
    plt.savefig(save_path + f'{optimizer_type}/meaningful_weight_vs_size.png')
    plt.close()

def plot_population_covariances(pop_data, save_path, meaningful_indices):
    """Plot covariance matrices for all populations in a single figure."""
    n_pops = len(pop_data)
    fig, axes = plt.subplots(1, n_pops, figsize=(6*n_pops, 5))
    if n_pops == 1:
        axes = [axes]
    
    vmin, vmax = float('inf'), float('-inf')
    # First pass to get common scale
    for pop in pop_data:
        X = pop['X'].cpu().numpy()
        cov = np.corrcoef(X.T)  # Using correlation instead of covariance for better visualization
        vmin = min(vmin, np.min(cov))
        vmax = max(vmax, np.max(cov))
    
    for idx, (ax, pop) in enumerate(zip(axes, pop_data)):
        X = pop['X'].cpu().numpy()
        cov = np.corrcoef(X.T)
        
        # Create mask for meaningful features
        meaningful = np.zeros(X.shape[1], dtype=bool)
        meaningful[pop['meaningful_indices']] = True
        
        im = ax.imshow(cov, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add grid to separate features
        ax.set_xticks(np.arange(-.5, X.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, X.shape[1], 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        
        # Highlight meaningful features
        for i in pop['meaningful_indices']:
            ax.add_patch(plt.Rectangle((i-0.5, -0.5), 1, X.shape[1], 
                                     fill=False, color='green', linewidth=2))
            ax.add_patch(plt.Rectangle((-0.5, i-0.5), X.shape[1], 1, 
                                     fill=False, color='green', linewidth=2))
        
        ax.set_title(f'Population {pop["pop_id"]}\nDataset: {pop.get("dataset_type", "Unknown")}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, label='Correlation', orientation='horizontal', pad=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_correlations.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def visualise_dataset(X, y, title, save_path=None, show=False, caption=None, log_scale=False):
    """
    Visualize a dataset of multidimensional tensors.

    Parameters:
    X (numpy.ndarray): The input dataset, a 1D array.
    y (numpy.ndarray): The target dataset, a 1D array.
    """
    if log_scale:
        y = np.log(y)
    # check for numpy
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().numpy()
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    if caption is not None:
        ax.text(0.5, 0.5, caption, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close('all')