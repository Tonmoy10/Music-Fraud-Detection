import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
import math

# --- CONFIGURATION ---
BASE_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"

# Analysis Menu
DATASET_CONFIG = {
    'Claude (Baseline)': 'raw_metadata_samples.csv',
    'ChatGPT': 'synthetic_data_gpt.csv',
    'Gemini': 'synthetic_data_gem.csv',
    'Copilot': 'synthetic_data_copilot.csv'
}

def get_best_distribution(data):
    """
    Tests Normal, Log-Normal, and Exponential distributions.
    Returns the name of the one with the lowest error (SSE).
    """
    dist_names = ['norm', 'lognorm', 'expon']
    best_dist = None
    best_sse = np.inf
    best_params = {}

    try:
        y, x = np.histogram(data, bins=50, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        for name in dist_names:
            dist = getattr(stats, name)
            try:
                params = dist.fit(data)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                if sse < best_sse:
                    best_sse = sse
                    best_dist = name
                    best_params = params
            except Exception:
                continue
    except Exception:
        return 'norm', {} # Fallback

    return best_dist, best_params

def analyze_dataset(name, filename):
    file_path = os.path.join(BASE_DIR, filename)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {filename}")
        return

    print(f"\n{'='*60}")
    print(f"ANALYZING: {name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include=[np.number])
    target_cols = numeric_df.columns.tolist()
    
    if not target_cols:
        print("No numeric columns found!")
        return

    # Setup Figure
    num_plots = len(target_cols)
    cols = 2
    rows = math.ceil(num_plots / cols) if num_plots > 1 else 1
    
    # Increase height to give everything room
    fig, axes = plt.subplots(rows, cols, figsize=(15, 7 * rows))
    
    # --- FIX 1: Push the Main Title way up ---
    fig.suptitle(f"Distribution Analysis: {name}", fontsize=18, weight='bold', y=0.98)
    
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(target_cols):
        ax = axes[i]
        data = numeric_df[col].dropna()
        if len(data) == 0: continue

        unique_count = data.nunique()
        
        # --- DISCRETE (Bar Chart) ---
        if unique_count < 30:
            print(f"[{col}] -> Discrete (Bar Chart)")
            value_counts = data.value_counts().sort_index()
            x_values = value_counts.index.astype(str)
            y_values = value_counts.values
            
            ax.bar(x_values, y_values, color='#404040', alpha=0.7, edgecolor='black')
            ax.set_title(f"{col}\n(Discrete Distribution)", fontsize=11, weight='bold', pad=15)
            ax.set_ylabel("Count")
            
            # Smart Tick Skipping
            if len(x_values) > 10:
                for index, label in enumerate(ax.xaxis.get_ticklabels()):
                    if index % 2 != 0: label.set_visible(False)
            
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

        # --- CONTINUOUS (Curve Fit) ---
        else:
            best_name, best_params = get_best_distribution(data)
            print(f"[{col}] -> Continuous (Fit: {best_name})")
            
            sns.histplot(data, kde=False, stat="density", color='#404040', 
                         element="step", alpha=0.2, ax=ax, label='Actual Data')
            
            if best_name:
                xmin, xmax = ax.get_xlim()
                x_line = np.linspace(xmin, xmax, 100)
                dist = getattr(stats, best_name)
                arg = best_params[:-2]
                loc = best_params[-2]
                scale = best_params[-1]
                p = dist.pdf(x_line, loc=loc, scale=scale, *arg)
                ax.plot(x_line, p, color='black', linewidth=2.5, label=f'Fit: {best_name}')
            
            ax.set_title(f"{col}\nBest Fit: {str(best_name).upper()}", fontsize=11, weight='bold', pad=15)
            ax.legend(frameon=False)

        sns.despine(ax=ax)
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)

    # Hide empty subplots
    if num_plots > 1:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    # --- FIX 2: Create "Buffer Zone" ---
    # top=0.88 means graphs start at 88% height (leaving 12% for title)
    # hspace=0.6 pushes rows apart vertically
    plt.subplots_adjust(top=0.88, bottom=0.1, hspace=0.6, wspace=0.3)
    
    print(f"-> Showing plot for {name}. Close window to continue...")
    plt.show()

if __name__ == "__main__":
    for name, filename in DATASET_CONFIG.items():
        analyze_dataset(name, filename)
    print("\nAll datasets analyzed.")