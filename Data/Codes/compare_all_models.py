import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"

DATASET_CONFIG = {
    'Claude (Baseline)': 'raw_metadata_samples.csv',
    'ChatGPT': 'synthetic_data_gpt.csv',
    'Gemini': 'synthetic_data_gem.csv',
    'Copilot': 'synthetic_data_copilot.csv'
}

COLORS = {
    'Claude (Baseline)': '#999999',  # Grey
    'ChatGPT': '#e41a1c',            # Red
    'Gemini': '#377eb8',             # Blue
    'Copilot': '#4daf4a'             # Green
}

def compare_models():
    print(f"--- GENERATING COMPARATIVE DASHBOARD ---")
    
    data_store = {}
    for name, filename in DATASET_CONFIG.items():
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                data_store[name] = df
                print(f"Loaded: {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Missing: {filename}")

    if not data_store:
        return

    # 1. Setup Figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    
    # --- ADJUSTMENT: Removed Title & Optimized Top Margin ---
    # fig.suptitle("Model Benchmark...", ...)  <-- REMOVED
    plt.subplots_adjust(hspace=0.6, top=0.98, bottom=0.08)

    # --- PLOT 1: DURATION ---
    ax = axes[0]
    target_col = 'metadata_duration_seconds'
    for name, df in data_store.items():
        if target_col in df.columns:
            sns.kdeplot(data=df[target_col].dropna(), ax=ax, 
                        label=name, color=COLORS[name], linewidth=2.5, cut=0)

    ax.set_title("1. Duration Distribution (Physics Check)", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Seconds", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 600)

    # --- PLOT 2: BITRATE ---
    ax = axes[1]
    target_col = 'metadata_bitrate'
    for name, df in data_store.items():
        if target_col in df.columns:
            counts = df[target_col].value_counts(normalize=True).sort_index()
            ax.plot(counts.index, counts.values, label=name, color=COLORS[name], marker='o', linewidth=2)

    ax.set_title("2. Bitrate Preferences (Quality Check)", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- PLOT 3: YEAR ---
    ax = axes[2]
    target_col = 'metadata_year'
    for name, df in data_store.items():
        if target_col in df.columns:
            clean_years = df[target_col].dropna()
            clean_years = clean_years[(clean_years > 1990) & (clean_years < 2027)]
            sns.kdeplot(data=clean_years, ax=ax, 
                        label=name, color=COLORS[name], linewidth=2.5, bw_adjust=1.5)

    ax.set_title("3. Content Timeline (Velocity Check)", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2010, 2026)

    print("-> Displaying Comparison Plot...")
    plt.show()

if __name__ == "__main__":
    compare_models()