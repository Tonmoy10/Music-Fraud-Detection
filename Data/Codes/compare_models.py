import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURATION ---
# Base folder where your CSVs live
BASE_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"

# MAPPING: Display Name -> Actual Filename
DATASET_CONFIG = {
    'Claude (Baseline)': 'raw_metadata_samples.csv',  # <--- Kept Original Name
    'ChatGPT': 'synthetic_data_gpt.csv',
    'Gemini': 'synthetic_data_gem.csv',
    'Copilot': 'synthetic_data_copilot.csv'
}

def get_file_path(filename):
    return os.path.join(BASE_DIR, filename)

def calculate_metrics(name, filename):
    file_path = get_file_path(filename)
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: File not found: {filename}")
        return None

    print(f"   -> Analyzing {name}...")
    try:
        df = pd.read_csv(file_path)
        
        # 1. HALLUCINATION TEST (Correlation Bias)
        # Force numeric, ignoring errors (Text -> NaN)
        dur = pd.to_numeric(df['metadata_duration_seconds'], errors='coerce')
        bit = pd.to_numeric(df['metadata_bitrate'], errors='coerce')
        
        # Create a clean dataframe for correlation (drop NaNs)
        valid_corr_data = pd.DataFrame({'d': dur, 'b': bit}).dropna()
        
        if len(valid_corr_data) < 10:
            correlation = 0.0 
        else:
            correlation = valid_corr_data['d'].corr(valid_corr_data['b'])
        
        # 2. RELIABILITY TEST (Completeness of Year)
        year_col = next((c for c in df.columns if 'year' in c.lower()), None)
        if year_col:
            # Check if value is valid number (not NaN, not text)
            valid_years = pd.to_numeric(df[year_col], errors='coerce').notna().sum()
            completeness = (valid_years / len(df)) * 100
        else:
            completeness = 0.0

        # 3. CHAOS TEST (Outliers)
        dur_clean = dur.dropna()
        if dur_clean.empty:
            outlier_count = 0
        else:
            Q1 = dur_clean.quantile(0.25)
            Q3 = dur_clean.quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((dur_clean < (Q1 - 1.5 * IQR)) | (dur_clean > (Q3 + 1.5 * IQR))).sum()

        return {
            'Model': name,
            'Correlation_Bias': abs(correlation), 
            'Metadata_Completeness': completeness,
            'Outlier_Count': outlier_count
        }

    except Exception as e:
        print(f"   [!] Error calculating metrics for {name}: {e}")
        return None

def generate_final_benchmark():
    print("--- STARTING BENCHMARK ---")
    results = []
    
    for display_name, filename in DATASET_CONFIG.items():
        metric = calculate_metrics(display_name, filename)
        if metric:
            results.append(metric)

    if not results:
        print("No results generated. Exiting.")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # --- PLOTTING ---
    plt.figure(figsize=(18, 6))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # Subplot 1: Correlation
    ax1 = plt.subplot(1, 3, 1)
    sns.barplot(data=df_results, x='Model', y='Correlation_Bias', palette='viridis', ax=ax1)
    ax1.set_title('Correlation Bias\n(Lower is Better)', weight='bold')
    ax1.set_ylabel('Abs. Correlation (Dur vs Bit)')
    ax1.axhline(0.05, color='red', linestyle='--', alpha=0.5, label='Realistic Threshold')
    for container in ax1.containers: ax1.bar_label(container, fmt='%.2f')

    # Subplot 2: Completeness
    ax2 = plt.subplot(1, 3, 2)
    sns.barplot(data=df_results, x='Model', y='Metadata_Completeness', palette='Reds_d', ax=ax2)
    ax2.set_title('Data Completeness\n(Higher is Better)', weight='bold')
    ax2.set_ylabel('Valid Year Data (%)')
    ax2.set_ylim(0, 115) 
    for container in ax2.containers: ax2.bar_label(container, fmt='%.1f%%')

    # Subplot 3: Outliers
    ax3 = plt.subplot(1, 3, 3)
    sns.barplot(data=df_results, x='Model', y='Outlier_Count', palette='Blues_d', ax=ax3)
    ax3.set_title('Anomaly Count\n(Proxy for Entropy)', weight='bold')
    ax3.set_ylabel('Outliers (Duration)')
    for container in ax3.containers: ax3.bar_label(container, fmt='%.0f')

    plt.suptitle('Benchmarking Synthetic Fraud Data Generation Models', fontsize=16, weight='bold', y=1.05)
    plt.tight_layout()

    # Save
    save_path = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Results\model_benchmark_comparison.png"
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSUCCESS: Comparison figure saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_final_benchmark()