import pandas as pd
import numpy as np
import os

# --- CHANGE THIS PATH FOR EACH DATASET ---
# 1. Ziyad's Data
# 2. Gemini Data
# 3. ChatGPT Data
# 4. Copilot Data
file1="\synthetic_data_gem.csv"
file2="\synthetic_data_copilot.csv"
file3="\synthetic_data_gpt.csv"
file4="\raw_metadata_samples.csv"
FILE_PATH = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"+file3

def get_comparison_data():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    df = pd.read_csv(FILE_PATH)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    print(f"--- ANALYZING: {os.path.basename(FILE_PATH)} ---")
    
    # 1. CORRELATION MATRIX
    print("\n[1] CORRELATION MATRIX:")
    print(numeric_df.corr().round(3).to_string())

    # 2. DETAILED STATISTICS (The Box Plot Data)
    print("\n[2] STATISTICAL SUMMARY (Quartiles):")
    stats = numeric_df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(stats.round(2).to_string())

    # 3. OUTLIER ANALYSIS (The Whiskers)
    print("\n[3] OUTLIER COUNTS (Data beyond 1.5x IQR):")
    outlier_summary = []
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
        outlier_summary.append((col, outliers))
    
    # Print neat table
    print(f"{'Column':<30} | {'Outlier Count'}")
    print("-" * 50)
    for name, count in outlier_summary:
        print(f"{name:<30} | {count}")

    print("\n--- END OF ANALYSIS ---")

if __name__ == "__main__":
    get_comparison_data()