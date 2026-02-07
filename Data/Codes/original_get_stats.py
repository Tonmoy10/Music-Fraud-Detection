import pandas as pd
import numpy as np
import os
import re

# --- CONFIGURATION ---
# POINT THIS TO ZIYAD'S ORIGINAL DATASET
FILE_PATH = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets\raw_metadata_samples.csv"

def get_robust_stats():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    print(f"--- ANALYZING: {os.path.basename(FILE_PATH)} (Robust Mode) ---")
    
    # 1. LOAD AND CLEAN
    df = pd.read_csv(FILE_PATH)
    
    # A. Rescue Year
    year_col = next((col for col in df.columns if 'year' in col.lower()), None)
    if year_col:
        def extract_year(val):
            if pd.isna(val): return None
            match = re.search(r'\d{4}', str(val))
            if match:
                y = int(match.group(0))
                return y if 1900 < y < 2030 else None
            return None
        df[year_col] = df[year_col].apply(extract_year)

    # B. Force Numeric Conversion (for everything else)
    # We create a new dataframe with only the numeric parts
    numeric_df = pd.DataFrame()
    for col in df.columns:
        # Check if it looks numeric-ish (not a name/ID)
        # We assume if it converts to float easily, we keep it.
        # We skip obvious text columns like 'title', 'artist' unless encoded.
        # For this statistical summary, we want RAW numbers (Year, Duration, Bitrate)
        
        # Try converting
        converted = pd.to_numeric(df[col], errors='coerce')
        
        # If >20% of the column is valid numbers, we keep it
        if converted.notna().sum() / len(df) > 0.2:
            numeric_df[col] = converted

    # 2. CORRELATION MATRIX (Ignors NaNs)
    print("\n[1] CORRELATION MATRIX (Pearson):")
    # Only show if we have enough matching rows
    print(numeric_df.corr().round(3).to_string())

    # 3. DETAILED STATISTICS
    print("\n[2] STATISTICAL SUMMARY (Valid Data Only):")
    stats = numeric_df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(stats.round(2).to_string())

    # 4. OUTLIER ANALYSIS
    print("\n[3] OUTLIER COUNTS (Data beyond 1.5x IQR):")
    outlier_summary = []
    for col in numeric_df.columns:
        # Drop NaNs for calculation
        valid_data = numeric_df[col].dropna()
        if len(valid_data) == 0:
            continue
            
        Q1 = valid_data.quantile(0.25)
        Q3 = valid_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((valid_data < lower_bound) | (valid_data > upper_bound)).sum()
        outlier_summary.append((col, outliers))
    
    print(f"{'Column':<30} | {'Outlier Count'}")
    print("-" * 50)
    for name, count in outlier_summary:
        print(f"{name:<30} | {count}")

    print("\n--- END OF ANALYSIS ---")

if __name__ == "__main__":
    get_robust_stats()