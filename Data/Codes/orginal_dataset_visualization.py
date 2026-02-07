import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import math
import os
import re

# --- CONFIGURATION ---
FILE_PATH = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets\raw_metadata_samples.csv"

def generate_honest_report():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    print(f"Loading Dataset: {os.path.basename(FILE_PATH)}...")
    df = pd.read_csv(FILE_PATH)
    
    # --- STEP 1: RESCUE YEAR (But DO NOT IMPUTE) ---
    print("\n--- EXTRACTING VALID YEARS ---")
    year_col = next((col for col in df.columns if 'year' in col.lower()), None)
    
    if year_col:
        def extract_year(val):
            if pd.isna(val): return None
            # Find 4 digits
            match = re.search(r'\d{4}', str(val))
            if match:
                y = int(match.group(0))
                # Keep if reasonable year, else NaN
                return y if 1900 < y < 2030 else None
            return None

        # Apply extraction - invalid years become NaN
        df[year_col] = df[year_col].apply(extract_year)
        
        valid_count = df[year_col].notna().sum()
        print(f"Valid Years Found: {valid_count} (Rows with missing years are ignored for the graph)")

    # --- STEP 2: PREPARE OTHER COLUMNS ---
    print("\n--- PREPARING NUMERIC DATA ---")
    
    # Create a copy for analysis
    # We convert text columns to category codes, but ONLY if they aren't mostly empty
    df_encoded = df.copy()
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' or df_encoded[col].dtype == 'bool':
             # Convert to codes, preserving NaNs as -1 usually, but here we want to handle numeric
             # Standard pandas factorize or cat.codes assigns -1 to NaN. 
             # For plotting, we actually want to DROP the NaNs, so we stick to numeric conversion.
             df_encoded[col] = df_encoded[col].astype('category').cat.codes
             # Note: cat.codes turns NaN into -1. We replace -1 with NaN to ignore it in stats.
             df_encoded[col] = df_encoded[col].replace(-1, float('nan'))

    numeric_df = df_encoded.select_dtypes(include=['number'])

    # --- VISUALIZATION SETUP ---
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    def make_zipper_labels(labels):
        return [f"\n{l}" if i % 2 == 1 else l for i, l in enumerate(labels)]

    # =========================================================
    # VISUAL 1: STATISTICAL TABLE (Honest Counts)
    # =========================================================
    print("1. Generating Statistical Table...")
    # .describe() automatically ignores NaNs
    stats = numeric_df.describe().T
    
    try:
        stats['mode'] = numeric_df.mode().iloc[0]
        cols_to_keep = ['count', 'mean', 'std', 'min', '50%', 'max', 'mode']
    except:
        cols_to_keep = ['count', 'mean', 'std', 'min', '50%', 'max']

    stats_clean = stats[cols_to_keep].round(2).rename(columns={'50%': 'Median'})

    # Height
    t_height = max(5, 2 + len(stats_clean) * 0.4)
    fig_table, ax_table = plt.subplots(figsize=(14, t_height), dpi=120)
    ax_table.axis('off')
    
    the_table = ax_table.table(
        cellText=stats_clean.values, colLabels=stats_clean.columns, rowLabels=stats_clean.index, 
        loc='center', cellLoc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.8)

    for (row, col), cell in the_table.get_celld().items():
        cell.set_linewidth(1.0)
        cell.set_edgecolor('black')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('black')
        elif col == -1:
            cell.set_text_props(weight='bold', color='black', ha='right')
            cell.set_facecolor('#e0e0e0')
            cell.set_width(0.35)
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='black')
    
    plt.title(f"Statistics (Valid Data Only): {os.path.basename(FILE_PATH)}", fontsize=14, weight='bold', y=0.99)


    # =========================================================
    # VISUAL 2: BOX PLOTS (Ignoring NaNs per column)
    # =========================================================
    print("2. Generating Box Plots...")
    n_cols = 3
    n_rows = math.ceil(len(numeric_df.columns) / n_cols)
    
    fig_box, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), dpi=120)
    axes_flat = axes.flatten() if len(numeric_df.columns) > 1 else [axes]

    for i, col in enumerate(numeric_df.columns):
        # GET DATA WITHOUT NaNs for this specific column
        clean_col_data = numeric_df[col].dropna()
        
        if len(clean_col_data) > 0:
            sns.boxplot(
                y=clean_col_data, ax=axes_flat[i], color='#bdc3c7', 
                width=0.5, linewidth=1.5, fliersize=3,
                flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"}
            )
            
            # Update title to show how many items are being plotted
            clean_title = str(col).replace('_', ' ').title()
            clean_title = "\n".join(textwrap.wrap(clean_title, 20))
            count_label = f"\n(N={len(clean_col_data)})"
            
            axes_flat[i].set_title(clean_title + count_label, weight='bold', fontsize=11, pad=10)
            axes_flat[i].set_ylabel("")
            axes_flat[i].grid(True, axis='y', linestyle=':', color='gray', alpha=0.5)
            
            for spine in axes_flat[i].spines.values():
                spine.set_edgecolor('black')
        else:
            axes_flat[i].set_title(f"{col}\n(No Valid Data)", color='red')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle("Outlier Analysis (Missing Data Ignored)", fontsize=16, weight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=3.0, w_pad=2.0)


    # =========================================================
    # VISUAL 3: HEATMAP
    # =========================================================
    print("3. Generating Heatmap...")
    # corr() automatically handles NaNs pairwise
    corr = numeric_df.corr()
    
    hm_size = max(10, len(numeric_df.columns) * 0.8)
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(hm_size+2, hm_size), dpi=120)
    
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap='Greys', center=0, 
        vmin=-1, vmax=1, square=True, linewidths=1, linecolor='white',
        cbar_kws={"shrink": .7}, ax=ax_heatmap
    )
    
    ax_heatmap.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=30)
    
    staggered = make_zipper_labels(numeric_df.columns)
    ax_heatmap.set_xticklabels(staggered, rotation=0, ha='center', fontsize=10)
    ax_heatmap.set_yticklabels(numeric_df.columns, rotation=0, va='center', fontsize=10)

    print("\nDONE. 3 Windows Open.")
    plt.show()

if __name__ == "__main__":
    generate_honest_report()