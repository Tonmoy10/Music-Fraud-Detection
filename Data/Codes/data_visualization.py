import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

# --- CONFIGURATION ---
file1="\synthetic_data_gem.csv"
file2="\synthetic_data_copilot.csv"
file3="\synthetic_data_gpt.csv"
file4="\raw_metadata_samples.csv"
FILE_PATH = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"+file4

def generate_final_polished_report():
    if not os.path.exists(FILE_PATH):
        print(f"Error: File not found at {FILE_PATH}")
        return

    print("Loading Dataset...")
    df = pd.read_csv(FILE_PATH)
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # --- THEME: STRICT MINIMALISM ---
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    # =========================================================
    # VISUAL 1: STATISTICAL TABLE (Title Lowered)
    # =========================================================
    print("1. Generating Monochrome Table...")
    stats = numeric_df.describe().T
    stats['mode'] = numeric_df.mode().iloc[0]
    cols_to_keep = ['count', 'mean', 'std', 'min', '50%', 'max', 'mode']
    stats_clean = stats[cols_to_keep].round(2).rename(columns={'50%': 'Median'})

    fig_table, ax_table = plt.subplots(figsize=(14, 6), dpi=120)
    ax_table.axis('off')
    
    the_table = ax_table.table(
        cellText=stats_clean.values, 
        colLabels=stats_clean.columns, 
        rowLabels=stats_clean.index, 
        loc='center', 
        cellLoc='center'
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
            cell.set_width(0.3)
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='black')
    
    # FIX: y=0.78 brings the title much closer to the top of the table
    plt.title("Descriptive Statistics Summary", fontsize=14, weight='bold', y=0.78)


    # =========================================================
    # VISUAL 2: BOX PLOTS (Filled Grey & Perfect Spacing)
    # =========================================================
    print("2. Generating Filled Box Plots...")
    cols = numeric_df.columns
    n_cols = 3
    n_rows = math.ceil(len(cols) / n_cols)
    
    fig_box, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows), dpi=120)
    axes_flat = axes.flatten() if len(cols) > 1 else [axes]

    for i, col in enumerate(cols):
        sns.boxplot(
            y=df[col], 
            ax=axes_flat[i], 
            color='#bdc3c7',  # Solid Light Grey Fill
            width=0.5,
            linewidth=1.5,
            fliersize=3,
            flierprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"}
        )
        
        clean_title = col.replace('_', ' ').title()
        axes_flat[i].set_title(clean_title, weight='bold', fontsize=12, pad=10)
        axes_flat[i].set_ylabel("Value")
        axes_flat[i].grid(True, axis='y', linestyle=':', color='gray', alpha=0.5)
        
        for spine in axes_flat[i].spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle("Outlier Analysis: Numerical Features", fontsize=16, weight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.0, w_pad=3.0)


    # =========================================================
    # VISUAL 3: HEATMAP (One Line Labels)
    # =========================================================
    print("3. Generating Grey-Scale Heatmap...")
    corr = numeric_df.corr()
    
    # Widen figure slightly to 14 inches to ensure horizontal labels fit
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 10), dpi=120)
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap='Greys', 
        center=0, 
        vmin=-1, vmax=1, 
        square=True, 
        linewidths=1, 
        linecolor='white',
        cbar_kws={"shrink": .7}, 
        ax=ax_heatmap
    )
    
    ax_heatmap.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)

    # FIX: No wrapping. Simple horizontal labels.
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.yticks(rotation=0, va='center', fontsize=10)

    print("\nDONE. 3 Windows Open.")
    plt.show()

if __name__ == "__main__":
    generate_final_polished_report()