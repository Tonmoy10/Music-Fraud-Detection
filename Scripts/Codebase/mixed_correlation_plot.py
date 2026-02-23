import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import os
import glob
import warnings
warnings.filterwarnings("ignore") # Suppress scipy warnings for perfect matrices

# ==========================================
# USER CONFIGURATION
# ==========================================
TARGET_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets\Ultimate_new"

# Columns to ignore (Unstructured IDs and Hashes that cannot be correlated)
UNSTRUCTURED_COLS = [
    'account_external_id', 'upload_external_id', 'content_ref', 
    'fingerprints_audio_hash', 'fingerprints_perceptual_hash', 
    'device_context_device_hash', 'metadata_title', 'metadata_album', 'display_name'
]

# Clean Presentation Labels for Matrix Axes
CLEAN_NAMES = {
    'account_type': 'Account Tier',
    'metadata_genre': 'Genre',
    'metadata_duration_seconds': 'Duration (sec)',
    'metadata_bitrate': 'Bitrate (kbps)',
    'metadata_format': 'Audio Format',
    'metadata_year': 'Upload Year',
    'profile_type': 'User Profile',
    'expected_category': 'Expected Category',
    'device_context_user_agent': 'User Agent'
}

# Distinct Solid Colors for Statistical Tests
TEST_COLORS = {
    'Spearman Rank': '#A9CCE3',      # Solid Blue
    'Point-Biserial': '#A9DFBF',     # Solid Green
    'Phi Coefficient': '#D7BDE2',    # Solid Purple
    'Cramérs V': '#F5CBA7',          # Solid Orange
    'Correlation Ratio': '#F1948A',  # Solid Red
    'Self': '#E5E7E9'                # Solid Grey
}
# Map names to integer indices for the discrete colormap
TEST_MAP = {k: i for i, k in enumerate(TEST_COLORS.keys())}
# ==========================================

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.size == 0: return 0.0
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n <= 1: return 0.0
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    min_dim = min((kcorr-1), (rcorr-1))
    return 0.0 if min_dim <= 0 else np.sqrt(phi2corr / min_dim)

def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    if cat_num <= 1: return 0.0
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures) if len(cat_measures) > 0 else 0
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    return 0.0 if denominator == 0 else np.sqrt(numerator / denominator)

def infer_data_type(series):
    """Dynamically categorizes variables."""
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() == 2:
            return 'Binary'
        elif series.nunique() > 20:
            return 'Continuous' 
        else:
            return 'Ordinal'    
    else:
        # We treat all string categories as Nominal to safely use Cramer's V / Eta
        return 'Nominal'        

def determine_test(type1, type2):
    types = {type1, type2}
    if types.issubset({'Continuous', 'Ordinal'}):
        return 'Spearman Rank'
    elif types == {'Binary'}:
        return 'Phi Coefficient'
    elif 'Binary' in types and ('Continuous' in types or 'Ordinal' in types):
        return 'Point-Biserial'
    elif 'Nominal' in types and ('Nominal' in types or 'Binary' in types):
        return 'Cramérs V'
    elif 'Nominal' in types and ('Continuous' in types or 'Ordinal' in types):
        return 'Correlation Ratio'
    return 'Unknown'

def clean_and_prepare_data(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # Drop pure unstructured identifiers
    cols_to_drop = [c for c in UNSTRUCTURED_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Safely drop high cardinality text (like unique IPs or random JSON lists) to prevent RAM freeze
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() > 50:
            df = df.drop(columns=[col])
            
    # Apply clean presentation names to columns
    df.rename(columns=CLEAN_NAMES, inplace=True)
    return df

def generate_datatype_table(df):
    """Generates Figure 1: The Data Type Classification Table. Only called once."""
    type_info = []
    for col in df.columns:
        dtype = infer_data_type(df[col])
        example = str(df[col].dropna().iloc[0])[:20]
        type_info.append([col, dtype, example])
        
    fig, ax = plt.subplots(figsize=(10, len(type_info) * 0.4 + 1))
    ax.axis('off')
    table = ax.table(cellText=type_info, colLabels=['Dataset Feature', 'Statistical Data Type', 'Example Value'], 
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d3d3d3')
        if row == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f9f9f9')
            
    plt.title("Data Architecture & Type Mapping", fontweight="bold", pad=20)
    print(">>> Displaying Data Type Table. CLOSE the window to continue...")
    plt.show()

def calculate_and_plot_mixed_correlation(df, dataset_name):
    cols = df.columns
    n = len(cols)
    
    color_matrix = np.zeros((n, n))
    annot_matrix = np.empty((n, n), dtype=object)
    method_matrix = np.empty((n, n), dtype=object) # Track methods for terminal output
    
    dtypes = {col: infer_data_type(df[col]) for col in cols}
    
    for i in range(n):
        for j in range(n):
            if i == j:
                color_matrix[i, j] = TEST_MAP['Self']
                annot_matrix[i, j] = "1.00"
                method_matrix[i, j] = "Self"
                continue
                
            c1, c2 = cols[i], cols[j]
            t1, t2 = dtypes[c1], dtypes[c2]
            method = determine_test(t1, t2)
            
            color_matrix[i, j] = TEST_MAP[method]
            method_matrix[i, j] = method
            
            mask = df[c1].notna() & df[c2].notna()
            v1, v2 = df.loc[mask, c1], df.loc[mask, c2]
            
            try:
                if method == 'Spearman Rank':
                    val, _ = ss.spearmanr(v1, v2)
                elif method == 'Phi Coefficient':
                    val, _ = ss.pearsonr(v1, v2) 
                elif method == 'Point-Biserial':
                    val, _ = ss.pointbiserialr(v1, v2) if t1 == 'Binary' else ss.pointbiserialr(v2, v1)
                elif method == 'Cramérs V':
                    val = cramers_v(v1, v2)
                elif method == 'Correlation Ratio':
                    val = correlation_ratio(v1, v2) if t1 == 'Nominal' else correlation_ratio(v2, v1)
                else:
                    val = 0.0
                annot_matrix[i, j] = f"{abs(val):.2f}"
            except Exception:
                annot_matrix[i, j] = "0.00"

    # --- TERMINAL RAW DATA OUTPUT ---
    print(f"\n{'='*70}")
    print(f"RAW CORRELATION DATA: {dataset_name}")
    print(f"{'='*70}")
    print(f"{'Feature 1':<22} | {'Feature 2':<22} | {'Value':<6} | {'Method Used'}")
    print("-" * 70)
    for i in range(n):
        for j in range(i + 1, n): # Upper triangle only to avoid duplicates
            print(f"{cols[i]:<22} | {cols[j]:<22} | {annot_matrix[i, j]:<6} | {method_matrix[i, j]}")
    print(f"{'='*70}\n")

    # --- HEATMAP VISUALIZATION ---
    fig, ax = plt.subplots(figsize=(10, 7)) # Reduced size to fit smaller screens
    
    cmap = mcolors.ListedColormap(list(TEST_COLORS.values()))
    
    sns.heatmap(color_matrix, cmap=cmap, vmin=0, vmax=len(TEST_COLORS)-1, 
                annot=annot_matrix, fmt="", cbar=False, ax=ax, 
                linewidths=1, linecolor='white',
                xticklabels=cols, yticklabels=cols,
                annot_kws={"size": 10, "weight": "bold", "color": "black"})

    # Construct Custom Legend
    legend_elements = [
        Patch(facecolor=TEST_COLORS['Spearman Rank'], edgecolor='black', label='Spearman (Cont. vs Cont.)'),
        Patch(facecolor=TEST_COLORS['Point-Biserial'], edgecolor='black', label='Point-Biserial (Cont. vs Binary)'),
        Patch(facecolor=TEST_COLORS['Phi Coefficient'], edgecolor='black', label='Phi Coefficient (Binary vs Binary)'),
        Patch(facecolor=TEST_COLORS['Cramérs V'], edgecolor='black', label="Cramér's V (Nom. vs Nom./Binary)"),
        Patch(facecolor=TEST_COLORS['Correlation Ratio'], edgecolor='black', label='Correlation Ratio (Nom. vs Cont.)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              title="Statistical Method\n(Text = Strength)", frameon=True, fontsize=9, title_fontsize=10)

    plt.title(f"Statistical Correlation Matrix\nDataset: {dataset_name}", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    print(f">>> Displaying Heatmap for {dataset_name}. CLOSE the window to proceed...")
    plt.show()

def main():
    csv_files = glob.glob(os.path.join(TARGET_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSVs found in {TARGET_DIR}")
        return

    table_drawn = False

    for file in csv_files:
        dataset_name = os.path.basename(file)
        print(f"\n--- Processing: {dataset_name} ---")
        
        df = pd.read_csv(file)
        df_clean = clean_and_prepare_data(df)
        
        # Draw the Data Type Table only for the very first dataset
        if not table_drawn:
            generate_datatype_table(df_clean)
            table_drawn = True
        
        calculate_and_plot_mixed_correlation(df_clean, dataset_name)
        
    print("\n✅ All datasets processed.")

if __name__ == "__main__":
    main()