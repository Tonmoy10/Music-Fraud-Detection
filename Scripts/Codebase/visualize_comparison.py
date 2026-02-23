import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import mplcursors # pip install mplcursors
from matplotlib.ticker import FixedFormatter

# ==========================================
# USER CONFIGURATION
# ==========================================
TARGET_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets\Ultimate_new"
# ==========================================

print(f"--- STARTING VISUALIZATION ENGINE ---")
print(f"Scanning: {TARGET_DIR}")

# 1. LOAD & MERGE DATASETS
csv_files = glob.glob(os.path.join(TARGET_DIR, "*.csv"))
if not csv_files:
    print(f"ERROR: No .csv files found in {TARGET_DIR}")
    exit()

all_data = []
stats_summary = []

# Universal Label Map (Now handles 1/0, True/False, and various text formats)
LABEL_MAP = {
    'legit': 'Legit', 'legitimate': 'Legit', 'allow': 'Legit', 'normal_user': 'Legit', 'normal': 'Legit', 
    '0': 'Legit', '0.0': 'Legit', 'false': 'Legit', 'human': 'Legit',
    'fraud': 'Fraud', 'bot_farm': 'Fraud', 'hacked_account': 'Fraud', 'impersonation': 'Fraud', 
    'bot': 'Fraud', 'reject': 'Fraud', '1': 'Fraud', '1.0': 'Fraud', 'true': 'Fraud', 'fake': 'Fraud'
}

# Common names for the target column
LABEL_CANDIDATES = ['expected_category', 'profile_type', 'label', 'is_fraud', 'target', 'class', 'bot_status', 'type']

for f in csv_files:
    try:
        name = os.path.basename(f)
        df = pd.read_csv(f)
        
        # 1. Standardize Column Names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        df['Dataset'] = name
        
        # 2. UNIVERSAL CASE INSENSITIVITY CLEANING
        for col in df.select_dtypes(include=['object']).columns:
            if col not in ['dataset', 'account_external_id', 'upload_external_id']: 
                df[col] = df[col].astype(str).str.strip().str.lower()
                
                if col == 'metadata_format':
                    df[col] = df[col].str.upper() # mp3 -> MP3
                elif col == 'metadata_genre':
                    df[col] = df[col].str.title() # pop -> Pop

        # 3. ROBUST LABEL LOGIC
        df['Label'] = 'Unknown'
        found_label_col = False
        
        for col in LABEL_CANDIDATES:
            if col in df.columns:
                clean_col = df[col].astype(str).str.lower().str.strip()
                df['Label'] = clean_col.map(lambda x: LABEL_MAP.get(x, 'Unknown'))
                
                # Fallback for partial matches
                mask_unknown = df['Label'] == 'Unknown'
                if mask_unknown.any():
                    df.loc[mask_unknown, 'Label'] = df.loc[mask_unknown, col].astype(str).str.lower().apply(
                        lambda x: 'Legit' if any(word in x for word in ['normal', 'human', '0', 'false']) 
                        else ('Fraud' if any(word in x for word in ['bot', 'hack', 'fraud', '1', 'true']) else 'Unknown')
                    )
                
                # If we successfully labeled at least some rows, stop searching for label columns
                if (df['Label'] != 'Unknown').any():
                    found_label_col = True
                    break

        if not found_label_col:
            print(f"WARNING: Could not find a recognizable label column in {name}. Rows will be dropped from graphs.")

        all_data.append(df)
        
        # --- CALCULATE STATS FOR TABLE (Calculated before 'Unknown' drop) ---
        stats = {
            'Dataset': name,
            'Total Rows': len(df),
            'Legit': len(df[df['Label'] == 'Legit']),
            'Fraud': len(df[df['Label'] == 'Fraud']),
            'Dur Mean': round(df['metadata_duration_seconds'].mean(), 1) if 'metadata_duration_seconds' in df.columns else 0,
            'Dur Std': round(df['metadata_duration_seconds'].std(), 1) if 'metadata_duration_seconds' in df.columns else 0,
            'Bitrate Mean': round(df['metadata_bitrate'].mean(), 0) if 'metadata_bitrate' in df.columns else 0,
            'Mode Bitrate': df['metadata_bitrate'].mode()[0] if 'metadata_bitrate' in df.columns else 0,
            'Year Range': f"{df['metadata_year'].min()}-{df['metadata_year'].max()}" if 'metadata_year' in df.columns else "N/A",
            'Unique Acc': df['account_external_id'].nunique() if 'account_external_id' in df.columns else 0
        }
        stats_summary.append(stats)
        print(f"Loaded: {name}")
        
    except Exception as e:
        print(f"Skipped {f}: {e}")

if not all_data: exit()
full_df = pd.concat(all_data, ignore_index=True)

# Drop Unknowns ONLY for the graphs
full_df = full_df[full_df['Label'] != 'Unknown']

if full_df.empty:
    print("\nERROR: All data was flagged as 'Unknown' label. Check your CSV column names.")
    exit()

# Create Stats DataFrame
stats_df = pd.DataFrame(stats_summary)

# Print Stats
print("\n" + "="*80)
print("DETAILED DATASET STATISTICS")
print("="*80)
print(stats_df.to_string(index=False))
print("="*80 + "\n")

# Set Style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
colors = sns.color_palette("tab10") 
binary_palette = {"Legit": "#2ecc71", "Fraud": "#e74c3c"}

# --- ADVANCED TOOLTIP HELPER ---
def add_tooltips():
    cursor = mplcursors.cursor(hover=mplcursors.HoverMode.Transient)
    
    @cursor.connect("add")
    def on_add(sel):
        sel.annotation.arrow_patch.set(visible=False)
        x, y = sel.target
        ax = sel.artist.axes
        
        x_val = x
        xticklabels = [label.get_text() for label in ax.get_xticklabels()]
        
        if xticklabels and abs(x - round(x)) < 0.1:
            idx = int(round(x))
            if 0 <= idx < len(xticklabels):
                x_val = xticklabels[idx]
        
        if abs(y - round(y)) < 0.001:
            y_val = f"{int(round(y))}"
        else:
            y_val = f"{y:.2f}"
            
        if isinstance(x_val, (int, float)):
             if abs(x_val - round(x_val)) < 0.001:
                 x_val = f"{int(round(x_val))}"
             else:
                 x_val = f"{x_val:.2f}"

        sel.annotation.set_text(f"{x_val}, {y_val}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.95, edgecolor="black", linewidth=0.5)

# ==========================================
# FIGURE 1: GLOBAL DISTRIBUTIONS
# ==========================================
print("Generating Figure 1...")
fig1, ax1 = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

if 'metadata_duration_seconds' in full_df.columns:
    sns.histplot(data=full_df, x='metadata_duration_seconds', hue='Dataset', element="step", fill=False, stat="density", common_norm=False, linewidth=2, ax=ax1[0], palette=colors)
    ax1[0].set_title("Duration Distribution", fontweight='bold')
    ax1[0].set_xlim(0, 600)

if 'metadata_bitrate' in full_df.columns:
    # Relaxed filter: Show the top 10 most common bitrates instead of hardcoding them
    top_bitrates = full_df['metadata_bitrate'].value_counts().nlargest(10).index
    bit_df = full_df[full_df['metadata_bitrate'].isin(top_bitrates)]
    
    if not bit_df.empty:
        sns.countplot(data=bit_df, x='metadata_bitrate', hue='Dataset', ax=ax1[1], palette=colors, edgecolor='black')
        ax1[1].set_title("Bitrate Preferences (Top 10 Most Common)", fontweight='bold')

add_tooltips()
fig1.suptitle("Part 1: Dataset Physics", fontsize=14, y=0.95)

# ==========================================
# FIGURE 2: FRAUD LOGIC
# ==========================================
print("Generating Figure 2...")
fig2, ax2 = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

if 'metadata_duration_seconds' in full_df.columns:
    sns.boxplot(data=full_df, x='Dataset', y='metadata_duration_seconds', hue='Label', ax=ax2[0], palette=binary_palette, showfliers=False)
    ax2[0].set_title("Duration Separation (Bots vs Humans)", fontweight='bold')

if 'metadata_bitrate' in full_df.columns:
    hq_df = full_df.copy()
    hq_df['is_high_quality'] = hq_df['metadata_bitrate'] >= 256
    hq_summary = hq_df.groupby(['Dataset', 'Label'])['is_high_quality'].mean().reset_index()
    hq_summary['is_high_quality'] *= 100
    sns.barplot(data=hq_summary, x='Dataset', y='is_high_quality', hue='Label', ax=ax2[1], palette=binary_palette, edgecolor='black')
    ax2[1].set_title("Quality Gap (% High Quality Uploads)", fontweight='bold')
    ax2[1].set_ylim(0, 100)

add_tooltips()
fig2.suptitle("Part 2: Fraud Separation Logic", fontsize=14, y=0.95)

# ==========================================
# FIGURE 3: CONTEXT
# ==========================================
print("Generating Figure 3...")
fig3, ax3 = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

if 'metadata_year' in full_df.columns:
    time_df = full_df[(full_df['metadata_year'] >= 2010) & (full_df['metadata_year'] <= 2027)]
    if not time_df.empty:
        sns.histplot(data=time_df, x='metadata_year', hue='Dataset', discrete=True, multiple="dodge", shrink=0.8, ax=ax3[0], palette=colors)
        ax3[0].set_title("Timeline Distribution", fontweight='bold')

if 'metadata_format' in full_df.columns:
    order = full_df['metadata_format'].value_counts().index
    sns.countplot(data=full_df, x='metadata_format', hue='Dataset', order=order, ax=ax3[1], palette=colors, edgecolor='black')
    ax3[1].set_title("Format Diversity", fontweight='bold')

add_tooltips()
fig3.suptitle("Part 3: Temporal & Format Context", fontsize=14, y=0.95)

# ==========================================
# FIGURE 4: DETAILED STATS TABLE
# ==========================================
print("Generating Figure 4 (Stats Table)...")
fig4, ax4 = plt.subplots(figsize=(14, len(stats_df) * 1.5 + 2)) 
ax4.axis('off')

table_data = [stats_df.columns.tolist()] + stats_df.values.tolist()
table = ax4.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.12] + [0.09]*(len(stats_df.columns)-1))

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2) 

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', weight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#f2f2f2')
    cell.set_edgecolor('black')
    cell.set_linewidth(0.5)

fig4.suptitle("Part 4: Detailed Statistical Comparison", fontsize=14, fontweight='bold', y=0.95)

print("\n--- DONE ---")
plt.show()