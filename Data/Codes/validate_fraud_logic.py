import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

# --- CONFIGURATION ---
BASE_DIR = r"E:\University of Aberdeen\Semester-2\Final Project\Data\Datasets"

# COMPARE ALL 4 DATASETS
# Batch 1: The Reference Models
# Batch 2: The Competitor Models
DATASET_GROUPS = [
    {
        'Claude (Baseline)': 'raw_metadata_samples.csv',
        'ChatGPT': 'synthetic_data_gpt.csv'
    },
    {
        'Gemini': 'synthetic_data_gem.csv',
        'Copilot': 'synthetic_data_copilot.csv'
    }
]

def analyze_logic_split():
    # Loop through the two batches (creates 2 separate figures)
    for batch_idx, batch_data in enumerate(DATASET_GROUPS):
        
        # Setup Figure: 2 Rows (Models) x 2 Columns (Duration & Bitrate)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        group_name = "Reference Models" if batch_idx == 0 else "Competitor Models"
        fig.suptitle(f"Internal Fraud Logic Validation: {group_name}", fontsize=16, weight='bold', y=0.98)

        axes = axes.flatten()
        plot_idx = 0
        
        for name, filename in batch_data.items():
            filepath = os.path.join(BASE_DIR, filename)
            
            # --- ERROR HANDLING ---
            if not os.path.exists(filepath):
                print(f"Skipping {name}: File not found ({filename})")
                plot_idx += 2
                continue
                
            df = pd.read_csv(filepath)
            
            # 1. FIND LABEL COLUMN
            col_name = None
            if 'profile_type' in df.columns: col_name = 'profile_type'
            elif 'metadata_profile_type' in df.columns: col_name = 'metadata_profile_type'
            
            if not col_name:
                print(f"Skipping {name}: No 'profile_type' column found.")
                # We essentially blank out these axes if data is missing
                axes[plot_idx].text(0.5, 0.5, "Data Missing", ha='center')
                axes[plot_idx+1].text(0.5, 0.5, "Data Missing", ha='center')
                plot_idx += 2
                continue

            # 2. CREATE CLASS LOGIC
            # Logic: If 'normal_user' or 'legit' -> Legit. Else -> Fraud.
            df['User_Class'] = df[col_name].apply(lambda x: 'Legit' if str(x).lower() in ['normal_user', 'legit'] else 'Fraud')
            
            # --- PLOT A: DURATION (Box Plot) ---
            ax_dur = axes[plot_idx]
            sns.boxplot(x='User_Class', y='metadata_duration_seconds', data=df, 
                        palette={'Legit': "#2ecc71", 'Fraud': "#e74c3c"}, 
                        order=['Legit', 'Fraud'], ax=ax_dur)
            ax_dur.set_title(f"{name}: Duration Logic", weight='bold')
            ax_dur.set_xlabel("")
            ax_dur.set_ylabel("Seconds")
            
            # --- PLOT B: BITRATE (Count Plot) ---
            ax_bit = axes[plot_idx + 1]
            sns.countplot(x='metadata_bitrate', hue='User_Class', data=df, 
                          palette={'Legit': "#2ecc71", 'Fraud': "#e74c3c"}, 
                          hue_order=['Legit', 'Fraud'], ax=ax_bit)
            ax_bit.set_title(f"{name}: Bitrate Logic", weight='bold')
            ax_bit.set_xlabel("Bitrate (kbps)")
            ax_bit.set_ylabel("Count")
            ax_bit.legend(title='', loc='upper right', fontsize='small')
            
            plot_idx += 2

        print(f"-> Displaying Batch {batch_idx + 1} ({group_name})...")
        plt.show()

if __name__ == "__main__":
    analyze_logic_split()