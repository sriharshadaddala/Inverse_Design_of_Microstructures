import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CONFIGURATION ---
train_path = "/Users/harsha/Desktop/PhD_project/Updated_Model/Trainingdata.csv"
gen_path = "/Users/harsha/Desktop/PhD_project/Updated_Model/Generated_RVE_Parameters(400).csv"
output_dir = "/Users/harsha/Desktop/PhD_project/Updated_Model/"

# Define your specific subset of interest
subset_cols = [
    'Input1', 'Input2', 'Input3', 'Input4', 'Input5', 
    'Porosity', 'Skew', 'Kurtosis', 'Mean', 'Variance'
]

# --- 2. DATA PROCESSING ---
# Load and subset only the columns you want
df_train = pd.read_csv(train_path)[subset_cols]
df_gen = pd.read_csv(gen_path)[subset_cols]

# Calculate Correlation Matrices
corr_train = df_train.corr(method='pearson')
corr_gen = df_gen.corr(method='pearson')

# Calculate Absolute Difference Matrix
corr_diff = np.abs(corr_train - corr_gen)

# --- 3. VISUAL DIAGNOSTIC (Targeted Heatmap) ---
plt.figure(figsize=(10, 8))

ax = sns.heatmap(corr_diff, 
                 annot=True, 
                 fmt=".2f", 
                 cmap="Reds", 
                 vmin=0, 
                 vmax=0.3, 
                 square=True, 
                 cbar_kws={'label': 'Absolute Correlation Error'},
                 annot_kws={"size": 10})

plt.title("Key Parameter Correlation Error Matrix", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()

# Save
plot_path = os.path.join(output_dir, "Targeted_Correlation_Diagnostic.png")
plt.savefig(plot_path, dpi=300)
print(f"✅ Targeted Heatmap saved to: {plot_path}")
plt.show()