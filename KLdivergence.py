import numpy as np
import pandas as pd
from scipy.stats import entropy
import os
import warnings

# Mute standard warnings to keep the terminal output clean
warnings.filterwarnings("ignore")

# ==========================================
# 1. FILE PATHS (Update these if they change)
# ==========================================
TRAIN_DATA_PATH = "/Users/harsha/Desktop/PhD_project/Updated_Model/Trainingdata.csv"
GEN_DATA_PATH = "/Users/harsha/Desktop/PhD_project/Updated_Model/Random_Baseline1.csv"
OUTPUT_REPORT_PATH = "/Users/harsha/Desktop/PhD_project/Updated_Model/2D_KL_Divergence_Report.csv"

# ==========================================
# 2. CORE MATH FUNCTION (Laplace Smoothed)
# ==========================================
def calculate_safe_2d_kl(real_x, real_y, gen_x, gen_y, bins=30, epsilon=1e-8):
    """
    Calculates the 2D KL Divergence safely by preventing 'Divide by Zero' 
    errors using Laplace smoothing (epsilon) on a 2D histogram grid.
    """
    # Define the exact same 2D grid boundaries for both datasets
    x_min = min(np.min(real_x), np.min(gen_x))
    x_max = max(np.max(real_x), np.max(gen_x))
    y_min = min(np.min(real_y), np.min(gen_y))
    y_max = max(np.max(real_y), np.max(gen_y))
    
    # Create the grid edges
    x_edges = np.linspace(x_min, x_max, bins + 1)
    y_edges = np.linspace(y_min, y_max, bins + 1)
    
    # Count how many points fall in each bin (2D Histogram)
    P_hist, _, _ = np.histogram2d(real_x, real_y, bins=[x_edges, y_edges])
    Q_hist, _, _ = np.histogram2d(gen_x, gen_y, bins=[x_edges, y_edges])
    
    # Add Laplace Smoothing (Microscopic mass to empty bins)
    P_hist += epsilon
    Q_hist += epsilon
    
    # Normalize so the whole board sums to 1.0 (Probability Density)
    P_prob = P_hist / np.sum(P_hist)
    Q_prob = Q_hist / np.sum(Q_hist)
    
    # Flatten the 2D grids into 1D arrays for SciPy's entropy function
    P_flat = P_prob.flatten()
    Q_flat = Q_prob.flatten()
    
    # Calculate KL Divergence: sum(P * log(P / Q))
    kl_div = entropy(P_flat, Q_flat)
    
    return kl_div

# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================
print("Loading datasets...")
try:
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_gen = pd.read_csv(GEN_DATA_PATH)
    print(f"✅ Training data loaded: {len(df_train)} samples")
    print(f"✅ Generated data loaded: {len(df_gen)} samples\n")
except FileNotFoundError as e:
    print(f"❌ ERROR: Could not find file. Please check the path.\n{e}")
    exit()


train_quantifying = df_train.iloc[:, 0:15]


train_inputs = df_train.iloc[:, 16:31]

gen_quantifying = df_gen.iloc[:, 0:15]
gen_inputs = df_gen.iloc[:, 16:31]

# 2. Add the drop command to remove the ghost column P completely:
df_train = df_train.drop(columns=[df_train.columns[15]])
df_gen = df_gen.drop(columns=[df_gen.columns[15]])

# Extract actual column names from the training data for the final report
quantifying_names = train_quantifying.columns.tolist()
input_names = train_inputs.columns.tolist()

results = []

print("Calculating 2D KL Divergence for all 225 pairs (30x30 Grid). This takes a few seconds...\n")

for i, q_name in enumerate(quantifying_names):
    for j, i_name in enumerate(input_names):
        
        # Extract the 1D arrays for this specific pair
        real_x = train_quantifying.iloc[:, i].values
        real_y = train_inputs.iloc[:, j].values
        
        gen_x = gen_quantifying.iloc[:, i].values
        gen_y = gen_inputs.iloc[:, j].values
        
        # Calculate the score 
        kl_score = calculate_safe_2d_kl(real_x, real_y, gen_x, gen_y, bins=30, epsilon=1e-8)
        
        results.append({
            'Quantifying_Parameter': q_name,
            'Input_Parameter': i_name,
            '2D_KL_Score': kl_score
        })

# ==========================================
# 4. REPORTING & SAVING
# ==========================================
df_results = pd.DataFrame(results)

# Calculate the Master Average Score across all pairs
mean_kl = df_results['2D_KL_Score'].mean()

print("-" * 50)
print(f"🎯 MASTER SCORE (Mean 2D KL Divergence): {mean_kl:.4f}")
print("-" * 50)

print("\n🏆 Top 3 BEST Modeled Pairs (Lowest Error):")
print(df_results.sort_values('2D_KL_Score').head(3).to_string(index=False))

print("\n⚠️ Top 3 WORST Modeled Pairs (Highest Error):")
print(df_results.sort_values('2D_KL_Score', ascending=False).head(3).to_string(index=False))

# Save the final report
df_results.to_csv(OUTPUT_REPORT_PATH, index=False)
print(f"\n✅ Full report saved successfully to:\n{OUTPUT_REPORT_PATH}")