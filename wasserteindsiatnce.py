import numpy as np
import pandas as pd
import ot  # Python Optimal Transport library (pip install POT)
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. LOAD DATASETS
# ==========================================
# Load Real, GAN, and Shuffled data
df_train = pd.read_csv("/Users/harsha/Desktop/PhD_project/Updated_Model/Trainingdata.csv")
df_gan = pd.read_csv("/Users/harsha/Desktop/PhD_project/Updated_Model/Generated_RVE_Parameters(400).csv") # UPDATE WITH YOUR ACTUAL GAN DATA
df_shuffled = pd.read_csv("/Users/harsha/Desktop/PhD_project/Updated_Model/Shuffled_Baseline.csv")

# Clean up ghost columns
if 'Unnamed: 15' in df_train.columns: df_train = df_train.drop(columns=['Unnamed: 15'])
if 'Unnamed: 15' in df_gan.columns: df_gan = df_gan.drop(columns=['Unnamed: 15'])
if 'Unnamed: 15' in df_shuffled.columns: df_shuffled = df_shuffled.drop(columns=['Unnamed: 15'])

# Isolate features
train_q = df_train.iloc[:, 0:15].values
train_i = df_train.iloc[:, 15:30].values

gan_q = df_gan.iloc[:, 0:15].values
gan_i = df_gan.iloc[:, 15:30].values

shuffled_q = df_shuffled.iloc[:, 0:15].values
shuffled_i = df_shuffled.iloc[:, 15:30].values

# ==========================================
# 2. CALCULATE 2D SLICED WASSERSTEIN
# ==========================================
print("Calculating 2D Sliced Wasserstein Distance for all 225 pairs...")

gan_swd_scores = []
shuffled_swd_scores = []

for idx_q in range(15):
    for idx_i in range(15):
        
        # 1. Extract the 2D point clouds for this specific pair
        # Shape becomes (N, 2)
        real_2d = np.column_stack((train_q[:, idx_q], train_i[:, idx_i]))
        gan_2d = np.column_stack((gan_q[:, idx_q], gan_i[:, idx_i]))
        shuffled_2d = np.column_stack((shuffled_q[:, idx_q], shuffled_i[:, idx_i]))
        
        # 2. Normalize the point clouds so the distance metric is fair across different scales
        # We fit standard scaler on real data, and apply to all to keep relative distances
        mean, std = real_2d.mean(axis=0), real_2d.std(axis=0) + 1e-8
        real_2d = (real_2d - mean) / std
        gan_2d = (gan_2d - mean) / std
        shuffled_2d = (shuffled_2d - mean) / std
        
        # 3. Calculate Sliced Wasserstein Distance (Lower is better)
        swd_gan = ot.sliced_wasserstein_distance(real_2d, gan_2d, n_projections=50)
        swd_shuf = ot.sliced_wasserstein_distance(real_2d, shuffled_2d, n_projections=50)
        
        gan_swd_scores.append(swd_gan)
        shuffled_swd_scores.append(swd_shuf)

# ==========================================
# 3. FINAL RESULTS
# ==========================================
mean_gan_swd = np.mean(gan_swd_scores)
mean_shuf_swd = np.mean(shuffled_swd_scores)

print("-" * 50)
print(f"🎯 GAN Sliced Wasserstein Distance:      {mean_gan_swd:.4f} (Lower = More Similar Shape)")
print(f"❌ Shuffled Sliced Wasserstein Distance: {mean_shuf_swd:.4f}")
print("-" * 50)

if mean_gan_swd < mean_shuf_swd:
    print("✅ PROOF SECURED: The GAN's 2D structural shapes are mathematically closer to the real data than the shuffled baseline.")