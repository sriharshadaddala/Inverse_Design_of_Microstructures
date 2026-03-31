import numpy as np
import torch
import pandas as pd
from sklearn.mixture import GaussianMixture
# 1. Set your path
#load input vector if RVE Generator
csv_path = "/Users/harsha/Desktop/PhD_project/Machine_Learning/X_unique.csv"
#load Pore Space parameters
csv_path_condition =  "/Users/harsha/Library/CloudStorage/OneDrive-UniversitéLibredeBruxelles/GENERATED_RVES/RVE_200(resolu)updated/PLY/PLOTS/Pore_Results3D.csv"
save_path = './Inputdata_Normalised'
df = pd.read_csv(csv_path,header=None)
df_c=pd.read_csv(csv_path_condition)


# Checks alignment
if len(df) != len(df_c):
    print(f"⚠️ WARNING: Row count mismatch! Y has {len(df)} rows, C has {len(df_c)} rows.")
else:
    print(f"✅ Row counts match perfectly: {len(df)} rows.")


condition_cols = ['Porosity','Mean', 'Variance', 'Skew', 'Kurtosis']


x = df.iloc[:, 0:15].to_numpy().copy()


# Calculate Bounds (one min and one max for each of the 15 columns)
Xmin = np.amin(x, axis=0) 
Xmax = np.amax(x, axis=0)
Bounds = [Xmin, Xmax]

# 4. Normalize using your Ganlib function
# This squashes each column into the [-1, 1] range

def Normalize(X, Xmin, Xmax):        ### X need to be 1D array 
  mean = (Xmax + Xmin)/2.0 #np.mean(X, axis=1)#
  std = (Xmax - Xmin)/2.0 #np.std(X, axis=1)#
 
  Y = X.copy().transpose()

  for i,varX in enumerate(Y):
    if(std[i]!=0.0): 
      Y[i] = (varX-mean[i])/std[i]
    else: 
      Y[i] = varX-mean[i]
 
  Y = Y.transpose()
  return Y    

y = Normalize(x, Xmin, Xmax)


def fit_optimal_1d_gmm(column_data, max_k=8):
    """
    Tests GMMs from K=1 to max_k, finds the lowest BIC, 
    and returns the best fitted model.
    """
    X = column_data.reshape(-1, 1)    # reshape the row vector to column vector
    
    lowest_bic = np.inf
    best_gmm = None
    best_k = 1
    
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, init_params='random_from_data')
        gmm.fit(X)
        
        current_bic = gmm.bic(X)
        
        if current_bic < lowest_bic:
            lowest_bic = current_bic
            best_gmm = gmm
            best_k = k
            
    return best_gmm, best_k

def transform_data_with_gmm(column_data, gmm):
    """
    Converts raw data into [Local_Scalar, One_Hot_1, One_Hot_2, ...]
    using the 4-Sigma rule for GAN normalization.
    """
    X = column_data.reshape(-1, 1)
    
    # 1. Predict which curve (cluster) each point belongs to
    labels = gmm.predict(X)
    
    # 2. Extract the means and standard deviations of the specific curves
    means = gmm.means_.flatten()
    # Add a tiny epsilon (1e-8) to prevent division by zero in perfectly flat clusters
    stds = np.sqrt(gmm.covariances_.flatten()) + 1e-8 
    
    # 3. Calculate the Local Scalar using the 4-Sigma rule
    # Formula: (x - mean) / (4 * std)
    # This safely forces 99.9% of the data to sit comfortably between -1.0 and 1.0
    local_scalars = (column_data - means[labels]) / (4 * stds[labels])
    
    # Clip extreme anomalies just in case they push slightly past -1 or 1
    local_scalars = np.clip(local_scalars, -1.0, 1.0)
    
    # 4. Create the One-Hot Encoding for the categories
    num_components = gmm.n_components
    one_hot_matrix = np.eye(num_components)[labels]
    
    # 5. Combine the scalar and the one-hot matrix
    # Shape will be (2000, 1 + K)
    transformed_column = np.column_stack([local_scalars, one_hot_matrix])
    
    return transformed_column


all_transformed_data = []
print("Starting GMM Normalization Pipeline...\n")

saved_gmm_models = {}

for col_name in condition_cols:
    print(f"Processing '{col_name}'...")
    raw_data = df_c[col_name].values
    
    # Step A: Find the perfect GMM
    best_gmm, best_k = fit_optimal_1d_gmm(raw_data, max_k=8)
    print(f"  -> Optimal Components (K) chosen: {best_k}")
    saved_gmm_models[col_name] = best_gmm
    # Step B: Transform the raw numbers into the GAN-friendly format
    transformed_matrix = transform_data_with_gmm(raw_data, best_gmm)
    print(f"  -> Transformed shape: {transformed_matrix.shape}")
    
    all_transformed_data.append(transformed_matrix)
    print("-" * 40)

final_gan_input = np.hstack(all_transformed_data)

# 5. Save for PyTorch training

torch.save({
    'Norm_Y': y,          # Your size 15 feature vectors
    'Norm_C': final_gan_input,     # Your size 10 condition vectors
    'Bounds_Y': Bounds,   # Bounds for Y
    'GMM_Models': saved_gmm_models
}, save_path)

print(f"Loaded {x.shape[0]} samples with {x.shape[1]} parameters.")
print(f"Features (y) shape: {y.shape}")

print("Data successfully normalized and saved.")



verification_df = df.iloc[:, 0:15].copy()
# Rename columns to 1-15 for clarity
verification_df.columns = [f"Param_{i+1}" for i in range(15)]

# Add the raw condition columns
for col in condition_cols:
    verification_df[col] = df_c[col]

verification_df.to_csv("RAW_Alignment_Verification.csv", index=False)

# 10. Final Status
print("-" * 30)
print(f"Total Aligned Samples: {x.shape[0]}")
print(f"Features (y) shape:    {y.shape}")
print(f"Conditions (c) shape:  {final_gan_input.shape}")
print("-" * 30)
print("Files Saved:")
print(f"1. PyTorch Input: {save_path}")
print(f"2. Excel Check:   RAW_Alignment_Verification.csv")
print("-" * 30)

# 2. Unpack the "backpack"
saved_data = torch.load(save_path, weights_only=False)

# 3. Pull out your specific variables using the dictionary keys
y = saved_data['Norm_Y']
c = saved_data['Norm_C']
bounds_y = saved_data['Bounds_Y']

# 4. Print the shapes to confirm the full dataset is there
print("=== FILE SUMMARY ===")
print(f"Features (Y) Shape:   {y.shape}")
print(f"Conditions (C) Shape: {c.shape}")
print("====================\n")






#  INVERSE NORMALIZATION CHECK
print("\n=== RUNNING INVERSE NORMALIZATION CHECKS ===")

saved_data = torch.load(save_path, weights_only=False)
y_saved = saved_data['Norm_Y']
c_saved = saved_data['Norm_C']
bounds_saved = saved_data['Bounds_Y']

# --- Check 1: Features (Y) ---
raw_row_0 = x[1258]
norm_row_0 = y_saved[1258]

Xmin_saved = bounds_saved[0]
Xmax_saved = bounds_saved[1]
mean_saved = (Xmax_saved + Xmin_saved) / 2.0
std_saved = (Xmax_saved - Xmin_saved) / 2.0

unnormalized_row_0 = (norm_row_0 * std_saved) + mean_saved
diff_Y = np.max(np.abs(raw_row_0 - unnormalized_row_0))

if diff_Y < 1e-4:
    print("✅ FEATURE RECONSTRUCTION: SUCCESS! The normalized data perfectly matches the original raw data.")
else:
    print("❌ FEATURE RECONSTRUCTION: FAILED!")
    print("Raw: ", np.round(raw_row_0, 4))
    print("Un-normalized: ", np.round(unnormalized_row_0, 4))

# --- Check 2: Conditions (C) ---
print("\n--- Condition Reconstruction ---")
raw_conditions_0 = df_c[condition_cols].iloc[1258].values #randomly check the vector parameter
norm_conditions_0 = c_saved[1258]

current_idx = 0 
for i, col_name in enumerate(condition_cols):
    gmm = saved_data['GMM_Models'][col_name]
    k_components = gmm.n_components
    chunk_size = 1 + k_components
    
    chunk = norm_conditions_0[current_idx : current_idx + chunk_size]
    local_scalar = chunk[0]
    one_hot_vector = chunk[1:]
    
    cluster_label = np.argmax(one_hot_vector)
    cluster_mean = gmm.means_.flatten()[cluster_label]
    cluster_std = np.sqrt(gmm.covariances_.flatten()[cluster_label]) + 1e-8
    
    reconstructed_val = (local_scalar * 4 * cluster_std) + cluster_mean
    original_val = raw_conditions_0[i]
    
    difference = np.abs(original_val - reconstructed_val)
    
    status = "✅ Match" if difference < 1e-4 else "❌ Mismatch"
    print(f"{col_name.ljust(10)} | Raw: {original_val:.5f} | Recon: {reconstructed_val:.5f} | {status}")
        
    current_idx += chunk_size