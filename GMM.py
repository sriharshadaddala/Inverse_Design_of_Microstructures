import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture


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

    X = column_data.reshape(-1, 1)
    
    # 1. Predict which curve (cluster) each point belongs to
    labels = gmm.predict(X)
    
    # 2. Extract the means and standard deviations of the specific curves
    means = gmm.means_.flatten()
    # Adding a tiny epsilon (1e-8) to prevent division by zero in perfectly flat clusters
    stds = np.sqrt(gmm.covariances_.flatten()) + 1e-8 
    
    
    # Formula: (x - mean) / (4 * std) , we use mean of the curve , where x belongs to (predicted above)
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


# 1. Load your Data
# (Change the path to match your actual file location)
file_path = "/Users/harsha/Library/CloudStorage/OneDrive-UniversitéLibredeBruxelles/GENERATED_RVES/RVE_200(resolu)updated/PLY/PLOTS/Pore_Results3D.csv"
df = pd.read_csv(file_path)

# Assuming your columns are named exactly like this. Adjust if necessary!
columns_to_normalize = ['Porosity','Mean', 'Variance', 'Skew', 'Kurtosis']
# This list will hold the transformed matrices for each column
all_transformed_data = []

print("Starting GMM Normalization Pipeline...\n")

for col_name in columns_to_normalize:
    print(f"Processing '{col_name}'...")
    raw_data = df[col_name].values
    
    # Step A: Find the perfect GMM using BIC (Bayesian Information Criterion) 
    best_gmm, best_k = fit_optimal_1d_gmm(raw_data, max_k=8)
    print(f"  -> Optimal Components (K) chosen: {best_k}")
    
    # Step B: Transform the raw numbers into the GAN-friendly format
    transformed_matrix = transform_data_with_gmm(raw_data, best_gmm)
    print(f"  -> Transformed shape: {transformed_matrix.shape}")
    
    all_transformed_data.append(transformed_matrix)
    print("-" * 40)

# 
# Concatenate all the transformed columns side-by-side
final_gan_input = np.hstack(all_transformed_data)

print(f"Original Data Shape: {df[columns_to_normalize].shape} (3000 RVEs, 4 raw numbers)")
print(f"Final GAN Input Shape: {final_gan_input.shape} (3000 RVEs, Categories + Scalars)")

# 3. Save the final matrix so your PyTorch Dataset can load it instantly
# Saving it as a NumPy array (.npy) is much faster for PyTorch than reading a CSV!
np.save("normalized_gmm_conditions.npy", final_gan_input)
print("\nSaved normalized data to 'normalized_gmm_conditions.npy'.")