import torch
import random
import numpy as np
import pandas as pd

# Load the file
save_path = '/Users/harsha/Desktop/PhD_project/Machine_Learning/Reverse_optimization/Inputdata_Normalised'
checkpoint = torch.load(save_path, weights_only=False)

# Extract the data
y_norm = checkpoint['Norm_Y']
c_norm = checkpoint['Norm_C']

print("--- DATA OVERVIEW ---")
print(f"Shape of Features (y): {y_norm.shape}")     # (2500, 15)
print(f"Shape of Conditions (c): {c_norm.shape}")   # (2500, 10)


print("\n--- SAMPLE CHECK (Row 0) ---")
print(f"Normalized Y: {y_norm[0]}")
print(f"Normalized C: {c_norm[0]}")


csv_output_path = "/Users/harsha/Desktop/PhD_project/Machine_Learning/Reverse_optimization/Normalized_Data_Output.csv"
combined_data = np.hstack((y_norm, c_norm))
df = pd.DataFrame(combined_data)
df.to_csv(csv_output_path, index=False)

print(f"Success! Normalized data saved to: {csv_output_path}")
print(f"File shape: {df.shape}")


def RandomShuffle(fileName, ratio):

  checkpoint = torch.load(fileName, weights_only=False, map_location='cpu')
  Y = checkpoint['Norm_Y']
  C = checkpoint['Norm_C']
  # Create indices for shuffling
  indices = list(range(Y.shape[0]))
  random.shuffle(indices)
# Select subset based on ratio
  train_size = int(ratio * len(indices))
  selected_indices = indices[:train_size]
  # Filter both arrays using the same shuffled indices
  Y_train = Y[selected_indices]
  C_train = C[selected_indices]
    
  return (torch.from_numpy(Y_train).float(), 
            torch.from_numpy(C_train).float())