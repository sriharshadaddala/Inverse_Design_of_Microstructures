import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

PSD_folder_path = '/Users/harsha/Desktop/PhD_project/Updated_Model/'
if PSD_folder_path not in sys.path:
    sys.path.append(PSD_folder_path)

from normalisedata import *
from GAN import Discriminator, Generator

# 1. Setup Device and Dimensions
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_vector_dim = 15
cond_dim = 49
Para_dim = 64
noiseVector_dim = 128

# 2. Load the Trained Model and GMMs
PATH = '/Users/harsha/Desktop/PhD_project/Updated_Model/GAN_Weights.pt'
checkpoint = torch.load(PATH, map_location=device, weights_only=False)
saved_data = torch.load('/Users/harsha/Desktop/PhD_project/Updated_Model/Inputdata_Normalised', weights_only=False)
loaded_gmms = saved_data['GMM_Models'] 

netD = Discriminator(Para_dim).to(device)
netD.load_state_dict(checkpoint['Discriminator_state_dict'])
netD.eval()
for param in netD.parameters():
    param.requires_grad = False

print("✅ Critic successfully loaded and frozen.\n")

# --- Setup Paths for Target CSV and Output Directory ---
TARGETS_CSV_PATH = '/Users/harsha/Desktop/PhD_project/Updated_Model/samplepoints.csv' 
OUTPUT_DIR = '/Users/harsha/Desktop/PhD_project/Updated_Model/'

os.makedirs(OUTPUT_DIR, exist_ok=True)
df_targets = pd.read_csv(TARGETS_CSV_PATH, header=None)
condition_cols = ['Porosity', 'Mean', 'Variance', 'Skew', 'Kurtosis', 'P10', 'P20', 'P30','P40', 'P50', 'P60','P70', 'P80', 'P90', 'P100']

bounds_y = saved_data['Bounds_Y']
Xmin = bounds_y[0]
Xmax = bounds_y[1]
mean_val = (Xmax + Xmin) / 2.0
std_val = (Xmax - Xmin) / 2.0

std_tensor = torch.tensor(std_val, device=device, dtype=torch.float)
mean_tensor = torch.tensor(mean_val, device=device, dtype=torch.float)
criterion = nn.BCELoss()
num_samples = 50
num_steps = 1000

# =========================================================================
# THE CRITICAL SPACING CONSTRAINT
# Max theoretical gap for 0.035 to 0.15 bounds is 0.023.
# 0.015 guarantees separation while allowing optimization.
# =========================================================================
MIN_BIN_GAP = 0.015  

print(f"Found {len(df_targets)} target conditions to process. Starting generation...\n")

for idx, row in df_targets.iterrows():
    file_id = int(row[1]) 
    
    print(f"\n=======================================================")
    print(f"🚀 Processing ID {file_id} ({idx + 1} of {len(df_targets)})")
    print(f"=======================================================")
    
    target_condition = [
        row[3], row[5], row[6], row[7], row[8], row[9], row[10], 
        row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18]
    ]
    normalised_condition = []

    for i, col_name in enumerate(condition_cols):
        single_raw_value = np.array([target_condition[i]])
        specific_gmm = loaded_gmms[col_name]
        transformed_piece = transform_data_with_gmm(single_raw_value, specific_gmm)
        normalised_condition.append(transformed_piece)

    condition_vector = np.hstack(normalised_condition)
    condition_tensor = torch.tensor(condition_vector, dtype=torch.float, device=device).repeat(num_samples, 1)
    
    z = torch.randn(num_samples, input_vector_dim, device=device)
    z.requires_grad_(True)
    optimizer = optim.Adam([z], lr=0.005)

    # --- OPTIMIZATION LOOP ---
    for step in range(num_steps):
        optimizer.zero_grad()
        
        generated_params = torch.tanh(z)
        combined_input = torch.cat((generated_params, condition_tensor), dim=1) 
        
        output = netD(combined_input)
        target_real = torch.ones_like(output)
        bce_loss = criterion(output, target_real)

        phys_params = (generated_params * std_tensor) + mean_tensor
        
        p4 = phys_params[:, 3]
        p5 = phys_params[:, 4]
        p6 = phys_params[:, 5]
        p7 = phys_params[:, 6]
        p8 = phys_params[:, 7]
        p9 = phys_params[:, 8]
        
        # DYNAMIC BUFFER: Scales with the available bounds, but never falls below MIN_BIN_GAP
        dynamic_buffer = torch.clamp((p5 - p4) * 0.15, min=MIN_BIN_GAP)
        
        gap1 = p6 - p4
        gap2 = p7 - p6
        gap3 = p8 - p7
        gap4 = p9 - p8
        gap5 = p5 - p9
        
        penalty = torch.relu(dynamic_buffer - gap1) + \
                  torch.relu(dynamic_buffer - gap2) + \
                  torch.relu(dynamic_buffer - gap3) + \
                  torch.relu(dynamic_buffer - gap4) + \
                  torch.relu(dynamic_buffer - gap5)
                  
        buffer_loss = penalty.mean()
        loss_prior = torch.mean(z**2)
        
        # HEAVY PENALTY WEIGHT (5.0) to force the optimizer to respect the gap
        total_loss = bce_loss + (5.0 * buffer_loss) + (0.1 * loss_prior)
        total_loss.backward()
        optimizer.step()
        
        if step % 500 == 0 or step == num_steps - 1:
            best_prob = output.max().item() * 100
            worst_prob = output.min().item() * 100
            print(f"  Step {step:03d}/{num_steps} | Best: {best_prob:.2f}% | Worst: {worst_prob:.2f}% | Loss: {total_loss.item():.4f}")

    # --- INVERSE NORMALIZATION & POST-PROCESSING ---
    with torch.no_grad():
        final_probs = output.detach().cpu().numpy().flatten()
        final_normalized_inputs = torch.tanh(z).detach().cpu().numpy()

    final_physical_parameters = (final_normalized_inputs * std_val) + mean_val

    # 1. INITIAL SORT
    final_physical_parameters[:, 5:9] = np.sort(final_physical_parameters[:, 5:9], axis=1)

    # 2. THE FORWARD-BACKWARD SPACING ALGORITHM (Guarantees Distinct Bins)
    for j in range(len(final_physical_parameters)):
        p4 = final_physical_parameters[j, 3] # Min inclusion bound
        p5 = final_physical_parameters[j, 4] # Max inclusion bound
        
        # Failsafe: Ensure the overall range is physically capable of holding 5 bins
        if p5 < p4 + (5 * MIN_BIN_GAP):
            p5 = p4 + (5 * MIN_BIN_GAP)
            final_physical_parameters[j, 4] = p5

        # FORWARD PASS: Push values UP from the bottom boundary
        final_physical_parameters[j, 5] = max(final_physical_parameters[j, 5], p4 + MIN_BIN_GAP)
        final_physical_parameters[j, 6] = max(final_physical_parameters[j, 6], final_physical_parameters[j, 5] + MIN_BIN_GAP)
        final_physical_parameters[j, 7] = max(final_physical_parameters[j, 7], final_physical_parameters[j, 6] + MIN_BIN_GAP)
        final_physical_parameters[j, 8] = max(final_physical_parameters[j, 8], final_physical_parameters[j, 7] + MIN_BIN_GAP)
        
        # BACKWARD PASS: Squeeze values DOWN from the top boundary
        final_physical_parameters[j, 8] = min(final_physical_parameters[j, 8], p5 - MIN_BIN_GAP)
        final_physical_parameters[j, 7] = min(final_physical_parameters[j, 7], final_physical_parameters[j, 8] - MIN_BIN_GAP)
        final_physical_parameters[j, 6] = min(final_physical_parameters[j, 6], final_physical_parameters[j, 7] - MIN_BIN_GAP)
        final_physical_parameters[j, 5] = min(final_physical_parameters[j, 5], final_physical_parameters[j, 6] - MIN_BIN_GAP)

    # --- FILTERING & SAVING ---
    successful_indices = np.where(final_probs > 0.90)[0]
    successful_parameters = final_physical_parameters[successful_indices]
    successful_probs = final_probs[successful_indices]

    print(f"  🎯 FOUND {len(successful_parameters)} PERFECT RECIPES for ID {file_id}!")

    if len(successful_parameters) > 0:
        col_names = [f"Param_{i+1}" for i in range(input_vector_dim)]
        df_generated = pd.DataFrame(successful_parameters, columns=col_names)
        df_generated['Confidence'] = successful_probs 
        
        save_filename = f"{file_id}.csv"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        
        df_generated.to_csv(save_path, index=False)
        print(f"  Saved to: {save_filename}")
    else:
        print(f"   No samples reached the 90% threshold for ID {file_id}.")

print("\n🎉 ALL CONDITIONS PROCESSED SUCCESSFULLY!")