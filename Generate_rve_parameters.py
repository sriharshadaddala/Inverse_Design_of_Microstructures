
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from normalisedata import transform_data_with_gmm
from GAN import Discriminator, Generator

# 1. Setup Device and Dimensions
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_vector_dim = 15
cond_dim = 21
Para_dim = 36
noiseVector_dim=64 
# 2. Load the Trained Model
PATH = '/Users/harsha/Desktop/PhD_project/Machine_Learning/no_condition/Weights.pt'
checkpoint = torch.load(PATH, map_location=device, weights_only=False)
saved_data = torch.load('/Users/harsha/Desktop/PhD_project/Machine_Learning/no_condition/Inputdata_Normalised', weights_only=False)
loaded_gmms = saved_data['GMM_Models'] # load the GMM models previously used for normalising real data



netD = Discriminator(Para_dim).to(device)
netD.load_state_dict(checkpoint['Discriminator_state_dict'])
netD.eval()
for param in netD.parameters():
    param.requires_grad = False

print("✅ Critic successfully loaded and frozen.\n")


target_condition= [0.10297, 0.018934063,7.65E-05,0.132716508,-0.771458044] # Pore space parameters for which we need to generate 15 parameters
condition_cols = ['Porosity','Mean', 'Variance', 'Skew', 'Kurtosis']
normalised_condition = []



for i, col_name in enumerate(condition_cols):
    # Get the specific raw number and format it for Scikit-Learn
    single_raw_value = np.array([target_condition[i]])
    
    # Grab the specific GMM model we saved for this column
    specific_gmm = loaded_gmms[col_name]
    
    # Run it through the transformation function
    transformed_piece = transform_data_with_gmm(single_raw_value, specific_gmm)
    
    # Save the piece
    normalised_condition.append(transformed_piece)

condition_vector = np.hstack(normalised_condition)
# 4. Generate Samples
num_samples = 10
condition_tensor = torch.tensor(condition_vector, dtype=torch.float, device=device).repeat(num_samples, 1)
z = torch.randn(num_samples, input_vector_dim, device=device)
z.requires_grad_(True)

optimizer = optim.Adam([z], lr=0.001)
criterion = nn.BCELoss()

def diversity_loss(batch_samples):
    # Move to CPU just for this calculation to avoid the MPS error
    samples_cpu = batch_samples.cpu()
    dist = torch.cdist(samples_cpu, samples_cpu)
    
    # Calculate the loss on CPU, then move the scalar result back to the device
    loss = -torch.mean(dist)
    return loss.to(device)

# In your loop:


num_steps = 2000
for step in range(num_steps):
    optimizer.zero_grad()
    
    # CRITICAL: Clamp the noise  outside so it will be  -1.0 to 1.0 during optimization loop 
    generated_params = torch.tanh(z)
    
    # add the morphing noise to target condition
    combined_input = torch.cat((generated_params, condition_tensor), dim=1) #add condition vector again in every loop after optimization 
    
    # using discriminator to get probabilty
    output = netD(combined_input)
    
    # We want  discriminator to output 1.0 (Real)
    target_real = torch.ones_like(output)
    loss = criterion(output, target_real)
    
    # Backpropagate the error directly into the NOISE
    loss_prior = torch.mean(z**2) 
    
    total_loss = loss + (2.0 * loss_prior) + (0.01 * diversity_loss(generated_params))
    total_loss.backward()
    #loss.backward()
    optimizer.step()
    sample_diversity = generated_params.std(dim=0).mean().item()
    
    
    if step % 1000 == 0 or step == num_steps - 1:
        # MAGIC REVEAL: Let's look at the highest and lowest probabilities in the batch!
        best_prob = output.max().item() * 100
        worst_prob = output.min().item() * 100
        avg_loss = loss.item()
        print(f"Step {step:03d}/{num_steps} | Best Sample: {best_prob:.2f}% | Worst Sample: {worst_prob:.2f}% | Avg Loss: {avg_loss:.4f}")

# --- 6. INVERSE NORMALIZATION (THE RESULT) ---



print("\n=== OPTIMIZATION COMPLETE ===")
with torch.no_grad():
    final_probs = output.detach().cpu().numpy().flatten()
    
    # Pass ONLY the optimized z (15 dimensions) to the blind Generator
    final_normalized_inputs = (z).cpu().numpy()
    


# Un-normalize using the saved bounds to get the physical dimensions
bounds_y = saved_data['Bounds_Y']
Xmin = bounds_y[0]
Xmax = bounds_y[1]
mean_val = (Xmax + Xmin) / 2.0
std_val = (Xmax - Xmin) / 2.0

final_physical_parameters = (final_normalized_inputs * std_val) + mean_val
# FILTERING: Only keep the samples that successfully reached at least 90% probability
successful_indices = np.where(final_probs > 0.80)[0]
successful_parameters = final_physical_parameters[successful_indices]
successful_probs = final_probs[successful_indices]

print(f"\n🎯 OUT OF {num_samples} ATTEMPTS, FOUND {len(successful_parameters)} PERFECT RECIPES!")

if len(successful_parameters) > 0:
    # --- 6. SAVE TO EXCEL/CSV ---
    col_names = [f"Param_{i+1}" for i in range(input_vector_dim)]
    df_generated = pd.DataFrame(successful_parameters, columns=col_names)
    
    # Add a column so you can see the Discriminator's confidence for each recipe
    df_generated['Confidence'] = successful_probs 
    
    save_path = "/Users/harsha/Desktop/PhD_project/Machine_Learning/no_condition/2981.csv"
    df_generated.to_csv(save_path, index=False)
    print(f"Data saved to: {save_path}")
else:
    print("⚠️ No samples reached the 90% threshold. Try generating a larger batch (e.g., num_samples = 50).")
print("===============================\n")