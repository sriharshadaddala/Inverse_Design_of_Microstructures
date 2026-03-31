
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from normalisedata import transform_data_with_gmm
from Discriminator import Discriminator

# 1. Setup Device and Dimensions
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
input_vector_dim = 15
cond_dim = 21
Para_dim = 36

# 2. Load the Trained Model
PATH = '/Users/harsha/Desktop/PhD_project/Machine_Learning/Reverse_optimization/Weights.pt'
checkpoint = torch.load(PATH, map_location=device, weights_only=False)
saved_data = torch.load('/Users/harsha/Desktop/PhD_project/Machine_Learning/Reverse_optimization/Inputdata_Normalised', weights_only=False)
loaded_gmms = saved_data['GMM_Models'] # load the GMM models previously used for normalising real data

netD = Discriminator(Para_dim).to(device)


netD.load_state_dict(checkpoint['Discriminator_state_dict'])
netD.eval()
for param in netD.parameters():
    param.requires_grad = False

print("✅ Critic successfully loaded and frozen.\n")


target_condition= [0.221689, 0.035867492,0.00017824,-0.0368189,-0.3677264] # Pore space parameters for which we need to generate 15 parameters
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
num_samples = 100
condition_tensor = torch.tensor(condition_vector, dtype=torch.float, device=device).repeat(num_samples, 1)
z = torch.randn(num_samples, input_vector_dim, device=device)
z.requires_grad_(True)

optimizer = optim.Adam([z], lr=0.005)
criterion = nn.BCELoss()



num_steps = 20000
for step in range(num_steps):
    optimizer.zero_grad()
    
    # CRITICAL: Clamp the noise  outside so it will be  -1.0 to 1.0 during optimization
    bounded_noise = torch.tanh(z)
    
    # add the morphing noise to target condition
    combined_input = torch.cat((bounded_noise, condition_tensor), dim=1) #add condition vector again in every loop after optimization 
    
    # using discriminator to get probabilty
    output = netD(combined_input)
    
    # We want  discriminator to output 1.0 (Real)
    target_real = torch.ones_like(output)
    loss = criterion(output, target_real)
    
    # Backpropagate the error directly into the NOISE
    loss.backward()
    optimizer.step()
    
    if step % 1000 == 0 or step == num_steps - 1:
        # MAGIC REVEAL: Let's look at the highest and lowest probabilities in the batch!
        best_prob = output.max().item() * 100
        worst_prob = output.min().item() * 100
        avg_loss = loss.item()
        print(f"Step {step:03d}/{num_steps} | Best Sample: {best_prob:.2f}% | Worst Sample: {worst_prob:.2f}% | Avg Loss: {avg_loss:.4f}")

# --- 6. INVERSE NORMALIZATION (THE RESULT) ---



print("\n=== OPTIMIZATION COMPLETE ===")
final_probs = output.detach().cpu().numpy().flatten()
final_normalized_inputs = torch.tanh(z).detach().cpu().numpy()


# Un-normalize using the saved bounds to get the physical dimensions
bounds_y = saved_data['Bounds_Y']
Xmin = bounds_y[0]
Xmax = bounds_y[1]
mean_val = (Xmax + Xmin) / 2.0
std_val = (Xmax - Xmin) / 2.0

final_physical_parameters = (final_normalized_inputs * std_val) + mean_val
# FILTERING: Only keep the samples that successfully reached at least 90% probability
successful_indices = np.where(final_probs > 0.90)[0]
successful_parameters = final_physical_parameters[successful_indices]
successful_probs = final_probs[successful_indices]

print(f"\n🎯 OUT OF {num_samples} ATTEMPTS, FOUND {len(successful_parameters)} PERFECT RECIPES!")

if len(successful_parameters) > 0:
    # --- 6. SAVE TO EXCEL/CSV ---
    col_names = [f"Param_{i+1}" for i in range(input_vector_dim)]
    df_generated = pd.DataFrame(successful_parameters, columns=col_names)
    
    # Add a column so you can see the Discriminator's confidence for each recipe
    df_generated['Confidence'] = successful_probs 
    
    save_path = "/Users/harsha/Desktop/PhD_project/Machine_Learning/Reverse_optimization/predicted_inputstoRVEgenerator.csv"
    df_generated.to_csv(save_path, index=False)
    print(f"Data saved to: {save_path}")
else:
    print("⚠️ No samples reached the 90% threshold. Try generating a larger batch (e.g., num_samples = 50).")
print("===============================\n")