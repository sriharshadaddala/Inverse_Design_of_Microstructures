import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
PSD_folder_path = '/Users/harsha/Desktop/PhD_project/Machine_Learning/Reverse_optimization/'
if PSD_folder_path not in sys.path:
    sys.path.append(PSD_folder_path)
from Discriminator import Discriminator
from checkdata import RandomShuffle


# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
#------------------------------------------------------------------------    

PATH = '/Users/harsha/Desktop/PhD_project/Machine_Learning/Reverse_optimization/Weights.pt' 



# Initialize BCELoss function
criterion = nn.BCELoss()

#  convention for real and fake labels during training
real_label = 0.95
fake_label = 0.
lrD = 0.0002
beta1 = 0.5



Para_dim = 36               #(condition+input vector)
cond_dim=21                 #normalised porosity parameters using GMM (5 parameters normalised to 21 parameters)
InputVector_dim=15          #(normalised input parameters)
# using  the Discriminator 
netD = Discriminator(Para_dim).to(device)
#optimizer
optimizerD = optim.AdamW(netD.parameters(), lr=lrD, betas=(beta1, 0.999))


if os.path.exists(PATH):
    print(f"--- Found existing model at {PATH}. Resuming training... ---")
    checkpoint = torch.load(PATH, weights_only=False, map_location=device)
    netD.load_state_dict(checkpoint['Discriminator_state_dict'])
    optimizerD.load_state_dict(checkpoint['D_optimizer_state_dict'])
    
else:
    print(f"--- No file found at {PATH}. Starting training from SCRATCH! ---")




fileName = '/Users/harsha/Desktop/PhD_project/Machine_Learning/PSD/Inputdata_Normalised'
checkpoint = torch.load(fileName, weights_only=False, map_location='cpu')
del checkpoint

num_epochs = 1000
Niter = 31
# Training Loop --------------------------------------------------------------------------
D_losses = []
iters = 0
ratio = 0.032
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
  # For each batch in the dataloader
  for i in range(Niter):
    real_y,real_c = RandomShuffle(fileName, ratio)
    real_y = real_y.to(device)
    real_c = real_c.to(device)
    netD.zero_grad()
    # Format batch
    b_size = real_y.size(0)
    d_real_input = torch.cat((real_y, real_c), dim=1)
    label = torch.full((b_size,1), real_label, dtype=torch.float, device=device)
    # Forward pass real batch through D
    output = netD(d_real_input)
    errD_real = criterion(output, label)
    # Calculate gradients for D in backward pass
    errD_real.backward()
    D_x = output.mean().item()

    ## Train with fake batch
    # Generate batch of random vectors
    noise = (torch.rand(b_size, InputVector_dim, device=device) * 2) - 1  # Noise between -1 and 1
    label.fill_(fake_label)

    d_fake_input = torch.cat((noise, real_c), dim=1)
    # Forward pass fake batch with D
    output = netD(d_fake_input)
    # Calculate D's loss on the all-fake batch
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    # Compute error of D as sum over the fake and the real batches
    errD = errD_real + errD_fake
    # optimize Discriminator
    optimizerD.step()

  

    # Output training stats
    if i % 10 == 0:
      loss_d_val = errD.detach().cpu().item()
      
      print('[%d/%d][%d/%d]\tLoss_D: %.4f\tD(Real): %.4f\tD(Fake): %.4f'
                  % (epoch, num_epochs, i, Niter, loss_d_val, D_x, D_G_z1))

      D_losses.append(loss_d_val)
  iters += 1

torch.save({
    'epoch': epoch, 
    'Discriminator_state_dict': netD.state_dict(),
    'D_optimizer_state_dict': optimizerD.state_dict(),
    'D_losses': D_losses, 
    'Para_dim': Para_dim, 
}, PATH)
 
  
        
    
    
    
    
    
