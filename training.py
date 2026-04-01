import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
PSD_folder_path = '/Users/harsha/Desktop/PhD_project/Machine_Learning/no_condition/'
if PSD_folder_path not in sys.path:
    sys.path.append(PSD_folder_path)
from GAN import Discriminator, Generator
from checkdata import RandomShuffle


# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
#------------------------------------------------------------------------    

PATH = '/Users/harsha/Desktop/PhD_project/Machine_Learning/no_condition/Weights.pt' 



# Initialize BCELoss function
criterion = nn.BCELoss()

#  convention for real and fake labels during training
real_label = 0.95
fake_label = 0.
lrD = 0.00001
lrG = 0.0002
beta1 = 0.5



Para_dim = 36               #(condition+input vector)
cond_dim=21                 #normalised porosity parameters using GMM (5 parameters normalised to 21 parameters)
InputVector_dim=15          #(normalised input parameters)
noiseVector_dim=64          # noise to Generator
# using  the Discriminator 
netD = Discriminator(Para_dim).to(device)
netG = Generator(noiseVector_dim).to(device)

#optimizer
optimizerD = optim.AdamW(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.AdamW(netG.parameters(), lr=lrG, betas=(beta1, 0.999))


if os.path.exists(PATH):
    print(f"--- Found existing model at {PATH}. Resuming training... ---")
    checkpoint = torch.load(PATH, weights_only=False, map_location=device)
    netG.load_state_dict(checkpoint['Generator_state_dict'])
    netD.load_state_dict(checkpoint['Discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['G_optimizer_state_dict'])
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
G_losses = []
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
    noise = torch.randn(b_size, noiseVector_dim, device=device)   # Noise between -1 and 1
    fake = netG(noise)
    #fake = netG(noise)
    label.fill_(fake_label)
    d_fake_input = torch.cat((fake.detach(), real_c), dim=1)
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

    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    # Since we just updated D, perform another forward pass of all-fake batch through D
    d_output_for_g = netD(torch.cat((fake, real_c), dim=1))
    # Calculate G's loss based on this output
    errG = criterion(d_output_for_g, label)
    # Calculate gradients for G
    errG.backward()
    # Update G
    optimizerG.step()

    # Output training stats
    if i % 10 == 0:
      loss_d_val = errD.detach().cpu().item()
      loss_g_val = errG.detach().cpu().item()
      print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                  % (epoch, num_epochs, i, Niter, loss_d_val, loss_g_val, D_x, D_G_z1))

        # Save Losses for plotting later
      G_losses.append(loss_g_val)
      D_losses.append(loss_d_val)
  iters += 1



torch.save({
    'epoch': epoch, 
    'Discriminator_state_dict': netD.state_dict(),
    'D_optimizer_state_dict': optimizerD.state_dict(),
    'D_losses': D_losses,
    'Generator_state_dict': netG.state_dict(), 
    'G_optimizer_state_dict': optimizerG.state_dict(),
    'Para_dim': Para_dim, 
    'G_losses': G_losses, 
}, PATH)
 
 
  
        
    
    
    
    
    
