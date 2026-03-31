import torch.nn as nn
import torch.optim as optim
import numpy as np
     

# Discriminator Code ----------------------------
class Discriminator(nn.Module):
  def __init__(self, Para_dim):
    super(Discriminator, self).__init__()
    self.Para_dim = Para_dim
    self.main = nn.Sequential(
      nn.Linear(Para_dim, 256),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.3),
      nn.Linear(256, 128),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.3),
      nn.Linear(128,64 ),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.3),
      nn.Linear(64, 32),
      nn.LeakyReLU(0.2, inplace=True),
      nn.Dropout(0.3),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )

  def forward(self, input):
    return self.main(input)


