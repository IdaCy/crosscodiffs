import torch
import torch.nn as nn
# import torch.nn.functional as F


class mlp2(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        # Slightly different activation, now sigmoid
        latent = torch.sigmoid(self.fc1(x))
        logits = self.fc2(latent)
        return latent, logits
