import torch
import torch.nn as nn
# import torch.nn.functional as F


class mlp1(nn.Module):
    def __init__(self, input_dim=2, latent_dim=8, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        # First layer -> ReLU -> latent representation
        latent = torch.relu(self.fc1(x))
        # Final output for classification
        logits = self.fc2(latent)
        return latent, logits
