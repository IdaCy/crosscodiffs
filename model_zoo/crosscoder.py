# import torch
import torch.nn as nn
# import torch.nn.functional as F


class CrossCoder(nn.Module):
    def __init__(self, input_dim=8, output_dim=8):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Simple linear transform from mlp1's latent to mlp2's latent
        return self.fc(x)
