# import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from model_zoo.crosscoder import CrossCoder


def train_crosscoder(
    latents1,
    latents2,
    input_dim=8,
    output_dim=8,
    epochs=5,
    lr=1e-3,
    batch_size=32
):
    """
    Train CrossCoder to map latents1 -> latents2.
    latents1, latents2 are Tensors of shape [N, latent_dim].
    Returns the trained crosscoder model
    """
    dataset = TensorDataset(latents1, latents2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    crosscoder = CrossCoder(input_dim, output_dim)
    optimizer = optim.Adam(crosscoder.parameters(), lr=lr)

    for epoch in range(epochs):
        for in_batch, out_batch in loader:
            pred = crosscoder(in_batch)
            loss = F.mse_loss(pred, out_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return crosscoder
