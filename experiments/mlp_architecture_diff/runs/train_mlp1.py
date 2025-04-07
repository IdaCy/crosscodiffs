# import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import ModelOne from the model zoo
from model_zoo.mlp2 import ModelOne


def train_model1(
    train_dataset,
    input_dim=2,
    latent_dim=8,
    output_dim=2,
    epochs=5,
    lr=1e-3,
    batch_size=32
):
    """
    Train ModelOne on the given dataset.
    Returns the trained model.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    model = ModelOne(input_dim, latent_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            _, outputs = model(X_batch)
            loss = F.cross_entropy(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
