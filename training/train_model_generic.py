# import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# import torch.nn.functional as F


def train_model_generic(
    model,
    train_dataset,
    criterion,
    epochs=10,
    lr=1e-3,
    batch_size=32,
    shuffle=True,
    device="cuda"
):
    """
    Trains ANY PyTorch model on the given dataset

    Parameters:
    model : torch.nn.Module
    train_dataset : torch.utils.data.Dataset
    criterion : callable - loss function, torch.nn.CrossEntropyLoss()/MSELoss
    epochs : int
    lr : float
    batch_size : int
    shuffle : bool - shuffle data each epoch?
    device : str

    Returns:
    model : torch.nn.Module
    """
    # Move model to the specified device (cpu or cuda)
    model.to(device)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle)

    # Define an optimizer (Adam as an example)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            # Move batch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            
            # If the model outputs multiple values (like latents and logits),
            # might do something like:
            # latents, logits = outputs
            # / check the type. for now: outputs is directly
            # the "prediction" used for loss calculation

            # Compute loss
            loss = criterion(outputs, y_batch)

            # Backprop & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        # print average loss per epoch
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    return model
