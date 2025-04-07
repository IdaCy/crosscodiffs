import torch
from torch.utils.data import DataLoader


def generate_latents(model, dataset, batch_size=32):
    """
    Given a trained model and dataset, return the latents (+ labels)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for X_batch, y_batch in loader:
            latents, _ = model(X_batch)
            all_latents.append(latents)
            all_labels.append(y_batch)

    # Concatenate all mini-batches
    latents_tensor = torch.cat(all_latents, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    return latents_tensor, labels_tensor
