import torch
import torch.utils.data as data
import numpy as np


def generate_toy_data(n_samples=1000, seed=0):
    """
    simple 2D dataset for binary classification.
    X ~ N(0, I) and label y = 1 if (x1 + x2 > 0), else 0.
    Returns train_dataset, test_dataset
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Convert to torch tensors
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    # Create a TensorDataset
    full_dataset = data.TensorDataset(X_tensor, y_tensor)

    # Split into train/test (80/20)
    train_size = int(0.8 * n_samples)
    test_size = n_samples - train_size
    train_dataset, test_dataset = data.random_split(full_dataset,
                                                    [train_size, test_size])

    return train_dataset, test_dataset
