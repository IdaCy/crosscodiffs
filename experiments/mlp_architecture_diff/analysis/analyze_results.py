import torch
import torch.nn.functional as F


def analyze_results(crosscoder, model1, model2, dataset):
    """
    Compute crosscoder's reconstruction error
       latent1 -> crosscoder -> latent2
    for the given dataset. Also calculates average dimension-wise correlation.

    Returns a dict with metrics: { 'mse': float, 'average_correlation': float,
                                   'dim_correlations': list }
    """
    crosscoder.eval()
    model1.eval()
    model2.eval()

    latents1 = []
    latents2 = []

    # Collect latents from model1 and model2
    for X, _ in dataset:
        X = X.unsqueeze(0)  # shape [1, input_dim]
        with torch.no_grad():
            l1, _ = model1(X)
            l2, _ = model2(X)
        latents1.append(l1)
        latents2.append(l2)

    latents1 = torch.cat(latents1, dim=0)  # shape [N, latent_dim]
    latents2 = torch.cat(latents2, dim=0)  # shape [N, latent_dim]

    # Crosscoder outputs
    with torch.no_grad():
        crosscoded = crosscoder(latents1)

    # MSE
    mse = F.mse_loss(crosscoded, latents2).item()

    # Dimension-wise correlation
    # We'll compute Pearson correlation dimension by dimension
    corrs = []
    for i in range(latents2.shape[1]):
        x_i = crosscoded[:, i]
        y_i = latents2[:, i]
        corr_matrix = torch.corrcoef(torch.stack([x_i, y_i]))
        corr_i = corr_matrix[0, 1].item()
        corrs.append(corr_i)

    avg_corr = sum(corrs) / len(corrs)

    return {
        "mse": mse,
        "average_correlation": avg_corr,
        "dim_correlations": corrs
    }
