# utils/metrics.py
"""
Core metrics for GP-AdaFL
- gradient similarity (cosine in latent space)
- data quality via predictive entropy
- fast accuracy proxy on a validation subset
All computations are torch-based for GPU acceleration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from pykalman import KalmanFilter


# ---------- gradient similarity ---------- #
def grad_sim(gk: torch.Tensor, G: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Cosine similarity between client gradient gk and global gradient G.
    Inputs are flattened tensors.
    """
    gk, G = gk.flatten(), G.flatten()
    dot = torch.dot(gk, G)
    norm = gk.norm() * G.norm() + eps
    return float((dot / norm).clamp(-1.0, 1.0))


# ---------- predictive entropy (data quality) ---------- #
def predictive_entropy(model, loader, device, max_batch: Optional[int] = None):
    """
    Compute average predictive entropy over dataloader.
    max_batch: early-stop for speed (useful on large sets)
    """
    model.eval()
    ent_list = []
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            if max_batch and idx >= max_batch:
                break
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            ent_list.append(ent)
    if not ent_list:
        return 0.0
    return float(torch.cat(ent_list).mean())


def data_quality(model, loader, device, max_batch: Optional[int] = 32) -> float:
    """
    Q_ent = 1 - (average predictive entropy / log(num_classes))
    Range in [0, 1], higher is better.
    """
    n_cls = loader.dataset.classes if hasattr(loader.dataset, 'classes') else 10
    n_cls = len(n_cls) if isinstance(n_cls, list) else n_cls
    ent = predictive_entropy(model, loader, device, max_batch)
    return 1.0 - float(ent / np.log(n_cls))


# ---------- fast accuracy proxy ---------- #
def accuracy_proxy(model, loader, device, max_batch: Optional[int] = 16):
    """
    Quick accuracy on first max_batch batches (for quality estimation).
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if idx >= max_batch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / (total + 1e-8)


# ---------- Kalman-based gradient tracking ---------- #
class GradientKalman:
    """
    Lightweight 1-D Kalman filter for online gradient norm smoothing
    state: [1]   observation: [1]
    """

    def __init__(self, process_var: float = 1e-4, measure_var: float = 1e-2):
        self.kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=1,
            transition_matrices=[[1.0]],
            observation_matrices=[[1.0]],
            transition_covariance=[[process_var]],
            observation_covariance=[[measure_var]],
            initial_state_mean=0.0,
            initial_state_covariance=[[1.0]],
        )
        self._state_mean = 0.0
        self._state_cov = 1.0

    def update(self, observation: float) -> float:
        self._state_mean, self._state_cov = self.kf.filter_update(
            self._state_mean, self._state_cov, observation
        )
        return float(self._state_mean)

    def uncertainty(self) -> float:
        return float(self._state_cov[0, 0])


# ---------- uncertainty-weighted fusion (generic) ---------- #
def bayesian_fusion(values: torch.Tensor, uncertainties: torch.Tensor) -> torch.Tensor:
    """
    values: [N]    uncertainties: [N] (std or variance)
    returns weighted mean and combined variance
    """
    inv_var = 1.0 / (uncertainties + 1e-8)
    weights = inv_var / inv_var.sum()
    fused = (values * weights).sum()
    fused_var = 1.0 / inv_var.sum()
    return fused, fused_var


# ---------- self-test ---------- #
if __name__ == "__main__":
    # 1. gradient similarity
    g1, g2 = torch.randn(100), torch.randn(100)
    print("Grad sim:", grad_sim(g1, g2))

    # 2. data quality (MNIST stub)
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    ds = datasets.MNIST(root="./data", train=True, download=True,
                        transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(),
                                torch.nn.Linear(28 * 28, 64),
                                torch.nn.ReLU(),
                                torch.nn.Linear(64, 10))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Data quality (entropy):", data_quality(model, loader, device, max_batch=4))

    # 3. Kalman smoothing
    kf = GradientKalman()
    for gn in np.linspace(1, 5, 20):
        smooth = kf.update(gn)
    print("Kalman smoothed gradient norm:", smooth, "uncertainty:", kf.uncertainty())
