# utils/data_utils.py
"""
Data-processing toolkit for GP-AdaFL
- Dirichlet non-IID split
- label-noise injection (medical annotation error simulation)
- local data-quality estimation (entropy / accuracy proxy)
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from typing import List, Tuple, Optional
import random
from sklearn.model_selection import train_test_split


# ---------- public non-IID splitter ---------- #
def dirichlet_split(dataset: torch.utils.data.Dataset,
                    n_clients: int,
                    alpha: float = 0.1,
                    seed: int = 42) -> List[List[int]]:
    """
    Returns: list of indices per client,  ∑|list| = len(dataset)
    Dirichlet(alpha) on label distribution → Non-IID
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    n_classes = len(dataset.classes)
    # label distribution per client
    dist = rng.dirichlet([alpha] * n_classes, n_clients)

    class2idx = {cls: np.where(targets == cls)[0] for cls in range(n_classes)}
    for cls in range(n_classes):
        rng.shuffle(class2idx[cls])

    client_indices = [[] for _ in range(n_clients)]
    for cls in range(n_classes):
        cls_idx = class2idx[cls]
        split = (dist[:, cls] * len(cls_idx)).astype(int)
        split[-1] = len(cls_idx) - split[:-1].sum()  # 补整
        pos = 0
        for cid, num in enumerate(split):
            client_indices[cid].extend(cls_idx[pos:pos + num].tolist())
            pos += num
    return client_indices


# ---------- medical annotation noise ---------- #
def add_label_noise(indices: List[int],
                    targets: List[int],
                    noise_rate: float = 0.1,
                    seed: int = 42) -> List[int]:
    """随机翻转标签，模拟医院标注错误"""
    rng = random.Random(seed)
    noisy = targets.copy()
    n_noisy = int(len(indices) * noise_rate)
    flip_pool = indices.copy()
    rng.shuffle(flip_pool)
    for idx in flip_pool[:n_noisy]:
        original = noisy[idx]
        choices = list(range(len(set(targets))))
        choices.remove(original)
        noisy[idx] = rng.choice(choices)
    return noisy


# ---------- local data-quality estimation ---------- #
class QualityEstimator:
    """
    1. entropy: 1 - avg(predictive entropy)
    2. acc_proxy: quick val-set accuracy (optional)
    3. scale: combine → [0,1]
    """

    def __init__(self, model, loader: torch.utils.data.DataLoader, device):
        self.model = model
        self.loader = loader
        self.device = device

    def entropy_score(self) -> float:
        """Q_ent = 1 - mean(softmax entropy)"""
        self.model.eval()
        ent_list = []
        with torch.no_grad():
            for x, _ in self.loader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                ent = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                ent_list.append(ent.cpu())
        ent = torch.cat(ent_list).mean().item()
        return 1.0 - float(ent / np.log(probs.shape[1]))  # 归一化

    def accuracy_proxy(self, val_ratio: float = 0.1) -> float:
        """在小验证集上快速估计准确率"""
        idx = list(range(len(self.loader.dataset)))
        train_idx, val_idx = train_test_split(idx, test_size=val_ratio,
                                              random_state=42,
                                              stratify=[self.loader.dataset.targets[i] for i in idx])
        val_set = Subset(self.loader.dataset, val_idx)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
        correct = total = 0
        self.model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total

    def combined(self, w_ent: float = 0.7, w_acc: float = 0.3) -> float:
        """Q_k = w_ent*entropy + w_acc*accuracy_proxy"""
        q_ent = self.entropy_score()
        q_acc = self.accuracy_proxy()
        return w_ent * q_ent + w_acc * q_acc


# ---------- small utility ---------- #
def split_train_val(dataset: Dataset, val_frac: float = 0.1, seed: int = 42):
    """返回 train_indices, val_indices (stratified)"""
    targets = np.array(dataset.targets)
    return train_test_split(
        np.arange(len(targets)),
        test_size=val_frac,
        random_state=seed,
        stratify=targets,
    )


# ---------- self-test ---------- #
if __name__ == "__main__":
    from torchvision import datasets, transforms

    ds = datasets.MNIST(root="./data", train=True, download=True,
                        transform=transforms.ToTensor())
    client_idx = dirichlet_split(ds, n_clients=10, alpha=0.1)
    print("Client 0 samples:", len(client_idx[0]))

    # 加噪声
    noisy_targets = add_label_noise(client_idx[0], ds.targets, noise_rate=0.1)
    print("Noisy labels for client 0 (first 10):", noisy_targets[:10])

    # 质量估计
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(28 * 28, 64),
                                torch.nn.ReLU(), torch.nn.Linear(64, 10))
    loader = torch.utils.data.DataLoader(
        Subset(ds, client_idx[0]), batch_size=64, shuffle=False)
    q_est = QualityEstimator(model, loader, device="cpu")
    print("Entropy score:", q_est.entropy_score())
    print("Combined quality:", q_est.combined())
