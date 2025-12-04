# tests/test_client.py
"""
Unit tests for LocalClient
pytest tests/test_client.py -v
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from client.client import LocalClient


@pytest.fixture
def dummy_cfg():
    return {
        "lr": 0.01,
        "E": 2,
        "sigma_max": 5.0,
        "clip_norm": 1.0,
        "delta": 1e-5,
        "epsilon_total": 10.0,
    }


@pytest.fixture
def dummy_loader():
    # 100 samples, 10 classes, 28×28 images
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=False)


@pytest.fixture
def dummy_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )


@pytest.fixture
def client(dummy_model, dummy_loader, dummy_cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LocalClient(
        client_id=0,
        model=dummy_model,
        train_loader=dummy_loader,
        cfg=dummy_cfg,
        device=device,
    )


# ---------- test cases ---------- #

def test_local_train_shape(client):
    """Output state_dict must have same keys & shape as global model."""
    global_sd = client.model.state_dict()
    updated = client.local_train(global_sd, noise_scale=0.5)
    assert isinstance(updated, dict)
    for k, v in global_sd.items():
        assert k in updated
        assert updated[k].shape == v.shape


def test_grad_norm_positive(client):
    """Gradient norm must be ≥ 0."""
    global_sd = client.model.state_dict()
    gn = client.grad_norm(global_sd)
    assert gn >= 0.0


def test_quality_score_range(client):
    """Quality score must be in [0, 1]."""
    q = client.quality_score()
    assert 0.0 <= q <= 1.0


def test_noise_injection_changes_params(client):
    """Ensure added noise actually changes parameters."""
    global_sd = client.model.state_dict()
    updated1 = client.local_train(global_sd, noise_scale=0.0)   # no noise
    updated2 = client.local_train(global_sd, noise_scale=1.0)   # high noise
    diff = 0.0
    for k in global_sd:
        diff += torch.norm(updated2[k] - updated1[k]).item()
    assert diff > 0.1, "Noise did not change parameters significantly"


def test_preserve_device(client):
    """Returned tensors must be on CPU."""
    global_sd = client.model.state_dict()
    updated = client.local_train(global_sd, noise_scale=0.5)
    for v in updated.values():
        assert v.device.type == "cpu"


def test_clip_norm_enforced(client):
    """Gradients are clipped to cfg['clip_norm']."""
    global_sd = client.model.state_dict()
    # trigger backward
    client.grad_norm(global_sd)
    total_norm = 0.0
    for p in client.model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    assert total_norm <= client.cfg["clip_norm"] + 1e-3


# ---------- run with pytest ---------- #
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
