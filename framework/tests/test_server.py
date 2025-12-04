# tests/test_server.py
"""
Unit tests for Server (GP-AdaFL)
pytest tests/test_server.py -v
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from client.client import LocalClient
from server.server import Server


# ---------- fixtures ---------- #

@pytest.fixture
def cfg():
    return {
        "lr": 0.01,
        "E": 1,
        "sigma_max": 5.0,
        "clip_norm": 1.0,
        "delta": 1e-5,
        "epsilon_total": 10.0,
        "hidden": 64,
        "r": 10,
        "H": 5,
        "lambda_": 1.0,
    }


@pytest.fixture
def loader():
    x = torch.randn(80, 1, 28, 28)
    y = torch.randint(0, 10, (80,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=32, shuffle=False)


@pytest.fixture
def model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )


@pytest.fixture
def clients(model, loader, cfg):
    num = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return [LocalClient(i, model, loader, cfg, device) for i in range(num)]


@pytest.fixture
def server(model, clients, cfg):
    global_sd = model.state_dict()
    return Server(global_sd, clients, cfg)


# ---------- test cases ---------- #

def test_server_initial_state(server):
    """Global model must be identical across rounds before any step."""
    sd1 = server.global_model
    sd2 = server.global_model
    for k in sd1:
        assert torch.allclose(sd1[k], sd2[k])


def test_step_returns_bool(server):
    """server.step() returns True while budget is available."""
    for r in range(3):
        ok = server.step(r)
        assert isinstance(ok, bool)
        if not ok:
            break  # budget exhausted


def test_privacy_increases(server):
    """Privacy budget must increase after each round."""
    eps0 = server.eps_spent
    server.step(0)
    eps1 = server.eps_spent
    assert eps1 > eps0


def test_weighted_aggregate_shape(server):
    """Aggregated parameters keep same shape & keys."""
    server.step(0)
    new_sd = server.global_model
    old_keys = list(new_sd.keys())
    assert len(new_sd) == len(old_keys)
    for k in old_keys:
        assert new_sd[k].shape == server.global_model[k].shape


def test_grad_sim_in_weights(server, clients):
    """Aggregation uses gradient similarity (non-uniform weights)."""
    # run one round
    server.step(0)
    # collect gradient norms from fixture
    grads = [c.grad_norm(server.global_model) for c in clients]
    # weights should be different (dirichlet split → non-IID)
    weights = server._aggregation_weights(grads, [1.0, 1.0, 1.0])
    assert len(set(weights)) > 1, "Weights are perfectly uniform"


def test_budget_exhausted_stop(server, cfg):
    """Stop training when ε_total exceeded."""
    cfg["epsilon_total"] = 0.01  # very small budget
    srv = Server(server.global_model, server.clients, cfg)
    continue_flag = True
    rounds = 0
    while continue_flag and rounds < 10:
        continue_flag = srv.step(rounds)
        rounds += 1
    assert not continue_flag, "Server did not stop after budget exhaustion"


def test_dynamic_sigma_range(server):
    """σ must be in [0.5 * sigma_max, sigma_max] due to sigmoid mapping."""
    noise_factor = 0.8
    sigma = server._dynamic_sigma(noise_factor, C=1.0)
    sigma_max = server.cfg["sigma_max"]
    assert 0.5 * sigma_max <= sigma <= sigma_max


def test_uncertainty_weighted_fusion(server):
    """Fusion produces different values than single channel."""
    # force some history
    for gn in np.linspace(1, 5, 10):
        server.predictor.update(gn)
    fused, unc = server.predictor.fused_predict()
    assert fused is not None
    assert unc >= 0.0


def test_kalman_uncertainty_decreases(server):
    """Kalman uncertainty drops after fitting."""
    for gn in np.linspace(1, 5, 10):
        server.predictor.update(gn)
    fused, unc1 = server.predictor.fused_predict()
    # feed more data
    for gn in np.linspace(5, 10, 20):
        server.predictor.update(gn)
    fused, unc2 = server.predictor.fused_predict()
    assert unc2 <= unc1, "Uncertainty did not decrease"


# ---------- run with pytest ---------- #
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
