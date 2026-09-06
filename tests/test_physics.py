"""Tests for physics residual loss computation."""

import numpy as np
import torch
import pytest


@pytest.fixture
def small_ybus():
    """Minimal 3-node Y-bus for unit testing."""
    G = np.array([[2.0, -1.0, -1.0],
                  [-1.0, 2.0, 0.0],
                  [-1.0, 0.0, 1.0]], dtype=np.float32)
    B = np.array([[-2.0, 1.0, 1.0],
                  [1.0, -2.0, 0.0],
                  [1.0, 0.0, -1.0]], dtype=np.float32)
    return G, B


def test_physics_loss_is_finite(small_ybus):
    from dsse.physics import physics_residual_loss
    G, B = [torch.tensor(x) for x in small_ybus]
    N = 3
    v_pred = torch.ones(N)
    v_meas = torch.tensor([1.0, 0.95, 0.98])
    p_meas = torch.tensor([0.5, -0.2, -0.3])
    q_meas = torch.tensor([0.1, -0.05, -0.05])
    mask = torch.ones(N)
    loss = physics_residual_loss(v_pred, v_meas, p_meas, q_meas, mask, mask, mask, G, B)
    assert torch.isfinite(loss), f"Physics loss is not finite: {loss}"


def test_physics_loss_non_negative(small_ybus):
    from dsse.physics import physics_residual_loss
    G, B = [torch.tensor(x) for x in small_ybus]
    N = 3
    v_pred = torch.tensor([1.0, 0.95, 0.98])
    v_meas = torch.tensor([1.0, 0.95, 0.98])
    p_meas = torch.zeros(N)
    q_meas = torch.zeros(N)
    mask = torch.ones(N)
    loss = physics_residual_loss(v_pred, v_meas, p_meas, q_meas, mask, mask, mask, G, B)
    assert loss >= 0.0, f"Physics loss must be non-negative, got {loss}"


def test_zero_lambda_returns_zero():
    from dsse.physics import PhysicsLoss
    G = torch.eye(3)
    B = torch.zeros(3, 3)
    pl = PhysicsLoss(G=G, B=B, lambda_physics=0.0)
    v = torch.ones(3)
    z = torch.zeros(3)
    mask = torch.ones(3)
    result = pl(v, z, z, z, mask, mask, mask)
    assert result.item() == 0.0
