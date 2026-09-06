"""Unit tests for all ML and baseline state estimation models."""

import numpy as np
import pytest
import torch

from dsse.models.linear import RidgeBaseline
from dsse.models.mlp import MLPBaseline
from dsse.models.tree import TreeBaseline
from dsse.models.gnn import GNNEstimator
from dsse.models.physics_gnn import PhysicsGNNEstimator
from dsse.baselines.wls import ThreePhaseWLS


def test_linear_baseline_fit_predict():
    """Ridge baseline fits and predicts correct shapes."""
    N = 278
    S = 20
    X = np.random.randn(S, 6 * N).astype(np.float32)
    y = np.random.randn(S, N).astype(np.float32)

    model = RidgeBaseline(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5, N)


def test_mlp_baseline_forward():
    """MLP forward pass produces correct output shape."""
    N = 278
    model = MLPBaseline(input_dim=6 * N, output_dim=N, hidden_dims=[64, 64])
    x = torch.randn(4, 6 * N)
    out = model(x)
    assert out.shape == (4, N)


def test_tree_baseline_fit_predict():
    """Tree baseline fits and predicts correct shapes."""
    N = 10
    S = 15
    X = np.random.randn(S, 6 * N)
    y = np.random.randn(S, N)

    model = TreeBaseline(n_estimators=10)
    model.fit(X, y)
    preds = model.predict(X[:3])
    assert preds.shape == (3, N)


def test_gnn_estimator_forward():
    """GNN estimator forward pass."""
    N = 278
    E = 300
    x = torch.randn(N, 6)  # node features
    edge_index = torch.randint(0, N, (2, E))
    edge_attr = torch.randn(E, 4)

    gnn = GNNEstimator(node_input_dim=6, edge_input_dim=4, hidden_dim=32, n_layers=2)
    out = gnn(x, edge_index, edge_attr)
    assert out.shape == (N,)


def test_physics_gnn_loss_computation():
    """Physics-regularised GNN computes total loss with finite scalar output."""
    N = 50
    G = torch.eye(N)
    B = torch.eye(N) * 0.1

    model = PhysicsGNNEstimator(
        node_input_dim=6,
        edge_input_dim=4,
        G=G,
        B=B,
        hidden_dim=32,
        n_layers=2,
        lambda_physics=0.05,
    )

    v_pred = torch.ones(N) * 1.02
    v_true = torch.ones(N) * 1.0
    z_v = torch.ones(N)
    z_p = torch.zeros(N)
    z_q = torch.zeros(N)
    mask = torch.ones(N)

    loss_tot, loss_sup, loss_phys = model.compute_total_loss(
        v_pred=v_pred,
        v_true=v_true,
        z_v=z_v,
        z_p=z_p,
        z_q=z_q,
        mask_v=mask,
        mask_p=mask,
        mask_q=mask,
    )

    assert torch.isfinite(loss_tot)
    assert loss_tot.item() > 0.0
    assert torch.isfinite(loss_sup)
    assert torch.isfinite(loss_phys)


def test_wls_estimator_runs():
    """Three-phase WLS estimator runs on mock system."""
    N = 10
    G = np.eye(N) * 2.0
    B = np.eye(N) * (-1.0)
    for i in range(N - 1):
        G[i, i + 1] = G[i + 1, i] = -1.0
        B[i, i + 1] = B[i + 1, i] = 0.5

    wls = ThreePhaseWLS(G=G, B=B, max_iter=5)
    z_v = np.ones(N) * 1.0
    z_p = np.zeros(N)
    z_q = np.zeros(N)
    mask_v = np.ones(N)
    mask_p = np.ones(N)
    mask_q = np.ones(N)

    res = wls.estimate(z_v, z_p, z_q, mask_v, mask_p, mask_q)
    assert "v_mag_pu" in res
    assert res["v_mag_pu"].shape == (N,)
