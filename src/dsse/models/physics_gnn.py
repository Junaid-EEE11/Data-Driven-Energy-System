"""Physics-regularised GNN: GNN + physics-consistency loss (proposed model)."""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsse.models.gnn import GNNEstimator
from dsse.physics import PhysicsLoss
from dsse.reproducibility import setup_logger

logger = setup_logger("model.physics_gnn")


class PhysicsGNNEstimator(nn.Module):
    """Proposed model: GNN + physics-consistency regularisation.

    Architecture:
    -------------
    - Same GNN backbone as GNNEstimator (ablation A1).
    - Additional PhysicsLoss term (lambda_physics * L_physics) during training.
    - At inference: returns point estimates (same as GNN alone).
    - PhysicsLoss uses the flat-angle admittance residual described in physics.py.

    Total loss:
        L_total = L_supervised + lambda_physics * L_physics

    Setting lambda_physics = 0 reproduces the GNN-only baseline (A1).

    Ablation map:
        A0: MLPBaseline (topology-agnostic NN)
        A1: GNNEstimator (lambda_physics = 0)
        A2: PhysicsGNNEstimator (lambda_physics > 0)   <-- this class
        A3: GNNEstimator + conformal calibration
        A4: PhysicsGNNEstimator + conformal calibration (full proposed method)
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        G: torch.Tensor,
        B: torch.Tensor,
        theta_nom: torch.Tensor | None = None,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0,
        lambda_physics: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn = GNNEstimator(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.physics_loss = PhysicsLoss(G=G, B=B, theta_nom=theta_nom, lambda_physics=lambda_physics)
        self.lambda_physics = lambda_physics
        logger.info(
            f"PhysicsGNNEstimator: node_in={node_input_dim}, edge_in={edge_input_dim}, "
            f"hidden={hidden_dim}, layers={n_layers}, lambda_physics={lambda_physics}"
        )

    def forward(
        self,
        x: torch.Tensor,           # [B, N, node_input_dim] or [N, node_input_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, edge_input_dim]
    ) -> torch.Tensor:
        """Returns V_mag_pu predictions. Shape: [B, N] or [N]."""
        return self.gnn(x, edge_index, edge_attr)

    def compute_total_loss(
        self,
        v_pred: torch.Tensor,      # [B, N] or [N]
        v_true: torch.Tensor,      # [B, N] or [N]
        z_v: torch.Tensor,         # [B, N] or [N]
        z_p: torch.Tensor,         # [B, N] or [N]
        z_q: torch.Tensor,         # [B, N] or [N]
        mask_v: torch.Tensor,      # [B, N] or [N]
        mask_p: torch.Tensor,      # [B, N] or [N]
        mask_q: torch.Tensor,      # [B, N] or [N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes (L_total, L_supervised, L_physics).

        Returns:
            (L_total, L_supervised, L_physics) - all scalar tensors.
        """
        L_sup = F.mse_loss(v_pred, v_true)
        L_phys = self.physics_loss(
            v_mag_pred=v_pred,
            v_mag_meas=z_v,
            p_meas=z_p,
            q_meas=z_q,
            mask_v=mask_v,
            mask_p=mask_p,
            mask_q=mask_q,
        )
        L_total = L_sup + L_phys
        return L_total, L_sup, L_phys
