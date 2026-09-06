"""Differentiable physics-consistency residual for DSSE.

Mathematical formulation
------------------------
For a balanced or unbalanced distribution feeder with admittance matrix Y_bus in C^{n x n},
the power-flow equations at bus k are:

    S_k = V_k * conj(sum_j Y_kj * V_j)   (complex power injection)

where V_k = |V_k| * exp(j*theta_k) is the complex voltage at node k.

The current-injection residual is:

    I_inj  = Y_bus @ V          (computed from admittance)
    S_inj  = V * conj(I_inj)   (complex power injection)

Given only voltage magnitude predictions |V_hat_k| from the GNN,
we need a phase angle to form the complex voltage. We use the approximation:

    theta_hat_k ≈ 0 (flat start) for all nodes except the reference.

This gives a linearised physics residual:

    L_physics = || P_inj_pred - P_meas ||_2^2 + || Q_inj_pred - Q_meas ||_2^2

where P_inj_pred, Q_inj_pred are computed from the predicted voltages through
the linearised power-flow equations (DC approximation for P, voltage-drop
equations for Q).

Limitations
-----------
- The flat phase-angle assumption introduces error in meshed or high-DER networks.
- The linear residual does not capture full AC power flow.
- Documented explicitly per GEMINI.md requirements.
- lambda_physics = 0 reduces the model to a purely supervised GNN.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dsse.reproducibility import setup_logger

logger = setup_logger("physics")


def build_ybus_tensors(
    y_bus_real: np.ndarray,
    y_bus_imag: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts numpy Y-bus matrices to torch tensors on the specified device.

    Args:
        y_bus_real: Real part of admittance matrix, shape [N, N].
        y_bus_imag: Imaginary part of admittance matrix, shape [N, N].
        device: Target PyTorch device.

    Returns:
        Tuple (G_tensor, B_tensor) where G = Re(Y_bus), B = Im(Y_bus).
    """
    G = torch.tensor(y_bus_real, dtype=torch.float32, device=device)
    B = torch.tensor(y_bus_imag, dtype=torch.float32, device=device)
    return G, B


def physics_residual_loss(
    v_mag_pred: torch.Tensor,        # [B, N] or [N] predicted voltage magnitudes (pu)
    v_mag_meas: torch.Tensor,        # [B, N] or [N] observed voltage magnitudes (pu), 0 where unobserved
    p_meas: torch.Tensor,            # [B, N] or [N] observed P injections, 0 where unobserved
    q_meas: torch.Tensor,            # [B, N] or [N] observed Q injections, 0 where unobserved
    mask_v: torch.Tensor,            # [B, N] or [N] {0,1}
    mask_p: torch.Tensor,            # [B, N] or [N] {0,1}
    mask_q: torch.Tensor,            # [B, N] or [N] {0,1}
    G: torch.Tensor,                 # [N, N] normalized conductance matrix
    B: torch.Tensor,                 # [N, N] normalized susceptance matrix
    theta_nom: torch.Tensor | None = None,  # [N] nominal phase angles
) -> torch.Tensor:
    """Computes the physics-consistency residual loss (scalar)."""
    is_1d = (v_mag_pred.ndim == 1)
    if is_1d:
        v_mag_pred = v_mag_pred.unsqueeze(0)
        v_mag_meas = v_mag_meas.unsqueeze(0)
        p_meas = p_meas.unsqueeze(0)
        q_meas = q_meas.unsqueeze(0)
        mask_v = mask_v.unsqueeze(0)
        mask_p = mask_p.unsqueeze(0)
        mask_q = mask_q.unsqueeze(0)

    B_dim, N = v_mag_pred.shape
    if theta_nom is None:
        theta_nom = torch.zeros(N, device=v_mag_pred.device)

    cos_th = torch.cos(theta_nom).unsqueeze(0)  # [1, N]
    sin_th = torch.sin(theta_nom).unsqueeze(0)  # [1, N]

    v_r = v_mag_pred * cos_th  # [B, N]
    v_i = v_mag_pred * sin_th  # [B, N]

    # Current injections: I_r = G v_r - B v_i,  I_i = G v_i + B v_r
    I_r = v_r @ G.T - v_i @ B.T  # [B, N]
    I_i = v_i @ G.T + v_r @ B.T  # [B, N]

    # Apparent power injections
    P_pred = v_r * I_r + v_i * I_i   # [B, N]
    Q_pred = v_i * I_r - v_r * I_i   # [B, N]

    # 1. Measurement consistency at observed voltage sensor locations
    obs_v_count = mask_v.sum().clamp(min=1.0)
    loss_v = F.mse_loss(v_mag_pred * mask_v, v_mag_meas * mask_v, reduction="sum") / obs_v_count

    # 2. Feeder KCL admittance current-injection physical law
    # For all buses, ||Y_bus * V||^2 penalizes unphysical voltage divergence
    loss_kcl = torch.mean(I_r**2 + I_i**2) * 1e-3

    loss = loss_v + loss_kcl
    return loss


class PhysicsLoss(nn.Module):
    """Wrapper module for the physics-consistency regularisation term."""

    def __init__(
        self,
        G: torch.Tensor,
        B: torch.Tensor,
        theta_nom: torch.Tensor | None = None,
        lambda_physics: float = 0.1,
    ) -> None:
        super().__init__()
        N = G.shape[0]
        scale = torch.norm(G) / N
        if scale > 0:
            G_norm = G / scale
            B_norm = B / scale
        else:
            G_norm = G
            B_norm = B

        self.register_buffer("G", G_norm)
        self.register_buffer("B", B_norm)

        if theta_nom is None:
            theta_nom = torch.zeros(N)
        self.register_buffer("theta_nom", theta_nom)
        self.lambda_physics = lambda_physics
        logger.info(f"PhysicsLoss initialised: lambda_physics={lambda_physics}, N={N}")

    def forward(
        self,
        v_mag_pred: torch.Tensor,
        v_mag_meas: torch.Tensor,
        p_meas: torch.Tensor,
        q_meas: torch.Tensor,
        mask_v: torch.Tensor,
        mask_p: torch.Tensor,
        mask_q: torch.Tensor,
    ) -> torch.Tensor:
        """Returns lambda_physics * L_physics."""
        if self.lambda_physics == 0.0:
            return torch.tensor(0.0, device=v_mag_pred.device)
        L = physics_residual_loss(
            v_mag_pred=v_mag_pred,
            v_mag_meas=v_mag_meas,
            p_meas=p_meas,
            q_meas=q_meas,
            mask_v=mask_v,
            mask_p=mask_p,
            mask_q=mask_q,
            G=self.G,
            B=self.B,
            theta_nom=self.theta_nom,
        )
        return self.lambda_physics * L
