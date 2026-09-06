"""Topology-agnostic MLP baseline for DSSE."""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn

from dsse.reproducibility import setup_logger

logger = setup_logger("model.mlp")


class MLPBaseline(nn.Module):
    """Multi-layer perceptron state estimator (topology-agnostic)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        layers: List[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 1.0)
        logger.info(f"MLPBaseline: {input_dim} -> {hidden_dims} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input: [B, input_dim]. Output: [B, output_dim]."""
        return self.net(x)
