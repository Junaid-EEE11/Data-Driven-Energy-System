"""Topology-aware Graph Neural Network for DSSE (no physics loss)."""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dsse.reproducibility import setup_logger

logger = setup_logger("model.gnn")


class GraphConvLayer(nn.Module):
    """Single message-passing layer with edge features via additive aggregation."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.msg_proj = nn.Linear(node_dim + edge_dim, out_dim)
        self.update_proj = nn.Linear(node_dim + out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # [B, N, node_dim] or [N, node_dim]
        edge_index: torch.Tensor,  # [2, E] directed source -> target
        edge_attr: torch.Tensor,   # [E, edge_dim]
    ) -> torch.Tensor:
        is_unbatched = (x.ndim == 2)
        if is_unbatched:
            x = x.unsqueeze(0)  # [1, N, node_dim]

        B, N, _ = x.shape
        src, tgt = edge_index[0], edge_index[1]
        E = edge_index.size(1)

        # Gather source node features for all edges: [B, E, node_dim]
        x_src = x[:, src, :]
        e_exp = edge_attr.unsqueeze(0).expand(B, -1, -1)  # [B, E, edge_dim]
        msg_in = torch.cat([x_src, e_exp], dim=-1)         # [B, E, node_dim + edge_dim]
        msgs = F.gelu(self.msg_proj(msg_in))               # [B, E, out_dim]

        # Aggregate messages into target nodes: [B, N, out_dim]
        deg = torch.bincount(tgt, minlength=N).float().clamp(min=1.0).view(1, N, 1).to(x.device)
        agg = torch.zeros(B, N, msgs.size(-1), device=x.device)
        tgt_exp = tgt.view(1, E, 1).expand(B, E, msgs.size(-1))
        agg.scatter_add_(1, tgt_exp, msgs)
        agg = agg / deg

        upd_in = torch.cat([x, agg], dim=-1)               # [B, N, node_dim + out_dim]
        out = F.gelu(self.update_proj(upd_in))             # [B, N, out_dim]
        out = self.drop(self.norm(out))

        return out.squeeze(0) if is_unbatched else out


class GNNEstimator(nn.Module):
    """Topology-aware Graph Neural Network for distribution system state estimation."""

    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_embed = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_input_dim, hidden_dim)
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Initialize output layer to 1.0 pu
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, 1.0)
        logger.info(f"GNNEstimator: node_in={node_input_dim}, edge_in={edge_input_dim}, hidden={hidden_dim}, layers={n_layers}")

    def forward(
        self,
        x: torch.Tensor,           # [B, N, node_input_dim] or [N, node_input_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr: torch.Tensor,   # [E, edge_input_dim]
    ) -> torch.Tensor:
        """Returns V_mag_pu predictions for each node. Shape: [B, N] or [N]."""
        is_unbatched = (x.ndim == 2)
        if is_unbatched:
            x = x.unsqueeze(0)

        h = F.gelu(self.node_embed(x))
        e = F.gelu(self.edge_embed(edge_attr))
        for layer in self.conv_layers:
            h = h + layer(h, edge_index, e)  # residual connections

        out = self.head(h).squeeze(-1)       # [B, N]
        return out.squeeze(0) if is_unbatched else out
