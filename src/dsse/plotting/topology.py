"""Visualization of IEEE 123 feeder graph, bus phases, and sensor locations."""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dsse.topology import FeederGraph


def plot_feeder_topology(
    graph: FeederGraph,
    meter_mask: Optional[np.ndarray] = None,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots the feeder graph with bus-phase connectivity and meter placement.

    Args:
        graph: Extracted FeederGraph object.
        meter_mask: Optional binary array [N_nodes] indicating sensor locations.
        save_path: Destination path for saving the figure.
        dpi: Image resolution.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    G = graph.networkx_graph

    # Generate layout using spring layout with deterministic seed
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    # Node categorization
    slack_nodes = [n.node_idx for n in graph.nodes if n.is_slack]
    load_nodes = [n.node_idx for n in graph.nodes if n.is_load and not n.is_slack]
    other_nodes = [n.node_idx for n in graph.nodes if not n.is_load and not n.is_slack]

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#bdc3c7", width=0.8, alpha=0.7)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, ax=ax, node_size=20, node_color="#95a5a6", label="Pass-through Bus")
    nx.draw_networkx_nodes(G, pos, nodelist=load_nodes, ax=ax, node_size=35, node_color="#3498db", label="Load Bus")
    nx.draw_networkx_nodes(G, pos, nodelist=slack_nodes, ax=ax, node_size=120, node_color="#e74c3c", node_shape="s", label="Substation Slack Bus")

    # Overlay meter locations if provided
    if meter_mask is not None:
        meter_indices = [i for i, m in enumerate(meter_mask) if m > 0.5 and i not in slack_nodes]
        nx.draw_networkx_nodes(
            G, pos, nodelist=meter_indices, ax=ax,
            node_size=60, node_color="#2ecc71", node_shape="^",
            label="Smart Meter (AMI)"
        )

    ax.set_title(f"IEEE 123-Node Distribution Feeder Topology ({graph.num_nodes} Bus-Phase Nodes)", fontsize=14, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
