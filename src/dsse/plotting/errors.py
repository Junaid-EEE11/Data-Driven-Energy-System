"""Plotting utilities for state estimation errors and voltage profiles."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error_vs_sparsity(
    df_results: pd.DataFrame,
    metric: str = "rmse",
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots estimation error vs sensor availability rate across all models.

    Args:
        df_results: DataFrame with columns ['model', 'availability', 'mae', 'rmse', ...]
        metric: Column to plot on y-axis ('rmse' or 'mae').
        save_path: Output file path.
        dpi: Image resolution.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    models = df_results["model"].unique()
    markers = ["o", "s", "^", "D", "v", "x"]
    colors = ["#7f8c8d", "#e67e22", "#3498db", "#9b59b6", "#2ecc71", "#e74c3c"]

    for i, model in enumerate(models):
        sub = df_results[df_results["model"] == model].sort_values("availability")
        ax.plot(
            sub["availability"] * 100.0,
            sub[metric],
            label=model,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Sensor Availability Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Voltage Magnitude {metric.upper()} (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("State Estimation Accuracy vs. Smart Meter Sparsity", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_voltage_profile(
    v_true: np.ndarray,
    v_pred: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    scenario_idx: int = 0,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots predicted vs true voltage magnitude profile across all nodes with conformal confidence bands."""
    fig, ax = plt.subplots(figsize=(12, 5))
    nodes = np.arange(len(v_true))

    ax.plot(nodes, v_true, label="Ground Truth (OpenDSS AC-PF)", color="#2c3e50", linewidth=1.5)
    ax.plot(nodes, v_pred, label="Predicted (Physics-GNN)", color="#e74c3c", linestyle="--", linewidth=1.5)

    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(
            nodes, lower_bound, upper_bound,
            color="#e74c3c", alpha=0.2, label="95% Conformal Prediction Interval"
        )

    ax.axhline(0.95, color="gray", linestyle=":", label="ANSI C84.1 Min (0.95 p.u.)")
    ax.axhline(1.05, color="gray", linestyle=":", label="ANSI C84.1 Max (1.05 p.u.)")

    ax.set_xlabel("Bus-Phase Node Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Voltage Magnitude (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title(f"Voltage Profile & Calibrated Uncertainty (Scenario #{scenario_idx})", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_spatial_errors(
    node_mae: np.ndarray,
    node_labels: Optional[List[str]] = None,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots node-wise error distribution sorted by bus-phase."""
    fig, ax = plt.subplots(figsize=(12, 4))
    nodes = np.arange(len(node_mae))
    ax.bar(nodes, node_mae, color="#3498db", width=0.8, alpha=0.85)

    ax.set_xlabel("Bus-Phase Node Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("Spatial Error Distribution Across Feeder Nodes", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
