"""Plotting utilities for conformal uncertainty calibration and interval evaluation."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_calibration_curve(
    nominal_levels: List[float],
    empirical_coverages: List[float],
    ood_empirical_coverages: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots reliability diagram: empirical coverage vs nominal coverage (1 - alpha)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    nom_pct = np.array(nominal_levels) * 100.0
    emp_pct = np.array(empirical_coverages) * 100.0

    # Ideal calibration line (y = x)
    ax.plot([50, 100], [50, 100], "k--", label="Perfect Calibration (y = x)", linewidth=1.5)

    # In-distribution coverage
    ax.plot(nom_pct, emp_pct, "o-", color="#2ecc71", linewidth=2.5, markersize=7, label="In-Distribution (Test Split)")

    # OOD coverage lines (showing calibration deterioration)
    if ood_empirical_coverages:
        styles = [("s--", "#e74c3c"), ("^--", "#e67e22"), ("d--", "#9b59b6")]
        for i, (ood_name, ood_covs) in enumerate(ood_empirical_coverages.items()):
            style, color = styles[i % len(styles)]
            ax.plot(
                nom_pct, np.array(ood_covs) * 100.0,
                style, color=color, linewidth=1.8, markersize=6,
                label=f"OOD: {ood_name}"
            )

    ax.set_xlabel("Nominal Coverage Level 1 - alpha (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Empirical Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Conformal Prediction Calibration & Reliability Diagram", fontsize=13, fontweight="bold")
    ax.set_xlim(50, 100)
    ax.set_ylim(40, 102)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_interval_width_vs_sparsity(
    df_intervals: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots conformal prediction interval width vs sensor availability rate."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sub = df_intervals.sort_values("availability")
    ax.plot(
        sub["availability"] * 100.0,
        sub["mean_width"],
        marker="o",
        color="#3498db",
        linewidth=2,
        markersize=6,
        label="Mean Interval Width (95% Nominal)",
    )

    if "median_width" in sub.columns:
        ax.plot(
            sub["availability"] * 100.0,
            sub["median_width"],
            marker="s",
            color="#2980b9",
            linestyle="--",
            linewidth=1.8,
            markersize=5,
            label="Median Interval Width",
        )

    ax.set_xlabel("Sensor Availability Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Prediction Interval Width (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("Conformal Prediction Interval Width vs. Sensor Sparsity", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
