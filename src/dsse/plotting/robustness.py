"""Plotting utilities for measurement robustness, OOD distribution shift, and ablations."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error_vs_noise(
    df_noise: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots estimation error vs Gaussian measurement noise std."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df_noise["model"].unique():
        sub = df_noise[df_noise["model"] == model].sort_values("noise_std")
        ax.plot(sub["noise_std"] * 100.0, sub["rmse"], marker="o", label=model, linewidth=2)

    ax.set_xlabel("Voltage Measurement Noise Std (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Voltage Magnitude RMSE (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("Robustness to Measurement Noise", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_error_vs_missingness(
    df_missing: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots estimation error vs random sensor missingness rate."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df_missing["model"].unique():
        sub = df_missing[df_missing["model"] == model].sort_values("missing_rate")
        ax.plot(sub["missing_rate"] * 100.0, sub["rmse"], marker="s", label=model, linewidth=2)

    ax.set_xlabel("Sensor Missingness / Dropout Rate (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Voltage Magnitude RMSE (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("Robustness to Communication Dropout & Missing Data", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_error_vs_bad_data(
    df_bad: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plots estimation error vs fraction of corrupted / bad data measurements."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df_bad["model"].unique():
        sub = df_bad[df_bad["model"] == model].sort_values("bad_data_fraction")
        ax.plot(sub["bad_data_fraction"] * 100.0, sub["rmse"], marker="^", label=model, linewidth=2)

    ax.set_xlabel("Bad Data / Gross Error Fraction (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Voltage Magnitude RMSE (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("Robustness to Malicious / Gross Sensor Data Corruption", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_ood_comparison(
    df_ood: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Bar plot comparing model performance under in-distribution and OOD test regimes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df_ood.pivot(index="regime", columns="model", values="rmse")
    pivot.plot(kind="bar", ax=ax, rot=0, width=0.8)

    ax.set_xlabel("Operating Regime / Distribution Shift", fontsize=11, fontweight="bold")
    ax.set_ylabel("Voltage Magnitude RMSE (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("Estimation Performance Under Operating Condition Shift (OOD)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_ablation_comparison(
    df_ablation: pd.DataFrame,
    save_path: Optional[Path | str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Bar plot comparing ablation configurations A0 through A4."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df_ablation["ablation_id"], df_ablation["rmse"], color="#34495e", width=0.6)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.0002, f"{yval:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Ablation Configuration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Voltage Magnitude RMSE (p.u.)", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Contribution of Topology, Physics, & Conformal", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig
