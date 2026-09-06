"""Evaluation metrics for state estimation: MAE, RMSE, MaxAE, P95AE, and uncertainty metrics."""

from __future__ import annotations
from typing import Dict, Optional

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def max_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(y_true - y_pred)))


def p95_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """95th-percentile absolute error."""
    return float(np.percentile(np.abs(y_true - y_pred), 95))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_labels: Optional[list] = None,
) -> Dict[str, float]:
    """Computes the full set of state-estimation accuracy metrics."""
    abs_err = np.abs(y_true.ravel() - y_pred.ravel())
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(abs_err ** 2))),
        "max_ae": float(np.max(abs_err)),
        "p95_ae": float(np.percentile(abs_err, 95)),
        "p50_ae": float(np.median(abs_err)),
        "std_ae": float(np.std(abs_err)),
    }


def compute_nodewise_metrics(
    y_true: np.ndarray,   # [S, N]
    y_pred: np.ndarray,   # [S, N]
) -> Dict[str, np.ndarray]:
    """Returns per-node error statistics across scenarios."""
    abs_err = np.abs(y_true - y_pred)  # [S, N]
    return {
        "node_mae": np.mean(abs_err, axis=0),
        "node_rmse": np.sqrt(np.mean(abs_err ** 2, axis=0)),
        "node_max_ae": np.max(abs_err, axis=0),
        "node_p95_ae": np.percentile(abs_err, 95, axis=0),
    }


def empirical_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Computes the empirical coverage of prediction intervals."""
    in_interval = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(in_interval))


def interval_width_stats(lower: np.ndarray, upper: np.ndarray) -> Dict[str, float]:
    """Computes width statistics of prediction intervals."""
    widths = upper - lower
    return {
        "mean_width": float(np.mean(widths)),
        "median_width": float(np.median(widths)),
        "std_width": float(np.std(widths)),
        "max_width": float(np.max(widths)),
    }
