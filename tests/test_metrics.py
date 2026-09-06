"""Unit tests for DSSE evaluation metrics: MAE, RMSE, MaxAE, P95AE, and coverage."""

import numpy as np
import pytest

from dsse.metrics import mae, rmse, max_ae, p95_ae, compute_all_metrics, empirical_coverage, interval_width_stats


def test_error_metrics_zero_on_identical():
    """Metrics must be zero when true and pred are identical."""
    y = np.array([1.0, 1.02, 0.98, 1.05])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert max_ae(y, y) == 0.0
    assert p95_ae(y, y) == 0.0


def test_error_metrics_correct_values():
    """Verify exact numerical computation of error metrics."""
    y_true = np.array([1.0, 1.0, 1.0, 1.0])
    y_pred = np.array([1.1, 0.9, 1.2, 0.8])
    # abs errors: [0.1, 0.1, 0.2, 0.2]
    assert np.isclose(mae(y_true, y_pred), 0.15)
    assert np.isclose(rmse(y_true, y_pred), np.sqrt((0.01 + 0.01 + 0.04 + 0.04) / 4))
    assert np.isclose(max_ae(y_true, y_pred), 0.2)

    res = compute_all_metrics(y_true, y_pred)
    assert "mae" in res
    assert "rmse" in res
    assert "max_ae" in res
    assert "p95_ae" in res


def test_empirical_coverage():
    """Test conformal prediction interval coverage calculation."""
    y_true = np.array([1.0, 1.0, 1.0, 1.0])
    lower = np.array([0.9, 0.9, 1.05, 0.9])  # 3rd is missed (1.0 < 1.05)
    upper = np.array([1.1, 1.1, 1.20, 1.1])
    cov = empirical_coverage(y_true, lower, upper)
    assert cov == 0.75

    stats = interval_width_stats(lower, upper)
    assert np.isclose(stats["mean_width"], (0.2 + 0.2 + 0.15 + 0.2) / 4)
