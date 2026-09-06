"""Unit tests for split conformal uncertainty calibration."""

import numpy as np
import pytest

from dsse.uncertainty import SplitConformalCalibrator, run_conformal_evaluation


def test_conformal_calibration_coverage():
    """Verify that conformal prediction achieves >= 1 - alpha coverage on exchangeable data."""
    rng = np.random.default_rng(42)
    M_cal = 500
    M_test = 500
    N = 10

    # True state + model prediction with Gaussian error
    y_true_all = rng.normal(1.0, 0.05, size=(M_cal + M_test, N))
    noise = rng.normal(0.0, 0.02, size=(M_cal + M_test, N))
    y_hat_all = y_true_all + noise

    y_cal = y_true_all[:M_cal]
    y_hat_cal = y_hat_all[:M_cal]
    y_test = y_true_all[M_cal:]
    y_hat_test = y_hat_all[M_cal:]

    calibrator = SplitConformalCalibrator(alpha=0.10)
    calibrator.calibrate(y_cal, y_hat_cal)

    results = calibrator.evaluate(y_test, y_hat_test, label="test_exchangeable")
    # For alpha=0.10, target is 90%
    assert results["empirical_coverage"] >= 0.85, f"Coverage too low: {results['empirical_coverage']}"
    assert results["mean_interval_width"] > 0


def test_conformal_does_not_access_test_labels():
    """Calibration must only use calibration data; test evaluation is purely feedforward."""
    rng = np.random.default_rng(100)
    y_cal = rng.normal(1.0, 0.02, size=(50, 10))
    y_hat_cal = rng.normal(1.0, 0.02, size=(50, 10))

    calibrator = SplitConformalCalibrator(alpha=0.05)
    calibrator.calibrate(y_cal, y_hat_cal)

    # Test predictions without test labels
    y_hat_test = rng.normal(1.0, 0.02, size=(20, 10))
    lower, upper = calibrator.predict_intervals(y_hat_test)
    assert lower.shape == (20, 10)
    assert upper.shape == (20, 10)
    assert np.all(upper >= lower)
