"""Split conformal prediction for calibrated voltage estimation uncertainty intervals."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from dsse.reproducibility import setup_logger

logger = setup_logger("uncertainty")


class SplitConformalCalibrator:
    """Split conformal prediction for DSSE voltage magnitude intervals.

    Procedure
    ---------
    1. On the *calibration set* (never used for training), compute non-conformity
       scores:  s_i = |y_i - y_hat_i|  for each node-scenario pair.
    2. For nominal coverage level alpha, compute the (1-alpha) quantile of the
       calibration scores (with finite-sample correction +1 in denominator):
           q_hat = Quantile(s_1,...,s_m; ceil((m+1)*(1-alpha))/m)
    3. At test time: interval = [y_hat - q_hat, y_hat + q_hat].

    Coverage guarantee: Provided calibration and test data are exchangeable,
    the empirical coverage is >= 1-alpha.  This guarantee does NOT hold under
    distribution shift (OOD scenarios); such failures are explicitly evaluated.

    Note: We use per-node quantiles to allow the interval width to reflect
    node-level uncertainty heterogeneity. Global quantile is also computed.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """
        Args:
            alpha: Miscoverage level. Coverage target = 1 - alpha.
                   E.g. alpha=0.05 targets 95% coverage.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.target_coverage = 1.0 - alpha
        self._global_quantile: Optional[float] = None
        self._nodewise_quantiles: Optional[np.ndarray] = None
        self._n_cal: int = 0

    def calibrate(
        self,
        y_cal: np.ndarray,      # [M, N] ground-truth voltages on calibration set
        y_hat_cal: np.ndarray,  # [M, N] model predictions on calibration set
    ) -> "SplitConformalCalibrator":
        """Computes non-conformity quantiles from the calibration partition.

        Args:
            y_cal:     Ground-truth voltage magnitudes (pu), shape [M_cal, N_nodes].
            y_hat_cal: Model predictions, shape [M_cal, N_nodes].

        Returns:
            self (for chaining)
        """
        if y_cal.shape != y_hat_cal.shape:
            raise ValueError("y_cal and y_hat_cal must have the same shape.")
        M, N = y_cal.shape
        self._n_cal = M

        # Non-conformity scores: |y - y_hat| per (scenario, node)
        scores = np.abs(y_cal - y_hat_cal)  # [M, N]

        # Global quantile: pool all node errors
        all_scores = scores.ravel()
        level = np.ceil((M * N + 1) * (1.0 - self.alpha)) / (M * N)
        level = min(level, 1.0)
        self._global_quantile = float(np.quantile(all_scores, level))

        # Per-node quantiles (allows heterogeneous interval widths)
        self._nodewise_quantiles = np.zeros(N, dtype=np.float64)
        for j in range(N):
            node_scores = scores[:, j]
            node_level = np.ceil((M + 1) * (1.0 - self.alpha)) / M
            node_level = min(node_level, 1.0)
            self._nodewise_quantiles[j] = np.quantile(node_scores, node_level)

        logger.info(
            f"Conformal calibration complete: alpha={self.alpha}, "
            f"target_coverage={self.target_coverage:.2%}, M_cal={M}, N_nodes={N}, "
            f"global_q={self._global_quantile:.6f}, "
            f"nodewise_q_mean={self._nodewise_quantiles.mean():.6f}"
        )
        return self

    def predict_intervals(
        self,
        y_hat: np.ndarray,   # [M_test, N] or [N]
        mode: str = "nodewise",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constructs prediction intervals.

        Args:
            y_hat: Predictions. Shape [M_test, N] or [N].
            mode:  'nodewise' (heterogeneous widths) or 'global' (uniform width).

        Returns:
            Tuple (lower, upper) arrays of the same shape as y_hat.
        """
        if self._global_quantile is None:
            raise RuntimeError("Must call calibrate() before predict_intervals().")

        if mode == "nodewise" and self._nodewise_quantiles is not None:
            q = self._nodewise_quantiles  # [N]
        else:
            q = self._global_quantile

        lower = y_hat - q
        upper = y_hat + q
        return lower, upper

    def evaluate(\
        self,
        y_true: np.ndarray,   # [M, N]
        y_hat: np.ndarray,    # [M, N]
        mode: str = "nodewise",
        label: str = "test",
    ) -> Dict[str, float]:
        """Evaluates conformal intervals on a given dataset split.

        Args:
            y_true:  Ground-truth voltage magnitudes, shape [M, N].
            y_hat:   Model predictions, shape [M, N].
            mode:    'nodewise' or 'global'.
            label:   Descriptive label (e.g. 'test', 'ood_high_loading').

        Returns:
            Dict with empirical coverage, coverage_error, mean_width, etc.
        """
        lower, upper = self.predict_intervals(y_hat, mode=mode)

        in_interval = (y_true >= lower) & (y_true <= upper)
        empirical_cov = float(np.mean(in_interval))
        coverage_error = empirical_cov - self.target_coverage
        widths = upper - lower

        results = {
            "label": label,
            "target_coverage": self.target_coverage,
            "empirical_coverage": empirical_cov,
            "coverage_error": coverage_error,
            "mean_interval_width": float(np.mean(widths)),
            "median_interval_width": float(np.median(widths)),
            "max_interval_width": float(np.max(widths)),
            "n_cal": self._n_cal,
            "alpha": self.alpha,
            "mode": mode,
        }
        logger.info(
            f"[{label}] Coverage: {empirical_cov:.3%} "
            f"(target={self.target_coverage:.2%}, error={coverage_error:+.3%}), "
            f"mean_width={results['mean_interval_width']:.6f}"
        )
        return results


def run_conformal_evaluation(
    calibrator_90: SplitConformalCalibrator,
    calibrator_95: SplitConformalCalibrator,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    label: str = "test",
) -> Dict[str, Dict]:
    """Runs both 90% and 95% conformal interval evaluations and returns combined results."""
    results = {
        "coverage_90": calibrator_90.evaluate(y_true, y_hat, label=label),
        "coverage_95": calibrator_95.evaluate(y_true, y_hat, label=label),
    }
    return results
