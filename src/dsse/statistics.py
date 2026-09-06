"""Statistical analysis utilities: bootstrap CI, multi-seed aggregation, and paired comparison."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    statistic: str = "mean",
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Nonparametric bootstrap confidence interval.

    Returns (point_estimate, lower_ci, upper_ci).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    stat_fn = {"mean": np.mean, "median": np.median, "std": np.std}[statistic]
    point = float(stat_fn(arr))

    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_stats[b] = stat_fn(sample)

    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    return point, lower, upper


def aggregate_seeds(
    results_per_seed: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregates scalar metric dicts from multiple random seeds into mean +/- std."""
    if not results_per_seed:
        return {}
    keys = results_per_seed[0].keys()
    aggregated: Dict[str, Dict[str, float]] = {}
    for key in keys:
        vals = np.array([r[key] for r in results_per_seed])
        aggregated[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return aggregated


def paired_bootstrap_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 0,
) -> Tuple[float, float]:
    """Paired bootstrap test for H0: mean(errors_A) == mean(errors_B).

    Returns (observed_delta, p_value_two_sided).
    The unit of analysis is each independent scenario/snapshot.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(errors_a)
    b = np.asarray(errors_b)
    assert len(a) == len(b), "Paired samples must have equal length"
    delta_obs = float(np.mean(a) - np.mean(b))
    diff = a - b
    centered = diff - np.mean(diff)  # center under H0

    count_extreme = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(centered), size=len(centered))
        boot_delta = float(np.mean(centered[idx]))
        if abs(boot_delta) >= abs(delta_obs):
            count_extreme += 1
    p_value = (count_extreme + 1) / (n_bootstrap + 1)
    return delta_obs, p_value
