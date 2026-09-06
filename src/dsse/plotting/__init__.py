"""Publication-ready plotting utilities for DSSE."""

from dsse.plotting.topology import plot_feeder_topology
from dsse.plotting.errors import plot_error_vs_sparsity, plot_voltage_profile, plot_spatial_errors
from dsse.plotting.robustness import (
    plot_error_vs_noise,
    plot_error_vs_missingness,
    plot_error_vs_bad_data,
    plot_ood_comparison,
    plot_ablation_comparison,
)
from dsse.plotting.calibration import plot_calibration_curve, plot_interval_width_vs_sparsity

__all__ = [
    "plot_feeder_topology",
    "plot_error_vs_sparsity",
    "plot_voltage_profile",
    "plot_spatial_errors",
    "plot_error_vs_noise",
    "plot_error_vs_missingness",
    "plot_error_vs_bad_data",
    "plot_ood_comparison",
    "plot_ablation_comparison",
    "plot_calibration_curve",
    "plot_interval_width_vs_sparsity",
]
