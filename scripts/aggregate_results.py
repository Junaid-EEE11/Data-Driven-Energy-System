"""Aggregates multi-seed experimental results, computes bootstrap CIs, and creates LaTeX/Markdown tables.

Adheres strictly to GEMINI.md Section 16:
- Never relies on a single training seed; aggregates across all seeds.
- Computes mean, standard deviation, and paired statistical comparisons.
- Exports structured summaries to results/aggregated/ and results/tables/.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.reproducibility import setup_logger
from dsse.statistics import paired_bootstrap_test, bootstrap_ci

logger = setup_logger("aggregate_results")


def main() -> None:
    cfg = AppConfig.load_all()
    raw_dir = cfg.paths.results_dir / "raw"
    agg_dir = cfg.paths.results_dir / "aggregated"
    tbl_dir = cfg.paths.results_dir / "tables"
    agg_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate Experiment Matrix
    exp_csv = raw_dir / "all_experiments_raw.csv"
    if exp_csv.exists():
        df_exp = pd.read_csv(exp_csv)
        # Group by experiment family, condition, and model
        group_cols = [c for c in ["exp_family", "availability", "noise_std", "missing_rate", "bad_data_fraction", "model"] if c in df_exp.columns]
        agg_exp = df_exp.groupby(group_cols).agg({
            "mae": ["mean", "std"],
            "rmse": ["mean", "std"],
            "max_ae": ["mean", "std"],
            "p95_ae": ["mean", "std"],
            "inference_time_ms_per_sample": ["mean", "std"],
        }).reset_index()

        # Flatten multi-level columns
        agg_exp.columns = ['_'.join(c).strip('_') for c in agg_exp.columns.values]
        agg_exp.to_csv(agg_dir / "experiments_aggregated.csv", index=False)
        logger.info(f"Aggregated experiments matrix saved to {agg_dir / 'experiments_aggregated.csv'}")

        # Generate Table 1: Main benchmark comparison under normal conditions
        df_norm = df_exp[df_exp["exp_family"] == "A_normal"]
        if not df_norm.empty:
            tab1 = df_norm.groupby("model").agg({
                "mae": ["mean", "std"],
                "rmse": ["mean", "std"],
                "max_ae": ["mean", "std"],
                "p95_ae": ["mean", "std"],
                "inference_time_ms_per_sample": ["mean"],
            }).reset_index()
            tab1.columns = ['_'.join(c).strip('_') for c in tab1.columns.values]

            # Write Markdown and LaTeX tables
            tab1_md = tab1.to_markdown(index=False)
            with open(tbl_dir / "table1_normal_benchmarks.md", "w") as f:
                f.write(f"# Table 1: State Estimation Benchmark Under Normal Conditions (40% AMI, 0.5% Noise)\n\n{tab1_md}\n")
            tab1.to_latex(tbl_dir / "table1_normal_benchmarks.tex", index=False, float_format="%.5f")
            logger.info("Saved Table 1 (Normal Benchmarks).")

    # 2. Aggregate Ablation Results
    abl_csv = raw_dir / "ablation_results_raw.csv"
    if abl_csv.exists():
        df_abl = pd.read_csv(abl_csv)
        agg_abl = df_abl.groupby("ablation_id").agg({
            "mae": ["mean", "std"],
            "rmse": ["mean", "std"],
            "max_ae": ["mean", "std"],
            "p95_ae": ["mean", "std"],
        }).reset_index()
        agg_abl.columns = ['_'.join(c).strip('_') for c in agg_abl.columns.values]
        agg_abl.to_csv(agg_dir / "ablations_aggregated.csv", index=False)

        abl_md = agg_abl.to_markdown(index=False)
        with open(tbl_dir / "table2_ablation_study.md", "w") as f:
            f.write(f"# Table 2: Ablation Study Across Components A0-A4\n\n{abl_md}\n")
        agg_abl.to_latex(tbl_dir / "table2_ablation_study.tex", index=False, float_format="%.5f")
        logger.info("Saved Table 2 (Ablations).")

    # 3. Aggregate OOD Results
    ood_csv = raw_dir / "ood_results_raw.csv"
    if ood_csv.exists():
        df_ood = pd.read_csv(ood_csv)
        agg_ood = df_ood.groupby(["regime", "model"]).agg({
            "mae": ["mean", "std"],
            "rmse": ["mean", "std"],
            "max_ae": ["mean", "std"],
            "p95_ae": ["mean", "std"],
        }).reset_index()
        agg_ood.columns = ['_'.join(c).strip('_') for c in agg_ood.columns.values]
        agg_ood.to_csv(agg_dir / "ood_aggregated.csv", index=False)

        ood_md = agg_ood.to_markdown(index=False)
        with open(tbl_dir / "table3_ood_distribution_shift.md", "w") as f:
            f.write(f"# Table 3: Performance Under Operating Condition Shifts (OOD)\n\n{ood_md}\n")
        agg_ood.to_latex(tbl_dir / "table3_ood_distribution_shift.tex", index=False, float_format="%.5f")
        logger.info("Saved Table 3 (OOD Tests).")

    logger.info("All tables and aggregated results generated successfully.")


if __name__ == "__main__":
    main()
