"""Script for calibrating split conformal prediction intervals and evaluating coverage.

Strict Non-Leakage:
- Uses ONLY the CALIBRATION partition for fitting non-conformity quantiles.
- Evaluates on the TEST partition and OOD sets.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.dataset import DSSEDataset
from dsse.metrics import compute_all_metrics
from dsse.models.physics_gnn import PhysicsGNNEstimator
from dsse.reproducibility import save_metadata, set_seed, setup_logger
from dsse.topology import extract_feeder_topology
from dsse.uncertainty import SplitConformalCalibrator, run_conformal_evaluation

logger = setup_logger("calibrate_conformal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--availability", type=float, default=0.40)
    p.add_argument("--noise_v", type=float, default=0.005)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--lambda_physics", type=float, default=0.10)
    return p.parse_args()


def load_model(cfg, graph, args, model_tag: str):
    device = torch.device("cpu")
    G = torch.tensor(graph.y_bus_real, dtype=torch.float32, device=device)
    B = torch.tensor(graph.y_bus_imag, dtype=torch.float32, device=device)
    N = graph.num_nodes
    nom_th = np.zeros(N, dtype=np.float32)
    for i, nd in enumerate(graph.nodes):
        p = nd.phase
        nom_th[i] = 0.0 if p == 1 else (-2.0 * np.pi / 3.0 if p == 2 else 2.0 * np.pi / 3.0)
    theta_nom = torch.tensor(nom_th, dtype=torch.float32, device=device)

    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device)

    model = PhysicsGNNEstimator(
        node_input_dim=6,
        edge_input_dim=4,
        G=G, B=B,
        theta_nom=theta_nom,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lambda_physics=args.lambda_physics,
    ).to(device)

    ckpt_path = cfg.paths.results_dir / "raw" / f"model_{model_tag}.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found. Running with initialized weights.")

    model.eval()
    return model, edge_index, edge_attr


def predict_dataset(model, dataset, indices: List[int], edge_index, edge_attr):
    x_node, _, y_true = dataset.get_tensors(indices)
    with torch.no_grad():
        preds = model(x_node, edge_index, edge_attr)
    return y_true.numpy(), preds.numpy()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = AppConfig.load_all()

    master_file = cfg.paths.feeder_dir / "IEEE123Master.dss"
    graph = extract_feeder_topology(master_file)

    tag = f"normal_avail{int(args.availability*100)}_noise{int(args.noise_v*1000)}_miss0_bad0"
    model_tag = f"physics_gnn_seed{args.seed}_avail{int(args.availability*100)}"

    model, edge_index, edge_attr = load_model(cfg, graph, args, model_tag)

    # Load data
    states_path = cfg.paths.data_dir / "raw" / "sim_states_normal.npz"
    meas_path = cfg.paths.data_dir / "interim" / f"measurements_{tag}.npz"
    dataset = DSSEDataset.load_from_npz(states_path, meas_path)

    with open(cfg.paths.data_dir / "splits" / "normal_splits.json") as f:
        splits = json.load(f)

    # 1. Predictions on CALIBRATION set
    y_cal_true, y_cal_pred = predict_dataset(model, dataset, splits["cal"], edge_index, edge_attr)

    # 2. Predictions on TEST set (In-Distribution)
    y_test_true, y_test_pred = predict_dataset(model, dataset, splits["test"], edge_index, edge_attr)

    # 3. Fit Conformal Calibrators for nominal levels
    eval_results = {}
    alphas = [0.10, 0.05]  # 90% and 95% nominal coverage

    for alpha in alphas:
        nominal = 1.0 - alpha
        calibrator = SplitConformalCalibrator(alpha=alpha)
        calibrator.calibrate(y_cal_true, y_cal_pred)

        # In-distribution test evaluation
        res_test = calibrator.evaluate(y_test_true, y_test_pred, label=f"in_distribution_test_nom_{int(nominal*100)}")
        eval_results[f"test_nom_{int(nominal*100)}"] = res_test
        logger.info(f"Nominal {nominal*100:.0f}% -> Empirical Test Coverage: {res_test['empirical_coverage']*100:.2f}%, Mean Width: {res_test['mean_interval_width']:.5f} pu")

    # 4. Evaluate under OOD distribution shifts
    for ood_regime in ["ood_high_loading", "ood_low_loading", "ood_high_solar_der"]:
        ood_states_path = cfg.paths.data_dir / "raw" / f"sim_states_{ood_regime}.npz"
        ood_meas_path = cfg.paths.data_dir / "interim" / f"measurements_{ood_regime}_avail{int(args.availability*100)}_noise{int(args.noise_v*1000)}_miss0_bad0.npz"

        if ood_states_path.exists() and ood_meas_path.exists():
            ood_dataset = DSSEDataset.load_from_npz(ood_states_path, ood_meas_path)
            ood_indices = list(range(min(len(ood_dataset), 300)))
            y_ood_true, y_ood_pred = predict_dataset(model, ood_dataset, ood_indices, edge_index, edge_attr)

            cal95 = SplitConformalCalibrator(alpha=0.05)
            cal95.calibrate(y_cal_true, y_cal_pred)
            res_ood = cal95.evaluate(y_ood_true, y_ood_pred, label=f"ood_{ood_regime}_nom_95")
            eval_results[f"ood_{ood_regime}_nom_95"] = res_ood
            logger.info(f"OOD {ood_regime} (Nominal 95%) -> Empirical Coverage: {res_ood['empirical_coverage']*100:.2f}%, Mean Width: {res_ood['mean_interval_width']:.5f} pu")

    results_dir = cfg.paths.results_dir / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"conformal_results_{model_tag}.json"
    save_metadata(eval_results, out_file)
    logger.info(f"Conformal evaluation results saved to {out_file}")


if __name__ == "__main__":
    main()
