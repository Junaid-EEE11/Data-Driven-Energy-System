"""Ablation study script evaluating components A0-A4 and physics loss weight lambda_physics sweep.

Ablations:
- A0: Topology-Agnostic Neural Network (MLP)
- A1: GNN Only (lambda_physics = 0)
- A2: GNN + Physics Loss (lambda_physics = 0.10)
- A3: GNN + Conformal Calibration
- A4: Full Physics-GNN + Conformal Framework
- Lambda Sweep: lambda in [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.dataset import DSSEDataset
from dsse.metrics import compute_all_metrics
from dsse.models.gnn import GNNEstimator
from dsse.models.mlp import MLPBaseline
from dsse.models.physics_gnn import PhysicsGNNEstimator
from dsse.reproducibility import save_metadata, set_seed, setup_logger
from dsse.topology import extract_feeder_topology
from dsse.uncertainty import SplitConformalCalibrator

logger = setup_logger("run_ablations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--availability", type=float, default=0.40)
    p.add_argument("--noise_v", type=float, default=0.005)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig.load_all()
    device = torch.device("cpu")

    master_file = cfg.paths.feeder_dir / "IEEE123Master.dss"
    graph = extract_feeder_topology(master_file)
    N = graph.num_nodes

    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device)

    tag = f"normal_avail{int(args.availability*100)}_noise{int(args.noise_v*1000)}_miss0_bad0"
    states_path = cfg.paths.data_dir / "raw" / "sim_states_normal.npz"
    meas_path = cfg.paths.data_dir / "interim" / f"measurements_{tag}.npz"
    dataset = DSSEDataset.load_from_npz(states_path, meas_path)

    with open(cfg.paths.data_dir / "splits" / "normal_splits.json") as f:
        splits = json.load(f)

    cal_ids = splits["cal"]
    test_ids = splits["test"]

    results_dir = cfg.paths.results_dir / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for seed in args.seeds:
        set_seed(seed)
        avail_tag = int(args.availability * 100)

        # 1. A0: Topology-agnostic MLP
        m_path = results_dir / f"model_mlp_seed{seed}_avail{avail_tag}.pt"
        if m_path.exists():
            mlp = MLPBaseline(input_dim=6 * N, output_dim=N).to(device)
            mlp.load_state_dict(torch.load(m_path, map_location=device))
            mlp.eval()
            X_test = np.stack([
                np.concatenate([
                    dataset[i]["z_v"].numpy(), dataset[i]["z_p"].numpy(), dataset[i]["z_q"].numpy(),
                    dataset[i]["mask_v"].numpy(), dataset[i]["mask_p"].numpy(), dataset[i]["mask_q"].numpy(),
                ]) for i in test_ids
            ])
            y_test = np.stack([dataset[i]["y"].numpy() for i in test_ids])
            with torch.no_grad():
                preds_a0 = mlp(torch.tensor(X_test, dtype=torch.float32, device=device)).numpy()
            m_a0 = compute_all_metrics(y_test, preds_a0)
            records.append({"ablation_id": "A0_MLP_Topology_Agnostic", "seed": seed, "has_topology": False, "has_physics": False, "has_conformal": False, **m_a0})

        # 2. A1: GNN only
        g_path = results_dir / f"model_gnn_seed{seed}_avail{avail_tag}.pt"
        if g_path.exists():
            gnn = GNNEstimator(node_input_dim=6, edge_input_dim=4, hidden_dim=64, n_layers=3).to(device)
            gnn.load_state_dict(torch.load(g_path, map_location=device))
            gnn.eval()

            # Predict on test
            preds_a1_list, trues_list = [], []
            with torch.no_grad():
                for i in test_ids:
                    s = dataset[i]
                    x_b = torch.stack([s["z_v"], s["z_p"], s["z_q"], s["mask_v"], s["mask_p"], s["mask_q"]], dim=-1)
                    preds_a1_list.append(gnn(x_b, edge_index, edge_attr).numpy())
                    trues_list.append(s["y"].numpy())
            y_test = np.stack(trues_list)
            preds_a1 = np.stack(preds_a1_list)
            m_a1 = compute_all_metrics(y_test, preds_a1)
            records.append({"ablation_id": "A1_GNN_Only", "seed": seed, "has_topology": True, "has_physics": False, "has_conformal": False, **m_a1})

        # 3. A2 & A4: Physics-GNN
        pg_path = results_dir / f"model_physics_gnn_seed{seed}_avail{avail_tag}.pt"
        if pg_path.exists():
            G_t = torch.tensor(graph.y_bus_real, dtype=torch.float32, device=device)
            B_t = torch.tensor(graph.y_bus_imag, dtype=torch.float32, device=device)
            nom_th = np.zeros(N, dtype=np.float32)
            for i, nd in enumerate(graph.nodes):
                p = nd.phase
                nom_th[i] = 0.0 if p == 1 else (-2.0 * np.pi / 3.0 if p == 2 else 2.0 * np.pi / 3.0)
            theta_nom_t = torch.tensor(nom_th, dtype=torch.float32, device=device)
            pgnn = PhysicsGNNEstimator(
                node_input_dim=6, edge_input_dim=4, G=G_t, B=B_t,
                theta_nom=theta_nom_t,
                hidden_dim=64, n_layers=3, lambda_physics=0.10
            ).to(device)
            pgnn.load_state_dict(torch.load(pg_path, map_location=device))
            pgnn.eval()

            # Calibration predictions
            preds_cal_list, trues_cal_list = [], []
            preds_test_list, trues_test_list = [], []
            with torch.no_grad():
                for i in cal_ids:
                    s = dataset[i]
                    x_b = torch.stack([s["z_v"], s["z_p"], s["z_q"], s["mask_v"], s["mask_p"], s["mask_q"]], dim=-1)
                    preds_cal_list.append(pgnn(x_b, edge_index, edge_attr).numpy())
                    trues_cal_list.append(s["y"].numpy())
                for i in test_ids:
                    s = dataset[i]
                    x_b = torch.stack([s["z_v"], s["z_p"], s["z_q"], s["mask_v"], s["mask_p"], s["mask_q"]], dim=-1)
                    preds_test_list.append(pgnn(x_b, edge_index, edge_attr).numpy())
                    trues_test_list.append(s["y"].numpy())

            y_cal = np.stack(trues_cal_list); preds_cal = np.stack(preds_cal_list)
            y_test = np.stack(trues_test_list); preds_test = np.stack(preds_test_list)

            # A2 (Metrics without conformal)
            m_a2 = compute_all_metrics(y_test, preds_test)
            records.append({"ablation_id": "A2_Physics_GNN", "seed": seed, "has_topology": True, "has_physics": True, "has_conformal": False, **m_a2})

            # A3 (GNN + Conformal)
            if g_path.exists():
                cal_gnn = SplitConformalCalibrator(alpha=0.05)
                # Calibrate on cal split using GNN predictions
                preds_cal_gnn = []
                with torch.no_grad():
                    for i in cal_ids:
                        s = dataset[i]
                        x_b = torch.stack([s["z_v"], s["z_p"], s["z_q"], s["mask_v"], s["mask_p"], s["mask_q"]], dim=-1)
                        preds_cal_gnn.append(gnn(x_b, edge_index, edge_attr).numpy())
                cal_gnn.calibrate(y_cal, np.stack(preds_cal_gnn))
                res_a3 = cal_gnn.evaluate(y_test, preds_a1)
                records.append({"ablation_id": "A3_GNN_Conformal", "seed": seed, "has_topology": True, "has_physics": False, "has_conformal": True, **m_a1, "empirical_coverage_95": res_a3["empirical_coverage"], "mean_width_95": res_a3["mean_interval_width"]})

            # A4 (Full Physics-GNN + Conformal)
            cal_full = SplitConformalCalibrator(alpha=0.05)
            cal_full.calibrate(y_cal, preds_cal)
            res_a4 = cal_full.evaluate(y_test, preds_test)
            records.append({"ablation_id": "A4_Full_Physics_GNN_Conformal", "seed": seed, "has_topology": True, "has_physics": True, "has_conformal": True, **m_a2, "empirical_coverage_95": res_a4["empirical_coverage"], "mean_width_95": res_a4["mean_interval_width"]})

    df_abl = pd.DataFrame(records)
    out_csv = results_dir / "ablation_results_raw.csv"
    df_abl.to_csv(out_csv, index=False)
    logger.info(f"Ablation study results saved to {out_csv} ({len(df_abl)} total rows)")


if __name__ == "__main__":
    main()
