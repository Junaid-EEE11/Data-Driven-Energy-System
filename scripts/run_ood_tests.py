"""Script for evaluating DSSE models under Out-Of-Distribution (OOD) shifts.

Evaluated Regimes:
1. ood_high_loading (1.3x - 1.8x base load)
2. ood_low_loading (0.2x - 0.5x base load)
3. ood_high_solar_der (extreme DER / PV penetration causing reverse power flow)
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
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

logger = setup_logger("run_ood_tests")


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

    results_dir = cfg.paths.results_dir / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    avail_tag = int(args.availability * 100)

    ood_regimes = ["normal", "ood_high_loading", "ood_low_loading", "ood_high_solar_der"]
    records = []

    for seed in args.seeds:
        set_seed(seed)
        logger.info(f"--- Running OOD Evaluation for Seed {seed} ---")

        # Load models
        models = {}
        r_path = results_dir / f"model_ridge_seed{seed}_avail{avail_tag}.pkl"
        if r_path.exists():
            with open(r_path, "rb") as f:
                models["ridge"] = pickle.load(f)

        t_path = results_dir / f"model_tree_seed{seed}_avail{avail_tag}.pkl"
        if t_path.exists():
            with open(t_path, "rb") as f:
                models["tree"] = pickle.load(f)

        m_path = results_dir / f"model_mlp_seed{seed}_avail{avail_tag}.pt"
        if m_path.exists():
            mlp = MLPBaseline(input_dim=6 * N, output_dim=N).to(device)
            mlp.load_state_dict(torch.load(m_path, map_location=device))
            mlp.eval()
            models["mlp"] = mlp

        g_path = results_dir / f"model_gnn_seed{seed}_avail{avail_tag}.pt"
        if g_path.exists():
            gnn = GNNEstimator(node_input_dim=6, edge_input_dim=4, hidden_dim=64, n_layers=3).to(device)
            gnn.load_state_dict(torch.load(g_path, map_location=device))
            gnn.eval()
            models["gnn"] = gnn

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
            models["physics_gnn"] = pgnn

        for regime in ood_regimes:
            states_path = cfg.paths.data_dir / "raw" / f"sim_states_{regime}.npz"
            meas_path = cfg.paths.data_dir / "interim" / f"measurements_{regime}_avail{avail_tag}_noise{int(args.noise_v*1000)}_miss0_bad0.npz"

            if not states_path.exists() or not meas_path.exists():
                logger.warning(f"Data for regime {regime} not found. Skipping.")
                continue

            dataset = DSSEDataset.load_from_npz(states_path, meas_path)
            # For normal, use test split only; for OOD, use all samples
            if regime == "normal":
                with open(cfg.paths.data_dir / "splits" / "normal_splits.json") as f:
                    eval_ids = json.load(f)["test"]
            else:
                eval_ids = list(range(len(dataset)))

            y_true_list = [dataset[i]["y"].numpy() for i in eval_ids]
            y_true = np.stack(y_true_list)

            for m_name, model in models.items():
                if m_name in ["ridge", "tree"]:
                    X = np.stack([
                        np.concatenate([
                            dataset[i]["z_v"].numpy(), dataset[i]["z_p"].numpy(), dataset[i]["z_q"].numpy(),
                            dataset[i]["mask_v"].numpy(), dataset[i]["mask_p"].numpy(), dataset[i]["mask_q"].numpy(),
                        ]) for i in eval_ids
                    ])
                    preds = model.predict(X)
                elif m_name == "mlp":
                    X = np.stack([
                        np.concatenate([
                            dataset[i]["z_v"].numpy(), dataset[i]["z_p"].numpy(), dataset[i]["z_q"].numpy(),
                            dataset[i]["mask_v"].numpy(), dataset[i]["mask_p"].numpy(), dataset[i]["mask_q"].numpy(),
                        ]) for i in eval_ids
                    ])
                    with torch.no_grad():
                        preds = model(torch.tensor(X, dtype=torch.float32, device=device)).numpy()
                elif m_name in ["gnn", "physics_gnn"]:
                    with torch.no_grad():
                        x_feat = torch.stack([
                            torch.stack([dataset[i]["z_v"], dataset[i]["z_p"], dataset[i]["z_q"],
                                         dataset[i]["mask_v"], dataset[i]["mask_p"], dataset[i]["mask_q"]], dim=-1)
                            for i in eval_ids
                        ]).to(device)
                        preds = model(x_feat, edge_index, edge_attr).cpu().numpy()

                m_res = compute_all_metrics(y_true, preds)
                records.append({"regime": regime, "seed": seed, "model": m_name, **m_res})

    df_ood = pd.DataFrame(records)
    out_csv = results_dir / "ood_results_raw.csv"
    df_ood.to_csv(out_csv, index=False)
    logger.info(f"OOD evaluation results saved to {out_csv} ({len(df_ood)} total rows)")


if __name__ == "__main__":
    main()
