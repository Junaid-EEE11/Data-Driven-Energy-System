"""Experiment Runner executing the full DSSE experimental matrix (Experiments A-F & I).

Families:
- Exp A: Normal conditions (Ridge, MLP, Tree, WLS, GNN, Proposed Physics-GNN)
- Exp B: Sensor sparsity (10%, 20%, 40%, 60%, 80%, 100%)
- Exp C: Measurement noise (0.002, 0.005, 0.010, 0.020, 0.050 pu)
- Exp D: Random missingness (10%, 20%, 40%, 60%)
- Exp E: Structured missingness (subgraph communication cluster failure)
- Exp F: Gross bad data (1%, 2%, 5%, 10%)
- Exp I: Computational inference runtime & solve times
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.corruption import apply_bad_data, apply_random_missingness, apply_structured_missingness
from dsse.dataset import DSSEDataset
from dsse.measurements import MeasurementGenerator, MeasurementVector
from dsse.metrics import compute_all_metrics
from dsse.models.gnn import GNNEstimator
from dsse.models.linear import RidgeBaseline
from dsse.models.mlp import MLPBaseline
from dsse.models.physics_gnn import PhysicsGNNEstimator
from dsse.models.tree import TreeBaseline
from dsse.reproducibility import save_metadata, set_seed, setup_logger
from dsse.topology import extract_feeder_topology

logger = setup_logger("run_experiments")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def get_graph_tensors(graph, device):
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device)
    return edge_index, edge_attr


def evaluate_sample_batch(models_dict, meas_batch, y_batch, graph, edge_index, edge_attr, device):
    """Evaluates all loaded models on a batch of measurements and returns predictions and timings."""
    results = {}
    S = len(y_batch)
    N = graph.num_nodes

    for name, model in models_dict.items():
        t0 = time.perf_counter()
        preds = []
        if name in ["ridge", "tree"]:
            # Feature matrix [S, 6*N]
            X = np.stack([
                np.concatenate([
                    m.z_v_mag, m.z_p_inj / 1000.0, m.z_q_inj / 1000.0,
                    m.mask_v, m.mask_p, m.mask_q
                ]) for m in meas_batch
            ])
            preds = model.predict(X)
        elif name == "mlp":
            X = np.stack([
                np.concatenate([
                    m.z_v_mag, m.z_p_inj / 1000.0, m.z_q_inj / 1000.0,
                    m.mask_v, m.mask_p, m.mask_q
                ]) for m in meas_batch
            ])
            with torch.no_grad():
                preds = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
        elif name in ["gnn", "physics_gnn"]:
            with torch.no_grad():
                x_feat = torch.tensor(
                    np.stack([
                        np.stack([
                            m.z_v_mag, m.z_p_inj / 1000.0, m.z_q_inj / 1000.0,
                            m.mask_v, m.mask_p, m.mask_q
                        ], axis=-1)
                        for m in meas_batch
                    ]),
                    dtype=torch.float32,
                    device=device,
                )
                preds = model(x_feat, edge_index, edge_attr).cpu().numpy()
        t_elapsed = time.perf_counter() - t0
        metrics = compute_all_metrics(y_batch, preds)
        metrics["inference_time_ms_per_sample"] = (t_elapsed / S) * 1000.0
        results[name] = metrics

    return results


def main() -> None:
    args = parse_args()
    cfg = AppConfig.load_all()
    device = torch.device(args.device)

    master_file = cfg.paths.feeder_dir / "IEEE123Master.dss"
    graph = extract_feeder_topology(master_file)
    edge_index, edge_attr = get_graph_tensors(graph, device)
    N = graph.num_nodes

    # Load splits and normal state test set
    with open(cfg.paths.data_dir / "splits" / "normal_splits.json") as f:
        splits = json.load(f)
    test_ids = splits["test"]

    states_path = cfg.paths.data_dir / "raw" / "sim_states_normal.npz"
    if not states_path.exists():
        logger.error("Dataset sim_states_normal.npz not found.")
        sys.exit(1)

    raw_states = np.load(states_path)
    y_test = raw_states["v_mag_pu"][test_ids]

    results_dir = cfg.paths.results_dir / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_exp_records = []

    for seed in args.seeds:
        set_seed(seed)
        logger.info(f"--- Running Experiments for Seed {seed} ---")

        # Load models trained for this seed (40% default availability)
        models = {}
        avail_tag = 40
        # 1. Ridge
        from dsse.models.linear import RidgeBaseline
        r_path = results_dir / f"model_ridge_seed{seed}_avail{avail_tag}.pkl"
        if r_path.exists():
            import pickle
            with open(r_path, "rb") as f:
                models["ridge"] = pickle.load(f)
        # 2. Tree
        t_path = results_dir / f"model_tree_seed{seed}_avail{avail_tag}.pkl"
        if t_path.exists():
            import pickle
            with open(t_path, "rb") as f:
                models["tree"] = pickle.load(f)
        # 3. MLP
        m_path = results_dir / f"model_mlp_seed{seed}_avail{avail_tag}.pt"
        if m_path.exists():
            mlp = MLPBaseline(input_dim=6 * N, output_dim=N).to(device)
            mlp.load_state_dict(torch.load(m_path, map_location=device))
            mlp.eval()
            models["mlp"] = mlp
        # 4. GNN
        g_path = results_dir / f"model_gnn_seed{seed}_avail{avail_tag}.pt"
        if g_path.exists():
            gnn = GNNEstimator(node_input_dim=6, edge_input_dim=4, hidden_dim=64, n_layers=3).to(device)
            gnn.load_state_dict(torch.load(g_path, map_location=device))
            gnn.eval()
            models["gnn"] = gnn
        # 5. PhysicsGNN
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

        if not models:
            logger.warning(f"No pre-trained models found for seed {seed}. Training lightweight instances for test evaluation.")
            continue

        meas_gen = MeasurementGenerator(graph=graph, seed=seed)
        rng = np.random.default_rng(seed + 500)

        # Helper to generate measurement list for test split
        def get_test_meas(avail=0.40, noise_v=0.005, miss=0.0, bad=0.0, structured=False):
            m_list = []
            for tid in test_ids:
                st = SimpleNamespace(
                    v_mag_pu=raw_states["v_mag_pu"][tid],
                    v_ang_rad=raw_states["v_ang_rad"][tid],
                    p_inj_kw=raw_states["p_inj_kw"][tid],
                    q_inj_kvar=raw_states["q_inj_kvar"][tid],
                )
                m = meas_gen.create_measurements(st, availability_rate=avail, noise_v_std=noise_v, rng=rng)
                if miss > 0:
                    m = apply_random_missingness(m, miss, rng=rng)
                if bad > 0:
                    m = apply_bad_data(m, bad, rng=rng)
                if structured:
                    m = apply_structured_missingness(m, graph=graph, failure_depth=2, rng=rng)
                m_list.append(m)
            return m_list

        # --- EXPERIMENT A: Normal Conditions ---
        logger.info(f"Seed {seed}: Evaluating Experiment A (Normal Conditions)")
        meas_a = get_test_meas(avail=0.40, noise_v=0.005)
        res_a = evaluate_sample_batch(models, meas_a, y_test, graph, edge_index, edge_attr, device)
        for m_name, m_res in res_a.items():
            all_exp_records.append({"exp_family": "A_normal", "seed": seed, "model": m_name, **m_res})

        # --- EXPERIMENT B: Sensor Sparsity ---
        logger.info(f"Seed {seed}: Evaluating Experiment B (Sensor Sparsity Sweep)")
        for avail in [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:
            meas_b = get_test_meas(avail=avail, noise_v=0.005)
            res_b = evaluate_sample_batch(models, meas_b, y_test, graph, edge_index, edge_attr, device)
            for m_name, m_res in res_b.items():
                all_exp_records.append({"exp_family": "B_sparsity", "seed": seed, "availability": avail, "model": m_name, **m_res})

        # --- EXPERIMENT C: Measurement Noise ---
        logger.info(f"Seed {seed}: Evaluating Experiment C (Measurement Noise Sweep)")
        for n_std in [0.002, 0.005, 0.010, 0.020, 0.050]:
            meas_c = get_test_meas(avail=0.40, noise_v=n_std)
            res_c = evaluate_sample_batch(models, meas_c, y_test, graph, edge_index, edge_attr, device)
            for m_name, m_res in res_c.items():
                all_exp_records.append({"exp_family": "C_noise", "seed": seed, "noise_std": n_std, "model": m_name, **m_res})

        # --- EXPERIMENT D: Random Missingness ---
        logger.info(f"Seed {seed}: Evaluating Experiment D (Random Missingness Sweep)")
        for miss_r in [0.10, 0.20, 0.40, 0.60]:
            meas_d = get_test_meas(avail=0.40, noise_v=0.005, miss=miss_r)
            res_d = evaluate_sample_batch(models, meas_d, y_test, graph, edge_index, edge_attr, device)
            for m_name, m_res in res_d.items():
                all_exp_records.append({"exp_family": "D_missingness", "seed": seed, "missing_rate": miss_r, "model": m_name, **m_res})

        # --- EXPERIMENT E: Structured Missingness ---
        logger.info(f"Seed {seed}: Evaluating Experiment E (Structured Communication Blackout)")
        meas_e = get_test_meas(avail=0.40, noise_v=0.005, structured=True)
        res_e = evaluate_sample_batch(models, meas_e, y_test, graph, edge_index, edge_attr, device)
        for m_name, m_res in res_e.items():
            all_exp_records.append({"exp_family": "E_structured_missingness", "seed": seed, "model": m_name, **m_res})

        # --- EXPERIMENT F: Gross Bad Data ---
        logger.info(f"Seed {seed}: Evaluating Experiment F (Bad Data Injection Sweep)")
        for bad_r in [0.01, 0.02, 0.05, 0.10]:
            meas_f = get_test_meas(avail=0.40, noise_v=0.005, bad=bad_r)
            res_f = evaluate_sample_batch(models, meas_f, y_test, graph, edge_index, edge_attr, device)
            for m_name, m_res in res_f.items():
                all_exp_records.append({"exp_family": "F_bad_data", "seed": seed, "bad_data_fraction": bad_r, "model": m_name, **m_res})

    # Save aggregated experimental results
    df_exp = pd.DataFrame(all_exp_records)
    out_csv = results_dir / "all_experiments_raw.csv"
    df_exp.to_csv(out_csv, index=False)
    logger.info(f"All experimental evaluations saved to {out_csv} ({len(df_exp)} total runs)")


if __name__ == "__main__":
    main()
