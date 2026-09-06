"""Training script for classical ML baselines: Ridge, MLP, Tree, WLS.

Usage:
    python scripts/train_baselines.py --model ridge --seed 42
    python scripts/train_baselines.py --model mlp --seed 42
    python scripts/train_baselines.py --model tree --seed 42
    python scripts/train_baselines.py --model wls   (no training needed, run inference)
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.dataset import DSSEDataset
from dsse.metrics import compute_all_metrics
from dsse.reproducibility import save_metadata, set_seed, setup_logger
from dsse.topology import extract_feeder_topology

logger = setup_logger("train_baselines")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ridge", "mlp", "tree", "wls"], default="ridge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--availability", type=float, default=0.40)
    p.add_argument("--noise_v", type=float, default=0.005)
    p.add_argument("--epochs", type=int, default=20, help="MLP only.")
    p.add_argument("--lr", type=float, default=1e-3, help="MLP only.")
    p.add_argument("--batch_size", type=int, default=64, help="MLP only.")
    return p.parse_args()


def load_data(cfg, splits, tag, partition="train"):
    """Loads and concatenates features for sklearn-style models."""
    states_path = cfg.paths.data_dir / "raw" / "sim_states_normal.npz"
    meas_path = cfg.paths.data_dir / "interim" / f"measurements_{tag}.npz"
    dataset = DSSEDataset.load_from_npz(states_path, meas_path)
    _, X, y = dataset.get_tensors(splits[partition])
    return X.numpy(), y.numpy()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = AppConfig.load_all()

    tag = f"normal_avail{int(args.availability*100)}_noise{int(args.noise_v*1000)}_miss0_bad0"
    splits_file = cfg.paths.data_dir / "splits" / "normal_splits.json"
    if not splits_file.exists():
        logger.error("Splits not found. Run make_splits.py first.")
        sys.exit(1)
    with open(splits_file) as f:
        splits = json.load(f)

    results_dir = cfg.paths.results_dir / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_tag = f"{args.model}_seed{args.seed}_avail{int(args.availability*100)}"

    start_time = time.time()

    if args.model in ["ridge", "tree"]:
        X_train, y_train = load_data(cfg, splits, tag, "train")
        X_val,   y_val   = load_data(cfg, splits, tag, "val")
        X_test,  y_test  = load_data(cfg, splits, tag, "test")
        logger.info(f"Loaded data: X_train={X_train.shape}, y_train={y_train.shape}")

        if args.model == "ridge":
            from dsse.models.linear import RidgeBaseline
            model = RidgeBaseline(alpha=1.0, seed=args.seed)
            model.fit(X_train, y_train)
        else:
            from dsse.models.tree import TreeBaseline
            model = TreeBaseline(seed=args.seed)
            model.fit(X_train, y_train)

        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_test)
        val_metrics  = compute_all_metrics(y_val,  val_pred)
        test_metrics = compute_all_metrics(y_test, test_pred)

        # Save model
        model_path = results_dir / f"model_{model_tag}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {args.model} model to {model_path}")

    elif args.model == "mlp":
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Subset
        from dsse.models.mlp import MLPBaseline

        states_path = cfg.paths.data_dir / "raw" / "sim_states_normal.npz"
        meas_path   = cfg.paths.data_dir / "interim" / f"measurements_{tag}.npz"
        dataset = DSSEDataset.load_from_npz(states_path, meas_path)
        N = dataset.y_all.shape[-1]
        input_dim = dataset.x_flat_all.shape[-1]

        _, X_train_t, y_train_t = dataset.get_tensors(splits["train"])
        _, X_val_t,   y_val_t   = dataset.get_tensors(splits["val"])
        _, X_test_t,  y_test_t  = dataset.get_tensors(splits["test"])

        device = torch.device("cpu")
        model = MLPBaseline(input_dim=input_dim, output_dim=N).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        best_val = float("inf")
        n_train = len(X_train_t)
        batch_size = args.batch_size

        patience = 0
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n_train)
            for b in range(0, n_train, batch_size):
                idx = perm[b:b+batch_size]
                x_b, y_b = X_train_t[idx], y_train_t[idx]
                optimizer.zero_grad()
                pred = model(x_b)
                loss = criterion(pred, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t).numpy()
            v_mae = compute_all_metrics(y_val_t.numpy(), val_pred)["mae"]
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1:2d}/{args.epochs} | val_mae={v_mae:.6f} pu")
            if v_mae < best_val:
                best_val = v_mae
                patience = 0
                torch.save(model.state_dict(), results_dir / f"model_{model_tag}.pt")
            else:
                patience += 1
                if patience >= 5:
                    logger.info(f"MLP Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(results_dir / f"model_{model_tag}.pt"))
        model.eval()
        with torch.no_grad():
            val_pred  = model(X_val_t).numpy()
            test_pred = model(X_test_t).numpy()
        val_metrics  = compute_all_metrics(y_val_t.numpy(),  val_pred)
        test_metrics = compute_all_metrics(y_test_t.numpy(), test_pred)

    elif args.model == "wls":
        # WLS runs at inference time (no training); evaluate on test set
        from dsse.baselines.wls import ThreePhaseWLS

        master_file = cfg.paths.feeder_dir / "IEEE123Master.dss"
        graph = extract_feeder_topology(master_file)

        if graph.y_bus_real is None:
            logger.error("Y-bus not available. Cannot run WLS.")
            sys.exit(1)

        wls = ThreePhaseWLS(G=graph.y_bus_real, B=graph.y_bus_imag)

        states_path = cfg.paths.data_dir / "raw" / "sim_states_normal.npz"
        meas_path   = cfg.paths.data_dir / "interim" / f"measurements_{tag}.npz"
        states = np.load(states_path)
        meas = np.load(meas_path)

        test_ids = splits["test"]
        wls_preds, wls_trues = [], []
        n_converged = 0

        for idx in test_ids[:200]:  # Limit for speed (WLS is slow at N=278)
            result = wls.estimate(
                z_v=meas["z_v_mag"][idx],
                z_p=meas["z_p_inj"][idx],
                z_q=meas["z_q_inj"][idx],
                mask_v=meas["mask_v"][idx],
                mask_p=meas["mask_p"][idx],
                mask_q=meas["mask_q"][idx],
            )
            wls_preds.append(result["v_mag_pu"])
            wls_trues.append(states["v_mag_pu"][idx])
            if result["converged"]:
                n_converged += 1

        logger.info(f"WLS convergence rate: {n_converged}/{len(test_ids[:200])}")
        val_metrics  = {"mae": float("nan"), "convergence_rate": n_converged / 200}
        test_metrics = compute_all_metrics(np.stack(wls_trues), np.stack(wls_preds))

    elapsed = time.time() - start_time
    meta = {
        "model": args.model,
        "seed": args.seed,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }
    save_metadata(meta, results_dir / f"baseline_results_{model_tag}.json")
    logger.info(f"Test MAE={test_metrics.get('mae', 'N/A'):.6f} pu | Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
