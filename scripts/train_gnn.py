"""Training script for GNN and PhysicsGNN models with fully-vectorised batch processing.

Usage:
    python scripts/train_gnn.py --model gnn --seed 42 --epochs 30
    python scripts/train_gnn.py --model physics_gnn --lambda_physics 0.1 --seed 42 --epochs 30
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.dataset import DSSEDataset, make_dataloader
from dsse.metrics import compute_all_metrics
from dsse.reproducibility import save_metadata, set_seed, setup_logger
from dsse.topology import extract_feeder_topology

logger = setup_logger("train_gnn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gnn", "physics_gnn"], default="gnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lambda_physics", type=float, default=0.10)
    p.add_argument("--availability", type=float, default=0.40)
    p.add_argument("--noise_v", type=float, default=0.005)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = AppConfig.load_all()
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    master_file = cfg.paths.feeder_dir / "IEEE123Master.dss"
    graph = extract_feeder_topology(master_file)
    N = graph.num_nodes

    # Load dataset
    tag = f"normal_avail{int(args.availability*100)}_noise{int(args.noise_v*1000)}_miss0_bad0"
    states_path = cfg.paths.data_dir / "raw" / "sim_states_normal.npz"
    meas_path = cfg.paths.data_dir / "interim" / f"measurements_{tag}.npz"

    if not states_path.exists() or not meas_path.exists():
        logger.error("Dataset not found. Run generate_dataset.py first.")
        sys.exit(1)

    dataset = DSSEDataset.load_from_npz(states_path, meas_path)

    # Load splits
    splits_file = cfg.paths.data_dir / "splits" / "normal_splits.json"
    if not splits_file.exists():
        logger.error("Splits file not found. Run make_splits.py first.")
        sys.exit(1)
    with open(splits_file) as f:
        splits = json.load(f)

    x_train, _, y_train = dataset.get_tensors(splits["train"])
    x_val,   _, y_val   = dataset.get_tensors(splits["val"])

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val   = x_val.to(device)
    y_val   = y_val.to(device)

    # Static graph edge tensors
    edge_index = graph.edge_index.to(device)
    edge_attr = graph.edge_attr.to(device)
    edge_dim = edge_attr.shape[-1] if edge_attr.numel() > 0 else 4
    node_input_dim = 6

    # Build model
    if args.model == "gnn":
        from dsse.models.gnn import GNNEstimator
        model = GNNEstimator(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_dim,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
        ).to(device)
        use_physics = False
    else:
        from dsse.models.physics_gnn import PhysicsGNNEstimator
        G = torch.tensor(graph.y_bus_real, dtype=torch.float32, device=device)
        B = torch.tensor(graph.y_bus_imag, dtype=torch.float32, device=device)
        nom_th = np.zeros(N, dtype=np.float32)
        for i, nd in enumerate(graph.nodes):
            p = nd.phase
            nom_th[i] = 0.0 if p == 1 else (-2.0 * np.pi / 3.0 if p == 2 else 2.0 * np.pi / 3.0)
        theta_nom = torch.tensor(nom_th, dtype=torch.float32, device=device)
        model = PhysicsGNNEstimator(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_dim,
            G=G, B=B,
            theta_nom=theta_nom,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lambda_physics=args.lambda_physics,
        ).to(device)
        use_physics = True

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    results_dir = cfg.paths.results_dir / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_tag = f"{args.model}_seed{args.seed}_avail{int(args.availability*100)}"
    best_val_loss = float("inf")
    patience_counter = 0
    train_history = []

    n_train = len(x_train)
    batch_size = args.batch_size

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = {"total": 0.0, "sup": 0.0, "phys": 0.0}
        n_batches = 0
        perm = torch.randperm(n_train)

        for b in range(0, n_train, batch_size):
            idx = perm[b:b+batch_size]
            x_b = x_train[idx]
            y_b = y_train[idx]

            optimizer.zero_grad()
            v_pred = model(x_b, edge_index, edge_attr)  # [B, N]

            if use_physics:
                z_v = x_b[..., 0]
                z_p = x_b[..., 1]
                z_q = x_b[..., 2]
                mv  = x_b[..., 3]
                mp  = x_b[..., 4]
                mq  = x_b[..., 5]
                L_total, L_sup, L_phys = model.compute_total_loss(
                    v_pred, y_b, z_v, z_p, z_q, mv, mp, mq
                )
                epoch_losses["sup"] += L_sup.item()
                epoch_losses["phys"] += L_phys.item()
                epoch_losses["total"] += L_total.item()
                L_total.backward()
            else:
                L_sup = criterion(v_pred, y_b)
                epoch_losses["sup"] += L_sup.item()
                epoch_losses["total"] += L_sup.item()
                L_sup.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n_batches += 1

        scheduler.step()

        # Batched Validation
        model.eval()
        with torch.no_grad():
            v_val_pred = model(x_val, edge_index, edge_attr)  # [150, N]
            val_preds_arr = v_val_pred.cpu().numpy()
            val_trues_arr = y_val.cpu().numpy()

        val_metrics = compute_all_metrics(val_trues_arr, val_preds_arr)
        val_mae = val_metrics["mae"]

        avg_total = epoch_losses["total"] / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info(f"Epoch {epoch:3d}/{args.epochs} | train_loss={avg_total:.6f} | val_mae={val_mae:.6f} pu")

        train_history.append({"epoch": epoch, "train_loss": avg_total, "val_mae": val_mae, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_mae < best_val_loss:
            best_val_loss = val_mae
            patience_counter = 0
            checkpoint_path = results_dir / f"model_{model_tag}.pt"
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    elapsed = time.time() - start_time
    meta = {
        "model": args.model,
        "seed": args.seed,
        "epochs_run": len(train_history),
        "best_val_mae": best_val_loss,
        "elapsed_seconds": elapsed,
        "args": vars(args),
        "history": train_history,
    }
    save_metadata(meta, results_dir / f"training_meta_{model_tag}.json")
    logger.info(f"Training complete. Best val MAE: {best_val_loss:.6f} pu | Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
