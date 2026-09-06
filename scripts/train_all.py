"""Trains all baseline models (Ridge, MLP, Tree, WLS) and neural models (GNN, Physics-GNN)
across all 5 random seeds (42, 43, 44, 45, 46).
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

SEEDS = [42, 43, 44, 45, 46]


def run_cmd(cmd_list):
    cmd_str = " ".join(cmd_list)
    print(f"\n[RUNNING] {cmd_str}")
    res = subprocess.run(cmd_list, check=True)
    return res.returncode


def main():
    print("=== STARTING FULL MULTI-SEED MODEL TRAINING PIPELINE ===")

    for seed in SEEDS:
        print(f"\n==================== SEED {seed} ====================")
        # 1. Ridge
        run_cmd([PYTHON, str(ROOT / "scripts/train_baselines.py"), "--model", "ridge", "--seed", str(seed)])

        # 2. Tree
        run_cmd([PYTHON, str(ROOT / "scripts/train_baselines.py"), "--model", "tree", "--seed", str(seed)])

        # 3. MLP
        run_cmd([PYTHON, str(ROOT / "scripts/train_baselines.py"), "--model", "mlp", "--seed", str(seed), "--epochs", "20"])

        # 4. GNN
        run_cmd([PYTHON, str(ROOT / "scripts/train_gnn.py"), "--model", "gnn", "--seed", str(seed), "--epochs", "10", "--lr", "0.001", "--hidden_dim", "64", "--n_layers", "3"])

        # 5. PhysicsGNN
        run_cmd([PYTHON, str(ROOT / "scripts/train_gnn.py"), "--model", "physics_gnn", "--seed", str(seed), "--epochs", "10", "--lr", "0.001", "--hidden_dim", "64", "--n_layers", "3", "--lambda_physics", "0.10"])

    # 6. WLS on seed 42 (model-based benchmark)
    run_cmd([PYTHON, str(ROOT / "scripts/train_baselines.py"), "--model", "wls", "--seed", "42"])

    print("\n=== ALL MODELS SUCCESSFULLY TRAINED ACROSS ALL 5 SEEDS ===")


if __name__ == "__main__":
    main()
