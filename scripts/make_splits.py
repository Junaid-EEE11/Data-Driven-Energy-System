"""Creates reproducible train/validation/calibration/test/OOD data splits.

Splitting strategy
------------------
- Normal-regime scenarios (e.g. 3000 samples) are split TEMPORALLY:
    Train: 60%  (indices 0 .. 0.60*N-1)
    Val:   15%  (indices 0.60*N .. 0.75*N-1)
    Cal:   10%  (indices 0.75*N .. 0.85*N-1)   <- conformal calibration ONLY
    Test:  15%  (indices 0.85*N .. N-1)

  Temporal ordering avoids accidentally placing near-identical load profiles
  in both train and test partitions (which would inflate apparent performance).

- OOD scenarios are kept entirely separate test sets.

Leakage prevention
------------------
- Scenario IDs are assigned once and stored in split files.
- Calibration partition IDs are never exposed during model training.
- Split files record the exact index lists for reproducibility.

Run:
    python scripts/make_splits.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.reproducibility import save_metadata, setup_logger

logger = setup_logger("make_splits")


SPLIT_FRACTIONS = {
    "train": 0.60,
    "val":   0.15,
    "cal":   0.10,
    "test":  0.15,
}


def make_normal_splits(n_samples: int, seed: int = 42) -> dict:
    """Creates temporal train/val/cal/test splits for normal regime samples."""
    assert abs(sum(SPLIT_FRACTIONS.values()) - 1.0) < 1e-9, "Fractions must sum to 1"

    indices = list(range(n_samples))
    n_train = int(np.round(n_samples * SPLIT_FRACTIONS["train"]))
    n_val   = int(np.round(n_samples * SPLIT_FRACTIONS["val"]))
    n_cal   = int(np.round(n_samples * SPLIT_FRACTIONS["cal"]))
    n_test  = n_samples - n_train - n_val - n_cal

    train_ids = indices[:n_train]
    val_ids   = indices[n_train:n_train + n_val]
    cal_ids   = indices[n_train + n_val:n_train + n_val + n_cal]
    test_ids  = indices[n_train + n_val + n_cal:]

    # Verify no overlap
    all_sets = [set(train_ids), set(val_ids), set(cal_ids), set(test_ids)]
    for i, si in enumerate(all_sets):
        for j, sj in enumerate(all_sets):
            if i != j:
                overlap = si & sj
                assert len(overlap) == 0, f"Partition overlap detected between partitions {i} and {j}: {overlap}"

    splits = {
        "train": train_ids,
        "val":   val_ids,
        "cal":   cal_ids,
        "test":  test_ids,
        "n_total": n_samples,
        "seed": seed,
        "strategy": "temporal_ordered",
    }
    logger.info(
        f"Normal splits: train={len(train_ids)}, val={len(val_ids)}, "
        f"cal={len(cal_ids)}, test={len(test_ids)}"
    )
    return splits


def main() -> None:
    cfg = AppConfig.load_all()
    splits_dir = cfg.paths.data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = cfg.paths.data_dir / "raw"
    normal_states_file = raw_dir / "sim_states_normal.npz"

    if not normal_states_file.exists():
        logger.error(
            f"Normal states file not found: {normal_states_file}\n"
            "Run: python scripts/generate_dataset.py --regime normal first."
        )
        sys.exit(1)

    normal_data = np.load(normal_states_file)
    n_normal = normal_data["v_mag_pu"].shape[0]

    # Normal regime splits
    splits = make_normal_splits(n_normal, seed=42)
    with open(splits_dir / "normal_splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved normal splits to {splits_dir / 'normal_splits.json'}")

    # OOD split files (all samples used as test)
    for ood_regime in ["ood_high_loading", "ood_low_loading", "ood_high_solar_der"]:
        ood_file = raw_dir / f"sim_states_{ood_regime}.npz"
        if ood_file.exists():
            ood_data = np.load(ood_file)
            n_ood = ood_data["v_mag_pu"].shape[0]
            ood_split = {
                "test": list(range(n_ood)),
                "n_total": n_ood,
                "regime": ood_regime,
                "strategy": "all_test_ood",
            }
            with open(splits_dir / f"{ood_regime}_splits.json", "w") as f:
                json.dump(ood_split, f, indent=2)
            logger.info(f"Saved OOD split for {ood_regime}: {n_ood} test samples")

    logger.info("All splits written. Calibration partition is RESERVED for conformal calibration only.")


if __name__ == "__main__":
    main()
