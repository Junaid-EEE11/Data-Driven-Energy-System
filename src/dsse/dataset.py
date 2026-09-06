"""PyTorch Dataset and DataLoader utilities for DSSE training."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dsse.reproducibility import setup_logger

logger = setup_logger("dataset")


@dataclass
class DSSESample:
    """Single scenario: inputs (noisy/masked measurements) + targets (ground-truth V_pu)."""
    z_v: np.ndarray       # [N] voltage magnitude measurements (pu, 0 where masked)
    z_p: np.ndarray       # [N] active power measurements (kW, 0 where masked)
    z_q: np.ndarray       # [N] reactive power measurements (kvar, 0 where masked)
    mask_v: np.ndarray    # [N] {0,1} voltage observable
    mask_p: np.ndarray    # [N] {0,1} P observable
    mask_q: np.ndarray    # [N] {0,1} Q observable
    y_v: np.ndarray       # [N] ground-truth V_mag_pu targets
    scenario_id: int
    regime: str = "normal"


class DSSEDataset(Dataset):
    """PyTorch Dataset wrapping a list of DSSESamples for mini-batch training."""

    def __init__(self, samples: List[DSSESample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        # Input feature vector: concatenate z_v, z_p, z_q, mask_v, mask_p, mask_q
        x = np.concatenate([s.z_v, s.z_p / 1000.0, s.z_q / 1000.0,
                            s.mask_v, s.mask_p, s.mask_q], axis=0).astype(np.float32)
        y = s.y_v.astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask_v": torch.from_numpy(s.mask_v.astype(np.float32)),
            "mask_p": torch.from_numpy(s.mask_p.astype(np.float32)),
            "mask_q": torch.from_numpy(s.mask_q.astype(np.float32)),
            "z_v": torch.from_numpy(s.z_v.astype(np.float32)),
            "z_p": torch.from_numpy((s.z_p / 1000.0).astype(np.float32)),
            "z_q": torch.from_numpy((s.z_q / 1000.0).astype(np.float32)),
            "scenario_id": torch.tensor(s.scenario_id, dtype=torch.long),
        }

    @classmethod
    def load_from_npz(
        cls,
        states_path: Path | str,
        meas_path: Path | str,
    ) -> "DSSEDataset":
        """Loads dataset from pre-saved .npz files (states + measurements)."""
        states = np.load(states_path)
        meas = np.load(meas_path)

        v_mag = states["v_mag_pu"]      # [S, N]
        z_v   = meas["z_v_mag"]         # [S, N]
        z_p   = meas["z_p_inj"]         # [S, N]
        z_q   = meas["z_q_inj"]         # [S, N]
        mask_v = meas["mask_v"]          # [S, N]
        mask_p = meas["mask_p"]          # [S, N]
        mask_q = meas["mask_q"]          # [S, N]

        S = v_mag.shape[0]
        samples = []
        for i in range(S):
            samples.append(DSSESample(
                z_v=z_v[i], z_p=z_p[i], z_q=z_q[i],
                mask_v=mask_v[i], mask_p=mask_p[i], mask_q=mask_q[i],
                y_v=v_mag[i],
                scenario_id=i,
            ))
        logger.info(f"Loaded {S} samples from {states_path}")
        ds = cls(samples)
        # Precompute stacked tensors for high-performance training
        ds.x_node_all = torch.tensor(
            np.stack([z_v, z_p / 1000.0, z_q / 1000.0, mask_v, mask_p, mask_q], axis=-1),
            dtype=torch.float32,
        )  # [S, N, 6]
        ds.x_flat_all = torch.tensor(
            np.concatenate([z_v, z_p / 1000.0, z_q / 1000.0, mask_v, mask_p, mask_q], axis=-1),
            dtype=torch.float32,
        )  # [S, 6*N]
        ds.y_all = torch.tensor(v_mag, dtype=torch.float32)  # [S, N]
        return ds

    def get_tensors(self, indices: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (x_node [B, N, 6], x_flat [B, 6*N], y [B, N]) for selected indices."""
        if indices is None:
            return self.x_node_all, self.x_flat_all, self.y_all
        idx = torch.tensor(indices, dtype=torch.long)
        return self.x_node_all[idx], self.x_flat_all[idx], self.y_all[idx]


def make_dataloader(
    dataset: DSSEDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Creates a DataLoader with reproducible settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
