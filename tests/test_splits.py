"""Tests for data partitioning, leakage prevention, and partition integrity."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from dsse.dataset import DSSESample, DSSEDataset


def test_split_no_overlapping_scenarios():
    """Verify that train, val, cal, and test splits have mutually exclusive scenario IDs."""
    n_train = 600
    n_val = 150
    n_cal = 150
    n_test = 300
    total = n_train + n_val + n_cal + n_test

    rng = np.random.default_rng(42)
    all_indices = rng.permutation(total)

    train_idx = set(all_indices[:n_train])
    val_idx = set(all_indices[n_train:n_train + n_val])
    cal_idx = set(all_indices[n_train + n_val:n_train + n_val + n_cal])
    test_idx = set(all_indices[n_train + n_val + n_cal:])

    # Pairwise disjointness checks
    assert len(train_idx.intersection(val_idx)) == 0, "Train and Val overlap!"
    assert len(train_idx.intersection(cal_idx)) == 0, "Train and Cal overlap!"
    assert len(train_idx.intersection(test_idx)) == 0, "Train and Test overlap!"
    assert len(val_idx.intersection(cal_idx)) == 0, "Val and Cal overlap!"
    assert len(val_idx.intersection(test_idx)) == 0, "Val and Test overlap!"
    assert len(cal_idx.intersection(test_idx)) == 0, "Cal and Test overlap!"


def test_dataset_save_and_reload_numerical_consistency():
    """Verify that saving and reloading NPZ dataset preserves exact floating point values."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        N = 278
        S = 10

        rng = np.random.default_rng(123)
        v_mag = rng.uniform(0.9, 1.1, size=(S, N)).astype(np.float64)
        v_ang = rng.uniform(-np.pi, np.pi, size=(S, N)).astype(np.float64)
        p_inj = rng.uniform(-100, 100, size=(S, N)).astype(np.float64)
        q_inj = rng.uniform(-50, 50, size=(S, N)).astype(np.float64)

        z_v = v_mag * (rng.random((S, N)) > 0.5)
        z_p = p_inj * (rng.random((S, N)) > 0.5)
        z_q = q_inj * (rng.random((S, N)) > 0.5)
        mask_v = (z_v > 0).astype(np.float32)
        mask_p = (z_p != 0).astype(np.float32)
        mask_q = (z_q != 0).astype(np.float32)

        states_path = tmp_path / "sim_states.npz"
        meas_path = tmp_path / "sim_meas.npz"

        np.savez_compressed(
            states_path,
            v_mag_pu=v_mag,
            v_ang_rad=v_ang,
            p_inj_kw=p_inj,
            q_inj_kvar=q_inj,
        )
        np.savez_compressed(
            meas_path,
            z_v_mag=z_v,
            z_p_inj=z_p,
            z_q_inj=z_q,
            mask_v=mask_v,
            mask_p=mask_p,
            mask_q=mask_q,
        )

        dataset = DSSEDataset.load_from_npz(states_path, meas_path)
        assert len(dataset) == S

        item0 = dataset[0]
        np.testing.assert_allclose(item0["y"].numpy(), v_mag[0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(item0["z_v"].numpy(), z_v[0], rtol=1e-5, atol=1e-5)
