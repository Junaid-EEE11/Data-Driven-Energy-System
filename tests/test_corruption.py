"""Tests for corruption mechanisms: missingness, bad data."""

import numpy as np
import pytest


@pytest.fixture
def sample_meas(base_state, feeder_graph, rng):
    from dsse.measurements import MeasurementGenerator
    gen = MeasurementGenerator(feeder_graph, seed=42)
    return gen.create_measurements(base_state, availability_rate=0.8, rng=rng)


def test_random_missingness_reduces_observations(sample_meas, feeder_graph):
    from dsse.corruption import apply_random_missingness
    rng = np.random.default_rng(42)
    before = sample_meas.mask_v.sum() + sample_meas.mask_p.sum() + sample_meas.mask_q.sum()
    corrupted = apply_random_missingness(sample_meas, missing_rate=0.40, rng=rng)
    after = corrupted.mask_v.sum() + corrupted.mask_p.sum() + corrupted.mask_q.sum()
    assert after < before, "Random missingness should reduce number of observations."


def test_bad_data_modifies_values(sample_meas):
    from dsse.corruption import apply_bad_data
    rng = np.random.default_rng(42)
    orig_v = sample_meas.z_v_mag.copy()
    corrupted = apply_bad_data(sample_meas, bad_data_fraction=0.10, rng=rng)
    assert not np.allclose(orig_v, corrupted.z_v_mag), "Bad data should modify measurement values."


def test_zero_missingness_is_identity(sample_meas):
    from dsse.corruption import apply_random_missingness
    rng = np.random.default_rng(0)
    result = apply_random_missingness(sample_meas, missing_rate=0.0, rng=rng)
    np.testing.assert_array_equal(result.mask_v, sample_meas.mask_v)


def test_masks_stay_binary_after_corruption(sample_meas):
    from dsse.corruption import apply_random_missingness
    rng = np.random.default_rng(7)
    result = apply_random_missingness(sample_meas, missing_rate=0.30, rng=rng)
    for mask in [result.mask_v, result.mask_p, result.mask_q]:
        unique = set(np.unique(mask))
        assert unique.issubset({0.0, 1.0})
