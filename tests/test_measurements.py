"""Tests for measurement generation: masks, noise, observable quantities."""

import numpy as np
import pytest


def test_measurement_shape(base_state, feeder_graph, rng):
    """Measurement vectors must match the node count."""
    from dsse.measurements import MeasurementGenerator
    gen = MeasurementGenerator(feeder_graph, seed=42)
    meas = gen.create_measurements(base_state, availability_rate=0.4, rng=rng)
    N = feeder_graph.num_nodes
    assert len(meas.z_v_mag) == N
    assert len(meas.mask_v) == N


def test_masks_are_binary(base_state, feeder_graph, rng):
    """All mask values must be exactly 0 or 1."""
    from dsse.measurements import MeasurementGenerator
    gen = MeasurementGenerator(feeder_graph, seed=42)
    meas = gen.create_measurements(base_state, availability_rate=0.4, rng=rng)
    for mask in [meas.mask_v, meas.mask_p, meas.mask_q]:
        unique = set(np.unique(mask))
        assert unique.issubset({0.0, 1.0}), f"Non-binary mask values: {unique}"


def test_unobserved_measurements_are_zero(base_state, feeder_graph, rng):
    """Zero-mask entries must correspond to zero measurement values."""
    from dsse.measurements import MeasurementGenerator
    gen = MeasurementGenerator(feeder_graph, seed=42)
    meas = gen.create_measurements(base_state, availability_rate=0.4, rng=rng)
    assert np.all(meas.z_v_mag[meas.mask_v < 0.5] == 0.0)
    assert np.all(meas.z_p_inj[meas.mask_p < 0.5] == 0.0)
    assert np.all(meas.z_q_inj[meas.mask_q < 0.5] == 0.0)


def test_availability_rate_approximately_correct(base_state, feeder_graph):
    """Sensor availability should roughly match requested rate."""
    from dsse.measurements import MeasurementGenerator
    rng = np.random.default_rng(99)
    gen = MeasurementGenerator(feeder_graph, seed=99)
    for rate in [0.10, 0.40, 0.80]:
        meas = gen.create_measurements(base_state, availability_rate=rate, rng=rng)
        observed = (meas.mask_v + meas.mask_p + meas.mask_q) > 0
        actual_rate = observed.mean()
        # Allow large tolerance due to substation always-on
        assert actual_rate >= rate * 0.5, f"Rate {rate:.0%}: got {actual_rate:.2%}"
