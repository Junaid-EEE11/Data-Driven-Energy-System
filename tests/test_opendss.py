"""Tests for OpenDSS interface: load, solve, extract state."""

import numpy as np
import pytest


def test_feeder_loads(feeder_interface):
    """Feeder should compile without errors."""
    assert feeder_interface is not None


def test_power_flow_converges(feeder_interface):
    """Base-case AC power flow must converge."""
    feeder_interface.reset_base_case()
    converged = feeder_interface.solve_power_flow()
    assert converged, "Base-case power flow did not converge."


def test_state_dimensions(feeder_interface, feeder_graph):
    """Extracted state vector must match topology node count."""
    feeder_interface.reset_base_case()
    feeder_interface.solve_power_flow()
    state = feeder_interface.extract_state(scenario_id=0, load_mult=1.0)
    assert len(state.v_mag_pu) == feeder_graph.num_nodes, (
        f"State dim {len(state.v_mag_pu)} != graph nodes {feeder_graph.num_nodes}"
    )


def test_voltage_in_physical_range(base_state):
    """Voltages must be in a physically plausible range (pu)."""
    assert np.all(base_state.v_mag_pu >= 0.70), "V_pu too low."
    assert np.all(base_state.v_mag_pu <= 1.20), "V_pu too high."


def test_state_is_finite(base_state):
    """No NaN or Inf values in extracted state."""
    assert np.all(np.isfinite(base_state.v_mag_pu))
    assert np.all(np.isfinite(base_state.v_ang_rad))


def test_stochastic_loads_still_converge(feeder_interface, rng):
    """Stochastic load modification should still converge."""
    feeder_interface.reset_base_case()
    feeder_interface.set_stochastic_loads(mean_mult=1.2, std_mult=0.1, rng=rng)
    converged = feeder_interface.solve_power_flow()
    assert converged
