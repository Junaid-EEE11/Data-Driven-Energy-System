"""Validates IEEE 123 feeder loading, power flow convergence, and topology extraction."""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from dsse.config import AppConfig
from dsse.opendss_interface import OpenDSSInterface
from dsse.reproducibility import setup_logger

logger = setup_logger("validate_feeder")


def validate_feeder() -> bool:
    """Executes end-to-end feeder validation tests."""
    cfg = AppConfig.load_all()
    master_file = cfg.paths.feeder_dir / "IEEE123Master.dss"

    logger.info(f"Validating feeder master file: {master_file}")
    if not master_file.exists():
        logger.error(f"Feeder master file does not exist: {master_file}")
        return False

    # 1. Initialize OpenDSS interface and check compilation
    interface = OpenDSSInterface(master_file=master_file)
    if interface.graph is None:
        logger.error("Failed to extract FeederGraph from OpenDSS.")
        return False

    graph = interface.graph
    logger.info(f"Topology extraction success: {graph.num_nodes} nodes, {graph.num_edges} edges.")

    # 2. Base case power flow convergence
    converged = interface.solve_power_flow()
    if not converged:
        logger.error("Base case power flow failed to converge.")
        return False
    logger.info("Base-case power flow converged successfully.")

    # 3. Ground truth state extraction
    state = interface.extract_state(scenario_id=0, load_mult=1.0)
    logger.info(
        f"Ground-truth extracted: V_pu min={state.v_mag_pu.min():.4f}, "
        f"max={state.v_mag_pu.max():.4f}, mean={state.v_mag_pu.mean():.4f}"
    )

    # Sanity checks on physical ranges
    assert 0.80 <= state.v_mag_pu.min() <= 1.05, f"V_pu min out of range: {state.v_mag_pu.min()}"
    assert 0.95 <= state.v_mag_pu.max() <= 1.15, f"V_pu max out of range: {state.v_mag_pu.max()}"
    assert len(state.v_mag_pu) == graph.num_nodes, "Voltage dimension mismatch."

    # 4. Deterministic indexing check
    interface_2 = OpenDSSInterface(master_file=master_file)
    state_2 = interface_2.extract_state(scenario_id=0, load_mult=1.0)
    diff = np.max(np.abs(state.v_mag_pu - state_2.v_mag_pu))
    assert diff < 1e-5, f"Non-deterministic state extraction detected: diff={diff}"
    logger.info("Deterministic indexing and state extraction verified.")

    # 5. Stochastic load modification test
    rng = np.random.default_rng(42)
    interface.set_stochastic_loads(mean_mult=1.2, std_mult=0.1, rng=rng)
    conv_stoch = interface.solve_power_flow()
    state_stoch = interface.extract_state(scenario_id=1, load_mult=1.2)
    assert conv_stoch, "Stochastic load power flow did not converge."
    logger.info(
        f"Stochastic load test passed: V_pu min={state_stoch.v_mag_pu.min():.4f}, "
        f"max={state_stoch.v_mag_pu.max():.4f}"
    )

    logger.info("ALL FEEDER VALIDATION TESTS PASSED SUCCESSFULLY.")
    return True


if __name__ == "__main__":
    success = validate_feeder()
    sys.exit(0 if success else 1)
