"""Pytest fixtures shared across the DSSE test suite."""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.reproducibility import set_seed


@pytest.fixture(scope="session")
def cfg():
    return AppConfig.load_all()


@pytest.fixture(scope="session")
def master_file(cfg):
    return cfg.paths.feeder_dir / "IEEE123Master.dss"


@pytest.fixture(scope="session")
def feeder_interface(master_file):
    from dsse.opendss_interface import OpenDSSInterface
    return OpenDSSInterface(master_file=master_file)


@pytest.fixture(scope="session")
def feeder_graph(master_file):
    from dsse.topology import extract_feeder_topology
    return extract_feeder_topology(master_file)


@pytest.fixture(scope="session")
def base_state(feeder_interface):
    feeder_interface.reset_base_case()
    feeder_interface.solve_power_flow()
    return feeder_interface.extract_state(scenario_id=0, load_mult=1.0)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)
