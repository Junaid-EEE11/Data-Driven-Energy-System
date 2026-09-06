"""Controlled measurement corruption mechanisms: noise, random/structured missingness, and gross bad data."""

from __future__ import annotations
import copy
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from dsse.measurements import MeasurementVector
from dsse.reproducibility import setup_logger
from dsse.topology import FeederGraph

logger = setup_logger("corruption")


def apply_random_missingness(
    meas: MeasurementVector,
    missing_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> MeasurementVector:
    """Simulates independent random sensor communication dropout by zeroing observed masks."""
    if missing_rate <= 0.0:
        return meas
    if rng is None:
        rng = np.random.default_rng()

    res = copy.deepcopy(meas)
    N = len(res.mask_v)

    # For each measurement type, randomly drop observed meters
    for mask_attr, z_attr in [("mask_v", "z_v_mag"), ("mask_p", "z_p_inj"), ("mask_q", "z_q_inj")]:
        mask = getattr(res, mask_attr)
        z = getattr(res, z_attr)

        observed_indices = np.where(mask > 0.5)[0]
        # Do not drop slack substation if possible
        num_to_drop = int(np.round(len(observed_indices) * missing_rate))
        if num_to_drop > 0:
            dropped_indices = rng.choice(observed_indices, size=num_to_drop, replace=False)
            mask[dropped_indices] = 0.0
            z[dropped_indices] = 0.0

    res.missing_rate = missing_rate
    res.is_corrupted = True
    return res


def apply_structured_missingness(
    meas: MeasurementVector,
    graph: FeederGraph,
    failure_depth: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> MeasurementVector:
    """Simulates localized communication failure / collector downtime affecting an entire feeder zone."""
    if rng is None:
        rng = np.random.default_rng()

    res = copy.deepcopy(meas)

    # Pick a random branch / collector hub
    candidate_buses = [n.bus_name for n in graph.nodes if not n.is_slack and len(graph.bus_to_nodes.get(n.bus_name, [])) > 0]
    if not candidate_buses:
        return res

    center_bus = rng.choice(candidate_buses)
    center_nodes = graph.bus_to_nodes[center_bus]

    # Find k-hop downstream subgraph in NetworkX
    affected_nodes = set(center_nodes)
    current_frontier = set(center_nodes)
    for _ in range(failure_depth):
        next_frontier = set()
        for u in current_frontier:
            if u in graph.networkx_graph:
                nbrs = list(graph.networkx_graph.neighbors(u))
                next_frontier.update(nbrs)
        affected_nodes.update(next_frontier)
        current_frontier = next_frontier

    affected_list = list(affected_nodes)
    res.mask_v[affected_list] = 0.0
    res.mask_p[affected_list] = 0.0
    res.mask_q[affected_list] = 0.0
    res.z_v_mag[affected_list] = 0.0
    res.z_p_inj[affected_list] = 0.0
    res.z_q_inj[affected_list] = 0.0

    res.is_corrupted = True
    logger.debug(f"Applied structured communication blackout on {len(affected_list)} nodes around bus {center_bus}")
    return res


def apply_bad_data(
    meas: MeasurementVector,
    bad_data_fraction: float = 0.05,
    magnitude_scale: float = 0.25,
    rng: Optional[np.random.Generator] = None,
) -> MeasurementVector:
    """Injects gross errors / bad data (large additive or multiplicative perturbations) into observed meters."""
    if bad_data_fraction <= 0.0:
        return meas
    if rng is None:
        rng = np.random.default_rng()

    res = copy.deepcopy(meas)

    # Target observed voltage and power measurements
    obs_v = np.where(res.mask_v > 0.5)[0]
    obs_p = np.where(res.mask_p > 0.5)[0]
    obs_q = np.where(res.mask_q > 0.5)[0]

    # Corrupt voltage measurements (e.g. +/- 15% to 30% voltage spike/sag)
    n_bad_v = int(np.ceil(len(obs_v) * bad_data_fraction))
    if n_bad_v > 0 and len(obs_v) > 0:
        bad_idx_v = rng.choice(obs_v, size=min(n_bad_v, len(obs_v)), replace=False)
        signs = rng.choice([-1.0, 1.0], size=len(bad_idx_v))
        res.z_v_mag[bad_idx_v] += signs * rng.uniform(magnitude_scale * 0.5, magnitude_scale, size=len(bad_idx_v))

    # Corrupt active and reactive power measurements (e.g. sign flip or 3x spike)
    n_bad_p = int(np.ceil(len(obs_p) * bad_data_fraction))
    if n_bad_p > 0 and len(obs_p) > 0:
        bad_idx_p = rng.choice(obs_p, size=min(n_bad_p, len(obs_p)), replace=False)
        for idx in bad_idx_p:
            res.z_p_inj[idx] = -1.5 * res.z_p_inj[idx] if abs(res.z_p_inj[idx]) > 1.0 else rng.uniform(50.0, 200.0)

    n_bad_q = int(np.ceil(len(obs_q) * bad_data_fraction))
    if n_bad_q > 0 and len(obs_q) > 0:
        bad_idx_q = rng.choice(obs_q, size=min(n_bad_q, len(obs_q)), replace=False)
        for idx in bad_idx_q:
            res.z_q_inj[idx] = -1.5 * res.z_q_inj[idx] if abs(res.z_q_inj[idx]) > 1.0 else rng.uniform(30.0, 100.0)

    res.bad_data_rate = bad_data_fraction
    res.is_corrupted = True
    return res
