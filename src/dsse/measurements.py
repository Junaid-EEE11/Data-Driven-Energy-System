"""Heterogeneous smart-meter measurement model, sensor placement masks, and observable quantities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dsse.opendss_interface import ElectricalState
from dsse.reproducibility import setup_logger
from dsse.topology import FeederGraph

logger = setup_logger("measurements")


@dataclass
class MeasurementVector:
    """Represents a noisy, sparse, potentially corrupted measurement snapshot for the feeder."""
    # Nodal measurement values (zeros where unmeasured / missing)
    z_v_mag: np.ndarray  # [N_nodes] in per unit
    z_p_inj: np.ndarray  # [N_nodes] in kW or pu
    z_q_inj: np.ndarray  # [N_nodes] in kvar or pu

    # Binary observation masks (1 = observed, 0 = unmeasured/missing)
    mask_v: np.ndarray  # [N_nodes] {0, 1}
    mask_p: np.ndarray  # [N_nodes] {0, 1}
    mask_q: np.ndarray  # [N_nodes] {0, 1}

    # Metadata
    availability_rate: float
    noise_v_std: float
    noise_pq_std: float
    missing_rate: float
    bad_data_rate: float
    is_corrupted: bool = False


class MeasurementGenerator:
    """Generates realistic heterogeneous smart meter and substation measurement sets."""

    def __init__(
        self,
        graph: FeederGraph,
        default_availability: float = 0.40,
        default_noise_v: float = 0.005,  # 0.5% std
        default_noise_pq: float = 0.02,  # 2.0% std
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.default_availability = default_availability
        self.default_noise_v = default_noise_v
        self.default_noise_pq = default_noise_pq
        self.rng = np.random.default_rng(seed)

    def generate_sensor_mask(
        self,
        availability_rate: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates heterogeneous sensor masks for V, P, Q across all bus-phase nodes."""
        if rng is None:
            rng = self.rng

        N = self.graph.num_nodes
        mask_v = np.zeros(N, dtype=np.float32)
        mask_p = np.zeros(N, dtype=np.float32)
        mask_q = np.zeros(N, dtype=np.float32)

        # 1. Slack bus nodes are ALWAYS monitored by substation SCADA / µPMU
        for i, node in enumerate(self.graph.nodes):
            if node.is_slack:
                mask_v[i] = 1.0
                mask_p[i] = 1.0
                mask_q[i] = 1.0

        # 2. Determine remaining candidate meter locations (load buses, DER buses, etc.)
        non_slack_indices = [i for i, node in enumerate(self.graph.nodes) if not node.is_slack]
        n_meters_to_place = int(np.round(len(non_slack_indices) * availability_rate))

        if n_meters_to_place > 0:
            selected_nodes = rng.choice(non_slack_indices, size=n_meters_to_place, replace=False)
            for idx in selected_nodes:
                # Realistic heterogeneous smart meter capabilities:
                # 60% of AMI meters report full (V, P, Q)
                # 20% report only (P, Q) power telemetry
                # 20% report only (V) voltage monitoring
                r = rng.random()
                if r < 0.60:
                    mask_v[idx] = 1.0
                    mask_p[idx] = 1.0
                    mask_q[idx] = 1.0
                elif r < 0.80:
                    mask_p[idx] = 1.0
                    mask_q[idx] = 1.0
                else:
                    mask_v[idx] = 1.0

        return mask_v, mask_p, mask_q

    def create_measurements(
        self,
        state: ElectricalState,
        availability_rate: Optional[float] = None,
        noise_v_std: Optional[float] = None,
        noise_pq_std: Optional[float] = None,
        custom_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> MeasurementVector:
        """Synthesizes realistic noisy smart meter measurements from ground-truth electrical state."""
        if rng is None:
            rng = self.rng

        avail = self.default_availability if availability_rate is None else availability_rate
        std_v = self.default_noise_v if noise_v_std is None else noise_v_std
        std_pq = self.default_noise_pq if noise_pq_std is None else noise_pq_std

        if custom_masks is not None:
            mask_v, mask_p, mask_q = custom_masks
        else:
            mask_v, mask_p, mask_q = self.generate_sensor_mask(avail, rng=rng)

        N = self.graph.num_nodes

        # Ground truth values
        v_true = state.v_mag_pu.copy()
        p_true = state.p_inj_kw.copy()
        q_true = state.q_inj_kvar.copy()

        # Add zero-mean Gaussian measurement noise
        noise_v = rng.normal(0.0, std_v, size=N)
        # Relative noise on P and Q scaled to magnitude, with floor
        p_scale = np.maximum(np.abs(p_true), 10.0)
        q_scale = np.maximum(np.abs(q_true), 10.0)
        noise_p = rng.normal(0.0, std_pq, size=N) * p_scale
        noise_q = rng.normal(0.0, std_pq, size=N) * q_scale

        z_v = (v_true + noise_v) * mask_v
        z_p = (p_true + noise_p) * mask_p
        z_q = (q_true + noise_q) * mask_q

        return MeasurementVector(
            z_v_mag=z_v,
            z_p_inj=z_p,
            z_q_inj=z_q,
            mask_v=mask_v,
            mask_p=mask_p,
            mask_q=mask_q,
            availability_rate=avail,
            noise_v_std=std_v,
            noise_pq_std=std_pq,
            missing_rate=0.0,
            bad_data_rate=0.0,
            is_corrupted=False,
        )
