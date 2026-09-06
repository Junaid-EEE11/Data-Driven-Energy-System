"""OpenDSS interface for the IEEE 123-node feeder: simulation, power flow solving, and state extraction."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import opendssdirect as dss

from dsse.reproducibility import setup_logger
from dsse.topology import FeederGraph, extract_feeder_topology

logger = setup_logger("opendss_interface")


@dataclass
class ElectricalState:
    """Ground-truth electrical state snapshot for a single power flow operating point."""
    scenario_id: int
    load_multiplier: float
    der_penetration: float
    converged: bool
    # Nodal state vectors aligned with feeder_graph.nodes
    v_mag_pu: np.ndarray  # [N_nodes] in per unit
    v_mag_kv: np.ndarray  # [N_nodes] in kV line-to-neutral
    v_ang_deg: np.ndarray  # [N_nodes] in degrees (-180 to 180)
    v_ang_rad: np.ndarray  # [N_nodes] in radians
    v_complex: np.ndarray  # [N_nodes] complex phasor in kV
    p_inj_kw: np.ndarray  # [N_nodes] net active power injection in kW
    q_inj_kvar: np.ndarray  # [N_nodes] net reactive power injection in kvar
    # System totals
    total_p_kw: float
    total_q_kvar: float
    total_loss_kw: float
    total_loss_kvar: float


class OpenDSSInterface:
    """Encapsulates all OpenDSS execution, power-flow solving, and parameter adjustment."""

    def __init__(self, master_file: Path | str, feeder_name: str = "ieee123") -> None:
        self.master_file = Path(master_file).resolve()
        self.feeder_name = feeder_name
        if not self.master_file.exists():
            raise FileNotFoundError(f"OpenDSS master file not found: {self.master_file}")
        self.graph: Optional[FeederGraph] = None
        self._initialize_circuit()

    def _initialize_circuit(self) -> None:
        """Clears and compiles the OpenDSS circuit."""
        dss.Basic.ClearAll()
        cmd = f'compile "{self.master_file}"'
        res = dss.Text.Command(cmd)
        if res:
            logger.debug(f"OpenDSS command output: {res}")
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            logger.warning("Initial base-case power flow did not converge!")
        if self.graph is None:
            self.graph = extract_feeder_topology(dss, feeder_name=self.feeder_name)
            logger.info(f"OpenDSS circuit initialized from {self.master_file}")

    def reset_base_case(self) -> None:
        """Recompiles the base-case circuit without re-extracting topology."""
        dss.Basic.ClearAll()
        dss.Text.Command(f'compile "{self.master_file}"')
        dss.Solution.Solve()

    def set_load_multiplier(self, multiplier: float) -> None:
        """Sets the global feeder load multiplier in OpenDSS."""
        dss.Solution.LoadMult(float(multiplier))

    def set_stochastic_loads(
        self,
        mean_mult: float,
        std_mult: float,
        rng: np.random.Generator,
        pf_range: Tuple[float, float] = (0.90, 0.98),
    ) -> None:
        """Applies realistic correlated stochastic load variations across individual loads."""
        load_names = dss.Loads.AllNames()
        for lname in load_names:
            dss.Loads.Name(lname)
            base_kw = dss.Loads.kW()
            # Multiplicative variation with clipping to realistic positive range
            delta = rng.normal(0.0, std_mult)
            mult = max(0.1, mean_mult + delta)
            dss.Loads.kW(base_kw * mult)

            # Randomize power factor slightly
            pf = rng.uniform(pf_range[0], pf_range[1])
            dss.Loads.PF(pf)

    def set_der_generation(
        self,
        penetration_fraction: float,
        rng: np.random.Generator,
    ) -> None:
        """Enables distributed solar PV generators scaled to match specified penetration."""
        # Check if generators exist
        gen_names = dss.Generators.AllNames()
        if not gen_names and penetration_fraction > 0.0:
            # Redirect SolarPV.dss
            solar_path = self.master_file.parent / "SolarPV.dss"
            if solar_path.exists():
                dss.Text.Command(f'redirect "{solar_path}"')
                gen_names = dss.Generators.AllNames()

        if gen_names:
            for gname in gen_names:
                dss.Generators.Name(gname)
                base_kw = 100.0  # nominal
                # Scale by penetration and stochastic solar irradiance factor
                solar_factor = rng.uniform(0.7, 1.3) if penetration_fraction > 0 else 0.0
                dss.Generators.kW(base_kw * penetration_fraction * solar_factor)

    def solve_power_flow(self) -> bool:
        """Solves the 3-phase AC power flow."""
        dss.Solution.Solve()
        return bool(dss.Solution.Converged())

    def extract_state(self, scenario_id: int = 0, load_mult: float = 1.0, der_pen: float = 0.0) -> ElectricalState:
        """Extracts the ground truth nodal voltage and injection state."""
        if self.graph is None:
            raise RuntimeError("FeederGraph is not initialized.")

        converged = bool(dss.Solution.Converged())
        N = self.graph.num_nodes

        v_mag_pu = np.zeros(N, dtype=np.float64)
        v_mag_kv = np.zeros(N, dtype=np.float64)
        v_ang_deg = np.zeros(N, dtype=np.float64)
        v_ang_rad = np.zeros(N, dtype=np.float64)
        v_complex = np.zeros(N, dtype=complex)
        p_inj_kw = np.zeros(N, dtype=np.float64)
        q_inj_kvar = np.zeros(N, dtype=np.float64)

        for i, node_meta in enumerate(self.graph.nodes):
            bus = node_meta.bus_name
            phase = node_meta.phase

            dss.Circuit.SetActiveBus(bus)
            voltages = dss.Bus.Voltages()  # [real1, imag1, real2, imag2, ...]
            nodes_in_bus = dss.Bus.Nodes()  # [1, 2, 3, ...]
            # opendssdirect provides Voltages() or puVmagAngle()
            pu_mag_ang = dss.Bus.puVmagAngle()  # [vmag_pu1, vang_deg1, vmag_pu2, vang_deg2, ...]
            if phase in nodes_in_bus:
                phase_idx = nodes_in_bus.index(phase)
                v_real = voltages[2 * phase_idx]
                v_imag = voltages[2 * phase_idx + 1]
                v_c = complex(v_real, v_imag) / 1000.0  # convert to kV
                v_complex[i] = v_c
                v_mag_kv[i] = abs(v_c)
                v_ang_rad[i] = np.angle(v_c)
                v_ang_deg[i] = np.degrees(v_ang_rad[i])

                if len(pu_mag_ang) >= 2 * (phase_idx + 1):
                    v_mag_pu[i] = pu_mag_ang[2 * phase_idx]
                    v_ang_deg[i] = pu_mag_ang[2 * phase_idx + 1]
                    v_ang_rad[i] = np.radians(v_ang_deg[i])
                else:
                    v_mag_pu[i] = abs(v_c) / node_meta.base_kv_ln
            else:
                v_mag_pu[i] = 1.0
                v_mag_kv[i] = node_meta.base_kv_ln
                v_ang_deg[i] = 0.0 if phase == 1 else (-120.0 if phase == 2 else 120.0)
                v_ang_rad[i] = np.radians(v_ang_deg[i])
                v_complex[i] = node_meta.base_kv_ln * np.exp(1j * v_ang_rad[i])

        # Compute net nodal power injections from Y_bus V
        # I = Y * V (in Amps / kA), S = V * conj(I)
        try:
            v_volts = v_complex * 1000.0  # Volts
            i_inj = self.graph.y_bus_dense @ v_volts  # Amps
            s_inj = v_volts * np.conj(i_inj)  # VA
            p_inj_kw = np.real(s_inj) / 1000.0  # kW
            q_inj_kvar = np.imag(s_inj) / 1000.0  # kvar
        except Exception:
            p_inj_kw = np.zeros(N)
            q_inj_kvar = np.zeros(N)

        total_power = dss.Circuit.TotalPower()
        total_loss = dss.Circuit.Losses()

        state = ElectricalState(
            scenario_id=scenario_id,
            load_multiplier=load_mult,
            der_penetration=der_pen,
            converged=converged,
            v_mag_pu=v_mag_pu,
            v_mag_kv=v_mag_kv,
            v_ang_deg=v_ang_deg,
            v_ang_rad=v_ang_rad,
            v_complex=v_complex,
            p_inj_kw=p_inj_kw,
            q_inj_kvar=q_inj_kvar,
            total_p_kw=float(total_power[0]) if len(total_power) > 0 else 0.0,
            total_q_kvar=float(total_power[1]) if len(total_power) > 1 else 0.0,
            total_loss_kw=float(total_loss[0]) / 1000.0 if len(total_loss) > 0 else 0.0,
            total_loss_kvar=float(total_loss[1]) / 1000.0 if len(total_loss) > 1 else 0.0,
        )
        return state
