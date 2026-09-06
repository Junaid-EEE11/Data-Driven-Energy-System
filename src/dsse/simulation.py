"""Simulation pipeline: stochastic power flow generation across normal and OOD operating regimes."""

from __future__ import annotations
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from dsse.config import AppConfig
from dsse.opendss_interface import ElectricalState, OpenDSSInterface
from dsse.reproducibility import save_metadata, setup_logger

logger = setup_logger("simulation")


def generate_simulation_dataset(
    master_file: Path | str,
    n_samples: int,
    regime: str = "normal",
    seed: int = 42,
    output_dir: Optional[Path | str] = None,
) -> Tuple[List[ElectricalState], Dict[str, Any]]:
    """Generates a batch of solved AC power flow states under a specified operating regime."""
    rng = np.random.default_rng(seed)
    interface = OpenDSSInterface(master_file=master_file)

    states: List[ElectricalState] = []
    metadata_list: List[Dict[str, Any]] = []

    logger.info(f"Generating {n_samples} power-flow snapshots for regime '{regime}' with seed {seed}...")

    # Configure regime distribution parameters
    if regime == "normal":
        mean_mult, std_mult, min_m, max_m = 1.0, 0.15, 0.60, 1.40
        der_fraction_range = (0.0, 0.05)
    elif regime == "ood_high_loading":
        mean_mult, std_mult, min_m, max_m = 1.70, 0.10, 1.45, 1.95
        der_fraction_range = (0.0, 0.05)
    elif regime == "ood_low_loading":
        mean_mult, std_mult, min_m, max_m = 0.38, 0.08, 0.20, 0.55
        der_fraction_range = (0.0, 0.05)
    elif regime == "ood_high_solar_der":
        mean_mult, std_mult, min_m, max_m = 0.90, 0.12, 0.65, 1.20
        der_fraction_range = (0.35, 0.55)
    else:
        raise ValueError(f"Unknown operating regime: {regime}")

    start_time = time.time()
    for sid in tqdm(range(n_samples), desc=f"Simulating [{regime}]"):
        # Reset base circuit
        interface.reset_base_case()

        # Sample load multiplier
        sampled_mult = float(np.clip(rng.normal(mean_mult, std_mult), min_m, max_m))
        der_pen = float(rng.uniform(der_fraction_range[0], der_fraction_range[1]))

        # Apply loads and DER
        interface.set_stochastic_loads(
            mean_mult=sampled_mult,
            std_mult=0.08,
            rng=rng,
            pf_range=(0.88, 0.98),
        )
        if der_pen > 0:
            interface.set_der_generation(penetration_fraction=der_pen, rng=rng)

        # Solve AC power flow
        converged = interface.solve_power_flow()
        if not converged:
            logger.warning(f"Power flow did not converge for scenario {sid} (load_mult={sampled_mult:.3f})")

        state = interface.extract_state(scenario_id=sid, load_mult=sampled_mult, der_pen=der_pen)
        states.append(state)

        metadata_list.append({
            "scenario_id": sid,
            "regime": regime,
            "load_multiplier": sampled_mult,
            "der_penetration": der_pen,
            "converged": state.converged,
            "total_p_kw": state.total_p_kw,
            "total_q_kvar": state.total_q_kvar,
            "v_min_pu": float(state.v_mag_pu.min()),
            "v_max_pu": float(state.v_mag_pu.max()),
            "v_mean_pu": float(state.v_mag_pu.mean()),
        })

    elapsed = time.time() - start_time
    summary_meta = {
        "regime": regime,
        "n_samples": n_samples,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "convergence_rate": sum(s.converged for s in states) / max(1, len(states)),
        "samples": metadata_list,
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Save compressed raw data
        v_mag_array = np.stack([s.v_mag_pu for s in states], axis=0)
        v_ang_array = np.stack([s.v_ang_rad for s in states], axis=0)
        p_inj_array = np.stack([s.p_inj_kw for s in states], axis=0)
        q_inj_array = np.stack([s.q_inj_kvar for s in states], axis=0)

        np.savez_compressed(
            out_path / f"sim_states_{regime}.npz",
            v_mag_pu=v_mag_array,
            v_ang_rad=v_ang_array,
            p_inj_kw=p_inj_array,
            q_inj_kvar=q_inj_array,
        )
        save_metadata(summary_meta, out_path / f"sim_metadata_{regime}.json")
        logger.info(f"Saved {n_samples} simulated states to {out_path / f'sim_states_{regime}.npz'}")

    return states, summary_meta
