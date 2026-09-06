"""Generates simulated power-flow datasets and corresponding measurement snapshots.

Run:
    python scripts/generate_dataset.py --regime normal --n_samples 3000
    python scripts/generate_dataset.py --regime ood_high_loading --n_samples 500
    python scripts/generate_dataset.py --regime ood_low_loading --n_samples 500
    python scripts/generate_dataset.py --regime ood_high_solar_der --n_samples 500
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dsse.config import AppConfig
from dsse.corruption import apply_bad_data, apply_random_missingness, apply_structured_missingness
from dsse.measurements import MeasurementGenerator
from dsse.reproducibility import save_metadata, set_seed, setup_logger
from dsse.simulation import generate_simulation_dataset
from dsse.topology import extract_feeder_topology

logger = setup_logger("generate_dataset")

AVAILABILITY_RATES = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
MISSING_RATES = [0.0, 0.10, 0.20, 0.40, 0.60]
BAD_DATA_RATES = [0.0, 0.01, 0.02, 0.05, 0.10]
NOISE_STD_V = [0.002, 0.005, 0.010, 0.020, 0.050]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate DSSE simulation dataset.")
    p.add_argument("--regime", type=str, default="normal",
                   choices=["normal", "ood_high_loading", "ood_low_loading", "ood_high_solar_der"])
    p.add_argument("--n_samples", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--availability", type=float, default=0.40,
                   help="Default smart-meter availability rate for the generated measurement set.")
    p.add_argument("--noise_v", type=float, default=0.005,
                   help="Voltage measurement noise std (pu).")
    p.add_argument("--missing_rate", type=float, default=0.0)
    p.add_argument("--bad_data_rate", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig.load_all()
    set_seed(args.seed)

    master_file = cfg.paths.feeder_dir / "IEEE123Master.dss"
    raw_dir = cfg.paths.data_dir / "raw"
    interim_dir = cfg.paths.data_dir / "interim"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run power flow simulations
    logger.info(f"Starting dataset generation: regime={args.regime}, n={args.n_samples}, seed={args.seed}")
    states, sim_meta = generate_simulation_dataset(
        master_file=master_file,
        n_samples=args.n_samples,
        regime=args.regime,
        seed=args.seed,
        output_dir=raw_dir,
    )

    # 2. Extract graph topology (for measurement generator)
    graph = extract_feeder_topology(master_file)
    meas_gen = MeasurementGenerator(
        graph=graph,
        default_availability=args.availability,
        default_noise_v=args.noise_v,
        default_noise_pq=0.02,
        seed=args.seed,
    )

    # 3. Generate measurement vectors for each scenario
    rng = np.random.default_rng(args.seed + 1000)
    z_v_list, z_p_list, z_q_list = [], [], []
    mv_list, mp_list, mq_list = [], [], []

    for sid, state in enumerate(states):
        meas = meas_gen.create_measurements(
            state=state,
            availability_rate=args.availability,
            noise_v_std=args.noise_v,
            rng=rng,
        )

        # Apply additional corruption if requested
        if args.missing_rate > 0:
            meas = apply_random_missingness(meas, args.missing_rate, rng=rng)
        if args.bad_data_rate > 0:
            meas = apply_bad_data(meas, args.bad_data_rate, rng=rng)

        z_v_list.append(meas.z_v_mag)
        z_p_list.append(meas.z_p_inj)
        z_q_list.append(meas.z_q_inj)
        mv_list.append(meas.mask_v)
        mp_list.append(meas.mask_p)
        mq_list.append(meas.mask_q)

    # 4. Save measurement arrays
    tag = (f"{args.regime}_avail{int(args.availability*100)}"
           f"_noise{int(args.noise_v*1000)}"
           f"_miss{int(args.missing_rate*100)}"
           f"_bad{int(args.bad_data_rate*100)}")
    meas_path = interim_dir / f"measurements_{tag}.npz"
    np.savez_compressed(
        meas_path,
        z_v_mag=np.stack(z_v_list),
        z_p_inj=np.stack(z_p_list),
        z_q_inj=np.stack(z_q_list),
        mask_v=np.stack(mv_list),
        mask_p=np.stack(mp_list),
        mask_q=np.stack(mq_list),
    )

    dataset_meta = {
        "regime": args.regime,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "availability": args.availability,
        "noise_v_std": args.noise_v,
        "missing_rate": args.missing_rate,
        "bad_data_rate": args.bad_data_rate,
        "states_file": str(raw_dir / f"sim_states_{args.regime}.npz"),
        "measurements_file": str(meas_path),
        "convergence_rate": sim_meta["convergence_rate"],
        "n_nodes": graph.num_nodes,
    }
    save_metadata(dataset_meta, interim_dir / f"dataset_meta_{tag}.json")
    logger.info(f"Dataset generation complete. Measurements saved to {meas_path}")


if __name__ == "__main__":
    main()
