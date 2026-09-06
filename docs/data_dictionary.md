# Data Dictionary & Machine-Readable Metadata

## 1. Electrical State Variables (`sim_states_*.npz`)

| Key | Shape | Type | Units | Description |
|---|---|---|---|---|
| `v_mag_pu` | `[S, 278]` | `float64` | p.u. | True nodal voltage magnitudes |
| `v_ang_rad` | `[S, 278]` | `float64` | radians | True nodal voltage phase angles |
| `p_inj_kw` | `[S, 278]` | `float64` | kW | Net active power injection at node |
| `q_inj_kvar` | `[S, 278]` | `float64` | kvar | Net reactive power injection at node |

## 2. Measurement Arrays (`measurements_*.npz`)

| Key | Shape | Type | Units | Description |
|---|---|---|---|---|
| `z_v_mag` | `[S, 278]` | `float32` | p.u. | Measured voltage magnitude (0 if unobserved) |
| `z_p_inj` | `[S, 278]` | `float32` | kW | Measured active power (0 if unobserved) |
| `z_q_inj` | `[S, 278]` | `float32` | kvar | Measured reactive power (0 if unobserved) |
| `mask_v` | `[S, 278]` | `float32` | {0, 1} | Binary indicator of voltage measurement |
| `mask_p` | `[S, 278]` | `float32` | {0, 1} | Binary indicator of active power measurement |
| `mask_q` | `[S, 278]` | `float32` | {0, 1} | Binary indicator of reactive power measurement |

## 3. Node Metadata (`FeederNode`)

Each index $i \in \{0, \dots, 277\}$ maps to:
- `node_idx`: Integer index in $[0, 277]$.
- `bus_name`: OpenDSS alphanumeric bus identifier (e.g. `'1'`, `'150'`, `'114'`).
- `phase`: Phase integer $p \in \{1, 2, 3\}$ (corresponding to Phase A, B, C).
- `node_id`: Composite string `"{bus_name}.{phase}"`.
- `base_kv_ln`: Nominal line-to-neutral base voltage (kV).
- `is_slack`: Boolean indicating substation slack bus conductor.
- `is_load`: Boolean indicating presence of spot/distributed customer load.
- `is_cap`: Boolean indicating shunt capacitor placement.
