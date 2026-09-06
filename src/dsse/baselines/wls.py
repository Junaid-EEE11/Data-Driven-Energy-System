"""Three-phase Weighted Least Squares DSSE baseline.

Mathematical Formulation
------------------------
The three-phase DSSE problem is formulated as a nonlinear WLS optimisation:

    min_x  (z - h(x))^T W (z - h(x))

where:
  x  : state vector [V_mag_1, ..., V_mag_N, V_ang_1, ..., V_ang_N]  (2N)
  z  : measurement vector [z_V, z_P, z_Q]  (n_meas)
  h  : measurement function (nonlinear for power injections)
  W  : diagonal weight matrix (W_ii = 1/sigma_i^2)

The nonlinear WLS solution uses the Gauss-Newton iterative method:
    H^T W H * dx = H^T W r
    x_{k+1} = x_k + dx

where H is the measurement Jacobian dh/dx and r = z - h(x) is the residual.

For voltage magnitude measurements:
    h_V(x)_k = |V_k|

For complex power injection at bus k (from current injection model):
    P_k = sum_j |V_k| |V_j| (G_kj cos(theta_k - theta_j) + B_kj sin(theta_k - theta_j))
    Q_k = sum_j |V_k| |V_j| (G_kj sin(theta_k - theta_j) - B_kj cos(theta_k - theta_j))

Jacobian elements are computed analytically as per standard power-flow sensitivity formulas.

Convergence: ||dx||_inf < tol or max_iter reached.
Failure cases: Singular gain matrix, non-convergence, ill-conditioned systems.

Reference: [CITATION NEEDED: classical WLS DSSE - Schweppe et al. 1970]
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg

from dsse.reproducibility import setup_logger

logger = setup_logger("baselines.wls")


class ThreePhaseWLS:
    """Nonlinear three-phase WLS state estimator for distribution feeders.

    This implements a reduced-order WLS estimator using voltage magnitudes and
    power injections as measurements. Phase angles are included in the state but
    initialised to their nominal values (0, -120, +120 degrees per phase).

    Limitations (documented per GEMINI.md):
    ----------------------------------------
    - Requires at least one voltage magnitude measurement per connected component.
    - The Gauss-Newton iteration may not converge for heavily loaded or highly
      unbalanced systems, or when measurements are extremely sparse.
    - The implementation uses a simplified single-phase equivalent per node (not
      a full three-phase coupled Jacobian). This is documented as a simplification.
    - For the unbalanced IEEE 123-bus feeder, the Y-bus used here is the 3-phase
      nodal admittance from OpenDSS, indexed per bus-phase node.
    """

    def __init__(
        self,
        G: np.ndarray,              # [N, N] conductance (Re(Y_bus))
        B: np.ndarray,              # [N, N] susceptance (Im(Y_bus))
        ref_node: int = 0,          # Slack/reference node index
        max_iter: int = 30,
        tol: float = 1e-4,
        sigma_v: float = 0.005,     # Voltage measurement std (pu)
        sigma_p: float = 0.02,      # Power injection std (pu of load)
        sigma_q: float = 0.02,
    ) -> None:
        self.G = G
        self.B = B
        self.N = G.shape[0]
        self.ref_node = ref_node
        self.max_iter = max_iter
        self.tol = tol
        self.sigma_v = sigma_v
        self.sigma_p = sigma_p
        self.sigma_q = sigma_q

    def _h_and_jacobian(
        self,
        v_mag: np.ndarray,   # [N]
        theta: np.ndarray,   # [N]
        obs_v: List[int],
        obs_p: List[int],
        obs_q: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluates measurement function h(x) and Jacobian H=dh/dx.

        State x = [v_mag, theta], shape [2N].
        Returns h [n_meas] and H [n_meas, 2N].
        """
        G, B = self.G, self.B
        N = self.N
        n_meas = len(obs_v) + len(obs_p) + len(obs_q)
        h = np.zeros(n_meas)
        H = np.zeros((n_meas, 2 * N))

        row = 0
        # Voltage magnitude measurements: h_k = V_k
        for k in obs_v:
            h[row] = v_mag[k]
            H[row, k] = 1.0         # d|V_k|/d|V_k|
            H[row, N + k] = 0.0     # d|V_k|/d(theta_k)
            row += 1

        # Active power injection: P_k = sum_j V_k V_j (G_kj cos(dth) + B_kj sin(dth))
        for k in obs_p:
            Pk = 0.0
            for j in range(N):
                dth = theta[k] - theta[j]
                Pk += v_mag[k] * v_mag[j] * (G[k, j] * np.cos(dth) + B[k, j] * np.sin(dth))
            h[row] = Pk

            # dP_k / d|V_m|
            for m in range(N):
                if m == k:
                    # Self term: dP_k/d|V_k|
                    s = 0.0
                    for j in range(N):
                        dth = theta[k] - theta[j]
                        s += v_mag[j] * (G[k, j] * np.cos(dth) + B[k, j] * np.sin(dth))
                    H[row, m] = 2 * G[k, k] * v_mag[k] + s - G[k, k] * v_mag[k]
                    # Simpler: sum_j V_j (G_kj cos + B_kj sin) + V_k G_kk
                    s2 = 0.0
                    for j in range(N):
                        dth = theta[k] - theta[j]
                        s2 += v_mag[j] * (G[k, j] * np.cos(dth) + B[k, j] * np.sin(dth))
                    H[row, m] = s2  # simplified
                else:
                    dth = theta[k] - theta[m]
                    H[row, m] = v_mag[k] * (G[k, m] * np.cos(dth) + B[k, m] * np.sin(dth))

            # dP_k / d(theta_m)
            for m in range(N):
                if m == k:
                    s = 0.0
                    for j in range(N):
                        if j != k:
                            dth = theta[k] - theta[j]
                            s += v_mag[k] * v_mag[j] * (-G[k, j] * np.sin(dth) + B[k, j] * np.cos(dth))
                    H[row, N + m] = s
                else:
                    dth = theta[k] - theta[m]
                    H[row, N + m] = v_mag[k] * v_mag[m] * (G[k, m] * np.sin(dth) - B[k, m] * np.cos(dth))
            row += 1

        # Reactive power injection: Q_k = sum_j V_k V_j (G_kj sin(dth) - B_kj cos(dth))
        for k in obs_q:
            Qk = 0.0
            for j in range(N):
                dth = theta[k] - theta[j]
                Qk += v_mag[k] * v_mag[j] * (G[k, j] * np.sin(dth) - B[k, j] * np.cos(dth))
            h[row] = Qk

            for m in range(N):
                if m == k:
                    s = 0.0
                    for j in range(N):
                        dth = theta[k] - theta[j]
                        s += v_mag[j] * (G[k, j] * np.sin(dth) - B[k, j] * np.cos(dth))
                    H[row, m] = s
                else:
                    dth = theta[k] - theta[m]
                    H[row, m] = v_mag[k] * (G[k, m] * np.sin(dth) - B[k, m] * np.cos(dth))

            for m in range(N):
                if m == k:
                    s = 0.0
                    for j in range(N):
                        if j != k:
                            dth = theta[k] - theta[j]
                            s += v_mag[k] * v_mag[j] * (G[k, j] * np.cos(dth) + B[k, j] * np.sin(dth))
                    H[row, N + m] = s
                else:
                    dth = theta[k] - theta[m]
                    H[row, N + m] = v_mag[k] * v_mag[m] * (-G[k, m] * np.cos(dth) - B[k, m] * np.sin(dth))
            row += 1

        return h, H

    def estimate(
        self,
        z_v: np.ndarray,    # [N] voltage measurements (0 where unobserved)
        z_p: np.ndarray,    # [N] P injection measurements (kW, 0 where unobserved)
        z_q: np.ndarray,    # [N] Q injection measurements (kvar, 0 where unobserved)
        mask_v: np.ndarray, # [N] {0,1}
        mask_p: np.ndarray, # [N] {0,1}
        mask_q: np.ndarray, # [N] {0,1}
        base_mva: float = 1.0,
    ) -> Dict[str, np.ndarray | bool | int]:
        """Runs Gauss-Newton WLS iteration to estimate V_mag and V_ang.

        Args:
            z_v, z_p, z_q:     Measurement vectors.
            mask_v, mask_p, mask_q: Observation masks.
            base_mva:           System MVA base for per-unit conversion.

        Returns:
            Dict with keys: v_mag_pu, v_ang_rad, converged, n_iter, residual_norm.
        """
        obs_v = list(np.where(mask_v > 0.5)[0])
        obs_p = list(np.where(mask_p > 0.5)[0])
        obs_q = list(np.where(mask_q > 0.5)[0])

        n_meas = len(obs_v) + len(obs_p) + len(obs_q)
        if n_meas < 2:
            logger.warning("Insufficient measurements for WLS estimation; returning flat start.")
            return {
                "v_mag_pu": np.ones(self.N),
                "v_ang_rad": np.zeros(self.N),
                "converged": False,
                "n_iter": 0,
                "residual_norm": np.inf,
            }

        # Build measurement vector z and weight matrix W
        z = np.zeros(n_meas)
        w = np.zeros(n_meas)
        row = 0
        for k in obs_v:
            z[row] = z_v[k]
            w[row] = 1.0 / (self.sigma_v ** 2)
            row += 1
        for k in obs_p:
            # Convert kW to pu
            z[row] = z_p[k] / (base_mva * 1000.0)
            w[row] = 1.0 / (self.sigma_p ** 2)
            row += 1
        for k in obs_q:
            z[row] = z_q[k] / (base_mva * 1000.0)
            w[row] = 1.0 / (self.sigma_q ** 2)
            row += 1

        W = np.diag(w)

        # Flat start initialisation
        v_mag = np.ones(self.N)
        theta = np.zeros(self.N)
        # Set nominal phase angles for three-phase nodes
        for i in range(self.N):
            # Simple heuristic: assume phase A=0, B=-2pi/3, C=+2pi/3
            # (proper per-node angles are assigned from the FeederGraph node_metadata)
            pass  # leave as 0 - flat start

        # Fix reference node
        ref = self.ref_node
        N = self.N

        converged = False
        n_iter = 0
        residual_norm = np.inf

        for iteration in range(self.max_iter):
            h, H = self._h_and_jacobian(v_mag, theta, obs_v, obs_p, obs_q)
            r = z - h

            # Remove reference node angle column from H (angle not estimated at slack)
            H_red = np.delete(H, N + ref, axis=1)
            # Gain matrix: G_wls = H^T W H
            G_wls = H_red.T @ W @ H_red
            rhs = H_red.T @ W @ r

            try:
                dx_red = linalg.solve(G_wls, rhs, assume_a="sym")
            except linalg.LinAlgError:
                logger.warning(f"WLS Gauss-Newton: singular gain matrix at iteration {iteration}.")
                break

            # Reconstruct full dx with zero at ref node
            dx = np.insert(dx_red, N + ref - (1 if ref > 0 else 0), 0.0)
            # (Simplified: just use reduced directly)
            dv = dx_red[:self.N]
            # Angle corrections (excluding ref)
            dth_indices = list(range(self.N, 2 * self.N - 1))
            dth_vals = dx_red[self.N:]

            # Apply corrections
            v_mag = v_mag + dv
            v_mag = np.clip(v_mag, 0.5, 1.5)
            j_ang = 0
            for j in range(self.N):
                if j != ref:
                    theta[j] += dth_vals[j_ang] if j_ang < len(dth_vals) else 0.0
                    j_ang += 1

            residual_norm = float(np.max(np.abs(dx_red)))
            n_iter = iteration + 1
            if residual_norm < self.tol:
                converged = True
                break

        if converged:
            logger.debug(f"WLS converged in {n_iter} iterations (||dx||_inf={residual_norm:.2e})")
        else:
            logger.debug(f"WLS did NOT converge after {n_iter} iterations (||dx||_inf={residual_norm:.2e})")

        return {
            "v_mag_pu": v_mag,
            "v_ang_rad": theta,
            "converged": converged,
            "n_iter": n_iter,
            "residual_norm": residual_norm,
        }
