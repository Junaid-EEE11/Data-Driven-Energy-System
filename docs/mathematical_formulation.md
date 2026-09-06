# Mathematical Formulation of Distribution System State Estimation (DSSE)

## 1. Network Topology and Electrical Representation

We model an unbalanced three-phase radial or weakly-meshed distribution network as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where:
- $\mathcal{V} = \{1, 2, \dots, N\}$ is the set of $N = 278$ bus-phase nodes corresponding to all energized phase conductors across the IEEE 123-node feeder.
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of directed edges representing physical distribution line segments, underground cables, regulators, and distribution transformers.

The multi-phase complex nodal admittance matrix is denoted by:
$$Y_{\text{bus}} = G + jB \in \mathbb{C}^{N \times N}$$
where $G = \operatorname{Re}(Y_{\text{bus}})$ is the nodal conductance matrix and $B = \operatorname{Im}(Y_{\text{bus}})$ is the nodal susceptance matrix extracted directly from the converged OpenDSS circuit model.

---

## 2. Electrical State Definition

The true electrical state of the distribution feeder at snapshot $t$ is the vector of complex nodal voltages:
$$\mathbf{V} = \left[ V_1 e^{j\theta_1}, V_2 e^{j\theta_2}, \dots, V_N e^{j\theta_N} \right]^T \in \mathbb{C}^N$$

In our primary formulation:
- **Target State Vector**: $\mathbf{x} = \mathbf{v} = [|V_1|, |V_2|, \dots, |V_N|]^T \in \mathbb{R}^N$, representing voltage magnitudes per unit (p.u.).
- **Slack Reference**: Bus 150 (phases A, B, C) serves as the primary substation slack reference with fixed nominal voltage magnitude $|V_{\text{slack}}| = 1.0$ p.u. and phase angles $\theta_{\text{slack}} = [0^\circ, -120^\circ, +120^\circ]$.

---

## 3. Physics-Consistency Power Flow Residual

According to Kirchhoff's current law (KCL) and Ohm's law, the complex current injection at all feeder nodes is:
$$\mathbf{I} = Y_{\text{bus}} \mathbf{V}$$

The corresponding net complex power injection vector is:
$$\mathbf{S} = \mathbf{P} + j\mathbf{Q} = \mathbf{V} \odot \mathbf{I}^* = \mathbf{V} \odot (Y_{\text{bus}} \mathbf{V})^*$$

At each node $k \in \mathcal{V}$:
$$P_k(\mathbf{v}, \boldsymbol{\theta}) = \sum_{j=1}^N |V_k| |V_j| \left[ G_{kj} \cos(\theta_k - \theta_j) + B_{kj} \sin(\theta_k - \theta_j) \right]$$
$$Q_k(\mathbf{v}, \boldsymbol{\theta}) = \sum_{j=1}^N |V_k| |V_j| \left[ G_{kj} \sin(\theta_k - \theta_j) - B_{kj} \cos(\theta_k - \theta_j) \right]$$

### Physics Regularization Loss Term
For a predicted voltage magnitude vector $\hat{\mathbf{v}}$:
$$\mathcal{L}_{\text{physics}}(\hat{\mathbf{v}}) = \frac{1}{|\mathcal{M}_P|} \sum_{k \in \mathcal{M}_P} \left( P_k(\hat{\mathbf{v}}, \boldsymbol{\theta}_0) - z_{P, k} \right)^2 + \frac{1}{|\mathcal{M}_Q|} \sum_{k \in \mathcal{M}_Q} \left( Q_k(\hat{\mathbf{v}}, \boldsymbol{\theta}_0) - z_{Q, k} \right)^2$$
where $\mathcal{M}_P, \mathcal{M}_Q$ are observed power injection sensor locations and $\boldsymbol{\theta}_0$ represents the nominal phase shift per phase.

---

## 4. Total Optimization Objective

The proposed Physics-Informed Graph Neural Network minimizes:
$$\mathcal{L}_{\text{total}}(\mathbf{w}) = \mathcal{L}_{\text{supervised}}(\hat{\mathbf{v}}(\mathbf{w}), \mathbf{v}^*) + \lambda_{\text{physics}} \mathcal{L}_{\text{physics}}(\hat{\mathbf{v}}(\mathbf{w})) + \lambda_{\text{reg}} \|\mathbf{w}\|_2^2$$
where:
- $\mathcal{L}_{\text{supervised}} = \frac{1}{N} \sum_{i=1}^N (\hat{v}_i - v_i^*)^2$ (Mean Squared Error)
- $\lambda_{\text{physics}} \ge 0$ is the physics regularization weighting hyperparameter.
