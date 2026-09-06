# Research Log: Physics-Aware and Uncertainty-Calibrated DSSE

## Project Title
**Physics-Aware and Uncertainty-Calibrated Distribution System State Estimation Under Sparse, Missing, and Corrupted Smart-Meter Measurements**

## Objective
Investigate whether topology-aware machine learning combined with electrical-physics consistency and calibrated predictive uncertainty improves distribution-system state estimation (DSSE) under sparse, noisy, missing, corrupted, and distribution-shifted measurements.

---

## 1. Technical Risks & Mitigation Strategies

| Technical Area | Risk / Challenge | Mitigation Strategy |
|---|---|---|
| **Phase Preserving Representation** | IEEE 123 is 3-phase unbalanced (1-phase, 2-phase, and 3-phase branches). Converting to balanced single-phase equivalent loses essential asymmetric physics. | Model each energized `(bus, phase)` tuple as a discrete graph node or each bus as a 3-phase node with explicit phase existence masks. Maintain deterministic node-phase mapping index throughout. |
| **Feeder Electrical Parameters** | Extracting exact branch series impedances ($Z_{ij} \in \mathbb{C}^{3\times 3}$) and shunt admittances from OpenDSS linecodes and geometries. | Parse OpenDSS full System Y-matrix (`dss.Circuit.SystemY()`) or compute branch line impedance matrices directly using `opendssdirect.py`. Verify complex branch flow vs nodal voltage equations against OpenDSS ground truth. |
| **Model-Based WLS Baseline** | Classical 3-phase branch-current / nodal-voltage WLS state estimation can encounter ill-conditioning or non-convergence when measurements are sparse or zero-injection nodes exist. | Implement a robust 3-phase Weighted Least Squares (WLS) state estimator using branch-current/nodal-voltage formulation or Gauss-Newton/Levenberg-Marquardt with pseudomeasurements (virtual zero-injection constraints) where needed. Document exact equations and convergence criteria. |
| **Differentiable Physics Loss** | AC power flow is non-linear. If predicting voltage magnitude only, phase angle is required for exact $I = Y V$ or $S = V (Y V)^*$. | Formulate a multi-target or physics-regularized residual: either jointly estimate voltage magnitude and angle with slack bus reference ($\theta_{\text{slack}} = [0, -120^\circ, 120^\circ]$) or use linearized 3-phase LinDistFlow / admittance current-mismatch residual $\mathcal{L}_{\text{phys}} = \| I_{\text{inj}} - Y_{\text{bus}} V \|_2^2$. Document mathematical derivation explicitly. |
| **Conformal Prediction Validity** | Split conformal prediction guarantees marginal coverage $1-\alpha$ under exchangeability, but fails under OOD / distribution shift. | Use strict split conformal calibration on a dedicated `CALIBRATION` split (never used in training or validation). Empirically evaluate coverage under both in-distribution and distinct OOD test sets (extreme loading, high DER penetration, topology change). Document coverage degradation transparently without claiming false OOD guarantees. |
| **PyTorch Architecture** | Graph message passing over 3-phase unbalanced networks. | Implement clean PyTorch Graph Neural Network (GNN) with EdgeConv / GAT / GraphSAGE message passing that operates on node features (meter measurements, masks, phase one-hot) and edge features (line impedance/admittance). Zero external non-reproducible dependencies. |

---

## 2. Phased Implementation Plan

- [x] **Phase 1: Environment, Configuration, Logging, Seeds & Repository Skeleton**
- [ ] **Phase 2: IEEE-123 / OpenDSS Interface and Topology Extraction**
- [ ] **Phase 3: Simulation & Data Generation Pipeline (Normal & OOD Scenarios)**
- [ ] **Phase 4: Heterogeneous Measurement & Corruption Models**
- [ ] **Phase 5: Data Partitioning (Train/Val/Cal/Test/OOD) & Leakage Tests**
- [ ] **Phase 6: Simple Baselines (Linear/Ridge, MLP, Tree)**
- [ ] **Phase 7: Conventional 3-Phase Model-Based WLS Estimator**
- [ ] **Phase 8: Topology-Aware Graph Neural Network (GNN)**
- [ ] **Phase 9: Differentiable Physics-Consistency Loss & Training Integration**
- [ ] **Phase 10: Split Conformal Uncertainty Calibration & Interval Evaluation**
- [ ] **Phase 11: Comprehensive Robustness & OOD Experiment Matrix**
- [ ] **Phase 12: Ablation Studies & Multi-Seed Paired Statistical Analysis**
- [ ] **Phase 13: Publication-Quality Automated Visualizations and Tables**
- [ ] **Phase 14: Mathematical Documentation, Data Dictionary, Manuscript Skeleton & Reproducibility Suite**

---

## 3. Log of Activities & Progress

- **2026-09-02**: Initial repository inspection. Verified Python 3.11.9 and installed core dependencies (`opendssdirect.py`, `dss-python`, `torch`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pytest`, `networkx`, `pyyaml`). Confirmed OpenDSS and PyTorch CPU functionality. Created implementation roadmap.
