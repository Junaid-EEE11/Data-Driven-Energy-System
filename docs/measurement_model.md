# Measurement Model and Corruption Protocols

## 1. Heterogeneous Measurement Types

Measurements are obtained from advanced metering infrastructure (AMI), distribution substation SCADA, and power quality monitors:
1. **Voltage Magnitude ($z_{V, i}$)**: Smart meter RMS voltage magnitude (p.u.).
2. **Active Power Injection ($z_{P, i}$)**: Net active power consumption/generation (kW).
3. **Reactive Power Injection ($z_{Q, i}$)**: Net reactive power consumption/generation (kvar).

## 2. Sensor Placement and Observability Masks

For each bus-phase node $i \in \{1, \dots, N\}$, binary observability indicators specify meter presence:
- $m_{V, i} \in \{0, 1\}$: Voltage magnitude measurement available.
- $m_{P, i} \in \{0, 1\}$: Active power measurement available.
- $m_{Q, i} \in \{0, 1\}$: Reactive power measurement available.

Sensor availability rate $\alpha_{\text{avail}} \in [0.10, 1.00]$ governs the fraction of non-slack buses equipped with smart meters. The substation slack bus (Bus 150) is always monitored ($m_{V, \text{slack}} = 1$).

## 3. Corruption Protocols

### A. Gaussian Measurement Noise
$$z_{V, i} = v_i^* + \epsilon_{V, i}, \quad \epsilon_{V, i} \sim \mathcal{N}(0, \sigma_V^2)$$
$$z_{P, i} = P_i^* + \epsilon_{P, i}, \quad \epsilon_{P, i} \sim \mathcal{N}(0, \sigma_P^2)$$
$$z_{Q, i} = Q_i^* + \epsilon_{Q, i}, \quad \epsilon_{Q, i} \sim \mathcal{N}(0, \sigma_Q^2)$$
Default parameters: $\sigma_V = 0.005$ p.u. (0.5%), $\sigma_P = \sigma_Q = 0.02 \cdot |P_i^*|$.

### B. Random Missingness / Sensor Dropout
A fraction $r_{\text{miss}} \in [0.10, 0.60]$ of active measurements is independently dropped:
$$m'_{i} = m_i \cdot \mathbb{I}(u_i > r_{\text{miss}}), \quad u_i \sim \mathcal{U}(0, 1)$$

### C. Structured Missingness (Communication Cluster Failure)
Simulates localized communication gateway failure. A connected subgraph $\mathcal{G}_{\text{cluster}}$ of size $K = 15$ nodes has all meter masks set to zero simultaneously.

### D. Gross Bad Data and Outliers
A fraction $r_{\text{bad}} \in [0.01, 0.10]$ of available sensors suffers gross measurement corruption:
$$z_{V, i}^{\text{corrupt}} = z_{V, i} \cdot (1 + \delta_i), \quad \delta_i \sim \mathcal{U}(\pm 0.15, \pm 0.35)$$
