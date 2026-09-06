# Experimental Protocol & Evaluation Methodology

## 1. Data Partitions & Leakage Prevention

- **TRAIN (60%)**: Used exclusively to optimize model parameters.
- **VALIDATION (15%)**: Used for hyperparameter tuning and early stopping.
- **CALIBRATION (10%)**: Strictly held out for split-conformal prediction quantile estimation. Never exposed during backpropagation.
- **TEST (15%)**: Evaluated only once per model checkpoint for final in-distribution metrics.
- **OOD TEST SETS**: Separate full datasets under distribution shift (high load, low load, high PV penetration).

## 2. Statistical Analysis & Hypothesis Testing

All neural and ML models are trained and evaluated across **5 independent random seeds** ($S = \{42, 43, 44, 45, 46\}$).
For model comparison, we report:
- Sample Mean ($\mu$) and Sample Standard Deviation ($\sigma$) across seeds.
- Non-parametric 95% Bootstrap Confidence Intervals with $B = 2,000$ resamples.
- Two-tailed Paired t-tests and Wilcoxon Signed-Rank tests on scenario-level MAE paired differences.
- Cohen's $d$ effect sizes.

## 3. Metrics

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{v}_i - v_i^*|$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (\hat{v}_i - v_i^*)^2}$$
3. **Maximum Absolute Error ($\text{MaxAE}$)**: $\max_i |\hat{v}_i - v_i^*|$
4. **95th Percentile Absolute Error ($\text{P95AE}$)**: $Q_{0.95}(\{|\hat{v}_i - v_i^*|\})$
5. **Empirical Conformal Coverage**:
   $$\text{Coverage}_{1-\alpha} = \frac{1}{N \cdot M} \sum_{m=1}^M \sum_{i=1}^N \mathbb{I}(v_{m, i}^* \in [\hat{l}_{m, i}, \hat{u}_{m, i}])$$
6. **Mean Interval Width**:
   $$\text{Width} = \frac{1}{N \cdot M} \sum_{m=1}^M \sum_{i=1}^N (\hat{u}_{m, i} - \hat{l}_{m, i})$$
