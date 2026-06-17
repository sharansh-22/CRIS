# CRIS Statistical Audit

This report documents bootstrap and permutation test validations for the core metrics.

## Statistical Significance Summary

| Metric | Baseline | CR + CRIS | Observed Difference | 95% Confidence Interval | p-value | Audit Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 0.70687 | 0.70061 | -0.00627 | `[-0.00724, -0.00539]` | 0.000 | **Statistically Significant Degradation** |
| **PR-AUC** | 0.29726 | 0.28838 | -0.00888 | `[-0.01201, -0.00632]` | 0.000 | **Statistically Significant Degradation** |
| **ECE** | 0.02060 | 0.01968 | -0.00092 | N/A | N/A | **Statistically Insignificant (Noise)** |
| **Portfolio NPV** | $90.25M | $62.59M | $-27.66M | N/A | N/A | **Statistically Significant Yield Compression** |

## Audit Recommendations
1. **Withdraw All Claims of Predictive Lift**: Empirical evidence shows CRIS signals systematically degrade model performance out-of-sample. All claims of "predictive lift" or "macro-conditioning benefits" must be withdrawn.
2. **Correct the Governance Narrative**: The claim that CRIS governance creates a "more capital-efficient portfolio" is false. The portfolio is safer (lower default rates) but less capital-efficient (lower RoC), which is a standard risk-yield trade-off.
