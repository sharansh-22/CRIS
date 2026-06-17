# Final Predictive Verdict

This report documents the final audit of direct macro-signal integration into borrower-level credit prediction.

## Quantitative Findings

- **ROC-AUC**: Credit Only = **0.70687** \| CR + CRIS = **0.70061** (Delta = **-0.00627**)
- **PR-AUC**: Credit Only = **0.29726** \| CR + CRIS = **0.28838** (Delta = **-0.00888**)
- **ECE**: Credit Only = **0.02060** \| CR + CRIS = **0.01968** (Delta = **-0.00092**)

## Statistical Confidence Intervals (95% Bootstrap)
- **ROC-AUC Difference**: `[-0.00724, -0.00539]` (Entirely below zero)
- **PR-AUC Difference**: `[-0.01201, -0.00632]` (Entirely below zero)

## Verdict
**FAIL**. The direct integration of CRIS environmental signals into borrower-level classifiers results in a statistically significant degradation of classification ranking. This failure is driven by panel-data overfitting (only 139 distinct months vs over 1 million loans) and information dilution of high-value borrower-intrinsic variables (FICO, DTI) by macro indicators.
