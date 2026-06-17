# CRIS vs. Credit Risk Champion — Statistical Validation Report

This report presents bootstrap statistical significance tests for the performance difference (Treatment - Control) on the LendingClub test set (100 bootstrap iterations).

## Statistical Significance Table

| Metric | Observed Difference | 95% Confidence Interval | p-value (CRIS >= CR) | Significant? |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | -0.00627 | [-0.00724, -0.00539] | 0.000 | YES (Degradation) |
| **PR-AUC** | -0.00888 | [-0.01201, -0.00632] | | |
| **NPV (60% Capacity)** | $-1,598,550 | [$-2,560,713, $-638,509] | | |

## Key Findings
- The degradation in out-of-sample ROC-AUC for the CRIS-conditioned model is **statistically significant** (the 95% confidence interval is entirely below zero).
- This confirms that the environmental signals introduced out-of-sample noise and did not provide predictive value.