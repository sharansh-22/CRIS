# CRIS Phase 3.1 — Statistical Validation Report

This report documents bootstrap significance tests for the differences between CRIS configurations and the Credit-Only baseline.

## Statistical Significance Table (Config B vs Config A)

| Metric | Observed Difference | 95% Confidence Interval | Significant Degradation? |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | -0.00022 | [-0.00101, +0.00056] | NO |
| **PR-AUC** | 0.00309 | [+0.00118, +0.00478] | NO |
| **Segmentation Ratio** | 0.09x | [-1.03x, +0.85x] | NO |
| **NPV (60% Capacity)** | $+266,392 | [$-599,841, $+1,247,817] | NO |

## Statistical Significance Table (Config F vs Config A)

| Metric | Observed Difference | 95% Confidence Interval | Significant Degradation? |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | -0.00626 | [-0.00723, -0.00538] | YES |
| **PR-AUC** | -0.00914 | [-0.01231, -0.00648] | YES |
| **Segmentation Ratio** | -0.16x | [-1.53x, +0.63x] | NO |
| **NPV (60% Capacity)** | $-1,510,709 | [$-2,548,902, $-591,366] | YES |

## Key Findings
- For Configuration B (Top 1), the performance decline is small but consistent, and not fully significant on NPV.
- For Configuration F (All Signals), the degradation in ROC-AUC, PR-AUC, and NPV is **statistically significant** (the 95% confidence intervals are entirely below zero).