# CRIS vs. Credit Risk Champion — Predictive Evaluation Report

This report compares the out-of-sample classification performance of the standalone Credit Risk model (Control) vs. the CRIS-conditioned model (Treatment) on the test split (year >= 2018).

## Performance Table

| Metric | Control (Credit Risk Champion) | Treatment (CR + CRIS) | Delta |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.70687 | 0.70061 | -0.00627 |
| **PR-AUC** | 0.29726 | 0.28838 | -0.00888 |
| **Brier Score** | 0.12421 | 0.12518 | +0.00097 |
| **Expected Calibration Error (ECE)** | 0.02060 | 0.01968 | -0.00092 |
| **Recall (at 20% PD threshold)** | 59.15% | 59.19% | +0.04% |
| **Precision (at 20% PD threshold)** | 27.17% | 27.08% | -0.09% |
| **F1 Score (at 20% PD threshold)** | 0.37238 | 0.37161 | -0.00077 |

## Error Classification Analysis

| Metric | Control (Credit Only) | Treatment (CR + CRIS) | Change |
| :--- | :---: | :---: | :---: |
| **True Positives (TP)** | 4,652 | 4,655 | +3 |
| **False Positives (FP)** | 12,468 | 12,533 | +65 |
| **False Negatives (FN)** | 3,213 | 3,210 | -3 |
| **True Negatives (TN)** | 29,667 | 29,602 | -65 |

## Key Findings
- The addition of CRIS signals to the model **degraded** out-of-sample ROC-AUC and PR-AUC slightly.
- This performance decline is consistent with panel-data overfitting, where the time-series macro signals are over-fit on the training set (2007-2015) but fail to generalize to the out-of-sample period (2018).
- Calibration error (ECE) shows a minor difference, indicating that the probability estimates remain relatively stable.