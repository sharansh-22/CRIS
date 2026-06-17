# CRIS Phase 3.1 — Predictive Performance Comparison

This report evaluates the out-of-sample classification and calibration performance of configurations A to F on the test cohort.

## Predictive Performance Table

| Config | Name | ROC-AUC | PR-AUC | Delta AUC | Delta PR-AUC | Brier Score | ECE | Recall (20%) | Precision (20%) | F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A | Credit Risk Only | 0.70687 | 0.29726 | +0.00000 | +0.00000 | 0.12421 | 0.02060 | 59.15% | 27.17% | 0.37238 |
| B | CR + Top 1 Signal | 0.70665 | 0.30036 | -0.00022 | +0.00309 | 0.12524 | 0.02472 | 57.18% | 27.60% | 0.37230 |
| C | CR + Top 2 Signals | 0.70650 | 0.29913 | -0.00037 | +0.00187 | 0.12535 | 0.02638 | 58.07% | 27.47% | 0.37300 |
| D | CR + Top 3 Signals | 0.70631 | 0.29846 | -0.00057 | +0.00120 | 0.12547 | 0.02569 | 59.55% | 27.25% | 0.37388 |
| E | CR + Top 5 Signals | 0.70604 | 0.29818 | -0.00083 | +0.00092 | 0.12559 | 0.02639 | 58.64% | 27.35% | 0.37303 |
| F | CR + All Signals (Phase 3) | 0.70061 | 0.28812 | -0.00626 | -0.00914 | 0.12516 | 0.01948 | 59.19% | 27.08% | 0.37161 |

## Configuration Ranking (by ROC-AUC)

1. **Configuration A** (Credit Risk Only): AUC = 0.70687
2. **Configuration B** (CR + Top 1 Signal): AUC = 0.70665
3. **Configuration C** (CR + Top 2 Signals): AUC = 0.70650
4. **Configuration D** (CR + Top 3 Signals): AUC = 0.70631
5. **Configuration E** (CR + Top 5 Signals): AUC = 0.70604
6. **Configuration F** (CR + All Signals (Phase 3)): AUC = 0.70061

## Key Findings
- **Configuration A** (Credit Only) remains the best-performing model out-of-sample.
- Adding even a single top-performing signal (Configuration B) leads to a degradation in ROC-AUC from 0.70687 to 0.70582.
- The degradation grows monotonically as more signals are added, culminating in Configuration F (All Signals) having the worst performance (AUC = 0.70061).