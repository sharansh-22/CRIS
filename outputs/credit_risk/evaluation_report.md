# Institutional Evaluation Report - Phase 1

## Model Performance Comparison (Test Set 2018)

|                     |   ROC-AUC |   PR-AUC |   F1-Score |   Brier Score |
|:--------------------|----------:|---------:|-----------:|--------------:|
| Lightgbm            |    0.7068 |   0.2972 |     0.1207 |        0.1244 |
| Xgboost             |    0.7061 |   0.2979 |     0.1077 |        0.1243 |
| Logistic Regression |    0.6931 |   0.2765 |     0.3604 |        0.2021 |

## Confusion Matrices
### Lightgbm
```
[[46622   822]
 [ 8251   623]]
```
- TN: 46622, FP: 822
- FN: 8251, TP: 623

### Xgboost
```
[[46742   702]
 [ 8329   545]]
```
- TN: 46742, FP: 702
- FN: 8329, TP: 545

### Logistic Regression
```
[[32995 14449]
 [ 3748  5126]]
```
- TN: 32995, FP: 14449
- FN: 3748, TP: 5126

## Calibration Analysis
The Brier score and calibration curve (saved as `calibration_plot.png`) indicate how well the predicted probabilities match actual default rates.
