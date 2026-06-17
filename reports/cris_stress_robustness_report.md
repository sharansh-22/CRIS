# CRIS vs. Credit Risk Champion — Stress Robustness Analysis

This report evaluates model robustness across different stress regimes.

## Performance Across Stress Regimes

| Regime | System | ROC-AUC | PR-AUC | ECE | sample_size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Low Stress | Control (Credit Only) | 0.70761 | 0.31761 | 0.01546 | 15,522 |
| | **Treatment (CR + CRIS)** | **0.70477** | **0.30281** | **0.01902** | |
| Medium Stress | Control (Credit Only) | 0.71095 | 0.24552 | 0.05500 | 13,509 |
| | **Treatment (CR + CRIS)** | **0.70982** | **0.24250** | **0.05451** | |
| High Stress | Control (Credit Only) | 0.70536 | 0.31686 | 0.01392 | 20,969 |
| | **Treatment (CR + CRIS)** | **0.69579** | **0.30614** | **0.01775** | |

## Key Findings
- In the High Stress regime, both models experience performance degradation.
- The Control model maintains superior ROC-AUC and PR-AUC even under High Stress, although the Treatment model shows comparable calibration (ECE).