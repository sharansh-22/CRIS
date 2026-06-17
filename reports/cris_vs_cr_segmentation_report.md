# CRIS vs. Credit Risk Champion — Risk Segmentation Report

This report compares the default rates across risk deciles for both models.

## Decile Default Rates

| Decile | Control (Credit Only) Default Rate | Treatment (CR + CRIS) Default Rate | Delta |
| :--- | :---: | :---: | :---: |
| D1 | 3.02% | 3.08% | +0.06% |
| D2 | 5.96% | 5.78% | -0.18% |
| D3 | 7.74% | 7.76% | +0.02% |
| D4 | 9.70% | 11.76% | +2.06% |
| D5 | 12.84% | 13.60% | +0.76% |
| D6 | 14.82% | 14.08% | -0.74% |
| D7 | 18.58% | 17.20% | -1.38% |
| D8 | 21.80% | 22.14% | +0.34% |
| D9 | 27.10% | 25.92% | -1.18% |
| D10 | 35.74% | 35.98% | +0.24% |

## Summary Metrics

| Metric | Control (Credit Only) | Treatment (CR + CRIS) | Delta |
| :--- | :---: | :---: | :---: |
| **D1 Default Rate (Safest)** | 3.02% | 3.08% | +0.06% |
| **D10 Default Rate (Riskiest)** | 35.74% | 35.98% | +0.24% |
| **Segmentation Ratio (D10 / D1)** | 11.83x | 11.68x | -0.15x |
| **D9+D10 Default Share (Top 20%)** | 39.95% | 39.35% | -0.60% |

## Key Findings
- The Control model achieved a higher segmentation ratio and concentrated more defaults in the riskiest deciles (D9 and D10) compared to the Treatment model.
- This indicates that the addition of CRIS signals slightly **diluted** the ranking quality of the credit risk model on the out-of-sample population.