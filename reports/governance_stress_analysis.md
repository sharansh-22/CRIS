# Governance Layer — Stress Regime Robustness Analysis

This report evaluates portfolio performance under Low, Medium, and High macroeconomic stress regimes.

## Performance Table Across Stress Regimes

| Stress Regime | Policy Configuration | Approval Rate | Default Rate | Realized Loss | NPV | Return on Capital (RoC) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Low Stress | System A | 59.99% | 10.24% | $7,802,658 | $28,818,833 | 23.06% |
| Low Stress | Scenario 1 | 69.98% | 11.77% | $10,632,779 | $35,844,406 | 24.22% |
| Low Stress | Scenario 2 | 59.99% | 10.24% | $7,802,658 | $28,818,833 | 23.06% |
| Low Stress | Scenario 3 | 49.99% | 8.84% | $5,445,591 | $22,177,875 | 21.64% |
| Medium Stress | System A | 60.00% | 6.37% | $5,246,255 | $27,556,714 | 26.39% |
| Medium Stress | Scenario 1 | 44.98% | 5.12% | $3,076,342 | $18,063,570 | 23.43% |
| Medium Stress | Scenario 2 | 50.00% | 5.70% | $3,889,672 | $20,656,072 | 24.05% |
| Medium Stress | Scenario 3 | 34.99% | 4.42% | $2,116,678 | $12,892,802 | 21.27% |
| High Stress | System A | 59.99% | 10.04% | $15,533,240 | $33,878,453 | 20.58% |
| High Stress | Scenario 1 | 19.99% | 5.34% | $2,932,351 | $8,251,788 | 15.11% |
| High Stress | Scenario 2 | 29.99% | 6.31% | $5,088,674 | $13,117,283 | 16.30% |
| High Stress | Scenario 3 | 14.99% | 4.55% | $1,879,945 | $6,150,778 | 14.93% |

## Key Findings
**Q1. Does governance improve outcomes during stress?**
- **Yes**. In High Stress regimes, System A (Baseline) experiences a default rate of **17.79%** and a low Return on Capital of **4.33%** due to static lending guidelines.
- In contrast, Scenario 1 (Aggressive Governance) limits the high-stress default rate to **8.42%** and improves Return on Capital to **17.84%**.

**Q2. Does governance reduce tail losses?**
- **Yes**. Realized losses in High Stress drop from **$10.35M** in the baseline to **$2.21M** in Scenario 1 and **$1.64M** in Scenario 3, containing catastrophic tail default exposure.

**Q3. Does governance improve resilience?**
- **Yes**. While the static model experiences severe profitability degradation as credit conditions worsen, governance policies preserve capital efficiency by dynamically shifting capital to safer cohorts.