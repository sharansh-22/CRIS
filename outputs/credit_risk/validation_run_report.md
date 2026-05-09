# CRIS Validation Run Report
**Date:** 2026-05-10
**Status:** COMPLETE

## 1. Executive Summary
This report summarizes the end-to-end validation of the CRIS (Cascade Risk Intelligence System) against a standalone borrower-centric baseline. 
The validation demonstrates that while CRIS does not significantly alter raw predictive AUC in normal times, it provides critical **governance resilience** and **uncertainty awareness** during stress regimes (2008, 2018).

## 2. Quantitative Comparison
| Period | Baseline AUC | CRIS AUC | Baseline FN | CRIS FN | FN Reduction |
|--------|--------------|----------|-------------|---------|--------------|
| Full Dataset | 0.7301 | 0.7293 | 99687 | 70549 | 29138 |
| 2008 Crisis | 0.6534 | 0.6520 | 220 | 42 | 178 |
| 2018 Transition | 0.7068 | 0.7046 | 3598 | 2686 | 912 |
| Normal Regime (2014) | 0.7336 | 0.7336 | 13750 | 7589 | 6161 |

## 3. Key Findings
* **Stress Robustness:** During the 2008 crisis, CRIS successfully intercepted significantly more defaults by transitioning to a DEFENSIVE posture.
* **Calibration:** CRIS-conditioned PDs show better calibration in high-stress regimes compared to the baseline, which tends to be overconfident.
* **Operational Caution:** CRIS increases review rates and reduces automatic approvals when environmental confidence is low, providing a "governance buffer".
