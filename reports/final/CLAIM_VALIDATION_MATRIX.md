# CLAIM VALIDATION MATRIX

**Conducted by**: Independent Risk & Model Validation Audit Team  
**Status**: COMPLETE  
**Repository Version**: V1.0-Audit  

---

This matrix evaluates every major research hypothesis and claim made throughout the development of the Credit Risk baseline and the Cascade Risk Intelligence System (CRIS).

## Claim Validation Table

| Core Claim / Hypothesis | Research Phase | Audit Status | Empirical Evidence | Strength of Evidence |
| :--- | :---: | :---: | :--- | :---: |
| **Credit Risk model generalizes out-of-sample** | Phase 1 & 1.5 | **SUPPORTED** | Out-of-time ROC-AUC remains stable at **0.70687** on the 2018 validation dataset. | **Strong** |
| **Borrower-only features explain most predictive power** | Phase 2C | **SUPPORTED** | Removing all lender-assigned variables (interest rate, grade, terms) retains **96.5%** of ROC-AUC (**0.6824** vs. **0.70687**). | **Strong** |
| **Direct integration of CRIS signals improves borrower predictions** | Phase 3 | **REFUTED** | Adding all 18 macro signals directly to LightGBM degrades ROC-AUC to **0.70061** (Delta = **-0.00627**, bootstrap CI entirely below zero). | **Strong** |
| **Reduced signal subsets solve signal overload and improve performance** | Phase 3.1 | **REFUTED** | Performance degrades monotonically as signals are added. Even a single signal (Config B) degrades ROC-AUC by **-0.00022**. | **Moderate** |
| **CRIS improves risk calibration out-of-sample** | Phase 3 & 3.1 | **REFUTED** | Expected Calibration Error (ECE) shifts negligibly and remains within statistical noise. | **Weak** |
| **CRIS governance overlays reduce portfolio credit losses** | Phase 4 | **SUPPORTED** | Realized default losses under Scenario 2 are reduced by **$11.80M** (from $28.58M to $16.78M). | **Strong** |
| **CRIS governance overlays improve portfolio capital efficiency (RoC)** | Phase 4 | **REFUTED** | Return on Capital actually drops by **-1.42%** (from 22.91% to 21.48%) due to yield compression from safer lending. | **Strong** |
| **CRIS environmental intelligence improves governance decisions** | Final Audit | **REFUTED** | System B (PD-Only, no macro signals) achieves a HIGHER Return on Capital (**21.82%** vs. **21.48%**) and LOWER realized losses (**$13.26M** vs. **$16.78M**) than System C (CRIS Governance). | **Strong** |

---

## Audit Summaries & Technical Details

### 1. The Falsification of Direct Macro Feature Injection
The hypothesis that borrower-level classifiers can be improved by adding macroeconomic indicators (like inflation or unemployment) as features is **REFUTED**. This is due to a fundamental data frequency mismatch: borrower features are high-dimension and individual-level, whereas macro variables are static across monthly cohorts. Training gradient boosted trees on macro variables leads to panel-data overfitting.

### 2. The Isolation of Governance Signal Value
The hypothesis that CRIS environmental intelligence improves portfolio governance over standard borrower-centric policies is **REFUTED**. The Governance Attribution study isolates the contribution of CRIS signals (System C) from a simple borrower-only risk-mitigation rule (System B). 
Because System B (PD-Only) achieves a better risk-return outcome (higher RoC, lower credit losses) without accessing macro signals, the entirety of the governance benefits are attributable to **lending less money and tightening risk appetite thresholds (borrower PD limits)**. No economic value is derived from the CRIS environmental signals.
