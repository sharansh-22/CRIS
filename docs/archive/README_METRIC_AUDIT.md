# README Metric Audit

**Date**: June 17, 2026  
**Auditor**: Model Risk Validation & Research Teams  
**Workspace**: CRIS Credit Risk Repository  

This audit cross-references every quantitative metric presented in the primary `README.md` against the underlying phase research reports to ensure consistency and transparency.

---

## Metric Consistency Matrix

| Metric in README | README Value | Verified Value in Reports | Source Report & Section | Status | Notes / Rationale |
|:---|:---:|:---:|:---|:---:|:---|
| **Loans Analyzed** | 1,345,350 | 1,345,350 | `credit_risk_economic_impact_report.md` (Part 1) | **VERIFIED** | Matches the total clean rows after data ingestion. |
| **Defaults Observed** | 268,599 | 268,599 | `credit_risk_economic_impact_report.md` (Part 1) | **VERIFIED** | Matches total defaults, representing a 19.96% base rate. |
| **Baseline ROC-AUC** | 0.7069 | 0.70687 | `credit_risk_phase2c_borrower_only_audit.md` (Section 4) | **VERIFIED** | Refers to the fully trained LightGBM model (`lightgbm.joblib`). |
| **Model Challenge ROC-AUC** | 0.7024 | 0.70235 | `credit_risk_phase1_champion_selection_report.md` (Section 1) | **VERIFIED** | Refers to the standardized model challenge benchmarker. |
| **Borrower-Only ROC-AUC** | 0.6824 | 0.68238 | `credit_risk_phase2c_borrower_only_audit.md` (Section 4) | **VERIFIED** | Retains 96.53% of the baseline model's ROC-AUC. |
| **Risk Segmentation Ratio** | 11.83× | 11.8344× | `credit_risk_phase2_default_concentration_report.md` (Section 4) | **VERIFIED** | D10 Default Rate (35.74%) divided by D1 Default Rate (3.02%). |
| **D1 Default Rate** | 3.02% | 3.02% | `credit_risk_phase2_default_concentration_report.md` (Section 3) | **VERIFIED** | Safest 10% risk decile. |
| **D10 Default Rate** | 35.74% | 35.74% | `credit_risk_phase2_default_concentration_report.md` (Section 3) | **VERIFIED** | Riskiest 10% risk decile. |
| **Default Concentration** | ~40% (39.95%) | 39.95% | `credit_risk_phase2_default_concentration_report.md` (Section 1) | **VERIFIED** | Share of total defaults contained in risk deciles D9 and D10. |
| **Economic Loss Reduction** | 78.4% | 78.39% | `credit_risk_economic_impact_report.md` (Part 5) | **VERIFIED** | Realized losses: $105.78M (Everyone) vs. $22.86M (LGBM) = $82.92M saved. |
| **Return on Capital Shift** | +5.49 pp | +5.49% | `credit_risk_economic_impact_report.md` (Part 5) | **VERIFIED** | Policy D (LGBM, 21.66%) minus Policy C (LR, 16.17%). |

---

## Clarification of Dual ROC-AUC Values

A potential point of confusion for reviewers is the presence of two different ROC-AUC metrics for the champion LightGBM model:
1. **0.7024 (Model Challenge)**: Appears in the Phase 1 benchmarking table. This model was trained using the uniform cross-dataset script `model_challenge.py` which downsamples the data for comparison speed.
2. **0.7069 (Full Baseline)**: Appears in the Phase 2C borrower-only audit and the main dashboard. This represents the final serialized model (`lightgbm.joblib`) trained on the full available dataset using `train.py` with early stopping and hyperparameter tuning.

Both values are correct within their respective contexts. The README clearly differentiates these two use cases.
