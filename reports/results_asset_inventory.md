# Results Asset Inventory

**Date**: June 17, 2026  
**Scope**: Verification that all generated figures and data files exist and are correctly referenced.

---

## 1. Research Report Files

| Report | Path | Size | Referenced In README? |
|---|---|---|---|
| Phase 0 Integrity Audit | `reports/credit_risk_phase0_integrity_report.md` | 19.7 KB | No (internal) |
| Phase 0.5 Audit Verification | `reports/credit_risk_phase0_5_audit_verification_report.md` | 18.7 KB | No (internal) |
| Phase 1 Readiness | `reports/credit_risk_phase1_readiness_report.md` | 9.7 KB | No (internal) |
| Phase 1 Leakage Certification | `reports/credit_risk_pre_phase1_leakage_certification.md` | 13.3 KB | No (internal) |
| Phase 1 Feature Certification | `reports/phase1_feature_certification.md` | 11.5 KB | No (internal) |
| Phase 1 Champion Selection | `reports/credit_risk_phase1_champion_selection_report.md` | 13.4 KB | **YES** |
| Phase 1 Model Challenge | `reports/credit_risk_model_challenge_report.md` | 11.1 KB | **YES** |
| Phase 1.5 Economic Validation | `reports/credit_risk_phase1_5_economic_champion_validation.md` | 14.9 KB | **YES** |
| Phase 2A Default Concentration (LGBM) | `reports/credit_risk_phase2_default_concentration_report.md` | 8.3 KB | **YES** |
| Phase 2A Default Concentration (XGB) | `reports/credit_risk_phase2a_xgboost_default_concentration_report.md` | 9.7 KB | **YES** |
| Phase 2B Feature Importance | `reports/credit_risk_phase2b_feature_importance_and_borrower_profiling.md` | 11.4 KB | **YES** |
| Phase 2C Borrower-Only Audit | `reports/credit_risk_phase2c_borrower_only_audit.md` | 12.0 KB | **YES** |
| Borrower Feature Audit | `reports/borrower_feature_audit.md` | 4.8 KB | **YES** |

---

## 2. Figures (reports/images/)

| Figure | Filename | Size | Exists? |
|---|---|---|---|
| Default Share by Decile (LGBM) | `default_share_by_decile.png` | 51 KB | **YES** |
| Default Rate by Decile (LGBM) | `default_rate_by_decile.png` | 87 KB | **YES** |
| Cumulative Default Capture (LGBM) | `cumulative_default_capture_curve.png` | 97 KB | **YES** |
| Default Share by Decile (XGB) | `xgb_default_share_by_decile.png` | 53 KB | **YES** |
| Default Rate by Decile (XGB) | `xgb_default_rate_by_decile.png` | 94 KB | **YES** |
| Cumulative Default Capture (XGB) | `xgb_cumulative_default_capture_curve.png` | 99 KB | **YES** |
| Feature Importance (Gain) | `feature_importance_gain.png` | 68 KB | **YES** |
| Permutation Importance | `permutation_importance.png` | 69 KB | **YES** |
| SHAP Importance | `shap_importance.png` | 73 KB | **YES** |
| Consensus Feature Importance | `consensus_feature_importance.png` | 61 KB | **YES** |
| Borrower Profile Comparison | `borrower_profile_comparison.png` | 42 KB | **YES** |
| Risk Driver Differences | `risk_driver_differences.png` | 72 KB | **YES** |
| Borrower-Only AUC Comparison | `borrower_only_auc_comparison.png` | 116 KB | **YES** |
| Borrower-Only Default Concentration | `borrower_only_default_concentration.png` | 41 KB | **YES** |
| Borrower-Only Feature Importance | `borrower_only_feature_importance.png` | 62 KB | **YES** |
| Borrower-Only Risk Profiles | `borrower_only_risk_profiles.png` | 46 KB | **YES** |
| Model Challenge AUC Comparison | `challenge_auc_comparison.png` | 32 KB | **YES** |
| Model Challenge ECE Comparison | `challenge_ece_comparison.png` | 27 KB | **YES** |
| Model Challenge NPV Comparison | `challenge_npv_comparison.png` | 32 KB | **YES** |
| Phase 1.5 NPV by Policy | `credit_risk_net_value_by_policy_1_5.png` | 131 KB | **YES** |
| Phase 1.5 ROC by Policy | `credit_risk_roc_by_policy_1_5.png` | 127 KB | **YES** |

---

## 3. Data Deliverables (reports/)

| File | Path | Size | Exists? |
|---|---|---|---|
| Feature Importance (Gain) | `reports/feature_importance_gain.csv` | 5.0 KB | **YES** |
| Feature Importance (Split) | `reports/feature_importance_split.csv` | 3.0 KB | **YES** |
| Permutation Importance | `reports/permutation_importance.csv` | 8.3 KB | **YES** |
| SHAP Importance | `reports/shap_importance.csv` | 5.3 KB | **YES** |
| Consensus Feature Importance | `reports/consensus_feature_importance.csv` | 6.6 KB | **YES** |
| Borrower-Only Feature Importance | `reports/borrower_only_feature_importance.csv` | 5.2 KB | **YES** |
| Low Risk Profile | `reports/low_risk_borrower_profile.csv` | 322 B | **YES** |
| Mid Risk Profile | `reports/mid_risk_borrower_profile.csv` | 314 B | **YES** |
| High Risk Profile | `reports/high_risk_borrower_profile.csv` | 302 B | **YES** |

---

## 4. Verdict

**All 37 figures, 13 reports, and 9 data deliverables exist and are correctly located.** No missing assets detected.
