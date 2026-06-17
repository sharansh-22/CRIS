# Credit Risk Pipeline Wiring Audit

**Date**: June 17, 2026  
**Scope**: End-to-end pipeline trace from raw data to final reports.

---

## Pipeline Trace

```
Raw CSV (data/credit_risk/accepted_2007_to_2018Q4.csv.gz)
  │  Loaded by: systems/credit_risk/features/ingestion.py → load_data()
  │  Config:    configs/credit_config.py → RAW_DATA_PATH
  ▼
Data Cleaning (ingestion.py → clean_data())
  │  Drops:     LEAKAGE_COLS (45 post-origination columns)
  │  Target:    loan_status → {Charged Off, Default} = 1, {Fully Paid} = 0
  │  Imputation: Median (numerical), "Unknown" (categorical)
  │  Output:    outputs/credit_risk/cleaned_data.parquet
  ▼
Feature Engineering (systems/credit_risk/features/engineering.py)
  │  Transforms: emp_length → emp_length_num, earliest_cr_line → cr_hist_years
  │  Encoding:   One-hot encoding for categorical variables
  │  Output:     outputs/credit_risk/engineered_data.parquet
  ▼
Temporal Split & Model Training (systems/credit_risk/models/train.py)
  │  Train:     year <= 2015
  │  Val:       2016–2017
  │  Test:      year >= 2018
  │  Models:    LR, XGBoost, LightGBM (with early stopping on validation)
  │  Scaler:    StandardScaler fit on train only
  │  Output:    systems/credit_risk/models/saved_models/*.joblib
  ▼
Phase 1 — Champion Model Selection (evaluation/model_challenge.py)
  │  Trains:    LR, DT, RF, XGB, LGBM on 100k train, evaluates on 50k test
  │  Metrics:   ROC-AUC, PR-AUC, F1, Brier, ECE, Default Capture, Segmentation
  │  Economic:  PD ≤ 15% approval policy, LGD = 70%
  │  Output:    reports/credit_risk_model_challenge_report.md
  ▼
Phase 1.5 — Economic Champion Validation (evaluation/economic_validation_phase1_5.py)
  │  Equal-size portfolios at 10%–60% approval buckets
  │  Output:    reports/credit_risk_phase1_5_economic_champion_validation.md
  ▼
Phase 2A — Default Concentration (evaluation/default_concentration_phase2.py)
  │  Decile analysis on 50k test set using LightGBM predictions
  │  Output:    reports/credit_risk_phase2_default_concentration_report.md
  ▼
Phase 2A-XGB — XGBoost Concentration (evaluation/default_concentration_xgb.py)
  │  Replication using XGBoost
  │  Output:    reports/credit_risk_phase2a_xgboost_default_concentration_report.md
  ▼
Phase 2B — Feature Importance & Profiling (evaluation/feature_importance_profiling_phase2b.py)
  │  Native Gain, Permutation, SHAP; borrower profiles by decile
  │  Output:    reports/credit_risk_phase2b_feature_importance_and_borrower_profiling.md
  ▼
Phase 2C — Borrower-Only Audit (evaluation/borrower_only_audit_phase2c.py)
  │  Removes lender features, retrains LightGBM, compares to full model
  │  Output:    reports/credit_risk_phase2c_borrower_only_audit.md
```

---

## Dependency Verification

| Check | Status | Evidence |
|---|---|---|
| `configs/credit_config.py` importable | **PASS** | All evaluation scripts import `SEED`, `OUTPUT_DIR` successfully |
| `engineered_data.parquet` exists | **PASS** | Used by all Phase 1–2C scripts |
| `saved_models/lightgbm.joblib` exists | **PASS** | 734 KB, loaded by Phase 2A/2B/2C |
| `saved_models/xgboost.joblib` exists | **PASS** | 967 KB, loaded by Phase 2A-XGB |
| `saved_models/scaler.joblib` exists | **PASS** | 7.8 KB, loaded for feature name mapping |
| No broken imports | **PASS** | All scripts run to completion |
| No invalid paths | **PASS** | All file paths resolve correctly |
| No stale config references | **PASS** | `RUN_EXTERNAL_VALIDATION = False` correctly isolates GMC/AB |
| No unused configs | **PASS** | `governance_config.py`, `macro_config.py`, `portfolio_config.py` serve the broader CRIS system |
