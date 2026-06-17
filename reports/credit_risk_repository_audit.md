# Credit Risk Repository Audit

**Date**: June 17, 2026  
**Scope**: Full structural audit of the Credit Risk subsystem within the CRIS repository.

---

## 1. Current Structure

```text
systems/credit_risk/
├── __init__.py
├── schema.py                          # Minimal placeholder (61 bytes)
├── calibration/
│   ├── confidence_estimation.py       # Scaffold placeholder
│   ├── probability_calibration.py     # Scaffold placeholder
│   └── threshold_tuning.py            # Scaffold placeholder
├── evaluation/
│   ├── __init__.py
│   ├── borrower_only_audit_phase2c.py       # Phase 2C script ✓
│   ├── calibration_metrics.py               # Scaffold placeholder
│   ├── default_concentration_phase2.py      # Phase 2A LightGBM ✓
│   ├── default_concentration_xgb.py         # Phase 2A XGBoost ✓
│   ├── economic_impact.py                   # CRIS economic simulation ✓
│   ├── economic_validation.py               # CRIS downstream validation ✓
│   ├── economic_validation_phase1_5.py      # Phase 1.5 script ✓
│   ├── feature_importance_profiling_phase2b.py  # Phase 2B script ✓
│   ├── metrics.py                           # Baseline evaluation ✓
│   ├── model_challenge.py                   # Phase 1 model challenge ✓
│   ├── precision_recall.py                  # Scaffold placeholder
│   ├── roc_analysis.py                      # Scaffold placeholder
│   └── stress_period_analysis.py            # Scaffold placeholder
├── explainability/
│   ├── borrower_explanations.py     # Scaffold placeholder
│   ├── feature_importance.py        # Scaffold placeholder
│   └── shap_analysis.py            # Scaffold placeholder
├── features/
│   ├── __init__.py
│   ├── behavioral_features.py       # Scaffold placeholder
│   ├── borrower_features.py         # Scaffold placeholder
│   ├── engineering.py               # Feature engineering pipeline ✓
│   ├── ingestion.py                 # Data ingestion pipeline ✓
│   ├── repayment_history.py         # Scaffold placeholder
│   ├── temporal_features.py         # Scaffold placeholder
│   └── utilization_metrics.py       # Scaffold placeholder
├── models/
│   ├── __init__.py
│   ├── ensemble.py                  # Scaffold placeholder
│   ├── lightgbm_model.py           # Scaffold placeholder
│   ├── logistic_regression.py       # Scaffold placeholder
│   ├── train.py                     # Model training pipeline ✓
│   ├── xgboost_model.py            # Scaffold placeholder
│   └── saved_models/
│       ├── lightgbm.joblib          # Saved model (734 KB) ✓
│       ├── logistic_regression.joblib
│       ├── scaler.joblib
│       └── xgboost.joblib           # Saved model (967 KB) ✓
└── overlays/                        # CRIS overlay system (not Credit Risk research)
```

---

## 2. Problems Found

| # | Issue | Severity | Location |
|---|---|---|---|
| 1 | **13 scaffold placeholder files** contain only `TODO: implementation scaffold` docstrings. They add no functionality and clutter the tree. | Medium | `evaluation/`, `features/`, `models/`, `explainability/` |
| 2 | **`scratch/audit_datasets.py`** is a temporary debugging script that should not be in the published repository. | Low | `scratch/` |
| 3 | **`columns.txt`** is a raw column dump with no documentation context. | Low | Root |
| 4 | **`CRIS_RELEASE_READINESS.md`** and **`BRANCHING.md`** are internal process files, not research deliverables. | Low | Root |
| 5 | **`borrower_only_lightgbm.joblib`** (360 KB) is stored in `reports/`, which is not the correct location for model artifacts. | Medium | `reports/` |
| 6 | **`__pycache__/`** directories exist in the working tree (gitignored but present locally). | None | Multiple |
| 7 | **No `requirements.txt`** exists. Only `environment.yml` is provided. Pip-only users cannot install. | Medium | Root |
| 8 | **Saved models are not gitignored** — `.joblib` files in `systems/credit_risk/models/saved_models/` are not in `.gitignore`. They are small enough to track but this should be a deliberate decision. | Low | `.gitignore` |

---

## 3. Recommended Cleanup Actions

1. **Keep scaffold placeholders as-is**. They represent planned architecture and signal intent. Adding `# Planned: <purpose>` comments would improve clarity, but removing them risks breaking `__init__.py` imports.
2. **Add `scratch/` to `.gitignore`** — already done (line 36).
3. **Move `borrower_only_lightgbm.joblib`** from `reports/` to `systems/credit_risk/models/saved_models/`.
4. **Create `requirements.txt`** for pip-only users.
5. **Remove `columns.txt`** or move it to `docs/` with context.

---

## 4. Missing Documentation

| Item | Status |
|---|---|
| README for Credit Risk subsystem | **Missing** — will be created as part of this task |
| `requirements.txt` | **Missing** |
| Quick Start for Credit Risk research | **Missing** — covered only for CRIS SAE in current README |
| Expected runtime documentation | **Missing** |
| Data download instructions | **Missing** — raw data is gitignored |

---

## 5. Missing Reproducibility Assets

| Asset | Status |
|---|---|
| `requirements.txt` | **Missing** |
| Saved models (`.joblib`) | **Present** in `saved_models/` |
| Engineered data (`.parquet`) | **Present locally** but gitignored (correct behavior for large files) |
| Raw data download link | **Not documented** |
