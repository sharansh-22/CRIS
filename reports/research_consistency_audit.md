# Research Consistency Audit

**Date**: June 17, 2026  
**Scope**: Cross-report metric consistency for Credit Risk Phases 0–2C.

---

## 1. Shared Constants Across All Reports

| Parameter | Value | Consistent? |
|---|---|---|
| Dataset | LendingClub ONLY | **YES** — all reports use LendingClub exclusively |
| Total Records | 1,345,350 | **YES** |
| Train Split | year ≤ 2015, 100,000 samples | **YES** |
| Test Split | year ≥ 2018, 50,000 samples | **YES** |
| Temporal Gap | 2016–2017 (2 years) | **YES** |
| Random Seed | 42 | **YES** |
| Default Definition | Charged Off / Default = 1 | **YES** |
| Total Defaults in Test | 7,865 | **YES** — confirmed in Phase 2A (LGBM), Phase 2A-XGB, Phase 2B, Phase 2C |
| Baseline Default Rate | 15.73% (7,865 / 50,000) | **YES** |
| Loss Given Default | 70% | **YES** |

---

## 2. Champion Model Metrics

| Metric | Phase 1 Report | Phase 2A/2B/2C Reports | Consistent? |
|---|---|---|---|
| Champion Model | LightGBM | LightGBM | **YES** |
| ROC-AUC (Phase 1 protocol) | 0.70235 | — | N/A (different protocol) |
| ROC-AUC (Phase 2C protocol) | — | 0.70687 | N/A (different sampling) |

> [!NOTE]
> Phase 1 and Phase 2C report slightly different ROC-AUC values (0.70235 vs. 0.70687). This is expected because Phase 1 trains on 100,000 samples using `model_challenge.py` with `StandardScaler` + `.fillna(0)`, while Phase 2C loads the `saved_models/lightgbm.joblib` which was trained via `train.py` with 200 estimators and early stopping on a validation set. The two pipelines are architecturally distinct but use the same temporal splits and data. This is **not an inconsistency** — it reflects two different training protocols producing two different (but close) model instances.

---

## 3. Default Concentration Metrics (Phase 2A)

| Metric | Phase 2A (LGBM) | Phase 2A-XGB | Consistent Format? |
|---|---|---|---|
| D1 Default Rate | 3.02% | 2.92% | **YES** |
| D10 Default Rate | 35.74% | 35.50% | **YES** |
| Segmentation Ratio | 11.83x | 12.16x | **YES** |
| D10 Default Share | 22.72% | 22.57% | **YES** |
| D9+D10 Default Share | 39.95% | 39.75% | **YES** |
| Total Defaults | 7,865 | 7,865 | **YES** |

---

## 4. Borrower Profile Metrics (Phase 2B vs. 2C)

| Metric | Phase 2B (Full Model D1) | Phase 2C (Borrower-Only D1) | Direction Consistent? |
|---|---|---|---|
| FICO (D1) | 760.74 | 756.38 | **YES** — both models place high-FICO borrowers in D1 |
| DTI (D1) | 12.39% | 11.66% | **YES** — both place low-DTI borrowers in D1 |
| Income (D1) | $106,065 | $104,369 | **YES** |

---

## 5. Feature Importance Rankings

| Feature | Phase 2B Full Model Rank | Phase 2C Borrower-Only Rank | Consistent Direction? |
|---|---|---|---|
| int_rate | 1 | Removed (Group B) | **Expected** |
| loan_amnt | 5 | 1 | **Expected** — rises when int_rate removed |
| fico_range_low | 8 | 2 | **Expected** — rises when int_rate removed |
| dti | 3 | 4 | **YES** |
| annual_inc | 4 | 3 | **YES** |

---

## 6. Contradictions Found

**None.** All cross-report metrics are internally consistent. The only apparent discrepancy (Phase 1 ROC-AUC 0.70235 vs. Phase 2C ROC-AUC 0.70687) is explained by different training pipelines (model_challenge.py vs. train.py) and is documented in both reports.

---

## 7. Unsupported Claims Found

**None.** Every quantitative claim in the research reports is directly traceable to script output and is reproducible under SEED=42.
