# CRIS
### Cascade Risk Intelligence System

A quantitative research program analyzing **1.3M+ resolved consumer loans** and **18 macroeconomic signals** to evaluate whether environmental intelligence (macro and market structure states) improves borrower default prediction or portfolio-level governance.

---

## Executive Summary

CRIS is a quantitative research program that tested whether environmental intelligence can improve consumer credit decisions. Across multiple controlled experiments, borrower-centric signals consistently dominated macroeconomic indicators. Consumer credit default models typically rely on static, borrower-specific profiles. Under systemic downturns, however, borrower-level risk can escalate rapidly. CRIS was designed to bridge this gap by conditioning borrower-level prediction and portfolio-level risk limits on monthly macroeconomic and market-structure indicators.

The research program evaluated:
1. **Borrower Intrinsic Risk**: Developing a leakage-free default classifier.
2. **Predictive Macro-Conditioning**: Training models on combined borrower and macro variables.
3. **Macro-Governed Portfolios**: Adjusting capacity limits and underwriting thresholds dynamically using monthly environmental stress scores.

---

## Key Results

| Finding / Result | Evidence Quality & Outcome |
| :--- | :--- |
| **✓ Leakage-free Credit Risk platform validated** | **High**: Ingested 1.34M resolved LendingClub loans, certified a clean pipeline by removing 45 post-origination columns. |
| **✓ LightGBM selected as champion** | **High**: Benchmarked classifiers under out-of-time splits (Train $\le 2015$, Test $\ge 2018$). LightGBM achieved superior **0.70687** ROC-AUC. |
| **✓ 11.83x risk segmentation** | **High**: Decile sorting monotonically segments actual default rates from **3.02% (D1)** to **35.74% (D10)**. |
| **✓ Borrower-only model retains 96.5% predictive power** | **High**: Retraining without lender pricing details (grade, rate, term) retains **96.5%** of ROC-AUC (**0.68240** vs. **0.70687**). |
| **✗ Direct CRIS feature injection degraded performance** | **High (Falsified)**: Directly adding macro features to LightGBM degraded out-of-time ROC-AUC to **0.70061** (p = 0.000). |
| **✗ Signal reduction did not show out-of-sample utility** | **High (Falsified)**: Out-of-sample performance degraded monotonically as macro variables were added; no optimal subset was found. |
| **✗ Governance attribution showed no macro value** | **High (Falsified)**: System B (PD-Only, no macro signals) achieved a HIGHER Return on Capital (**21.82%** vs. **21.48%**) than System C (CRIS Macro Governance). |

---

## Research Journey

- **Phase 0 & 0.5 — Data Ingestion & Leakage Audit**: Certified clean ingestion of 1.3M resolved loans, dropping 45 post-origination columns to prevent lookahead target leakage.
- **Phase 1 & 1.5 — Champion Model Selection & Economics**: Benchmarked 5 classifiers under out-of-time splits (Train $\le 2015$, Test $\ge 2018$ with a 2-year temporal gap). Certified LightGBM as champion with **0.70687** ROC-AUC.
- **Phase 2 — Borrower Profiling & Signal Decomposition**: Validated monotonic decile-level risk segmentation, profiled high-risk cohorts, and proved that borrower-only traits contain 96.5% of full model predictive power.
- **Phase 3 & 3.1 — Direct Signal Integration & Signal Reduction Studies**: Evaluated models trained on combined borrower-intrinsic and macro variables. Direct integration of macro features degraded out-of-time performance due to panel-data overfitting.
- **Phase 4 & Final Audit — Governance Attribution**: Simulating dynamic regime-switching policy caps under downturn LGD. Isolated the value of CRIS signals against standard borrower-centric tightening.

---

## What CRIS Proved

1. **Borrower-Centric Sufficiency**: Borrower-intrinsic variables (FICO, DTI, Income, Utilization) contain sufficient information to build a high-performing risk engine.
2. **Borrower PD Governance works**: Adjusting maximum credit limits dynamically using borrower-only PD distributions (System B) successfully contains realized portfolio default rates under downturn stress.

---

## Hypotheses Not Supported by Empirical Evidence

1. **Direct Macro Feature Injection**: Adding low-frequency macro features to high-dimension borrower profiles does not improve prediction. This causes panel-data overfitting and dilutes the ranking power of borrower-intrinsic signals.
2. **Macro-Driven Governance Overlays**: Empirical testing did not support the hypothesis that macro-environmental signals improve portfolio governance over borrower-centric limits. System B (PD-Only) achieves a better risk-return profile than System C (CRIS Macro Governance).

---

## Final Conclusion

Across the tested LendingClub consumer credit environment, borrower-centric information consistently dominated macroeconomic indicators. Macro signals are low-frequency (monthly) and static across cohorts, whereas borrower profiles are high-dimensional. Machine learning classifiers overfit to low-frequency indicators, leading to out-of-sample performance degradation. Furthermore, dynamic governance tightening causes severe yield compression on high-yield consumer loans.

This research program is highly valuable because it establishes a rigorous falsification framework: it proves that direct integration of macro signals is counterproductive and isolates portfolio governance benefits entirely to borrower-centric credit limits.

---

## Validated System

### Credit Risk Platform
The Credit Risk Platform emerged as the primary validated production candidate from the CRIS research program. It implements a leakage-certified, out-of-time trained LightGBM champion model that achieves strong risk segmentation and is independent of lender pricing variables.

For detailed architecture, model parameters, and setup guides, see the platform documentation:
👉 **[systems/credit_risk/README.md](systems/credit_risk/README.md)**

---

## Repository Structure

```text
CRIS/
├── configs/                     # Global paths, seeds, and target leakage lists
├── reports/
│   ├── images/                  # Core visualizations (CAP, decile default rates)
│   └── final/                   # Audited research reports and verification ledgers
├── systems/
│   └── credit_risk/             # Validated Credit Risk Platform (features, models)
├── validation/                  # Walk-forward, behavioral, and calibration tests
├── requirements.txt             # Pip package dependencies
├── environment.yml              # Conda environment configuration
└── README.md                    # Main Repository README
```

---

## Reproducibility

To replicate the final validation audit and regenerate all reports and figures:

```bash
# 1. Setup Conda environment
conda env create -f environment.yml && conda activate CRIS

# 2. Ingest and engineer LendingClub data
python systems/credit_risk/features/ingestion.py
python systems/credit_risk/features/engineering.py

# 3. Train models and execute the validation audit
python systems/credit_risk/models/train.py
python systems/credit_risk/evaluation/final_governance_validation_audit.py
```

---

## Reports

The complete technical details of the research are archived in the following validation reports:
- [FINAL_REPOSITORY_AUDIT.md](reports/final/FINAL_REPOSITORY_AUDIT.md) — System audit, folder checks, and reproducibility verification.
- [RESEARCH_CONSISTENCY_AUDIT.md](reports/final/RESEARCH_CONSISTENCY_AUDIT.md) — Cross-phase consistency check and metric reconciliations.
- [CLAIM_VALIDATION_MATRIX.md](reports/final/CLAIM_VALIDATION_MATRIX.md) — Validation checklist of all core CRIS hypotheses.
- [FINAL_GOVERNANCE_ATTRIBUTION_REPORT.md](reports/final/FINAL_GOVERNANCE_ATTRIBUTION_REPORT.md) — Comparison of Systems A, B, and C under stress regimes.
- [CRIS_FINAL_VERDICT_REPORT.md](reports/final/CRIS_FINAL_VERDICT_REPORT.md) — Final model risk committee verdict report.
