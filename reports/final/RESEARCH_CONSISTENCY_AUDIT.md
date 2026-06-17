# RESEARCH CONSISTENCY AUDIT

**Conducted by**: Independent Risk & Model Validation Audit Team  
**Status**: COMPLETE  
**Repository Version**: V1.0-Audit  

---

## 1. Scope & Objective

This audit examines the consistency of key performance indicators (KPIs), model rankings, predictive power metrics, and economic outputs across all ten research phases (Phase 0 through Phase 4) and the Final Governance Attribution Study. Our objective is to ensure that all documentation shares a single source of truth and to highlight any historical metric drift or contradictory interpretations.

---

## 2. Model Rankings & AUC Consistency

### Champion Model Selection (Phase 1 vs. Downstream Phases)
- **Phase 1 Benchmark ROC-AUC**: LightGBM was certified as the champion model with a score of **0.70687** (rounded to 0.7069 in README). 
- **Downstream Verification**:
  - Phase 1.5, Phase 2A, Phase 2B, and Phase 2C use the exact same champion LightGBM model configuration.
  - Phase 3 baseline ROC-AUC matches at **0.70687**.
  - Phase 3.1 baseline ROC-AUC matches at **0.70687**.
  - **Rankings**: The model ranking remains strictly consistent across all benchmarking experiments: `LightGBM > XGBoost > Random Forest > Decision Tree > Logistic Regression`.

### Borrower-Only Audit (Phase 2C vs. README)
- **ROC-AUC**: Full model = **0.7069** \| Borrower-Only model = **0.6824** (Delta = **-0.0245**).
- **PR-AUC**: Full model = **0.2973** \| Borrower-Only model = **0.2718** (Delta = **-0.0255**).
- **Risk Segmentation Ratio**: Full model = **11.83×** \| Borrower-Only model = **8.86×** (Delta = **-2.97×**).
- **Consistency Status**: **100% Consistent**. The figures reported in Phase 2C align exactly with the TL;DR dashboard in the current README.

---

## 3. Economic Validation & Metrics Consistency

### Base Economic Framework (Phase 1.5 vs. Phase 3 & 4)
- **LGD Model**: Phase 1.5 used a flat **70% LGD** (Loss Given Default) for cash flow simulations, yielding a baseline NPV of **$91.58M** at 60% capacity.
- **Phase 4 Stress LGD**: Phase 4 introduced a pro-cyclical downturn LGD:
  - Low Stress: **55%**
  - Medium Stress: **70%**
  - High Stress: **85%**
- **Impact of LGD Shift on System A**: Under the downturn LGD model, System A's simulated out-of-time NPV shifts from **$91.58M** (flat 70% LGD) to **$90.25M** (regime-specific LGD), and the baseline Return on Capital is **22.91%**.
- **Consistency Status**: **Reconciled**. The shift in absolute baseline NPV from $91.58M to $90.25M is mathematically justified by the transition from flat to regime-specific LGD.

---

## 4. Contradictions & Interpretive Errors Identified

During our audit of earlier reports, we identified the following contradictions and errors:

### 1. The Governance Return on Capital (RoC) Error
- **The Contradiction**: In the Phase 4 validation report (`reports/governance_statistical_validation.md`), the summary table reported an observed RoC difference of **-1.42%** (Scenario 2 vs. System A), indicating a *reduction* in capital efficiency. However, the text below it stated: *"The increase in Return on Capital (+0.21%) is statistically significant, validating that governance layer CRIS creates a more capital-efficient portfolio."*
- **Audit Findings**: The table was correct; the text was false. Scenario 2 (Moderate Governance) reduces Return on Capital. This error occurred because the writer expected governance to improve capital efficiency, whereas the empirical reality of consumer loan pricing shows that tighter risk limits compress yield faster than they reduce default losses.

### 2. Double-counting of Default Savings
- **The Contradiction**: Phase 4 reports highlighted the savings of **$11.80M** in default losses under Scenario 2 as a pure benefit of CRIS governance.
- **Audit Findings**: This is an incomplete representation of portfolio economics. Scenario 2 avoids $11.80M in defaults but foregoes **$39.46M** in interest income by rejecting borrowers, resulting in a net NPV drag of **-$27.66M**. Marketing this as a "capital preservation success" without acknowledging the opportunity cost of foregone yield is an interpretive error.

---

## 5. Summary Audit Table

| Metric Group | Consistency Status | Identified Contradictions | Corrected Value |
| :--- | :--- | :--- | :--- |
| **ROC-AUC (Phase 1)** | Consistent | None | 0.70687 (LGBM Champion) |
| **Borrower-Only AUC** | Consistent | None | 0.6824 |
| **NPV (Baseline 60%)** | Reconciled | Flat LGD ($91.58M) vs. Stress LGD ($90.25M) | Both are mathematically correct in context |
| **RoC (Governance Phase 4)** | **FAILED** | Table showed -1.42%, text claimed +0.21% | **-1.42%** (RoC is reduced by governance) |
| **Economic Drag** | **FAILED** | Loss savings highlighted, opportunity cost omitted | Net drag is **-$27.66M** |
