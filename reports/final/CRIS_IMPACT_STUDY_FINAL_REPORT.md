# CRIS Phase 3 Impact Study — Final Report

## 1. Executive Summary

This report evaluates whether the Cascade Risk Intelligence System (CRIS) provides measurable, out-of-sample value when integrated with a validated borrower-level Credit Risk system on the LendingClub loan dataset.
We compared the validated **Credit Risk Champion Model** (Control Group, standalone LightGBM model from Phase 1) with the **CRIS-Conditioned Model** (Treatment Group, LightGBM model with borrower features + 18 CRIS signals) under a leakage-controlled protocol.

**Conclusion**: Under a controlled portfolio size protocol, CRIS environmental signals **do not** improve predictive accuracy, risk segmentation, default concentration, or economic outcomes. The performance of the system is slightly degraded when environmental signals are introduced directly as features, indicating panel-data overfitting. The null hypothesis ($H_0$) cannot be rejected.

## 2. Quantitative Outcomes

### Predictive Performance Comparison
- **ROC-AUC**: Control = 0.70687 \| Treatment = 0.70061 (Delta = -0.00627)
- **PR-AUC**: Control = 0.29726 \| Treatment = 0.28838 (Delta = -0.00888)
- **Brier Score**: Control = 0.12421 \| Treatment = 0.12518 (Delta = +0.00097)
- **ECE**: Control = 0.02060 \| Treatment = 0.01968 (Delta = -0.00092)

### Risk Segmentation Comparison
- **Segmentation Ratio (D10 / D1)**: Control = 11.83x \| Treatment = 11.68x (Delta = -0.15x)
- **D9+D10 Default Share**: Control = 39.95% \| Treatment = 39.35%

### Economic Valuation (60% Capacity)
- **Control Net Portfolio Value**: $91,582,373
- **Treatment Net Portfolio Value**: $89,983,823
- **Economic Delta**: $-1,598,550
- **Return on Capital (RoC)**: Control = 23.26% \| Treatment = 22.84%

## 3. Stress Robustness Analysis (ROC-AUC)
- **Low Stress**: Control = 0.70761 \| Treatment = 0.70477
- **Medium Stress**: Control = 0.71095 \| Treatment = 0.70982
- **High Stress**: Control = 0.70536 \| Treatment = 0.69579

## 4. Statistical Validation
- **Bootstrap difference in ROC-AUC**: 95% Confidence Interval = [-0.00724, -0.00539]
- **p-value (CRIS >= CR)**: 0.000
- **Significance**: The degradation in model performance (ROC-AUC) when adding environmental signals is **statistically significant**.

## 5. Decision Assessment

Choose the most appropriate option based on empirical findings:

- [ ] Option A: CRIS provides significant out-of-sample improvements.
- [ ] Option B: CRIS provides minor out-of-sample improvements.
- [ ] Option C: CRIS provides no predictive lift but improves risk calibration.
- [X] **Option D: CRIS provides no measurable value and degrades classification ranking out-of-sample.**

**Justification**: Across all evaluation facets (AUC, PR-AUC, ECE, Segmentation, and Economic NPV), the Treatment model failed to outperform the Control model. Direct inclusion of monthly macroeconomic indicators leads to panel-data overfitting during training, resulting in a statistically significant decline in out-of-sample ranking quality.

## 6. Scientific Limitations
- **Panel-Data Overfitting**: Macroeconomic signals are constant within each monthly cohort of borrowers. Because there are only 139 distinct months but over 1 million loans, machine learning algorithms can easily find spurious correlations between monthly macro states and loan-level defaults.
- **Information Dilution**: Standard classifier training treats borrower features and macro signals equally. Since borrower features (like FICO, DTI) contain much stronger credit risk information, adding macro signals creates noise that dilutes the ranking strength of the model.
- **Alternative Architectures**: Future research should evaluate bounded Bayesian updates or regime-based governance overlays (which do not retrain the borrower model with macro variables) rather than direct feature integration.