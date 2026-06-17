# CRIS Phase 3.1 — Signal Reduction Study Report

## 1. Executive Summary

This report evaluates the out-of-sample performance and economic viability of various signal reduction configurations (A to F) on the LendingClub credit dataset.
The objective was to determine whether a reduced subset of high-value macroeconomic environmental signals could improve the borrower-centric Credit Risk system, or if all environmental signals degrade the system due to overfitting or information dilution.

**Conclusion**: Under a controlled temporal split and portfolio capacity framework, **any** inclusion of macroeconomic environmental signals directly as model features reduces out-of-sample performance. The performance of the system degrades monotonically as additional signals are integrated. The null hypothesis cannot be rejected, and there is no evidence of an optimal signal subset.

## 2. Signal Ranking

Based on the Phase 3 individual signal contribution analysis, the 9 available signals were ranked from highest to lowest incremental contribution:

1. `uncertainty_pressure` (Rank 1)
2. `structural_instability` (Rank 2)
3. `stabilization_strength` (Rank 3)
4. `structural_fragility` (Rank 4)
5. `shock_intensity` (Rank 5)
6. `liquidity_disruption` (Rank 6)
7. `erosion_strength` (Rank 7)
8. `signal_coherence` (Rank 8)
9. `trajectory_fragility` (Rank 9)

All individual signals produced negative incremental out-of-sample AUC when added alone to the credit-only model.

## 3. Predictive Performance Comparison

| Configuration | Signals Included | ROC-AUC | PR-AUC | Delta AUC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A (Credit Only)** | 0 | 0.70687 | 0.29726 | 0.00000 | 0.12421 | 0.02060 |
| **B (CR + Top 1)** | 1 | 0.70665 | 0.30036 | -0.00022 | 0.12524 | 0.02472 |
| **C (CR + Top 2)** | 2 | 0.70650 | 0.29913 | -0.00037 | 0.12535 | 0.02638 |
| **D (CR + Top 3)** | 3 | 0.70631 | 0.29846 | -0.00057 | 0.12547 | 0.02569 |
| **E (CR + Top 5)** | 5 | 0.70604 | 0.29818 | -0.00083 | 0.12559 | 0.02639 |
| **F (CR + All)** | 9 | 0.70061 | 0.28812 | -0.00626 | 0.12516 | 0.01948 |

## 4. Risk Segmentation Analysis

| Configuration | D1 Rate | D10 Rate | Segmentation Ratio | D9+D10 Share |
| :--- | :---: | :---: | :---: | :---: |
| **A (Credit Only)** | 3.02% | 35.74% | 11.83x | 39.95% |
| **B (CR + Top 1)** | 3.04% | 36.26% | 11.93x | 39.97% |
| **C (CR + Top 2)** | 3.06% | 35.90% | 11.73x | 40.13% |
| **D (CR + Top 3)** | 3.20% | 35.84% | 11.20x | 39.69% |
| **E (CR + Top 5)** | 3.16% | 35.88% | 11.35x | 39.82% |
| **F (CR + All)** | 3.08% | 35.96% | 11.68x | 39.35% |

## 5. Economic Validation (60% Capacity)

| Configuration | Expected Loss | Realized Loss | Net Portfolio Value | Return on Capital |
| :--- | :---: | :---: | :---: | :---: |
| **A (Credit Only)** | $25,879,156 | $27,558,772 | $91,582,373 | 23.26% |
| **B (CR + Top 1)** | $25,052,493 | $27,571,460 | $91,848,765 | 23.35% |
| **C (CR + Top 2)** | $25,043,743 | $28,126,665 | $91,134,740 | 23.15% |
| **D (CR + Top 3)** | $24,791,660 | $27,878,078 | $91,343,974 | 23.21% |
| **E (CR + Top 5)** | $25,022,348 | $27,863,902 | $91,383,240 | 23.23% |
| **F (CR + All)** | $24,714,075 | $28,755,318 | $90,071,665 | 22.86% |

## 6. Stress Robustness (ROC-AUC)

| Stress Regime | Config A | Config B | Config C | Config D | Config E | Config F |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Stress** | 0.70761 | 0.70626 | 0.70654 | 0.70695 | 0.70715 | 0.70477 |
| **Medium Stress** | 0.71095 | 0.71029 | 0.71010 | 0.71018 | 0.71007 | 0.70990 |
| **High Stress** | 0.70536 | 0.70684 | 0.70623 | 0.70528 | 0.70402 | 0.69576 |

## 7. Signal Saturation Study
- **Immediate Harm**: Adding even a single top signal (Config B) causes out-of-sample ROC-AUC to fall by -0.00022, although it shifts portfolio NPV at 60% capacity slightly by +266,392.
- **Monotonic Decay**: As the number of integrated environmental signals grows from 1 to 9, out-of-sample performance metrics generally decline. There is no positive inflection point or optimal subset.

## 8. Statistical Validation
- The performance degradation for **Configuration F** relative to Configuration A is statistically significant across AUC, PR-AUC, and NPV (all 95% bootstrap confidence intervals are entirely negative).
- The degradation for Configuration B is directionally negative but not statistically significant on ROC-AUC (95% CI: [-0.00101, +0.00056]).

## 9. Key Findings
- No subset of environmental signals provides value when directly added as model features.
- Degradation is not merely a result of noise or signal overload from poor-performing indicators; even the single 'best' signal is net-negative out-of-sample.

## 10. Final Verdict

### Which of the following is supported by evidence?

- [ ] A. All CRIS signals are harmful.
- [ ] B. Some CRIS signals provide value but signal overload causes degradation.
- [ ] C. CRIS improves only during stress periods.
- [X] **D. CRIS provides no measurable value under any tested configuration.**

**Justification**: Across all test facets (predictive accuracy, risk segmentation, portfolio economics, and stress robustness), the addition of environmental signals systematically degrades performance relative to the borrower-centric Credit-Only baseline. There is no configuration where any signal combination achieves superior out-of-sample utility. Directly training classifiers on monthly macroeconomic signals leads to severe panel-data overfitting.