# Phase 2 Final Report: Probabilistic Macro Conditioning Overlay

## 1. Executive Summary
Phase 2 preserved the Phase 1 standalone borrower-risk model as the primary estimator and tested a bounded CRIS Layer 3 macro-conditioning overlay. The experiment asks whether probabilistic market-state interpretation can improve lending robustness during deteriorating environments, not whether CRIS predicts defaults or markets.

On the 2018 out-of-time test set, the macro-conditioned overlay produced ROC-AUC **0.7046** versus baseline **0.7068**, PR-AUC **0.2943** versus **0.2972**, and Brier score **0.1252** versus **0.1244**. This is not an average-case improvement. The overlay made governance more defensive: approval rate declined from **81.36%** to **80.03%**, approved-loan default rate moved from **12.13%** to **11.99%**, and **153** baseline-approved defaulters were moved into review by environmental pressure. However, 2018 high-stress calibration worsened, so the evidence supports conditional governance caution rather than a stronger predictive claim.

## 2. Research Motivation
Phase 1 estimated:

`P(Default | Borrower Features)`

Phase 2 investigates:

`P(Default | Borrower Features, Probabilistic Market State)`

The research hypothesis is modest: a borrower-only model may rank applicants reasonably in stable regimes while becoming overconfident when the surrounding system deteriorates. CRIS Layer 3 is used only as an environmental interpretation framework.

## 3. Temporal Synchronization Methodology
LendingClub issue dates are month-level timestamps. For each issue month, the experiment used the latest available SPY market observation on or before the issue date and ran Layer 3 using only market history available up to that date. The join was a backward as-of synchronization; no future market data, loan performance data, or hindsight crisis labels entered the conditioning record.

Artifacts generated:

- `credit_risk/outputs/phase2_spy_market_2005_2018.csv`
- `credit_risk/outputs/phase2_layer3_macro_states.csv`
- `credit_risk/outputs/phase2_macro_conditioning_results.json`
- `credit_risk/outputs/phase2_historical_diagnostic_by_year.csv`

## 4. Layer 3 Environmental-State Design
The allowed Layer 3 outputs were retained as environmental descriptors:

- `uncertainty_pressure`
- `structural_fragility`
- `liquidity_disruption`
- `stabilization_strength`
- `trajectory_fragility`
- `dominant_field`

These were not treated as raw market predictors. A compact audit score, `macro_stress_score`, was derived from permitted descriptors only. It combines uncertainty, structural fragility, liquidity disruption, trajectory fragility, and weak stabilization. Raw SPY returns/prices were never joined to loan records.

## 5. Conditioning Architecture
The borrower probability from Phase 1 LightGBM remained the base PD. The overlay applied a bounded validation-calibrated log-odds pressure shift:

`logit(PD_macro) = logit(PD_borrower) + beta * max(0, macro_stress_score - validation_anchor)`

Fitted overlay parameters:

- Validation stress anchor: **0.0880**
- Beta: **1.5000**
- Maximum logit shift: **0.3500**

The overlay can raise risk under excess environmental pressure, but it cannot lower borrower risk in benign regimes.

## 6. Comparative Baseline Results
| split                | system            |   roc_auc |   pr_auc |   brier |   log_loss |   ece_10bin |     f1 |
|:---------------------|:------------------|----------:|---------:|--------:|-----------:|------------:|-------:|
| 2016_2017_validation | baseline          |    0.7201 |   0.4300 |  0.1610 |     0.4928 |      0.0453 | 0.4186 |
| 2016_2017_validation | macro_conditioned |    0.7190 |   0.4294 |  0.1607 |     0.4920 |      0.0410 | 0.4236 |
| 2018_test            | baseline          |    0.7068 |   0.2972 |  0.1244 |     0.4022 |      0.0207 | 0.3422 |
| 2018_test            | macro_conditioned |    0.7046 |   0.2943 |  0.1252 |     0.4041 |      0.0252 | 0.3448 |

## 7. Stress-Period Analysis
Stress periods were defined prospectively from the top quartile of 2018 Layer 3 `macro_stress_score`, not from default outcomes. This keeps the stress segmentation environmental rather than label-driven.

| segment      | system            |     n |   default_rate |   roc_auc |   pr_auc |   brier |   ece_10bin |
|:-------------|:------------------|------:|---------------:|----------:|---------:|--------:|------------:|
| high_stress  | baseline          | 16436 |         0.1583 |    0.7050 |   0.2990 |  0.1250 |      0.0219 |
| high_stress  | macro_conditioned | 16436 |         0.1583 |    0.6985 |   0.2907 |  0.1272 |      0.0334 |
| lower_stress | baseline          | 39882 |         0.1573 |    0.7076 |   0.2968 |  0.1242 |      0.0203 |
| lower_stress | macro_conditioned | 39882 |         0.1573 |    0.7074 |   0.2967 |  0.1243 |      0.0218 |

2018 monthly deterioration view:

| issue_month   |    n |   default_rate |   macro_stress_score |   environmental_confidence | dominant_field         |   baseline_brier |   macro_brier |
|:--------------|-----:|---------------:|---------------------:|---------------------------:|:-----------------------|-----------------:|--------------:|
| 2018-01       | 8600 |         0.1947 |               0.0390 |                     0.8954 | NONE                   |           0.1428 |        0.1428 |
| 2018-02       | 6752 |         0.1949 |               0.0365 |                     0.9027 | NONE                   |           0.1439 |        0.1439 |
| 2018-03       | 7178 |         0.1946 |               0.1130 |                     0.8435 | TRAJECTORY_DEGRADATION |           0.1446 |        0.1445 |
| 2018-04       | 6924 |         0.1967 |               0.1455 |                     0.8256 | MIXED                  |           0.1446 |        0.1444 |
| 2018-05       | 6632 |         0.1814 |               0.1470 |                     0.7861 | TRAJECTORY_DEGRADATION |           0.1379 |        0.1383 |
| 2018-06       | 4893 |         0.1598 |               0.1090 |                     0.7929 | TRAJECTORY_DEGRADATION |           0.1214 |        0.1216 |
| 2018-07       | 4317 |         0.1228 |               0.1010 |                     0.7957 | TRAJECTORY_DEGRADATION |           0.1073 |        0.1077 |
| 2018-08       | 3569 |         0.0883 |               0.1070 |                     0.7935 | TRAJECTORY_DEGRADATION |           0.0888 |        0.0897 |
| 2018-09       | 2423 |         0.0714 |               0.0970 |                     0.7971 | TRAJECTORY_DEGRADATION |           0.0804 |        0.0809 |
| 2018-10       | 2150 |         0.0395 |               0.0805 |                     0.8093 | NONE                   |           0.0680 |        0.0680 |
| 2018-11       | 1632 |         0.0147 |               0.2455 |                     0.7321 | MIXED                  |           0.0507 |        0.0649 |
| 2018-12       | 1248 |         0.0104 |               0.1955 |                     0.7171 | TRANSITIONAL           |           0.0452 |        0.0541 |

Historical structural-break diagnostic:

|   year | validation_status    |      n |   default_rate |   macro_stress_score |   baseline_auc |   macro_auc |   baseline_brier |   macro_brier |
|-------:|:---------------------|-------:|---------------:|---------------------:|---------------:|------------:|-----------------:|--------------:|
|   2007 | in_sample_diagnostic |    251 |         0.1793 |               0.3000 |         0.7577 |      0.7626 |           0.1400 |        0.1347 |
|   2008 | in_sample_diagnostic |   1562 |         0.1581 |               0.4370 |         0.6534 |      0.6520 |           0.1297 |        0.1281 |
|   2009 | in_sample_diagnostic |   4716 |         0.1260 |               0.4038 |         0.6524 |      0.6515 |           0.1063 |        0.1082 |
|   2010 | in_sample_diagnostic |  11536 |         0.1289 |               0.2606 |         0.7142 |      0.7132 |           0.1048 |        0.1063 |
|   2011 | in_sample_diagnostic |  21721 |         0.1518 |               0.2037 |         0.7124 |      0.7111 |           0.1190 |        0.1196 |
|   2012 | in_sample_diagnostic |  53367 |         0.1620 |               0.1185 |         0.7035 |      0.7043 |           0.1254 |        0.1253 |
|   2013 | in_sample_diagnostic | 134804 |         0.1560 |               0.0952 |         0.7118 |      0.7116 |           0.1220 |        0.1221 |
|   2014 | in_sample_diagnostic | 223103 |         0.1845 |               0.1054 |         0.7336 |      0.7336 |           0.1337 |        0.1338 |
|   2015 | in_sample_diagnostic | 375546 |         0.2019 |               0.1047 |         0.7536 |      0.7533 |           0.1386 |        0.1385 |
|   2016 | validation           | 293105 |         0.2329 |               0.1206 |         0.7235 |      0.7218 |           0.1606 |        0.1601 |
|   2017 | validation           | 169321 |         0.2313 |               0.0531 |         0.7142 |      0.7142 |           0.1618 |        0.1618 |
|   2018 | out_of_time          |  56318 |         0.1576 |               0.1026 |         0.7068 |      0.7046 |           0.1244 |        0.1252 |

The 2007-2015 rows are included to inspect crisis-era behavior and environmental alignment only. They are not out-of-time evidence because the saved Phase 1 borrower model was trained on those years.

## 8. Calibration Analysis
The overlay was tuned on 2016-2017 validation data using Brier score with log-loss tie breaking. On 2018, calibration changed from Brier **0.1244** to **0.1252** and 10-bin ECE from **0.0207** to **0.0252**.

This is not a universal calibration victory. The overlay is useful only if Layer 3 pressure corresponds to borrower-model underconfidence or overconfidence in a given regime. In late 2018, Layer 3 correctly saw market deterioration, but realized defaults in the closed-loan sample were low for those issue months; the overlay therefore raised risk when the observed credit sample did not validate the increase.

## 9. Failure-Mode Analysis
The overlay is designed to matter most when borrower risk is near the governance boundary and Layer 3 reports elevated uncertainty or deterioration. In this experiment, that design reduced approvals and intercepted some baseline false negatives, but it did not improve 2018 ranking or stress-segment calibration.

Observed failure modes:

- Month-level loan timestamps limit precise daily synchronization.
- SPY is a broad environmental proxy, not a consumer-credit macro panel.
- 2007-2015 diagnostics are in-sample for the saved borrower model and should not be read as out-of-time validation.
- The 2018 test window contains deterioration episodes, but not a full credit crisis.
- A single monotone overlay cannot model every borrower-macro interaction.
- The fitted beta reached the allowed cap, which is a warning sign: the validation window favored stronger defensive pressure than the bounded institutional policy allowed, while the 2018 test window did not reward that pressure on average.

## 10. Institutional Interpretation
Phase 2 does not show that CRIS predicts defaults. It shows that an uncertainty-aware market-state overlay can make lending governance less aggressive when environmental pressure rises. This is aligned with the CRIS philosophy: probabilistic market stress should influence how financial systems behave.

## 11. Practical Limitations
The current overlay uses one market proxy and one bounded coefficient fitted on validation data. It should be treated as a governance-control experiment, not a production credit policy. Future work should test additional pre-specified environmental sources, preserve strict release lags, and validate across lenders or vintages before institutional deployment.

## 12. Future Layer 4 Governance Implications
Layer 4 can use Phase 2 outputs as defensive policy inputs:

- route high-uncertainty approvals to manual review,
- raise capital or reserve buffers during macro deterioration,
- adapt approval thresholds without replacing borrower PDs,
- monitor false-negative concentration in stressed regimes,
- separate model confidence from environmental confidence.

Final institutional rule retained: CRIS interprets probabilistic macro stress; it does not claim causal certainty or market/default prediction.
