# CRIS Signal Contribution Report

This report ranks the 18 CRIS environmental signals based on their incremental contribution (lift relative to the standalone Credit Risk model).

## Signal Contribution Ranking Table

| Rank | Signal | Family | Incremental AUC | Incremental NPV (60% Capacity) |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `uncertainty_pressure` | Layer3.Meta | -0.00022 | $+266,392 |
| 2 | `structural_instability` | Layer3.Slow | -0.00101 | $+417,136 |
| 3 | `stabilization_strength` | Layer3.Meta | -0.00196 | $-797,969 |
| 4 | `structural_fragility` | Layer3.Slow | -0.00254 | $+8,260 |
| 5 | `shock_intensity` | Layer3.Fast | -0.00369 | $+201,991 |
| 6 | `liquidity_disruption` | Layer3.Fast | -0.00537 | $+550,633 |
| 7 | `erosion_strength` | Layer3.Decay | -0.00603 | $-91,676 |
| 8 | `signal_coherence` | Layer3.Meta | -0.00653 | $-803,419 |
| 9 | `trajectory_fragility` | Layer3.Decay | -0.00697 | $-1,569,662 |

## Key Findings
- Almost all individual environmental signals produce negative incremental out-of-sample AUC when added to the borrower-level credit model.
- This indicates that no single macro or market structure signal successfully improves the model's generalization performance out-of-sample.