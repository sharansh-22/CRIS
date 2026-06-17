# CRIS Signal Ranking Verified

This report documents the verified ranking of environmental signals based on their incremental out-of-sample contributions.

## Verified Signal Ranking Table

| Rank | Signal | Family | Incremental AUC | Incremental Segmentation | Incremental Economic Impact (NPV) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 1 | `uncertainty_pressure` | Layer3.Meta | -0.00022 | -0.08x | $+266,392 |
| 2 | `structural_instability` | Layer3.Slow | -0.00101 | -0.12x | $+417,136 |
| 3 | `stabilization_strength` | Layer3.Meta | -0.00196 | -0.22x | $-797,969 |
| 4 | `structural_fragility` | Layer3.Slow | -0.00254 | -0.19x | $+8,260 |
| 5 | `shock_intensity` | Layer3.Fast | -0.00369 | -0.31x | $+201,991 |
| 6 | `liquidity_disruption` | Layer3.Fast | -0.00537 | -0.42x | $+550,633 |
| 7 | `erosion_strength` | Layer3.Decay | -0.00603 | -0.39x | $-91,676 |
| 8 | `signal_coherence` | Layer3.Meta | -0.00653 | -0.48x | $-803,419 |
| 9 | `trajectory_fragility` | Layer3.Decay | -0.00697 | -0.55x | $-1,569,662 |

## Key Takeaways
- **All** individual signals show a negative incremental contribution to ROC-AUC, reflecting that no single signal is sufficient to improve out-of-sample performance.
- Several signals show a minor positive economic impact on portfolio NPV under specific underwriting capacities, but overall predictive rank-ordering degrades across the board.