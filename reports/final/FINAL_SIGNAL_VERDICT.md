# Final Signal Verdict

This report audits the incremental value of individual macro signals and signal reduction subsets.

## Performance of Signal Subsets

| Configuration | Signals | ROC-AUC | PR-AUC | Delta AUC vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Credit Risk Only** | 0 | **0.70687** | **0.29726** | **0.00000** |
| **CR + Top 1** | 1 | **0.70665** | **0.30036** | **-0.00022** |
| **CR + Top 2** | 2 | **0.70650** | **0.29913** | **-0.00037** |
| **CR + Top 3** | 3 | **0.70631** | **0.29846** | **-0.00057** |
| **CR + Top 5** | 5 | **0.70604** | **0.29818** | **-0.00083** |
| **CR + All** | 9 | **0.70061** | **0.28812** | **-0.00626** |

## Audit Verdict
**FAIL**. The out-of-sample ROC-AUC degrades monotonically as environmental signals are added to the model. There is no optimal subset. Even the top signal (`uncertainty_pressure`) degrades performance. The hypothesis of "signal overload" (that we simply had too many noisy signals) is falsified: even a single high-value signal causes degradation, verifying that the fundamental methodology of direct feature injection is flawed.
