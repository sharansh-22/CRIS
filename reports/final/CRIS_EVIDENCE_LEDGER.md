# CRIS Evidence Ledger

| Experiment | Hypothesis | Empirical Result | Supports CRIS? | Contradicts CRIS? | Evidence Strength |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Direct predictive Integration** | CRIS signals improve borrower default prediction | ROC-AUC fell by **-0.00627**; PR-AUC fell by **-0.00888** | NO | YES | **Strong** (Bootstrap CI below 0) |
| **Signal Reduction subsets** | Top signals can out-perform Credit Only baseline | Even Top 1 signal degrades ROC-AUC by **-0.00022** | NO | YES | **Moderate** |
| **Stress regime calibration** | CRIS improves calibration in stress periods | ECE shifted from 0.02060 to 0.01968 | NO | YES | **Weak** |
| **Dynamic Governance Overlay** | Macro overlays improve capital efficiency | Realized loss falls by **$11.80M** but Return on Capital drops by **-1.42%** | MIXED | YES | **Strong** (NPV & RoC decline) |
| **Governance Attribution** | CRIS governance outperforms simple borrower-based tightening | System B (PD-Only) matches System C (CRIS-Gov) within **$0.22M** NPV | NO | YES | **Strong** (Attribution to CRIS is negligible) |
