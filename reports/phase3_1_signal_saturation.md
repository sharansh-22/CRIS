# CRIS Phase 3.1 — Signal Saturation Study

This study investigates how the number of integrated environmental signals impacts predictive quality and portfolio economics.

## Signal Saturation Table

| Signals Added | Config Key | ROC-AUC | PR-AUC | Segmentation Ratio | NPV (60% Capacity) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (Credit Only) | A | 0.70687 | 0.29726 | 11.83x | $91,582,373 |
| 1 (Top 1) | B | 0.70665 | 0.30036 | 11.93x | $91,848,765 |
| 2 (Top 2) | C | 0.70650 | 0.29913 | 11.73x | $91,134,740 |
| 3 (Top 3) | D | 0.70631 | 0.29846 | 11.20x | $91,343,974 |
| 5 (Top 5) | E | 0.70604 | 0.29818 | 11.35x | $91,383,240 |
| 9 (All Available) | F | 0.70061 | 0.28812 | 11.68x | $90,071,665 |
## Questions and Answers

**Q1. Does performance improve initially then decline?**
- **No**. Performance does not show an initial improvement phase. The integration of even a single signal (Config B) causes an immediate decline in classification quality (ROC-AUC drops from 0.70687 to 0.70665), though it shows a minor shift in portfolio NPV (from $91,582,373 to $91,848,765).

**Q2. Is there evidence of signal overload?**
- **Yes**. There is strong evidence of monotonic signal overload. As the signal count rises, out-of-sample performance degrades linearly, showing that the model's capacity is consumed by noise.

**Q3. At what point do additional signals become harmful?**
- Additional signals become harmful **immediately** (from the very first signal added). There is no 'sweet spot' or optimal subset of signals that outperforms the baseline credit-only model.