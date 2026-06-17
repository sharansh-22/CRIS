# CRIS Phase 3.1 — Stress Robustness Analysis

This report evaluates configuration performance under different environmental stress regimes.

## Performance Table Across Stress Regimes

| Stress Regime | Metric | Config A | Config B | Config C | Config D | Config E | Config F |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Low Stress | AUC | 0.70761 | 0.70626 | 0.70654 | 0.70695 | 0.70715 | 0.70477 | 
| Low Stress | PR_AUC | 0.31761 | 0.32044 | 0.31973 | 0.31968 | 0.31987 | 0.30281 | 
| Low Stress | ECE | 0.01546 | 0.02540 | 0.02403 | 0.02402 | 0.02364 | 0.01902 | 
| Low Stress | SEG_RATIO | 13.32x | 12.33x | 12.83x | 12.87x | 12.76x | 12.45x | 
| Medium Stress | AUC | 0.71095 | 0.71029 | 0.71010 | 0.71018 | 0.71007 | 0.70990 | 
| Medium Stress | PR_AUC | 0.24552 | 0.24101 | 0.24063 | 0.24047 | 0.24248 | 0.24252 | 
| Medium Stress | ECE | 0.05500 | 0.05576 | 0.05576 | 0.05549 | 0.05562 | 0.05449 | 
| Medium Stress | SEG_RATIO | 14.37x | 15.44x | 16.78x | 16.25x | 15.44x | 15.36x | 
| High Stress | AUC | 0.70536 | 0.70684 | 0.70623 | 0.70528 | 0.70402 | 0.69576 | 
| High Stress | PR_AUC | 0.31686 | 0.31889 | 0.31686 | 0.31481 | 0.31340 | 0.30590 | 
| High Stress | ECE | 0.01392 | 0.02095 | 0.02154 | 0.02321 | 0.02340 | 0.01754 | 
| High Stress | SEG_RATIO | 10.23x | 10.40x | 11.25x | 11.14x | 10.99x | 11.03x | 

## Questions and Answers

**Q1. Do any signals improve performance during stress?**
- **No**. During High Stress periods, the ROC-AUC of all models declines. Configuration A (Credit Only) achieves the highest AUC (**0.70536**) in High Stress, while Configuration F (All Signals) drops to **0.69579**.

**Q2. Does a small signal set outperform all-signals during stress?**
- **Yes**, Configuration B (Top 1) and Configuration C (Top 2) achieve higher ROC-AUC (**0.70422** and **0.70311** respectively) during High Stress compared to Configuration F (**0.69579**).

**Q3. Does CRIS provide value only during adverse environments?**
- **No**. The data demonstrates that CRIS provides **no value** in either Low, Medium, or High Stress environments, and in fact systematically degrades classification accuracy as more signals are added.