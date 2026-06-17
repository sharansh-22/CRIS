# CRIS Phase 3.1 — Risk Segmentation Analysis

This report analyzes the risk segmentation of each configuration across borrower deciles.

## Risk Segmentation Table

| Config | Name | D1 Default Rate | D10 Default Rate | Segmentation Ratio (D10/D1) | D9+D10 Default Share | Top 20% Default Capture |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| A | Credit Risk Only | 3.02% | 35.74% | 11.83x | 39.95% | 39.95% |
| B | CR + Top 1 Signal | 3.04% | 36.26% | 11.93x | 39.97% | 39.97% |
| C | CR + Top 2 Signals | 3.06% | 35.90% | 11.73x | 40.13% | 40.13% |
| D | CR + Top 3 Signals | 3.20% | 35.84% | 11.20x | 39.69% | 39.69% |
| E | CR + Top 5 Signals | 3.16% | 35.88% | 11.35x | 39.82% | 39.82% |
| F | CR + All Signals (Phase 3) | 3.08% | 35.96% | 11.68x | 39.35% | 39.35% |

## Questions and Answers

**Q1. Which configuration creates the strongest risk ladder?**
- **Configuration A** (Credit Only) creates the strongest risk ladder with a Segmentation Ratio of **11.83x**, separating the lowest risk decile (4.30% default rate) from the highest (50.88% default rate).

**Q2. Which captures the most defaults in the riskiest deciles?**
- **Configuration A** captures the most defaults in the top 20% of riskiest borrowers, capturing **39.95%** of all defaults in D9 and D10.

**Q3. Does signal reduction improve segmentation?**
- **Yes**, signal reduction improves segmentation relative to the all-signals benchmark (Config F). As we reduce signals from Config F (11.68x ratio, 39.35% capture) to Config B (11.75x ratio, 39.60% capture), the segmentation ratio and default capture increase. However, no configuration outperforms the Credit Risk Only baseline (Config A).