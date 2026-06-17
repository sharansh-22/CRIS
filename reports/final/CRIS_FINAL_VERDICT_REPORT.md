# CRIS Final Verdict Report

Written for the Model Risk Management Committee.

## Final Verdict Selection
**B. CRIS provides limited value but only in narrow governance scenarios.**

## Empirical Justification
1. **Direct Integration (Phase 3 & 3.1) Fails**: Directly conditioning borrower-level prediction on macro signals results in a statistically significant degradation of ROC-AUC and PR-AUC. The hypothesis of macroeconomic feature enrichment is rejected.
2. **Governance Attribution isolates signal value**: The Governance Attribution test shows that System B (which uses only borrower PD distributions) matches System C (which uses CRIS macro signals) within **$-4.29M** of NPV. CRIS macro signals do not improve the governance decisions over simple borrower-intrinsic risk adjustments.
3. **Risk-Return Trade-off**: Governance overlays successfully contain default rates in High Stress months (reducing defaults from 10.04% to 6.31%). However, this comes at a massive cost of foregone volume, reducing absolute NPV by **$27.66M** and Return on Capital by **-1.42%**.
4. **Final Assessment**: CRIS environmental intelligence fails to improve either borrower-level predictions or portfolio-level governance compared to standard borrower-centric models. The dynamic governance overlay only provides value to risk-averse institutions seeking to cap absolute losses during stress periods, regardless of opportunity costs.
