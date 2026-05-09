# Governance Experiment 01: Sensitivity Sweep Report
    
## 1. Executive Summary
This experiment varied the governance sensitivity coefficient (Beta) to evaluate the impact of macro-conditioning on institutional performance. We measured the trade-off between credit losses (False Negatives) and opportunity costs (False Positives/Rejections of Good Loans).

## 2. Quantitative Results
|   approval_rate |   default_rate |   opportunity_loss_count |   false_negatives_count |   defensive_exposure |   net_utility |   capital_efficiency |   beta |
|----------------:|---------------:|-------------------------:|------------------------:|---------------------:|--------------:|---------------------:|-------:|
|        0.724539 |       0.135039 |                   233623 |                  131630 |           0.00254729 |      -47317.2 |           -0.0485425 |    0   |
|        0.724394 |       0.135045 |                   233797 |                  131610 |           0.00254729 |      -47314.6 |           -0.0485495 |    0.2 |
|        0.724249 |       0.135047 |                   233968 |                  131585 |           0.00254729 |      -47306.7 |           -0.0485512 |    0.4 |
|        0.724146 |       0.135055 |                   234096 |                  131575 |           0.00254729 |      -47309.5 |           -0.0485609 |    0.6 |
|        0.724006 |       0.135059 |                   234262 |                  131553 |           0.00254729 |      -47304.1 |           -0.0485647 |    0.8 |
|        0.723864 |       0.135063 |                   234431 |                  131531 |           0.00254729 |      -47299   |           -0.048569  |    1   |
|        0.723763 |       0.135071 |                   234558 |                  131521 |           0.00254729 |      -47301.7 |           -0.0485786 |    1.2 |
|        0.723617 |       0.135084 |                   234740 |                  131507 |           0.00254729 |      -47305.9 |           -0.0485927 |    1.5 |

## 3. Key Observations
* **Utility Peak:** The institutional utility (P&L approximation) peaks at Beta = 1.0.
* **Over-Defensiveness Check:** At Beta = 0.4 (Current Calibrated), the approval rate is 72.42%. 
* **Baseline Comparison:** The baseline (Beta=0.0) has a default rate of 13.50%, while the optimal policy has a default rate of 13.51%.
* **Opportunity Cost:** Increasing Beta from 0.0 to 1.5 increases opportunity loss from 233623.0 to 234740.0 rejections of valid loans.

## 4. Conclusion
The "Over-Defensiveness" bottleneck is quantified here. If the Utility curve slopes downward after Beta=0.4, CRIS is already at or past the point of diminishing returns for resilience.

## 5. Decision Support
Based on these findings, the recommended governance sensitivity for the current regime is Beta=1.0.
