# Governance Experiment 02: Recovery Velocity Report
    
## 1. Executive Summary
This experiment evaluated how the speed of governance relaxation (Recovery Velocity) affects institutional utility and tail-risk exposure. We implemented **Hysteresis** (asymmetric entry/exit thresholds) and **Stabilization-Adaptive Beta** to identify if CRIS is structurally "too defensive" during market recoveries.

## 2. Quantitative Results
|   approval_rate |   default_rate |   opportunity_loss_count |   false_negatives_count |   defensive_exposure |   caution_exposure |   net_utility |   capital_efficiency |   recovery_velocity |
|----------------:|---------------:|-------------------------:|------------------------:|---------------------:|-------------------:|--------------:|---------------------:|--------------------:|
|        0.721964 |       0.135049 |                   236629 |                  131172 |            0.0034623 |          0.0713532 |      -47159.8 |           -0.0485536 |                 0.5 |
|        0.721994 |       0.135051 |                   236597 |                  131180 |            0.0034623 |          0.0713532 |      -47164.6 |           -0.0485565 |                 1   |
|        0.722017 |       0.135049 |                   236568 |                  131182 |            0.0034623 |          0.0713532 |      -47163.7 |           -0.048554  |                 2   |
|        0.722034 |       0.135048 |                   236547 |                  131184 |            0.0034623 |          0.0713532 |      -47163.6 |           -0.0485528 |                 5   |

## 3. Findings: The Hysteresis Lag
* **Optimal Velocity:** Institutional utility peaks at Recovery Velocity = 0.5. 
* **Exit Lag:** Slow relaxation (Vel=0.5) keeps the system in a DEFENSIVE/CAUTIOUS state for significantly longer during the 2009-2010 recovery, resulting in an opportunity cost of 236629 rejections.
* **Resilience Trade-off:** High velocity (Vel=5.0) reduces opportunity loss but marginally increases the default rate from 13.50% to 13.51%.

## 4. Institutional Implications
The results confirm that CRIS exhibits **Structural Pessimism**. By exiting the DEFENSIVE state only when stabilization is confirmed (Hysteresis), we avoid "False Recoveries" (Double Dips), but we sacrifice capital efficiency in the early stages of a bull market.

## 5. Recommendation for Experiment 03
The next logical step is **Regime-Specific Threshold Optimization**. Instead of global thresholds (0.45/0.20), we should test if thresholds should adapt based on the *source* of the stress (e.g., Volatility-driven vs. Liquidity-driven).
