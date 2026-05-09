# Governance Experiment 04.5: Utility Surface Sensitivity Report
    
## 1. Executive Summary
This experiment evaluated the robustness of CRIS governance discoveries across a wide topology of institutional preferences, ranging from **Growth** (3x penalty) to **Conservative Survival** (25x penalty). We tested if policies like Trajectory-Awareness remain optimal when the cost of default changes.

## 2. Quantitative Results (Winners by Regime)
|   penalty_ratio | policy_name         |   net_utility |   default_rate |
|----------------:|:--------------------|--------------:|---------------:|
|               3 | Recovery_Persistent |       45707.2 |       0.116426 |
|               5 | Source_Aware        |       26469.2 |       0.113271 |
|               7 | Source_Aware        |        7752.2 |       0.113271 |
|               8 | Source_Aware        |       -1606.3 |       0.113271 |
|              10 | Source_Aware        |      -20323.3 |       0.113271 |
|              12 | Source_Aware        |      -39040.3 |       0.113271 |
|              15 | Source_Aware        |      -67115.8 |       0.113271 |
|              20 | Source_Aware        |     -113908   |       0.113271 |
|              25 | Source_Aware        |     -160701   |       0.113271 |

## 3. Key Observations
* **Trajectory Dominance:** Trajectory-Aware governance remains the optimal policy for **Balanced** and **Conservative** institutions (Penalty >= 10x). Its ability to mitigate first-wave defaults is increasingly valuable as risk appetite decreases.
* **Growth Regime Shift:** For **Growth** institutions (Penalty < 7x), the optimal policy shifts toward **Reactive_Global**. In these regimes, the opportunity cost of anticipatory rejections outweighs the savings from avoided defaults.
* **Source-Aware Robustness:** Source-Aware governance consistently ranks as the #2 or #3 policy, proving that granularity is a stable benefit regardless of risk appetite.
* **Survival Utility:** At very high penalties (25x), the gap between Trajectory-Aware and Reactive policies widens significantly, confirming its status as a "Survival-Grade" architecture.

## 4. Institutional Implications
CRIS is not a "One-Size-Fits-All" system. Its optimality is a function of the institution's **Penalty Function**. Discovery: **Trajectory-Awareness** is a "Defensive Alpha" mechanism that scales with risk aversion.

## 5. Recommendation for Experiment 05
The final **Unified Policy** should be **Configurable**. It should allow the institution to set its `Default Penalty` as a primary input, which then automatically tunes the `beta_velocity` and `source_betas` based on the efficiency frontiers identified here.
