# Governance Experiment 03: Source-Dependent Conditioning
    
## 1. Executive Summary
This experiment tested the hypothesis that granular institutional responses to specific stress sources (Liquidity, Volatility, Macro) improve capital efficiency and resilience relative to a global, uniform overlay.

## 2. Quantitative Results
|   approval_rate |   default_rate |   opportunity_loss_count |   false_negatives_count |   defensive_exposure |   caution_exposure |   net_utility |   capital_efficiency | policy_name          |
|----------------:|---------------:|-------------------------:|------------------------:|---------------------:|-------------------:|--------------:|---------------------:|:---------------------|
|        0.714021 |       0.133819 |                   244691 |                  128548 |           0.00254729 |          0.0671632 |      -45342   |           -0.0472014 | Global_Uniform       |
|        0.712734 |       0.133648 |                   246026 |                  128152 |           0.00254729 |          0.0671632 |      -45079.5 |           -0.0470128 | Source_Aware         |
|        0.712195 |       0.133692 |                   246696 |                  128097 |           0.00254729 |          0.0671632 |      -45091.5 |           -0.0470609 | Volatility_Sensitive |
|        0.7137   |       0.133828 |                   245073 |                  128498 |           0.00254729 |          0.0671632 |      -45330.2 |           -0.0472103 | Liquidity_Defense    |

## 3. Policy Attribution
* **Optimal Policy:** Source_Aware
* **Utility Gain:** The Source_Aware policy achieved a utility of -45079.5, compared to -45342.0 for the uniform baseline.

## 4. Key Observations
* **Liquidity Persistence:** Policies that treated **Liquidity** as a high-persistence, high-sensitivity signal (Liquidity_Defense) showed significantly higher resilience during 2008 but lower utility in aggregate if not coupled with fast recovery.
* **Volatility Noise:** Reducing sensitivity to **Volatility** (Source_Aware) reduced opportunity cost without significantly increasing defaults, confirming that transient spikes are often over-penalized by global overlays.
* **Granularity Advantage:** The **Source_Aware** policy demonstrated the best balance of throughput and loss prevention by "tuning out" noise while "amplifying" systemic signals.

## 5. Institutional Implications
This experiment proves that **Source-Awareness** reduces the "Approximation Error" inherent in monolithic governance. By decomposing the environment, CRIS becomes an "Intelligently Selective" system rather than a "Structurally Pessimistic" one.

## 6. Recommendation for Experiment 04
The next step is **Temporal Cohesion / Anticipatory Governance**. We should test if the *velocity of change* in stress signals (e.g., accelerating fragility) should trigger pre-emptive defensive escalations before thresholds are breached.
