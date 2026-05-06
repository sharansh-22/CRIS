# Convergence Design: Probabilistic Temporal Coordination

## 1. Objective
The **Convergence Engine** is the central coordinator of the CRIS Layer 3 framework. Its role is to synchronize the outputs of independent stress_fields into a coherent, smoothed, and operationally usable probability stream.

---

## 2. Independent Input Streams
The coordinator receives three primary probability vectors every trading day:
1.  **Fast Shock Intensity** ($P_{fast}$)
2.  **Slow Structural Stress** ($P_{slow}$)
3.  **Trajectory Fragility** ($P_{decay}$)

Each input is accompanied by a **Confidence Score** ($C$) based on signal signal-to-noise ratios and local data quality.

---

## 3. The Arbitration Mechanism

### 3.1 Adaptive Weighting
Weights are not static. They evolve based on the **Dominance Field**.
*   In acute, high-velocity crashes, $W_{fast}$ expands to capture immediate reflexive panic.
*   In slow, sideways grinds, $W_{decay}$ expands to capture structural deterioration that volatility metrics miss.
*   The transition between weights is governed by a **Hysteresis Governor**, preventing oscillation between engines.

### 3.2 Bounded Inter-Layer Influence
Before final aggregation, the engine performs a "partner check":
*   If FAST and SLOW both report high stress, their respective confidence scores are boosted.
*   The maximum boost is hard-capped at **5%** to prevent runaway feedback loops and "echo-chamber" reinforcement.

---

## 4. Meta-Dynamics

### 4.1 Uncertainty Pressure
The coordinator computes a **Signal Coherence** metric. 
*   High divergence between engines (e.g., FAST = 0.9, SLOW = 0.1) produces high **Uncertainty Pressure**.
*   This pressure actively dampens the "Overall Risk" probability, preventing high-conviction actions on incoherent signals.

### 4.2 Stabilization Strength
A persistent internal state tracks the **Healing Rate** of the market.
*   Stress accumulates instantly (Panic is reflexive).
*   Stabilization rebuilds gradually via an asymmetric EMA (Healing is structural).
*   Until stabilization strength recovers to baseline, the system remains "wary," even if raw returns turn positive.

---

## 5. Operational Output
The final `convergence` field provides:
*   `overall_risk`: Smoothed, coordinated stress probability.
*   `overall_confidence`: Aggregated interpretation certainty.
*   `uncertainty_score`: Signal divergence intensity.
*   `stabilization_strength`: Current structural healing progress.
