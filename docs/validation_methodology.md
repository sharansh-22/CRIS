# Validation Methodology: Institutional Behavioral Testing

## 1. Philosophical Grounding: Falsification
The CRIS Layer 3 framework is validated through **Strict Falsification**. We do not look for "backtest alpha"; we actively try to break the system's interpretive logic using adversarial scenarios.

![Validation Pipeline](diagrams/validation_pipeline.png)

---

## 2. Multi-Horizon Scenario Testing
The system is subjected to five "Standard Archetypes" that represent the spectrum of market behavior:
1.  **Acute Reflexive Shock:** (e.g., COVID 2020) - Tests FAST layer reflexivity.
2.  **Structural Selloff:** (e.g., Q4 2018) - Tests SLOW layer persistence.
3.  **Resilience Degradation:** (e.g., 2022 Bear) - Tests TRAJECTORY engine erosion detection.
4.  **Positive Noise:** (e.g., Vaccine Rally 2020) - Tests false-positive suppression during high-volatility upside.
5.  **Equilibrium Noise:** (e.g., 2019 Calm) - Tests baseline stability.

---

## 3. Core Validation Metrics

### 3.1 Probabilistic Stability (The "Flip" Test)
We measure the number of major interpretation transitions over long-duration simulations. An institutional system must provide a stable interpretive baseline.
*   **Target:** < 3 major flips per year during multi-crisis decades.

### 3.2 Trend-Bias Decoupling
We explicitly measure the correlation between `erosion_strength` and pure price trend.
*   **Goal:** The system must identify structural "grinds" without falsely flagging healthy secular pullbacks.

### 3.3 Asymmetric Recovery Lag
We measure the time taken for stabilization strength to rebuild after a crash.
*   **Target:** Gradual rebuilding (15–20 trading days) to ensure "scar memory" prevents premature re-risking.

---

## 4. Cross-Asset Normalization
The framework is validated across equities, indices, and high-volatility proxies (e.g., TSLA).
*   **Pass Criteria:** Assets with wildly different baseline volatilities must yield identical probabilistic stress intensities when subjected to equivalent relative structural shocks.

---

## 5. Systemic Integrity
Every validation run includes a **Pipeline Consistency Check**:
*   Ensuring zero NaN propagation.
*   Verifying bounded weights sum to 1.0.
*   Confirming LSTM influence is capped at 10%.
