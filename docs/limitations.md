# Institutional Limitations: Intellectual Honesty in Modeling

## 1. Scope Boundaries
The CRIS Layer 3 framework is a **Structural Market Interpretation System**, not a crystal ball.

### 1.1 Exogenous Shocks
The system interprets stress *after* the structure begins to fail. It cannot predict a sudden, exogenous geopolitical event (e.g., a "Sunday night surprise") before it reflects in liquidity, volatility, or price trajectories.

### 1.2 Latency vs. Robustness
By design, the system uses EMA smoothing and temporal coordination to ensure interpretation stability. This introduces a slight "recognition lag" (typically 1–3 days) compared to high-frequency momentum traders. This is an intentional trade-off to eliminate whipsaw and false positives.

---

## 2. Technical Dependencies

### 2.1 Data Quality
The entropy and trajectory engines are sensitive to "dirty" data. Missing days, incorrect adjustment factors, or extreme gaps in the price feed will severely degrade the `uncertainty_score`.

### 2.2 In-Sample Memory
While the LSTM is self-supervised and bounded, its pattern-recognition capability is naturally limited to structural archetypes present in its training set. In a truly "new" era of market physics, the LSTM contribution will decay to zero (by design), leaving the system to rely purely on its deterministic entropy/volatility layers.

---

## 3. Interpreting "mixed" Signals
The system preserves uncertainty. In rare cases of extreme divergence (e.g., FAST = 1.0, SLOW = 0.0), the system may produce an "UNCLEAR" or "MIXED" interpretation for several days. Institutional allocators must be prepared to handle these periods of neutral ambiguity.

---

## 4. Asset Breadth
The system is optimized for liquid indices and equities. It is not currently calibrated for highly illiquid assets, OTC markets, or extremely low-frequency instruments where daily returns do not carry structural signal.
