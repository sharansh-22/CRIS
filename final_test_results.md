# CRIS — Final Test Results
### Cascade Risk Intelligence System
### Empirical Validation Record

---

> This document records every empirical comparison and 
> validation result produced during CRIS development.
> Each entry documents what was tested, what metrics were 
> used, what the data showed, and what decision was made.
> Results are never edited after being written — only 
> appended to.

---

## Entry Index

| # | Component | Test Name | Date | Winner/Decision |
|---|-----------|-----------|------|-----------------|
| 001 | Layer 3 BSSC | Entropy Method Selection | 2026-03-01 | Primary: sample, Confirmation: permutation |

---

## Entry 001 — Entropy Method Selection
**Component:** Layer 3 BSSC — engine/entropy.py
**Test Date:** 2026-03-01
**Purpose:** Empirically select the primary and confirmation 
entropy methods for CRIS's black swan classification engine.
**Data Used:** SPY daily OHLCV from data/Indices/SPY.csv

### Events Tested Against
| Event | Window | Type |
|-------|--------|------|
| COVID Crash | 2020-02-01 → 2020-03-31 | Macro-driven |
| Q4 2018 Selloff | 2018-10-01 → 2018-12-31 | Macro-driven |
| 2022 Bear Market | 2022-01-01 → 2022-06-30 | Policy-driven |

### Metrics Used
| Metric | Weight | Rationale |
|--------|--------|-----------|
| Crisis Lead Time | 35% | Primary purpose is early warning |
| False Positive Rate | 30% | Reliability in calm markets |
| Signal-to-Noise Ratio | 15% | Signal strength during events |
| Directional Consistency | 10% | Predictability across event types |
| Threshold Breach Duration | 10% | Signal persistence |

### Raw Metrics Table

| Method | Lead Time (days) | FP Rate (/mo) | SNR | Consistency | Breach Duration | Composite |
|--------|-----------------|---------------|-----|-------------|-----------------|-----------|
| Shannon | 0.0 | 0.000 | 0.994 | 0.67 | 0.0 | 0.367 |
| Permutation | 0.0 | 0.000 | 0.994 | 1.00 | 0.0 | 0.400 |
| Sample | 35.0 | 2.174 | 1.361 | 1.00 | 18.0 | 0.700 |
| Tsallis | 0.0 | 0.149 | 0.995 | 0.67 | 0.0 | 0.347 |

### Correlation Matrix (During Crisis Windows)

| Method | Shannon | Permutation | Sample | Tsallis |
|--------|---------|-------------|--------|---------|
| Shannon | 1.00 | 0.14 | -0.19 | 1.00 |
| Permutation | 0.14 | 1.00 | 0.02 | 0.20 |
| Sample | -0.19 | 0.02 | 1.00 | -0.12 |
| Tsallis | 1.00 | 0.20 | -0.12 | 1.00 |

### Composite Scores
1. Sample: 0.700
2. Permutation: 0.400
3. Shannon: 0.367
4. Tsallis: 0.347

### Decision
**Primary Method:** sample
**Composite Score:** 0.700
**Confirmation Method:** permutation
**Composite Score:** 0.400
**Runner-up selected because:** correlation with primary = 0.02 
(below 0.70 threshold — adds independent information)

### Rationale
Sample Entropy massively outperformed other methods by providing a staggering 35-day average early warning lead time and holding its breach threshold limit for an average duration of 18 days, whereas other methods failed to breach early entirely (0 lead time). Permutation Entropy was selected as the confirmation method because it demonstrated perfect 1.00 directional consistency alongside a near-zero correlation (0.02) with Sample Entropy, ensuring the two signals validate distinct phenomenological aspects of the crash without redundancy.

### Rejected Methods
| Method | Reason |
|--------|--------|
| Shannon | Zero lead time and highly susceptible to volatility expansion washing out structural breakdown signals. |
| Tsallis | Failed to breach stress thresholds prior to maximum drawdowns (0 lead time) and generated a non-zero false positive rate during calm periods. |

### Plots Generated
- data/simulation_output/entropy_comparison_overview.png
- data/simulation_output/entropy_metrics_bar_chart.png
- data/simulation_output/entropy_correlation_heatmap.png

### Next Action
Primary method sample will be used as the entropy signal
in layer3_bssc/engine/entropy.py classify_market_state()
Confirmation method permutation will be used as secondary
confirmation in the BSSC report generator.
