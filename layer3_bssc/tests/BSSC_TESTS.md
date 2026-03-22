# BSSC — Component Test Suite
## Layer 3: Individual Component Tests

> System-level stress tests are in CRIS_TESTS.md.
> This file records component-level tests only.

---

## Test Index

| ID | Component | Test Name | Date | Status |
|----|-----------|-----------|------|--------|
| TS-001 | entropy.py | Entropy Method Selection | 2026-03-01 | Superseded |
| TS-002 | entropy.py | Baseline Calibration | 2026-03-22 | Complete |
| VR-001 | slippage.py | Monte Carlo Validation | 2026-03-18 | Complete |
| PD-001 | detector.py | Persistence Gate Diagnostic | 2026-03-22 | Complete |

---

## TS-001 — Entropy Method Selection
**Component:** entropy.py  
**Date:** 2026-03-01  
**Status:** Superseded by TS-002 and Decision 4  

### Why Conducted
Determine which entropy method best detects black swans in market return distributions. Shannon entropy was known to fall during crashes. Sample, Permutation, and Tsallis were alternatives.

### Candidates
Shannon, Permutation, Sample, Tsallis

### Events Used
COVID 2020, Q4 2018 selloff, 2022 Fed bear market

### Metrics
| Metric | Weight |
|--------|--------|
| Crisis Lead Time | 35% |
| False Positive Rate | 30% |
| Signal-to-Noise Ratio | 15% |
| Directional Consistency | 10% |
| Breach Duration | 10% |

### Why Superseded
Sample entropy selected here (score 0.700) failed in full pipeline stress testing. The 35-day lead time result was not reproducible. See ADR Decision 4 for full investigation. Primary signal replaced with volatility ratio in TS-002.

---

## TS-002 — Baseline Calibration Study
**Component:** entropy.py (volatility regime)  
**Date:** 2026-03-22  
**Status:** Complete ✅

### Why Conducted
After replacing entropy with volatility ratio, empirically determine which calm period baseline definition produces the most accurate classification. Four candidates tested representing different approaches to defining market normality.

### Candidates
| ID | Definition |
|----|-----------|
| A | Fixed 6 months 2018 |
| B | 3 months 2018 + 3 recent calm months |
| C | Most recent calm 6 months |
| D | Full year 2018 |

### Events Used (labeled ground truth)
| Event | Expected |
|-------|----------|
| COVID crash | BLACK_SWAN |
| Q4 2018 selloff | STRESS |
| Bear 2022 | STRESS |
| Vaccine rally | NORMAL |
| Calm 2019 H1 | NORMAL |
| COVID pre-crash | STRESS |

### Metrics
| Metric | Weight |
|--------|--------|
| Correct Classification Rate | 40% |
| Stress False Positive Rate | 30% |
| Black Swan Lead Time | 20% |
| Threshold Stability | 10% |

---

## VR-001 — Slippage Monte Carlo Validation
**Component:** slippage.py  
**Date:** 2026-03-18  
**Status:** Complete ✅

### Why Conducted
Validate that entropy-conditioned slippage model produces meaningfully higher execution costs under Jump-Diffusion paths than GBM paths. Confirms the entropy-slippage connection works correctly.

### Validation Gate
JD/GBM mean slippage ratio must exceed 1.5x.

### Parameters
2000 GBM paths + 2000 JD paths  
Execution anchored to max drawdown peak  
eta=0.3 (upper bound — recalibrate at backtesting)  

---

## PD-001 — Persistence Gate Diagnostic
**Component:** detector.py  
**Date:** 2026-03-22  
**Status:** Complete ✅

### Why Conducted
Empirically determine how many consecutive days above the volatility stress threshold before STRESS is confirmed as sustained (not a brief spike). And how many days before BLACK_SWAN is confirmed.

### Method
Measured maximum consecutive days above each threshold across all known events. Gate sits above the false positive maximum and below the true positive minimum.

### Key Measurement
Vaccine rally stress streak: 8 days (false positive)  
COVID stress max streak: 35 days (true positive)  
COVID black swan streak: 23 days  
Vaccine black swan streak: 0 days  
