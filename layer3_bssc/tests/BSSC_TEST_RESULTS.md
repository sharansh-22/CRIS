# BSSC — Component Test Results
## Layer 3: Individual Component Test Results

> System-level stress test results in CRIS_TEST_RESULTS.md
> This file records component-level results only.

---

## Results Index

| ID | Test | Date | Winner/Decision |
|----|------|------|-----------------|
| TS-001 | Entropy Method Selection | 2026-03-01 | Sample (superseded) |
| TS-002 | Baseline Calibration | 2026-03-22 | B Split (0.714%) |
| VR-001 | Slippage Monte Carlo | 2026-03-18 | PASSED 2.29x |
| PD-001 | Persistence Gate | 2026-03-22 | 10-day / 5-day |

---

## TS-001 — Entropy Method Selection Results
**Date:** 2026-03-01  
**Status:** Superseded

### Raw Results
| Method | Lead Time | FP Rate | SNR | Consistency | Duration | Score |
|--------|-----------|---------|-----|-------------|----------|-------|
| Sample | 35 days | 2.174/mo | 1.361 | 1.00 | 18 days | 0.700 |
| Permutation | 0 days | 0.000/mo | 0.994 | 1.00 | 0 days | 0.400 |
| Shannon | 0 days | 0.000/mo | 0.994 | 0.67 | 0 days | 0.367 |
| Tsallis | 0 days | 0.149/mo | 0.995 | 0.67 | 0 days | 0.347 |

### Why Superseded
Sample entropy selected here failed in stress testing. The adaptation problem: after ~2 weeks of crashes, falling becomes the new normal and entropy resets. The 35-day result was not reproducible in validation. Replaced by volatility ratio (TS-002).

---

## TS-002 — Baseline Calibration Results
**Date:** 2026-03-22  
**Status:** Complete ✅

### Results
| Candidate | Baseline | Correct | FP Rate | Lead Time | Score |
|-----------|----------|---------|---------|-----------|-------|
| 🥇 B Split | 0.714% | 5/6 | 0.0% | 13 days | 0.700 |
| A 6m-2018 | 0.739% | 4/6 | 0.0% | 13 days | 0.400 |
| D Full2018 | 0.740% | 4/6 | 0.0% | 13 days | 0.400 |
| C Recent | 0.546% | 4/6 | 13.9% | 15 days | 0.280 |

### Decision
```
╔══════════════════════════════════════════════════════╗
║  WINNER: B Split                Score: 0.700        ║
║  Baseline: 0.714% daily move                        ║
║  NORMAL/STRESS boundary:  1.071% (1.5x)             ║
║  STRESS/BLACK SWAN boundary: 2.142% (3.0x)          ║
╚══════════════════════════════════════════════════════╝
```

### Why B Split Won
Only candidate to correctly classify Q4 2018 as STRESS. Lower baseline (0.714%) vs A and D (0.739%, 0.740%) means Q4 2018 elevated moves cross the 1.5x threshold. Zero false positive rate on calm periods.

### Why C Was Rejected
13.9% false positive rate on Calm 2019 H1. Lower baseline (0.546%) causes routine modern volatility to look stressed — baseline contamination.

---

## VR-001 — Slippage Monte Carlo Results
**Date:** 2026-03-18  
**Status:** Complete ✅

### Distribution Results (2000+2000 paths)
| Metric | GBM (Normal) | Jump-Diffusion | Ratio |
|--------|-------------|----------------|-------|
| Mean | 214.59 bps | 490.58 bps | 2.29x |
| Median | 210.55 bps | 241.91 bps | 1.15x |
| P95 | 447.25 bps | 1948.61 bps | 4.36x |
| P99 | 548.48 bps | 2729.41 bps | 4.98x |
| Max | 696.72 bps | 4492.89 bps | 6.45x |

### Validation Gate
```
╔══════════════════════════════════════════════════════╗
║  Gate: JD/GBM ratio > 1.5x                         ║
║  Actual: 2.29x                                      ║
║  Result: ✅ PASSED                                  ║
╚══════════════════════════════════════════════════════╝
```

### Key Finding
Median ratio 1.15x vs mean ratio 2.29x confirms fat right tail — most JD paths calm, a minority experience catastrophic slippage that dominates the mean. Correct behavior for jump-diffusion model.

### Regime Breakdown
| Regime | GBM | Jump-Diffusion |
|--------|-----|----------------|
| NORMAL | 2000 | 1887 |
| STRESS | 0 | 112 |
| BLACK_SWAN | 0 | 1 |

### Known Limitations
eta=0.3 is upper bound of Almgren-Chriss calibration range (0.1-0.3). Recalibrate against empirical SPY execution costs during backtesting phase.

---

## PD-001 — Persistence Gate Diagnostic Results
**Date:** 2026-03-22  
**Status:** Complete ✅

### Streak Measurements
| Event | Stress Max Streak | BS Max Streak | Expected |
|-------|------------------|---------------|----------|
| COVID | 35 days | 23 days | TRUE POSITIVE |
| Q4 2018 | 26 days | 0 days | TRUE POSITIVE |
| Vaccine | 8 days | 0 days | FALSE POSITIVE |
| Bear 2022 | 31 days | 0 days | TRUE POSITIVE |
| Calm 2019 | 0 days | 0 days | TRUE NEGATIVE |

### Decision
```
╔══════════════════════════════════════════════════════╗
║  STRESS gate:      10 consecutive days              ║
║  Rationale: filters vaccine 8-day streak            ║
║             confirms Q4 2018 first wave (12 days)   ║
║                                                     ║
║  BLACK SWAN gate:  5 consecutive days               ║
║  Rationale: vaccine never fires (0 days)            ║
║             confirms COVID (23-day streak)          ║
╚══════════════════════════════════════════════════════╝
```

### Permutation Entropy Alarm Validation
| Event | Alarm First Fires | Vol Ratio Fires | Lead Time |
|-------|------------------|-----------------|-----------|
| COVID | Jan 2, 2020 | Feb 27, 2020 | 56 days ✅ |
| Q4 2018 | Oct 5, 2018 | Oct 16, 2018 | 11 days ✅ |
| Vaccine | Never | Never | N/A ✅ |

Alarm correctly silent on vaccine rally: 0 false alarms.
