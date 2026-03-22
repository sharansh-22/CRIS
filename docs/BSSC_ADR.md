# BSSC — Architecture Decision Record
## Layer 3: Black Swan and Slippage Computer
**Status:** Complete
**Last Updated:** 2026-03-22

---

## Overview
BSSC (Black Swan and Slippage Computer) serves as Layer 3 of CRIS. It is responsible for detecting black swan market conditions and modeling the execution cost of trading during those conditions. It relies on a two-signal architecture (Permutation Entropy Alarm + Volatility Ratio Confirmation) to accurately and robustly classify the market regime and calculate slippage using an entropy-conditioned Almgren-Chriss square root law.

---

## Decision Log

### Decision 1 — Price Path Simulation Method
**Date:** Early development  
**Question:** How to model extreme price paths?  
**Candidates:**  
  - Standard GBM (Geometric Brownian Motion)
  - Merton Jump-Diffusion  

**Decision:** Merton Jump-Diffusion  
**Rationale:** GBM cannot produce the fat-tailed return distributions observed during black swans. Jump-Diffusion adds sudden discontinuous jumps.  
**Validated Result:**  
  GBM kurtosis: 0.158 | JD kurtosis: 80.58  
  GBM min return: -2.58% | JD min return: -21.29%  
**Status:** FINAL ✅

---

### Decision 2 — Primary Entropy Method (TS-001)
**Date:** 2026-03-01  
**Question:** Which entropy method best detects black swans in market return distributions?  
**Candidates tested:**  
  Shannon, Permutation, Sample, Tsallis  
**Metrics:** Lead time (35%), FP rate (30%), SNR (15%), consistency (10%), breach duration (10%)  

**TS-001 Result:**  

| Method | Lead Time | FP Rate | Score |
|--------|-----------|---------|-------|
| Sample | 35 days | 2.174/mo | 0.700 |
| Permutation | 0 days | 0.000/mo | 0.400 |
| Shannon | 0 days | 0.000/mo | 0.367 |
| Tsallis | 0 days | 0.149/mo | 0.347 |

**Decision:** Sample Entropy (primary), Permutation Entropy (confirmation)  
**Status:** SUPERSEDED — see Decision 4

---

### Decision 3 — Slippage Model Architecture
**Date:** Early development  
**Question:** How to model execution costs?  
**Decision:** Almgren-Chriss with entropy conditioning  
**Key design decisions:**  
  - IS not VWAP (avoids self-contamination)
  - Regime multiplier on spread only, not market impact (prevents double-counting path volatility)
  - Execution anchored to max drawdown peak (forced liquidation scenario)
  - 2000 paths for stable P99 estimation  

**Validated Result (VR-001):**  
  JD/GBM ratio: 2.29x (gate >1.5x PASSED)  
  P99 JD IS: 2729.41 bps  
**Status:** FINAL ✅

---

### Decision 4 — Primary Detection Signal
**Date:** 2026-03-21  
**Question:** Why did the stress test fail 3/5 with Sample Entropy as primary signal?  

**Root Cause Investigation:**  
Shannon entropy issue (known): falls during crashes because returns concentrate on left tail — confirmed in plots showing Shannon dropping during COVID.

Sample entropy issue (discovered in stress test): also fails for sustained crashes. After ~2 weeks of consistent negative returns, falling becomes the new normal. Sample entropy measures self-similarity — a market falling every day is highly self-similar (low entropy). Signal adapts and resets to baseline.

This is called the adaptation problem: entropy compares to recent experience. After 2 weeks of crashing, crashing IS recent experience. Signal disappears.

TS-001 35-day lead time was not reproducible: The result depended on specific baseline computation that happened to produce one or two threshold breaches. Not a robust signal.

**Candidates evaluated:**  
  1. Invert entropy (detect downward breaches)
  2. Volatility ratio (compare to fixed baseline)
  3. Directional entropy (sequence patterns)
  4. Combined multi-signal approach

**Baseline calibration study (TS-002):**  
Four baseline definitions tested:  

| Candidate | Correct | FP Rate | Score |
|-----------|---------|---------|-------|
| B Split (3m2018+3mRecent) | 5/6 | 0% | 0.700 |
| A (6m 2018) | 4/6 | 0% | 0.400 |
| D (Full 2018) | 4/6 | 0% | 0.400 |
| C (Recent 6m) | 4/6 | 13.9% | 0.280 |

Winner: B Split — baseline 0.714% daily move

**Persistence gate validation:**  
  Vaccine rally stress streak: 8 days  
  COVID stress max streak: 35 days  
  Q4 2018 stress max streak: 26 days  
  → 10-day gate filters vaccine, confirms Q4 2018  
  COVID black swan streak: 23 days  
  Vaccine black swan streak: 0 days  
  → 5-day gate sufficient for black swan confirmation  

**Decision:** Two-signal architecture  
  Signal 1: Permutation Entropy Alarm
    No persistence gate — fires immediately
    Threshold: perm_entropy < baseline - 0.05
    Role: early warning (11-56 day lead time)
  Signal 2: Volatility Ratio Confirmation
    STRESS: >1.5x for 10+ days
    BLACK_SWAN: >3.0x for 5+ days
    Baseline: 0.714% (TS-002 B Split)

**Rationale:** Volatility ratio compares to fixed 2018 baseline — does not adapt. COVID on day 30 is still 5x the 2018 baseline. Permutation entropy retained as alarm because it detects directional pattern shift before magnitude becomes extreme.

**Validated result (stress test):**  
  5/5 scenarios passing  
  COVID lead time: 56 days (alarm)  
  Q4 2018 lead time: 11 days (alarm)  
  Vaccine false alarms: 0  

**Status:** FINAL ✅

---

### Decision 5 — Data Contracts
**Date:** Mid development  
**Question:** How to pass data between pipeline stages?  
**Decision:** Pydantic v2 BaseModel  
**Rationale:** Runtime enforcement — invalid values raise ValidationError at object creation not silently downstream. In financial risk systems silent errors are more dangerous than loud ones.  
**Key design:** ConfigDict(extra='ignore') for forward compatibility with JSON loading.  
**Status:** FINAL ✅

---

## Known Limitations

| Limitation | Impact | Trigger for Revision |
|------------|--------|---------------------|
| Baseline staleness | Stress threshold may produce false positives as market structure evolves | Recalibrate every 3 years |
| eta=0.3 upper bound | Slippage estimates may be conservative | Recalibrate at backtesting phase |
| 3 calibration events | Insufficient for statistical significance | Backtesting module post all layers |
| Single asset (SPY) | Results not validated on other assets | Multi-asset testing planned |
| Synthetic order book | Not real Level-2 data | Alpaca free tier when available |

---

## Performance Summary

| Metric | Value | Validation |
|--------|-------|------------|
| Detection lead time (COVID) | 56 days | Empirical |
| Detection lead time (Q4 2018) | 11 days | Empirical |
| False positive rate | 0% | 5-scenario stress test |
| Classification accuracy | 5/5 (100%) | Stress test |
| JD/GBM slippage ratio | 2.29x | VR-001 |
| Crisis P99 IS | 2729 bps | VR-001 |

**Note on financial quantification:**
Financial value of early warning not yet computed. Requires out-of-sample backtesting across events not used for calibration. Planned in backtesting module after all four layers are complete.
