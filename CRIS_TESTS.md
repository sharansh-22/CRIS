# CRIS — Test Registry

## Rules
- Every test **must** initialize a `wandb.init()` block and log all results to the WandB cloud as the system of record.
- No hardcoded numbers — all metrics must come from the engine's return values.
- System resource usage (RAM, RSS) must be tracked via `psutil` during every run.

---

## TS-001: Entropy Method Selection

**Objective:** Select the optimal entropy method for detecting market regime changes before black swan events.

**Runner:** `layer3_bssc/engine/entropy_comparison.py`

**Engine:** `run_entropy_method_selection()` in `layer3_bssc/engine/entropy.py`

**WandB Project:** `CRIS` · **Run Name:** `TS-001-Entropy-Comparison`

### Candidates Evaluated
| # | Method |
|---|--------|
| 1 | Shannon Entropy |
| 2 | Permutation Entropy |
| 3 | Sample Entropy |
| 4 | Tsallis Entropy |

### Historical Events Tested
| Event | Start | End |
|-------|-------|-----|
| COVID-2020 | 2020-02-01 | 2020-03-31 |
| Q4 2018 Selloff | 2018-10-01 | 2018-12-31 |
| Fed 2022 Tightening | 2022-01-01 | 2022-06-30 |

### Evaluation Metrics (logged to WandB)
- Lead Time (days before crisis trough)
- False Positive Rate (breaches per calm month)
- Magnitude (sigma shift from baseline)
- Directional Consistency (fraction of events with same signal direction)
- Composite Score (weighted aggregate: 40% lead time, 30% FP rate, 20% magnitude, 10% consistency)

### WandB Dashboard Sections
| Section | Contents |
|---------|----------|
| Summary Cards | Winner method, confirmation method, scores |
| Full Metrics Table | All 4 methods with 6 metrics side-by-side |
| Ranking Table | Methods sorted by composite score |
| Rejection Table | Rejected methods with reasons |
| Bar Charts | Composite scores, lead times, FP rates |
| Decision Summary | Primary/confirmation selection with rationale |
| System Metrics | RAM total/used, process RSS |

### How to Reproduce
```bash
conda run -n CRIS python -m layer3_bssc.engine.entropy_comparison
```

### Design Note DN-001 — Slippage Execution Anchoring
**Component:** Layer 3 BSSC — `engine/slippage.py`
**Decision Date:** 2026-03-18

**Decision:** Execution in `run_monte_carlo_slippage()` is anchored to the maximum drawdown peak of each path.

**Rationale:** Random execution point averaging masked the crisis impact — an initial implementation using random execution produced JD slippage lower than GBM slippage, contradicting financial reality. Anchoring to max drawdown peak reflects forced liquidation dynamics: margin calls, redemption requests, and risk limit breaches concentrate at peak loss periods.

**Known conservatism:** voluntary traders experience lower slippage. Conservatism is intentional for risk assessment.

**Trigger for revision:** actual crisis Implementation Shortfall (IS) in backtesting consistently below modeled IS by more than 40%.

### Design Note DN-002 — Permutation Entropy Deferral
**Component:** Layer 3 BSSC — `engine/slippage.py`
**Decision Date:** 2026-03-18

**Deferred:** `compute_execution_speed_recommendation()` using Permutation Entropy score as directional filter for optimal execution speed.

**Reason for deferral:** Permutation Entropy's role as execution speed signal requires validation against real directional market events. Layer 2 MMAD will provide additional microstructure regime data that strengthens this validation. Implementing before that data exists risks calibrating on insufficient evidence.

**Trigger for implementation:** Layer 2 MMAD complete and microstructure regime comparison test suite run.

---

## Validation Runs

### VR-001 — Slippage Monte Carlo Validation
**Component:** Layer 3 BSSC — `engine/slippage.py`
**Date:** 2026-03-18
**Purpose:** Confirm JD slippage > GBM slippage at >1.5x ratio
**Result:** PASSED — 2.29x ratio at mean, 4.98x at P99
