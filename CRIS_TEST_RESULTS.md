# CRIS — System Stress Test Results
> This file documents system-level stress test results only.
> One entry per layer.
> Individual component test results live inside each
> layer's tests/ directory.

---

## Results Index

| ID | Layer | Date | Result |
|----|-------|------|--------|
| ST-L3 | Layer 3 BSSC | 2026-03-22 | ✅ 5/5 PASSED |

---

## ST-L3 — Layer 3 BSSC Results
**Date:** 2026-03-22
**Overall:** ✅ 5/5 PASSED

### Results

| ID | Scenario | Type | State | Action | Result |
|----|----------|------|-------|--------|--------|
| ST-001 | COVID Crash | TRUE_POSITIVE | BLACK_SWAN | LIQUIDATE | ✅ |
| ST-002 | Q4 2018 Selloff | TRUE_POSITIVE | STRESS | REDUCE | ✅ |
| ST-003 | Vaccine Rally | TRUE_NEGATIVE | NORMAL | HOLD | ✅ |
| ST-004 | Calm Bull 2019 | TRUE_NEGATIVE | NORMAL | HOLD | ✅ |
| ST-005 | 2022 Bear Market | TRUE_POSITIVE | STRESS | REDUCE | ✅ |

### Verdict
```
╔══════════════════════════════════════════════════╗
║  Layer 3 BSSC: 5/5 PASSED                      ║
║  TRUE POSITIVES:  3/3 correctly detected        ║
║  TRUE NEGATIVES:  2/2 correctly suppressed      ║
║  Ready to merge: ✅ (already merged to main)    ║
╚══════════════════════════════════════════════════╝
```

### Key Metrics
| Metric | Value |
|--------|-------|
| Detection lead time (COVID) | 56 days |
| Detection lead time (Q4 2018) | 11 days |
| False positive rate | 0% |
| Crisis P99 slippage | 2729 bps |
| JD/GBM ratio | 2.29x |

### Known Limitation
In-sample validation: scenarios were used both
to calibrate thresholds and to validate the system.
Out-of-sample validation planned in backtesting
module after all four layers are complete.

---

## Future Results
| ID | Layer | Result |
|----|-------|--------|
| ST-L2 | Layer 2 MMAD | Pending |
| ST-L1 | Layer 1 Signal | Pending |
| ST-L4 | Layer 4 Credit | Pending |
| ST-CV | Convergence | Pending |
