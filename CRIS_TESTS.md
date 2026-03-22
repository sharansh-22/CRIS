# CRIS — System Stress Tests
> This file documents system-level stress tests only.
> One entry per layer.
> Individual component tests live inside each
> layer's tests/ directory.

---

## Test Index

| ID | Layer | Date | Status |
|----|-------|------|--------|
| ST-L3 | Layer 3 BSSC | 2026-03-22 | ✅ 5/5 PASSED |

---

## ST-L3 — Layer 3 BSSC Stress Test
**Layer:** Layer 3 BSSC
**Date:** 2026-03-22
**Status:** ✅ 5/5 PASSED

### Why Conducted
Validate the complete Layer 3 pipeline end to end
against five real historical scenarios representing
structurally different market conditions.
Three true positives — system must detect stress.
Two true negatives — system must stay silent.

### Scenarios
| ID | Scenario | Type | Severity |
|----|----------|------|----------|
| ST-001 | COVID Crash | TRUE_POSITIVE | CRITICAL |
| ST-002 | Q4 2018 Selloff | TRUE_POSITIVE | HIGH |
| ST-003 | Vaccine Rally | TRUE_NEGATIVE | CRITICAL |
| ST-004 | Calm Bull Market 2019 | TRUE_NEGATIVE | HIGH |
| ST-005 | 2022 Fed Bear Market | TRUE_POSITIVE | HIGH |

### Pass Criteria
| Scenario | Expected State | Expected Action |
|----------|---------------|-----------------|
| COVID Crash | BLACK_SWAN | LIQUIDATE |
| Q4 2018 Selloff | STRESS or BLACK_SWAN | REDUCE or LIQUIDATE |
| Vaccine Rally | NORMAL or STRESS | HOLD or REDUCE |
| Calm 2019 | NORMAL | HOLD |
| 2022 Bear | STRESS or BLACK_SWAN | REDUCE or LIQUIDATE |

### How To Run
  conda run -n CRIS python3 -m \
    layer3_bssc.tests.stress_test_layer3 --no-wandb

---

## Future Entries
| ID | Layer | Status |
|----|-------|--------|
| ST-L2 | Layer 2 MMAD | Not Started |
| ST-L1 | Layer 1 Signal | Not Started |
| ST-L4 | Layer 4 Credit | Not Started |
| ST-CV | Convergence | Not Started |
