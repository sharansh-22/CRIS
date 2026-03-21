# CRIS Layer 3 — Stress Test Results
**Date:** 2026-03-21T00:51:41.514337
**Total Scenarios:** 5
**Passed:** 2 / 5
**Overall Status:** ❌ 3 FAILED

---

## Results Summary

| ID | Scenario | Type | Severity | State | Action | Breach | P99 | Ratio | Result |
|----|----------|------|----------|-------|--------|--------|-----|-------|--------|
| ST-001 | COVID Crash | TRUE_POSITIVE | CRITICAL | NORMAL | HOLD | 0 | 2729.4 | 2.29x | ❌ |
| ST-002 | Q4 2018 Selloff | TRUE_POSITIVE | HIGH | NORMAL | HOLD | 0 | 2729.4 | 2.29x | ❌ |
| ST-003 | Vaccine Rally | TRUE_NEGATIVE | CRITICAL | NORMAL | HOLD | 0 | 2729.4 | 2.29x | ✅ |
| ST-004 | Calm Bull Market 2019 | TRUE_NEGATIVE | HIGH | NORMAL | HOLD | 0 | 2729.4 | 2.29x | ✅ |
| ST-005 | 2022 Fed Bear Market | TRUE_POSITIVE | HIGH | NORMAL | HOLD | 0 | 2729.4 | 2.29x | ❌ |

---

## Detailed Results

### ST-001 — COVID Crash ❌
**Test Type:** TRUE_POSITIVE
**Description:** Macro-driven sharp crash. Primary BSSC validation event.

#### Check Results
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Market State | BLACK_SWAN | NORMAL | ❌ |
| Recommended Action | LIQUIDATE | HOLD | ❌ |
| Breach Duration | >= 15 days | 0 days | ❌ |
| JD/GBM Ratio | >= 1.5x | 2.29x | ✅ |
| Pipeline Consistency | No warnings | 0 warnings | ✅ |

#### Key Metrics
- Entropy Delta: -0.0346
- Peak Entropy Date: 2020-02-03
- Mean Slippage: 490.58 bps
- P99 Slippage: 2729.41 bps

### ST-002 — Q4 2018 Selloff ❌
**Test Type:** TRUE_POSITIVE
**Description:** Fed rate hike driven selloff. Moderate crisis, slower development.

#### Check Results
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Market State | STRESS or BLACK_SWAN | NORMAL | ❌ |
| Recommended Action | REDUCE or LIQUIDATE | HOLD | ❌ |
| Breach Duration | >= 8 days | 0 days | ❌ |
| JD/GBM Ratio | >= 1.5x | 2.29x | ✅ |
| Pipeline Consistency | No warnings | 0 warnings | ✅ |

#### Key Metrics
- Entropy Delta: -0.0233
- Peak Entropy Date: 2018-10-05
- Mean Slippage: 490.58 bps
- P99 Slippage: 2729.41 bps

### ST-003 — Vaccine Rally ✅
**Test Type:** TRUE_NEGATIVE
**Description:** Pfizer announcement Nov 2020. Positive shock — must NOT fire BLACK_SWAN. Critical false positive test.

#### Check Results
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Market State | NORMAL or STRESS | NORMAL | ✅ |
| Recommended Action | HOLD or REDUCE | HOLD | ✅ |
| Breach Duration | <= 6 days | 0 days | ✅ |
| JD/GBM Ratio | >= 1.5x | 2.29x | ✅ |
| Pipeline Consistency | No warnings | 0 warnings | ✅ |

#### Key Metrics
- Entropy Delta: 0.0601
- Peak Entropy Date: 2020-11-09
- Mean Slippage: 490.58 bps
- P99 Slippage: 2729.41 bps

### ST-004 — Calm Bull Market 2019 ✅
**Test Type:** TRUE_NEGATIVE
**Description:** Low volatility bull market H1 2019. Baseline test — system must stay silent.

#### Check Results
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Market State | NORMAL | NORMAL | ✅ |
| Recommended Action | HOLD | HOLD | ✅ |
| Breach Duration | <= 3 days | 0 days | ✅ |
| JD/GBM Ratio | >= 1.5x | 2.29x | ✅ |
| Pipeline Consistency | No warnings | 0 warnings | ✅ |

#### Key Metrics
- Entropy Delta: -0.0331
- Peak Entropy Date: 2019-02-22
- Mean Slippage: 490.58 bps
- P99 Slippage: 2729.41 bps

### ST-005 — 2022 Fed Bear Market ❌
**Test Type:** TRUE_POSITIVE
**Description:** Policy-driven slow bear market. Tests structural stress detection without a single crash day.

#### Check Results
| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Market State | STRESS or BLACK_SWAN | NORMAL | ❌ |
| Recommended Action | REDUCE or LIQUIDATE | HOLD | ❌ |
| Breach Duration | >= 8 days | 0 days | ❌ |
| JD/GBM Ratio | >= 1.5x | 2.29x | ✅ |
| Pipeline Consistency | No warnings | 0 warnings | ✅ |

#### Key Metrics
- Entropy Delta: 0.0541
- Peak Entropy Date: 2022-03-11
- Mean Slippage: 490.58 bps
- P99 Slippage: 2729.41 bps

---

## Stability Check (ST-001 COVID — 3 Runs)

| Run | P99 Slippage (bps) |
|-----|-------------------|
| 1 | 2729.41 |
| 2 | 2729.41 |
| 3 | 2729.41 |

**Variance:** 0.00% — STABLE

---

## Failed Scenarios

### ST-001 — COVID Crash
- **Failed:** Market State
  - Expected: BLACK_SWAN
  - Actual: NORMAL
- **Failed:** Recommended Action
  - Expected: LIQUIDATE
  - Actual: HOLD
- **Failed:** Breach Duration
  - Expected: >= 15 days
  - Actual: 0 days
- *Suggested investigation:* Check models.py defaults, or entropy baseline calibration.

### ST-002 — Q4 2018 Selloff
- **Failed:** Market State
  - Expected: STRESS or BLACK_SWAN
  - Actual: NORMAL
- **Failed:** Recommended Action
  - Expected: REDUCE or LIQUIDATE
  - Actual: HOLD
- **Failed:** Breach Duration
  - Expected: >= 8 days
  - Actual: 0 days
- *Suggested investigation:* Check models.py defaults, or entropy baseline calibration.

### ST-005 — 2022 Fed Bear Market
- **Failed:** Market State
  - Expected: STRESS or BLACK_SWAN
  - Actual: NORMAL
- **Failed:** Recommended Action
  - Expected: REDUCE or LIQUIDATE
  - Actual: HOLD
- **Failed:** Breach Duration
  - Expected: >= 8 days
  - Actual: 0 days
- *Suggested investigation:* Check models.py defaults, or entropy baseline calibration.

---

## WandB Run
https://wandb.ai/sharanshpandeyxib-adgitm/CRIS/runs/vvyrthi8

---
*Generated by CRIS Layer 3 Stress Test*
*Layer 3 BSSC v1.0*
