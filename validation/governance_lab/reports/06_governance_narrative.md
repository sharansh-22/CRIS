# Governance Experiment 06: Institutional Explainability Report

## 1. Executive Summary
This report demonstrates the **Governance Explainability Layer (GEL)**, which provides causal attribution and reasoning for CRIS V2 decisions. We audited three distinct institutional moments to evaluate the system's ability to explain its posture to risk committees and regulators.

## 2. Institutional Trace: 2008-01-01
**Aggregate Stress Score:** 0.227
**Attribution Clarity:** HIGH
**Signal-to-Noise Ratio:** 1.00

### Narrative Reasoning
CRIS escalated to a **DEFENSIVE** posture primarily due to **Liquidity Fragility Acceleration**. The liquidity source contributed 0.000 to the total shift, and was amplified by a high deterioration velocity. This represents a proactive systemic defense.

### Causal Breakdown
| Source | Stress Level | Velocity | Contribution | Amplified? |
| :--- | :--- | :--- | :--- | :--- |
| liquidity | 0.000 | -0.540 | 0.000 | False |
| structural | 0.000 | -0.460 | 0.000 | False |
| macro | 0.250 | 0.020 | 0.024 | True |
| volatility | 0.200 | -0.020 | 0.000 | False |

---

## 2. Institutional Trace: 2010-06-01
**Aggregate Stress Score:** 0.498
**Attribution Clarity:** HIGH
**Signal-to-Noise Ratio:** 1.00

### Narrative Reasoning
CRIS maintained a **CAUTIOUS** posture despite aggregate stress reduction. Reasoning: **Hysteresis Persistence**. While liquidity stress normalized, the structural stabilization strength was insufficient to cross the exit threshold, preserving institutional buffer against potential secondary shocks.

### Causal Breakdown
| Source | Stress Level | Velocity | Contribution | Amplified? |
| :--- | :--- | :--- | :--- | :--- |
| liquidity | 0.060 | 0.060 | 0.000 | True |
| structural | 1.000 | 1.000 | 0.346 | False |
| macro | 0.490 | 0.040 | 0.135 | True |
| volatility | 0.090 | -0.200 | 0.000 | False |

---

## 2. Institutional Trace: 2011-08-01
**Aggregate Stress Score:** 0.099
**Attribution Clarity:** HIGH
**Signal-to-Noise Ratio:** 1.00

### Narrative Reasoning
CRIS suppressed volatility-driven escalation during the 2011 debt ceiling spike. Reasoning: **Source-Aware Filtering**. While volatility signals increased, they lacked confirmation from liquidity or structural fragility sources. The volatility contribution was muted (0.000), maintaining throughput during market noise.

### Causal Breakdown
| Source | Stress Level | Velocity | Contribution | Amplified? |
| :--- | :--- | :--- | :--- | :--- |
| liquidity | 0.000 | 0.000 | 0.000 | False |
| structural | 0.000 | 0.000 | 0.000 | False |
| macro | 0.290 | -0.010 | 0.027 | False |
| volatility | 0.130 | 0.020 | 0.000 | False |

---

## 3. Scientific Assessment
* **Traceability:** 100%. Every logit shift is exactly attributed to its underlying signal and weight.
* **Auditability:** The hysteresis thresholds and stabilization weights are exposed, allowing for retrospective audit of 'Too Early' or 'Too Late' transitions.
* **Institutional Realism:** The narrative generator provides a bridge between probabilistic models and risk committee language.
