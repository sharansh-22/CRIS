# CRIS — Test Results

## TS-001: Entropy Method Selection

**WandB Project:** `CRIS` · **Run Name:** `TS-001-Entropy-Comparison`  
**WandB Dashboard:** [View Run](https://wandb.ai/sharanshpandeyxib-adgitm/CRIS/runs/wbmhteam)  
**Source of Truth:** `data/simulation_output/entropy_method_selection.json`

### A. Performance Metrics

All values logged live to WandB by `entropy_comparison.py` — extracted from `run_entropy_method_selection()`.

| Method | Lead Time | FP Rate/mo | Magnitude (σ) | Consistency | Composite |
|--------|-----------|------------|----------------|-------------|-----------|
| Shannon | 0.0 days | 0.000 | 0.088 | 0.67 | 0.377 |
| Permutation | 0.0 days | 0.000 | 0.178 | 0.67 | 0.400 |
| **Sample** | **35.0 days** | 2.174 | **0.832** | **1.00** | **0.700** |
| Tsallis | 0.0 days | 0.149 | 0.045 | 0.67 | 0.346 |

### B. Final Ranking
1. 🥇 **Sample Entropy** — Composite: 0.700
2. 🥈 **Permutation Entropy** — Composite: 0.400
3. 🥉 Shannon Entropy — Composite: 0.377
4. Tsallis Entropy — Composite: 0.346

### C. Decision

**Primary Method: Sample Entropy** (score: 0.700)  
**Confirmation Method: Permutation Entropy** (score: 0.400)

### D. Key Findings

1. **Sample Entropy dominates** — 35-day average lead time across 3 crises. No other method breached the stress threshold before the crisis trough.
2. **Perfect directional consistency** — Sample Entropy rose in all 3 events (consistency = 1.00).
3. **Permutation complements, not duplicates** — Low correlation (< 0.7) with Sample ensures independent validation.
4. **Shannon and Tsallis are noise** — Zero lead time in all events, low composite scores, rejected.

### E. Rejected Methods
| Method | Composite | Reason |
|--------|-----------|--------|
| Shannon | 0.377 | Zero lead time, did not meet selection criteria |
| Tsallis | 0.346 | Zero lead time, did not meet selection criteria |

### F. System Metrics (from WandB)

| Metric | Value |
|--------|-------|
| System RAM Total | 15.3 GB |
| RAM Used | 40.8% |
| Process RSS | 375.6 MB |

### G. WandB Dashboard

**Project:** [CRIS](https://wandb.ai/sharanshpandeyxib-adgitm/CRIS)  
**Run:** [TS-001-Entropy-Comparison](https://wandb.ai/sharanshpandeyxib-adgitm/CRIS/runs/wbmhteam)

Interactive dashboard includes: full metrics table, ranking chart, composite score bar chart, lead time bar chart, FP rate bar chart, decision summary, and system resource tracking.
