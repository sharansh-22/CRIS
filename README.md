# CRIS — Cascade Risk Intelligence System

> A four-layer financial risk intelligence system
> that detects black swan events before they
> materialize in markets.

## What CRIS Does
CRIS monitors four independent market domains simultaneously. It generates actionable execution insights and fires systemic alerts only when multiple domains mathematically confirm structural stress.

- **Layer 1 Signal** — Leading market indicators and technical price signals.
- **Layer 2 MMAD** — Macroeconomic, Monetary, and alternative Demographic data streams.
- **Layer 3 BSSC** — Black Swan detection and execution Slippage modeling via Jump-Diffusion.
- **Layer 4 Credit** — Corporate credit health and systemic debt market stress.
- **Convergence Engine** — Multi-layer alert generation combining signals from the four core layers.
- **Dashboard** — Centralized UI for active monitoring and risk breakdown.

## Repository Structure

CRIS follows a clean architecture model designed to prevent component bleeding:
- **Modular Branches:** Each sequence/layer is built independently on its own branch (e.g., `feature/bssc`). Branches only squash merge to `main` when comprehensively stress-tested and finalized.
- **Shared Data:** The `data/` folder is committed on `main` and contains all independent input structures (OHLCV, FRED, GDELT) so that any feature branch can pull raw metrics reliably.
- **Silod Outputs:** Generated metrics and simulation outputs live inside the respective layer's folder (e.g., `layer3_bssc/outputs/`). These only become visible on `main` after the sequence successfully completes.

> **Architecture & Testing Documentation:**
> - [CRIS_TREE.md](CRIS_TREE.md) — System-wide component tree and current development state.
> - [CRIS_TESTS.md](CRIS_TESTS.md) — System-level end-to-end stress tests and pass criteria.
> - [CRIS_TEST_RESULTS.md](CRIS_TEST_RESULTS.md) — Current benchmark scores for the overall system.
> - [BRANCHING.md](BRANCHING.md) — Strict guidelines on git workflows and module separation rules.

## Development Status

| Branch | Layer / Module | Status |
|--------|----------------|--------|
| `main` | **Layer 3 BSSC** | ✅ **COMPLETE** — 5/5 PASSED |
| `feature/mmad` | Layer 2 MMAD | Not Started |
| `feature/signal` | Layer 1 Signal | Not Started |
| `feature/credit` | Layer 4 Credit | Not Started |
| `feature/convergence` | Convergence Engine | Not Started |
| `feature/dashboard` | Dashboard | Not Started |

## Key Performance Metrics (Layer 3 BSSC)

The Layer 3 BSSC pipeline has been rigorously stress-tested across 5 major historical scenarios.

| Metric | Benchmark Result |
|--------|------------------|
| **Stress Test Pass Rate** | 100% (5/5 Scenarios) |
| **Detection Lead Time (COVID)** | 56 Days Before Peak |
| **Detection Lead Time (2018 Selloff)** | 11 Days Before Peak |
| **JD/GBM Slippage Ratio** | 2.29x - 2.39x (High Duress) |
| **Crisis P99 Slippage** | ~2624 - 2729 bps |
| **Monte Carlo Stability** | ✅ STABLE (0.00% Variance) |

### Out-of-Sample Findings
To ensure robustness, the system was tested against events outside its calibration set:
- **SVB Regional Bank Crisis (2023):** ✅ **PASSED**. Correctly identified liquidity duress and recommended `REDUCE`.
- **Late 2023 Bull Run:** ✅ **PASSED**. Maintained `NORMAL` / `HOLD` status, suppressing false positives.
- **August 2024 VIX Spike:** ⚠️ **LIMITATION**. The system's memory window (252-day) dampened sensitivity to this ultra-compressed 1-day shock. This confirms the need for Layer 2 (MMAD) and Layer 1 (Signal) integration for high-speed flash crash detection.
