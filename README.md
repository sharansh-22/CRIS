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
