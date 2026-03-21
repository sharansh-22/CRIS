# CRIS — Cascade Risk Intelligence System

> A four-layer financial risk intelligence system
> that detects black swan events before they
> materialize in markets.

## What CRIS Does
Monitors four independent domains simultaneously
and fires an alert only when multiple domains
confirm stress simultaneously:

- Layer 3 BSSC — Execution damage modeling
- Convergence Engine — Multi-layer alert generation

## Repository Structure
Each layer is built on its own branch.
Branches merge to main when complete.
data/ on main contains all shared input data.
Generated outputs live inside each layer folder
and become visible on main after branch merges.
See BRANCHING.md and CRIS_TREE.md for details.

## Branch Status
| Branch | Layer | Status |
|--------|-------|--------|
| feature/bssc | Layer 3 BSSC | In Progress |
| feature/mmad | Layer 2 MMAD | Not Started |
| feature/signal | Layer 1 Signal | Not Started |
| feature/credit | Layer 4 Credit | Not Started |
| feature/convergence | Convergence | Not Started |
| feature/dashboard | Dashboard | Not Started |
