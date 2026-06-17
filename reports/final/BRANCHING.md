# CRIS — Branching Strategy

## Core Rules
- `main` is always stable and verified.
- Layer logic resides on dedicated `feature/` branches during research.
- Input data schemas are centralized in `data/` on `main`.
- Generated probabilistic reports and validation artifacts reside within layer sub-directories.

## Branch Naming
| Branch | Focus Area |
|--------|------------|
| `feature/layer3` | Layer 3 Interpretation Engine |
| `feature/mmad` | Layer 2 MMAD |
| `feature/signal` | Layer 1 Signal Harvester |
| `feature/credit` | Layer 4 Credit Risk |
| `feature/convergence` | Multi-Layer Convergence Coordinator |
| `feature/dashboard` | Risk Dynamics Visualization |

## Validation Hierarchy
- `CRIS_TESTS.md`: System-level behavioral stress tests.
- `CRIS_TEST_RESULTS.md`: System-level validation scores.
- `layer*/audit/`: Detailed adversarial validation and institutional reports.

## Documentation on main
- `docs/*_ADR.md`: Architecture Decision Records (ADRs) documenting major research pivots.
- `docs/*_FLOW.md`: Structural dynamics and theory per layer.

## Merge Requirements
- All structural engines complete and decoupled.
- All adversarial audits (`audit_v*.py`) passing.
- Institutional validation reports updated.
- `CRIS_TESTS.md` and `CRIS_TEST_RESULTS.md` synchronized.

## Workflow Pipeline
1. Research and architecture hardening on `feature/` branch.
2. Adversarial audit and falsification testing.
3. Generation of institutional validation reports.
4. Merge to `main` via Squash & Merge.
5. Update `CRIS_TREE.md` to reflect finalized integration.
