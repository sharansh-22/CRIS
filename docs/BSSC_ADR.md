# BSSC — Architecture Decision Record
## Layer 3: Black Swan and Slippage Computer

### Initial Hypothesis
BSSC detects black swan events by measuring entropy
in market return distributions. High entropy signals
structural breakdown. Sample Entropy was selected
as primary method via empirical TS-001 test showing
35-day lead time across three historical events.

### Components Designed
- simulation.py  Merton Jump-Diffusion price paths
- entropy.py     Sample Entropy market state classifier
- slippage.py    Implementation Shortfall pipeline
- models.py      Pydantic v2 typed data contracts
- detector.py    Hybrid orchestrator and assembler
- report.py      Atomic JSON and markdown output

### Hypothesis Changes
#### Change 1 — Shannon vs Sample Entropy
Initial implementation used Shannon entropy.
Shannon falls during directional crashes because
returns concentrate on the left tail — concentrated
distributions have lower entropy.
Decision: replaced with Sample Entropy which rises
during self-similarity breakdown.
Status: RESOLVED

#### Change 2 — Sample Entropy Adaptation Problem
Sample Entropy also fails for sustained crashes.
During COVID after day 7, falling becomes the norm.
The market is highly self-similar (all large negative
returns). Sample Entropy detects this repetition and
reports LOW entropy — signal inverts.
TS-001's 35-day lead time was not reproducible.
Investigation confirmed entropy adapts to sustained
crashes and resets to baseline.
Status: UNDER INVESTIGATION

#### Change 3 — Threshold Direction
Data shows entropy moves DOWNWARD during crashes
not upward. Downward breach detection may be the
correct approach. Volatility ratio as alternative
primary signal under consideration.
Status: UNDER INVESTIGATION

### Current State
Entropy fix in progress on feature/bssc.
Stress tests: 2/5 passing.
Target: 5/5 before merge to main.

### Key Validated Results
- Merton JD kurtosis: 80.58 vs GBM 0.158 ✅
- Slippage JD/GBM ratio: 2.29x (gate >1.5x) ✅
- Pydantic runtime enforcement working ✅
- Atomic JSON persistence working ✅
