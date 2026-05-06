# Adaptive Calibration Implementation Report

## 1. Executive Summary

Implemented a governed adaptive calibration system for CRIS Layer 3 focused only on slow environmental normalization. The implementation distinguishes environmental context from fixed stress semantics:

- Environmental anchors may adapt slowly: volatility baseline, FAST permutation-entropy baseline, and SLOW sample-entropy baseline.
- Stress semantics remain fixed: FAST/SLOW/TRAJECTORY identities, stress multipliers, convergence logic, uncertainty semantics, and probability mappings are unchanged.

The system behaves like slow climate adaptation rather than weather reaction. Current inference uses the previously approved anchors; candidate updates are evaluated only after inference and can affect only future calls. This prevents same-observation threshold chasing.

## 2. Environmental Anchor Architecture

New module:

- `layer3/calibration.py`

New state:

- `CalibrationState`

Maintained anchors:

- `volatility_anchor`
- `fast_perm_entropy_anchor`
- `slow_sample_entropy_anchor`

The anchors are versioned, bounded, and exposed through `Layer3Output.calibration` metadata. The detector APIs remain unchanged; the orchestrator passes approved anchors through the existing `baseline_vol`, `baseline_perm`, and `baseline_sample` pathways.

No stress thresholds, model weights, confidence floors, convergence caps, or trajectory semantics adapt.

## 3. Stress-Filtering Logic

Candidate anchors are computed from raw long-horizon returns using robust environmental statistics:

- Volatility uses filtered/winsorized absolute returns.
- FAST entropy uses long-horizon rolling permutation entropy.
- SLOW entropy uses long-horizon rolling sample entropy.

Extreme absolute-return observations are filtered from environmental volatility candidates so crisis spikes do not dominate the definition of normal. If filtering leaves too little data, the system falls back to winsorization rather than accepting distorted raw crisis tails.

This is environmental filtering, not label learning. It does not use PnL, future outcomes, crisis labels, or prediction performance.

## 4. Calibration Freeze Logic

Calibration updates freeze when current stress context is extreme:

- FAST shock >= `0.85`
- SLOW structural stress >= `0.80`
- TRAJECTORY erosion >= `0.80`
- uncertainty pressure >= `0.70`
- optional systemic stress >= `0.70`

Freeze behavior:

- Candidate anchors may still be recorded for audit.
- Approved anchors do not move.
- `freeze_active` and `freeze_reason` are stored in calibration metadata.

Purpose: prevent the system from learning `crisis = normal`.

## 5. Adaptation-Velocity Governance

Approved anchors cannot move faster than `5%` per governed update. This cap applies separately to volatility, FAST entropy, and SLOW entropy anchors.

Update cadence:

- Minimum history: `756` trading days.
- Governed update interval: `63` observations, approximately quarterly.
- Initial long-horizon update is allowed once sufficient history exists.

Anchor bounds:

- Volatility anchor: `[0.0005, 0.08]`
- Entropy anchors: `[0.05, 1.0]`

This prevents abrupt jumps such as `15 -> 19` and instead enforces gradual movement such as `15 -> 15.75`.

## 6. Multi-Horizon Memory Design

Candidate anchors blend three environmental horizons:

- Short anchor: `252` trading days
- Medium anchor: `756` trading days
- Deep anchor: `1260` trading days

Blend weights:

- Short: `10%`
- Medium: `30%`
- Deep: `60%`

The deep anchor dominates when available. This avoids single-era bias while still allowing slow recognition that the market environment has changed.

If callers provide only short rolling slices, calibration remains inert. This preserves current walk-forward behavior unless the caller supplies sufficient environmental history.

## 7. Versioning & Auditability

`CalibrationState` tracks:

- `anchor_version`
- `effective_date`
- approved anchors
- `calibration_window_days`
- `observations_since_update`
- `freeze_active`
- `freeze_reason`
- `last_candidate`
- `last_update_approved`
- `last_update_reason`
- `max_allowed_change`

`Layer3Output` now includes optional `calibration` metadata showing the anchors used for that specific output. Historical outputs can therefore be reconstructed and interpreted with the exact anchor version used at inference time.

## 8. Validation Results

Targeted calibration governance suite:

- `conda run -n CRIS python layer3/validation/calibration_tests.py`
- Result: `5 passed, 0 failed`

Full Layer 3 behavioral suite:

- `conda run -n CRIS python layer3/validation/behavioral_suite.py`
- Result: `20 passed, 0 failed`

The full suite verified that existing stress interpretation, recovery behavior, oscillation stability, uncertainty behavior, and raw-data dominance remained intact after the calibration hook was added.

## 9. Crisis-Normalization Resistance Findings

The new validation suite confirmed:

- Calibration freezes during extreme FAST stress.
- Frozen calibration does not absorb crisis volatility.
- Long high-volatility samples cannot move anchors abruptly when stress context is elevated.

This directly addresses the main institutional risk of adaptive baselines: accidentally normalizing crisis conditions.

## 10. Cross-Era Stability Findings

The multi-horizon candidate test confirmed that a recent high-volatility year does not dominate the blended environmental anchor when deeper history exists.

The bounded-velocity test confirmed that even when long-horizon data supports a different environmental baseline, the approved anchor moves only within the governed cap.

The insufficient-history test confirmed that short samples do not trigger adaptive calibration.

## 11. Remaining Calibration Risks

Slow adaptation will lag genuine structural change. This is intentional; the system prioritizes semantic stability over fast adaptation.

Stress freezes can delay recognition of a permanently changed volatility environment if stress remains elevated for a long time. This should be handled through governance review, not automatic threshold chasing.

Entropy anchors are estimator-sensitive and should remain more conservatively governed than volatility anchors.

Callers must provide sufficient long-horizon history for adaptive calibration to activate. Existing short-window usage remains stable but will not benefit from environmental adaptation.

## 12. Interpretability Preservation Findings

For every adaptive mechanism:

- What adapts: environmental baseline anchors only.
- Why it adapts: long-term normalization realism.
- How fast it adapts: quarterly cadence with 5% capped movement.
- What constrains it: hard bounds, minimum history, robust statistics, and freeze logic.
- What freezes it: extreme FAST/SLOW/TRAJECTORY stress, high uncertainty, or optional systemic stress.
- What prevents runaway drift: no outcome optimization, no recursive feedback, no online learning, and no stress-threshold adaptation.

The implementation remains explainable and auditable through `Layer3Output.calibration`.

## 13. Final Institutional Calibration Assessment

CRIS Layer 3 now has a lightweight adaptive calibration system that slowly recognizes environmental evolution while preserving fixed stress semantics.

The implementation does not create adaptive AI, predictive optimization, fast threshold chasing, hidden regimes, or recursive learning loops. It improves institutional realism by letting background anchors evolve only under strict governance.

Final assessment: the system now better reflects that the world evolves, while preserving CRIS's bounded, interpretable, probabilistic architecture.
