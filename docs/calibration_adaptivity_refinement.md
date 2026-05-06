# Calibration Adaptivity Refinement Report

## 1. Executive Summary

CRIS Layer 3 currently uses a mix of asset-specific baseline inputs, fixed structural thresholds, and bounded convergence constants. This is institutionally understandable and stable, but partially exposed to historical staleness. A baseline calibrated to one volatility or entropy era can become too permissive in a high-volatility decade, too reactive in a low-volatility decade, or semantically misaligned as market microstructure evolves.

The refinement objective is not to make Layer 3 smarter, more predictive, or more reactive. The objective is slow environmental normalization: let baseline anchors adapt conservatively to long-run market structure while preserving fixed semantic meaning for stress multipliers, drawdown severity, uncertainty thresholds, and convergence governors.

Recommended direction: introduce a lightweight calibration-governance concept around long-horizon environmental anchors, primarily for `baseline_vol`, `baseline_perm`, and `baseline_sample`. These anchors should update slowly, remain bounded versus their prior approved values, and be explicitly frozen during acute stress. They should not optimize thresholds, learn from outcomes, or feed recursively from Layer 3 outputs.

## 2. Static Baseline Findings

Primary fixed baseline dependencies:

- `SPY_BASELINE_VOL = 0.00714`: default daily volatility anchor used by FAST and SLOW.
- `SPY_BASELINE_PERM_ENTROPY = 0.95`: default FAST permutation-entropy anchor.
- `ENTROPY_SAMPLE_BASELINE = 0.50`: default SLOW sample-entropy anchor.
- `VOL_STRESS_MULTIPLIER = 1.5` and `VOL_CRITICAL_MULTIPLIER = 3.0`: structural volatility-ratio semantics.
- `FAST_VOL_SPIKE_THRESHOLD = 2.0` and `FAST_VOL_EXTREME_THRESHOLD = 4.0`: FAST shock-ratio semantics.
- `ENTROPY_STRESS_THRESHOLD = 0.15` and `ENTROPY_CRITICAL_THRESHOLD = 0.30`: entropy-delta severity bands.
- Persistence constants such as `STRESS_PERSISTENCE_DAYS = 10`, `CRITICAL_PERSISTENCE_DAYS = 5`, `DECAY_PERSISTENCE_THRESHOLD = 30`, and `DECAY_MIN_CONFIDENCE_DAYS = 15`.
- TRAJECTORY fixed scales such as 2% drawdown detection, 30-day half-life normalization, 1.5% bounce detection, and drawdown severity constants.
- Convergence thresholds such as `UNCERTAINTY_DISAGREEMENT_THRESHOLD = 0.40`, `UNCERTAINTY_LOW_CONFIDENCE_THRESHOLD = 0.30`, `PARTNER_INFLUENCE_CAP = 0.05`, and `WEIGHT_SMOOTHING_ALPHA = 0.15`.
- Stabilization parameters such as `STABILIZATION_GROWTH_RATE = 0.05` and `STABILIZATION_SHOCK_PENALTY = 0.20`.

Static calibration remains acceptable for semantic constants that define interpretation philosophy: partner influence caps, LSTM influence caps, engine priors, persistence day counts, uncertainty disagreement bands, and drawdown/recovery semantics. These should not adapt automatically because they define the meaning of stress, not the market environment.

Slow adaptivity would improve realism for environmental anchors: baseline volatility, FAST entropy baseline, SLOW entropy baseline, and potentially long-run liquidity/slippage reference assumptions if execution-cost modules become operational. These are descriptive baselines, not stress definitions.

Adaptation would risk overfitting if applied to multipliers, stress thresholds, convergence caps, confidence floors, or trajectory severity definitions. Those values should remain governed constants unless changed through explicit research review.

## 3. Environmental Drift Findings

In a prolonged low-volatility secular environment, a stale high-volatility baseline can understate stress because current volatility ratios remain muted even when the environment is meaningfully unstable relative to its new norm. FAST and SLOW may both underreact to structural breaks from a compressed-volatility baseline.

In a structurally elevated volatility environment, a stale low-volatility baseline can overstate ordinary turbulence as persistent stress. FAST may repeatedly interpret normal elevated-volatility fluctuations as shock-like, while SLOW may maintain excessive structural stress.

Entropy distributions can also drift. Market structure changes, index composition, liquidity fragmentation, or persistent macro intervention can change the normal range of permutation and sample entropy. Static entropy baselines may interpret a new normal as either false disorder or false calm.

TRAJECTORY is less dependent on external baselines because it uses price-path recovery structure, drawdown duration, failed bounces, and participation asymmetry. Its fixed windows and severity scales are still historical assumptions, but they are more semantic than environmental. Adaptivity here should be avoided except through explicit long-horizon validation.

Convergence is mostly protected from environmental drift because it consumes normalized engine outputs. However, if baseline drift contaminates FAST/SLOW outputs, convergence inherits the bias. The correct solution is upstream environmental normalization, not convergence redesign.

## 4. Adaptive-Normalization Recommendations

Add slow environmental anchors only for baseline normalization inputs:

- Adaptive volatility anchor: long-horizon rolling median or winsorized mean of absolute daily returns.
- Adaptive FAST entropy anchor: long-horizon rolling median of short-window permutation entropy in non-stress history.
- Adaptive SLOW entropy anchor: long-horizon rolling median of rolling sample entropy in non-stress history.
- Optional calibration metadata: anchor value, approved floor/cap, last update date, effective window length, and freeze status.

Use long windows only. Reasonable horizons are 756 to 1260 trading days, or approximately three to five years. Quarterly recalculation is acceptable; daily anchor mutation is not. The system can compute daily candidate anchors, but production anchor adoption should occur on a governed schedule.

Prefer robust statistics over reactive EMAs. Rolling medians, trimmed means, and percentile bands are better than short-window averages because they resist crisis spikes and low-volatility compression artifacts.

Keep the existing stress multipliers as ratios to the adaptive anchor. For example, a 2.0 FAST vol ratio should still mean "twice the approved environmental baseline." Only the denominator evolves slowly; the shock semantics do not.

## 5. Governance & Adaptation Constraints

Recommended governance rules:

- Minimum calibration window: 756 trading days before an adaptive anchor is considered reliable.
- Preferred window: 1260 trading days for production-grade environmental calibration.
- Update cadence: quarterly or monthly at most; no intraday or short-window updates.
- Maximum anchor drift: cap approved anchor movement to a small percentage per update, such as 5% quarterly or 10% semiannually.
- Absolute guardrails: anchor values must remain within pre-approved asset-class ranges.
- Freeze during stress: do not update approved anchors when FAST, SLOW, TRAJECTORY, or uncertainty are elevated beyond governance thresholds.
- Freeze after discontinuities: pause anchor updates around splits, data-quality breaks, missing-data bursts, or structural feed changes.
- Audit trail: persist previous anchor, candidate anchor, approved anchor, window length, update date, and freeze reason.

The adaptation mechanism should observe raw historical data only. It should not use realized Layer 3 success, downstream actions, PnL, drawdown outcomes, or post-hoc labels. That separation prevents prediction optimization from entering baseline governance.

## 6. Temporal-Stability Findings

Adaptive normalization can destabilize convergence if anchors move too quickly, because engine outputs would shift for calibration reasons rather than market reasons. This is the central risk. Slow cadence, drift caps, and stress freezes are therefore mandatory.

Historical comparability is preserved if each output can be associated with the anchor version used at that timestamp. Without calibration metadata, a 0.70 FAST shock in 2020 and a 0.70 FAST shock in 2026 may become harder to compare.

Probabilistic consistency is preserved if adaptive anchors alter only environmental denominators, not probability mapping curves or convergence logic. A stress multiplier should retain the same semantic interpretation across periods.

Semantic continuity requires separating two ideas: "the environment has changed" and "the current market is stressed." Adaptive baselines should update only the first. The engines should still decide the second from current observations relative to the approved environment.

## 7. Cross-Period Robustness Findings

In a 2008-style environment, a slow anchor should not immediately absorb crisis volatility as normal. Freeze conditions are essential. After the crisis, the anchor may gradually recognize a structurally higher volatility environment only if elevated volatility persists beyond the freeze and long-horizon window.

In a 2020-style environment, the system should preserve acute FAST sensitivity during the shock, then avoid permanently treating the post-shock volatility distribution as abnormal if market structure remains elevated for years. Slow adaptation helps here, but only after crisis contamination is controlled.

In prolonged low-volatility eras, adaptive volatility anchors improve realism by preventing stale high baselines from desensitizing FAST and SLOW. Drift caps prevent the anchor from collapsing too far and turning every modest move into a shock.

In prolonged inflationary or macro-instability eras, adaptive anchors reduce chronic false alarm pressure by recognizing that volatility and entropy norms have structurally shifted. The system remains stress-aware because multipliers and persistence requirements are unchanged.

In structurally elevated volatility regimes, adaptive normalization improves interpretability by making outputs relative to the current institutional environment rather than an obsolete historical era.

## 8. Interpretability Preservation Findings

Each adaptive mechanism must answer five questions:

- What adapts: only the baseline anchor, not the stress rule.
- How fast it adapts: quarterly or monthly approval from multi-year history.
- Why it adapts: to normalize environmental drift, not improve forecasts.
- What constrains it: hard caps, minimum windows, robust statistics, asset-class bounds, and freeze conditions.
- What prevents runaway drift: no recursive dependence on Layer 3 outputs, no outcome optimization, and no short-window updates.

The adaptive state should be exposed as calibration metadata. Operators should be able to inspect the current volatility anchor, entropy anchors, drift from prior anchor, candidate anchor, and freeze reason.

Do not hide adaptivity inside detector math. If an anchor is adaptive, it should be named and governed as an environmental baseline.

## 9. Remaining Calibration Risks

Slow anchors may lag genuine structural change. That lag is acceptable and intentional; the alternative is market-chasing behavior.

Stress freezes may over-exclude data during long crisis eras, delaying recognition that the environment has permanently shifted. This can be mitigated with review-based override, not automatic self-learning.

Asset-specific anchors may reduce cross-asset comparability if each instrument adapts independently without asset-class governance. Calibration metadata and asset-class bounds are required.

Entropy anchors are more fragile than volatility anchors because entropy estimators are sensitive to window size, data quality, and missing observations. Entropy adaptivity should be more conservative than volatility adaptivity.

If anchor updates are not versioned, backtests and live outputs may become semantically mixed.

## 10. Anti-Overfitting Safeguards

Do not adapt stress multipliers, probability curves, confidence floors, convergence weights, partner influence caps, or LSTM influence caps.

Do not update baselines from short recent windows.

Do not optimize anchors to reduce false positives, improve drawdown response, improve Sharpe ratio, or match historical case labels.

Do not use downstream performance as an input.

Do not allow anchors to move during acute stress.

Do not create separate hidden regimes with different calibration constants.

Do not allow adaptive anchors to feed each other recursively.

Do not change the meaning of FAST, SLOW, or TRAJECTORY. Calibration adaptivity should only improve environmental normalization.

## 11. Recommended Adaptive Mechanisms

Recommended minimal mechanism:

1. Add a `CalibrationState` concept that stores approved baseline anchors and metadata.
2. Compute candidate anchors from long-horizon raw data using robust statistics.
3. Apply governance filters: minimum window, stress freeze, data-quality checks, drift caps, and asset-class bounds.
4. Pass approved anchors into the existing `run_layer3` baseline parameters.
5. Preserve current detector identities and convergence behavior.

Suggested anchors:

- `volatility_anchor`: rolling 3- to 5-year median absolute daily return, winsorized before aggregation.
- `fast_perm_entropy_anchor`: rolling 3- to 5-year median of FAST permutation entropy, updated more slowly than volatility.
- `slow_sample_entropy_anchor`: rolling 3- to 5-year median of SLOW sample entropy, with stricter missing-data checks.

Suggested metadata:

- `anchor_version`
- `effective_date`
- `calibration_window_days`
- `candidate_value`
- `approved_value`
- `prior_approved_value`
- `max_allowed_change`
- `freeze_active`
- `freeze_reason`

This is intentionally small. It creates environmental awareness without adding a calibration framework, optimizer, regime classifier, or self-learning loop.

## 12. Final Calibration-Stability Assessment

Layer 3 should not abandon fixed constants. Many constants are part of its interpretive grammar and should remain stable. The correct refinement is to distinguish environmental baselines from semantic stress definitions.

Final recommendation: make baseline volatility and entropy anchors slowly adaptive under strict governance, while keeping multipliers, persistence thresholds, convergence caps, and trajectory semantics fixed. This makes CRIS less historically fragile and more institutionally realistic without creating reactive intelligence, hidden regimes, or prediction-chasing behavior.

The architecture can become environmentally aware while remaining probabilistic, bounded, interpretable, decoupled, and calibration-stable.
