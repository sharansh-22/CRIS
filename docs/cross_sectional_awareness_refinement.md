# Cross-Sectional Awareness Refinement Report

## 1. Executive Summary

CRIS Layer 3 is mature as a single-asset probabilistic interpretation engine. It can distinguish FAST shock, SLOW structural instability, TRAJECTORY deterioration, and META uncertainty within one instrument. It is also cross-asset normalized in the sense that equivalent relative shocks can produce comparable probabilities across assets with different baseline volatility.

That is not the same as cross-sectional systemic awareness. The current architecture does not yet reason about whether stress is isolated, synchronized, propagating, sector-clustered, or fragmenting across a market universe. This creates an institutional limitation because systemic risk often appears through relationships before it is fully visible in any single asset's local stress path.

Recommended direction: add a lightweight, modular, observational cross-sectional layer outside Layer 3 internals. It should consume per-asset `Layer3Output` streams plus raw return panels, then emit bounded systemic context fields such as `contagion_intensity`, `synchronization_pressure`, `fragmentation_pressure`, `systemic_fragility_pressure`, and `systemic_uncertainty_pressure`. These outputs should inform downstream governance, not rewrite FAST/SLOW/TRAJECTORY probabilities.

## 2. Current Cross-Sectional Limitations

Layer 3 currently reasons primarily within one asset. FAST reads that asset's shock intensity and entropy breakdown. SLOW reads that asset's volatility and entropy persistence. TRAJECTORY reads that asset's recovery half-lives, failed rebounds, and holding failure. META coordinates those fields for that asset.

Existing cross-asset work mainly validates normalization. The `cross_asset_calibration_audit.py` checks robustness across volatility scales, but it does not model inter-asset propagation. A TSLA-like asset and SPY-like asset can be made comparable, but CRIS does not yet ask whether TSLA stress is spreading into semiconductors, indices, credit proxies, or defensive sectors.

The documentation already points toward multi-asset cross-correlation in `docs/future_work.md`, and `docs/architecture_overview.md` mentions cross-sectional correlation clusters as part of SLOW's intended signal language. In code, however, SLOW remains asset-local. This is a documentation-to-implementation gap, not a design flaw.

Current blind points:

- Sector contagion is not distinguished from isolated asset deterioration.
- Cross-asset stress propagation is not measured.
- Correlation concentration and dispersion collapse are not represented.
- Asynchronous stress can look like unrelated single-asset events.
- Liquidity spillover is inferred only through local FAST liquidity disruption.
- Index-level synchronization is not separated from constituent-level deterioration.
- Systemic healing is not tracked after broad contagion.

## 3. Systemic Blindspot Findings

CRIS can identify that Asset A is deteriorating and Asset B is deteriorating, but it does not yet model whether A and B are deteriorating together in a structurally meaningful way.

Asset deterioration means the local Layer 3 fields are stressed. Network deterioration means many assets, sectors, or macro proxies are becoming jointly fragile, more correlated under stress, or more fragmented in ways that reduce diversification reliability.

The current architecture may understate systemic seriousness when many assets show moderate but synchronized stress. No single asset needs to show extreme Layer 3 risk for a portfolio-level fragility event to be meaningful.

It may also overstate systemic seriousness if one large asset is under severe stress while the broader market remains dispersed and resilient. A cross-sectional layer should help distinguish concentrated idiosyncratic stress from spreading systemic stress.

Macro shock diffusion is especially underrepresented. Layer 3 can interpret the local effects after a shock reaches each instrument, but it does not map the order, breadth, or coherence of propagation across a universe.

## 4. Relationship-Awareness Findings

Most valuable relationship signals are structural, not predictive:

- Stress breadth: how many assets have elevated FAST/SLOW/TRAJECTORY fields.
- Stress co-movement: whether elevated fields are rising together.
- Correlation concentration: whether pairwise or sector-level correlations compress toward one dominant movement.
- Dispersion collapse: whether diversification is failing because assets stop behaving independently.
- Sector synchronization: whether stress is clustered inside a sector or spreading across sectors.
- Volatility contagion: whether elevated local volatility appears sequentially across related groups.
- Synchronized rebound failure: whether multiple assets recover poorly after shared stress.
- Cross-sectional uncertainty: whether assets disagree sharply about the state of the system.
- Propagation order: whether stress migrates from leaders to peers to broad indices.

Lower priority mechanisms:

- Large dense covariance matrices updated everywhere.
- Fully pairwise asset graphs for all names.
- Prediction-oriented lead-lag networks.
- Black-box graph embeddings.
- Outcome-optimized contagion scores.

The highest-value path is to summarize relationship structure into a small number of interpretable systemic fields.

## 5. Contagion Modeling Recommendations

Model contagion as observed propagation, not prediction. The systemic layer should ask:

- Is stress broadening?
- Is stress synchronizing?
- Is stress moving from one cluster to another?
- Are recoveries failing together?
- Is diversification reliability falling?

Recommended contagion primitives:

- `stress_breadth`: fraction of assets with elevated Layer 3 stress fields.
- `stress_intensity_breadth`: cross-sectional average of bounded local stress, optionally sector-weighted.
- `contagion_intensity`: bounded measure of stress broadening across sectors over a medium window.
- `synchronization_pressure`: bounded measure of rising co-movement among stressed assets.
- `rebound_failure_breadth`: fraction of assets with elevated TRAJECTORY rebound failure after shared stress.
- `liquidity_spillover_pressure`: breadth of elevated FAST liquidity disruption.

Keep these primitives observational. They should not feed back into Layer 3 engines. They should not adjust per-asset shock intensity, structural instability, or trajectory erosion. They can provide a systemic overlay and downstream governance context.

Avoid recursive network simulation. CRIS should not attempt to simulate cascades asset by asset. It should measure whether cascading behavior is already becoming visible in probabilistic outputs and return relationships.

## 6. Correlation-Breakdown Findings

Rising correlation concentration is meaningful when it occurs together with rising stress breadth, elevated FAST/SLOW probabilities, or falling dispersion. High correlation during calm index drift is less concerning than high correlation during drawdowns, failed rebounds, or liquidity disruption.

Sudden decorrelation is ambiguous. It can mean fragmentation, rotation, idiosyncratic repricing, or data noise. It should raise systemic uncertainty only when paired with elevated dispersion, inconsistent Layer 3 fields, or sector-level stress disagreement.

Synchronized panic is meaningful when many assets show high FAST shock or liquidity disruption simultaneously and correlations compress. That implies diversification is failing under stress.

Sector-wide deterioration is meaningful when SLOW and TRAJECTORY stress broaden inside a sector even without a market-wide FAST event. This captures grinding systemic weakness that would be missed by index-only readings.

Flight-to-safety behavior should not be treated as simple decorrelation. A defensive asset rallying while risk assets deteriorate may be coherent systemic stress, not fragmentation. The systemic layer should distinguish "protective divergence" from incoherent fragmentation by using asset-group roles.

Statistical noise is likely when correlation changes occur over short windows, small universes, missing data, or low-volatility drift with no Layer 3 stress confirmation.

## 7. Systemic-Uncertainty Findings

Systemic uncertainty should rise during fragmented stress: some sectors panic, others remain calm, and cross-sectional relationships are unstable. This is different from single-asset uncertainty because the uncertainty is about propagation and diversification reliability.

Systemic uncertainty should also rise during unstable synchronization. When many assets suddenly move together under stress, the system may be clear that risk is elevated but uncertain about how broad and persistent the systemic event will become.

Partial contagion should produce moderate uncertainty. A sector cluster under stress with healthy outside sectors is not fully systemic, but the propagation boundary matters.

Asynchronous stress should increase uncertainty when stress migrates over time across clusters. This can indicate diffusion even if no single day appears broadly synchronized.

Recommended output: `systemic_uncertainty_pressure`, bounded in `[0, 1]`, driven by disagreement among sector clusters, unstable correlation structure, and divergence between asset-local Layer 3 fields and cross-sectional co-movement.

Do not create hidden systemic regimes such as "contagion mode" or "fragmentation mode." Use continuous pressures.

## 8. Temporal-Systemic-Memory Findings

Systemic stress should retain lightweight memory. Broad contagion affects institutional behavior after the initial event because allocators distrust diversification, liquidity, and rebound quality for some time.

Recommended memory primitives:

- `contagion_memory`: bounded residue after broad stress synchronization.
- `synchronization_residue`: slow-decaying memory of recent correlation concentration under stress.
- `fragmentation_inertia`: persistence of unstable sector disagreement.
- `systemic_healing_lag`: slow recovery metric after broad rebound failure.

These should be convergence-like memory fields for the systemic layer only. They should not modify local Layer 3 engine outputs and should not recursively amplify one another.

Systemic healing should require breadth improvement, correlation normalization, and reduced rebound failure breadth. It should not reset immediately because index price recovered for a few days.

## 9. Recommended Cross-Sectional Architecture

Add a modular `systemic_context` layer outside Layer 3 internals:

```text
asset returns panel ───────────────┐
sector/group metadata ────────────┤
per-asset Layer3Output streams ───┤
                                   └── Systemic Context Overlay
                                           ├── contagion_intensity
                                           ├── synchronization_pressure
                                           ├── fragmentation_pressure
                                           ├── systemic_fragility_pressure
                                           ├── systemic_uncertainty_pressure
                                           └── systemic_healing_lag
```

Inputs:

- Per-asset Layer 3 outputs.
- Return panel for a governed universe.
- Optional sector/group map.
- Optional liquidity bucket metadata.

Outputs:

- Continuous bounded systemic pressures.
- Sector-level summaries.
- Top contributing clusters.
- Calibration and coverage metadata.

Design constraints:

- Observational, not predictive.
- Additive, not entangled with Layer 3 engines.
- Modular, so Layer 3 remains independently usable.
- Bounded outputs in `[0, 1]`.
- No dense graph ML.
- No recursive feedback into asset-level probabilities.
- No automatic trading actions.

Implementation should begin with sector/group aggregation rather than all-pairs networks. A small number of interpretable cross-sectional summaries will be more governable than a giant relationship matrix.

## 10. Downstream Governance Value

Systemic outputs are useful for Layer 4 and portfolio governance because they describe diversification reliability and propagation pressure, not asset direction.

Most useful downstream fields:

- `systemic_fragility_pressure`: portfolio-level structural vulnerability.
- `contagion_intensity`: breadth and speed of stress propagation.
- `synchronization_pressure`: likelihood that diversification is failing under stress.
- `fragmentation_pressure`: degree of unstable cross-sectional disagreement.
- `rebound_failure_breadth`: how many assets are failing to recover together.
- `systemic_uncertainty_pressure`: confidence penalty for allocation decisions.

Operational uses:

- Exposure throttling when systemic fragility is elevated.
- Concentration review when stress clusters inside related groups.
- Diversification haircut when synchronization pressure is high.
- Risk-budget tightening when contagion intensity is broadening.
- Neutrality preference when fragmentation pressure and uncertainty are high.

These outputs should guide governance posture, not predict next-period returns.

## 11. Interpretability & Governance Safeguards

For each proposed mechanism:

- `stress_breadth` measures how widespread local Layer 3 stress is. It does not predict direction.
- `contagion_intensity` measures broadening of stress across groups. It does not simulate future cascade paths.
- `synchronization_pressure` measures co-movement under stress. It does not claim all correlations are permanently changed.
- `fragmentation_pressure` measures unstable disagreement across groups. It does not classify market regimes.
- `rebound_failure_breadth` measures shared recovery weakness. It does not forecast another drawdown.
- `systemic_uncertainty_pressure` measures ambiguity in relationship structure. It does not override local asset interpretation.

What propagates:

- Only bounded systemic context to downstream governance.
- No systemic score should overwrite local FAST/SLOW/TRAJECTORY fields.

What persists:

- Lightweight systemic memory residues after broad contagion, unstable synchronization, or fragmented stress.

What is bounded:

- All systemic outputs and memories remain in `[0, 1]`.
- Sector and universe coverage constraints should cap confidence.
- Missing data should raise uncertainty rather than force conclusions.

What prevents runaway recursion:

- The systemic layer consumes Layer 3 outputs but does not feed back into Layer 3.
- Memory fields update from current observed cross-sectional inputs, not from each other.
- No optimization loop, no outcome learning, no graph neural network, no hidden regimes.

## 12. Anti-Overfitting Safeguards

Do not build all-pairs dependency models as the default architecture.

Do not tune relationship thresholds to historical crisis labels.

Do not optimize systemic outputs for drawdown reduction, Sharpe improvement, or allocation performance.

Do not introduce learned graph embeddings.

Do not infer causality from correlation.

Do not treat every correlation spike as systemic stress.

Do not allow small-universe artifacts to produce high-confidence systemic conclusions.

Do not let systemic context change per-asset Layer 3 probabilities.

Do not create discrete regimes such as "contagion regime" or "fragmented regime." Preserve continuous pressures.

## 13. Remaining Systemic Risks

Universe definition risk: systemic conclusions depend on which assets are included. A narrow universe can falsely imply broad contagion or miss external propagation.

Sector taxonomy risk: poor grouping can hide real transmission channels. Governance should support stable, reviewed group mappings.

Data quality risk: missing returns, stale prices, and asynchronous holidays can distort correlation and breadth measures.

Correlation illusion risk: correlation can rise mechanically during volatility spikes without true structural transmission. Layer 3 stress confirmation should be required before correlation concentration becomes high systemic pressure.

Latency risk: a conservative systemic layer may recognize propagation after it is already visible. This is acceptable because the objective is interpretation, not prediction.

False comfort risk: low current synchronization does not guarantee low systemic risk if stress is latent or exogenous. The layer should preserve uncertainty when coverage is weak.

## 14. Final Cross-Sectional Maturity Assessment

CRIS is mature at asset-local probabilistic interpretation and cross-asset normalization. It is not yet mature at systemic relationship interpretation. The next architectural step should be a lightweight systemic context overlay that observes breadth, synchronization, fragmentation, contagion, and shared rebound failure across a governed universe.

This should not be a Layer 3 redesign. FAST, SLOW, TRAJECTORY, and META should remain stable. The systemic layer should sit above or beside Layer 3, consuming its outputs and adding bounded relationship context for downstream governance.

Final assessment: CRIS can evolve from single-asset probabilistic interpretation toward cross-sectional systemic awareness without becoming a predictive network AI. The right design is modular, observational, bounded, interpretable, and explicitly anti-recursive.
