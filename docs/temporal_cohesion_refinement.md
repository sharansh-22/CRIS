# Temporal Cohesion Refinement Report

## 1. Executive Summary

Layer 3 is mathematically coherent and appropriately decoupled, but its temporal behavior can still feel more like three parallel observations than one continuously evolving probabilistic organism. The main gap is not signal intelligence or predictive calibration. The gap is contextual continuity: recent stress, failed healing, ambiguity, and recovery skepticism are represented in pieces, but they are not governed by a shared lightweight temporal memory primitive.

The current architecture already protects the right philosophy: independent FAST, SLOW, and TRAJECTORY engines; bounded 5% partner influence; no circular feedback; no hard regime machine; and uncertainty-aware convergence. Those constraints should remain intact. Refinement should focus on better use of convergence-local temporal context, especially post-shock scar memory, uncertainty inertia, and recovery skepticism residue.

Recommended refinement direction: add small bounded convergence-level memory fields that observe prior outputs but do not feed recursively into the engines. These fields should shape meta-dynamics and narrative continuity only, with explicit decay, caps, and non-recursive update rules.

## 2. Temporal Cohesion Findings

FAST -> SLOW continuity is structurally plausible but psychologically thin. FAST can cool rapidly after acute shock, while SLOW only retains memory through rolling volatility and consecutive breach streaks. Because SLOW persistence is based on current consecutive threshold breach, stress memory can disappear abruptly when the breach condition clears, even if the recent market path remains institutionally serious.

SLOW -> TRAJECTORY continuity is better because TRAJECTORY uses multi-horizon recovery half-life, failed rebound accumulation, participation quality, and holding failure. However, this continuity is still local to price windows. It does not explicitly know whether the current recovery is following a recent FAST panic or a prior SLOW structural episode, except through the price/return path itself.

Convergence-level continuity exists through smoothed weights, an evolution tracker, uncertainty pressure EMA, and stabilization strength. This is the correct location for shared continuity because it preserves engine independence. The weakness is that convergence currently stores history mainly as risk history, dominant-field history, and smoothed weights; it does not store a bounded contextual memory of recent instability severity or healing quality.

Scenario spot checks in the `CRIS` environment showed this texture clearly. In a sudden-shock path, FAST cooled to `0.00` by the end while TRAJECTORY remained elevated around `0.28` and uncertainty remained around `0.36`; that is coherent. But stabilization returned to `1.00`, which reads too healed relative to lingering trajectory skepticism. In recovery and double-dip paths, stabilization stayed near `0.00` through the final window, which is defensible under persistent stress but can feel binary rather than gradually healing.

## 3. Transition-Continuity Findings

The current transition logic tracks dominant-field changes over a rolling window and computes an evolution score from risk volatility, risk change, and dominant-field instability. This is valuable but observational. It helps identify active transition, yet it does not encode the emotional or institutional memory of the transition.

Dominant-field changes can therefore feel semantically abrupt. A sequence can move from FAST_SHOCK to MIXED to TRAJECTORY_DEGRADATION without a visible continuity primitive explaining that the later skepticism is the scar of the earlier shock. The math can be right while the narrative feels fragmented.

Weight smoothing reduces abrupt switching, but it governs allocation, not state memory. A smoothed weight transition can still leave the user asking why the system feels either too cleanly recovered or too persistently impaired.

## 4. Contextual Memory Findings

Current memory sources:

- FAST memory: short rolling windows inside the FAST detector; intentionally reactive and fast-cooling.
- SLOW memory: consecutive breach streaks and rolling volatility/entropy signals.
- TRAJECTORY memory: multi-horizon recovery quality, failed bounces, half-life risk, holding failure, and LSTM advisory if trained.
- META memory: weight EMA, risk/dominant/weight history, uncertainty EMA, stabilization accumulator.

The system remembers recent systemic conditions, but unevenly. TRAJECTORY has the strongest natural memory. FAST intentionally has the weakest. SLOW has memory when breaches remain consecutive, but less contextual memory after an acute shock clears. META has memory, but it is under-specified: stabilization and uncertainty are scalar smoothers rather than contextual residues.

Memory persistence is sometimes too weak after acute shocks because stabilization can rebuild while trajectory skepticism remains. It is sometimes too strong during extended recovery because stabilization can remain pinned near zero while other fields partially normalize. The issue is not magnitude alone; it is lack of contextual healing stage.

## 5. Recovery-Evolution Findings

Fake recoveries are partially handled by TRAJECTORY through failed rebound accumulation and holding failure. This is a strong local design because it derives skepticism from observed recovery quality rather than engine coupling.

V-shaped recoveries are where narrative continuity needs more nuance. The system should allow FAST to cool quickly, but META should retain a modest scar-memory residue until SLOW and TRAJECTORY both demonstrate healing quality. That residue should not boost SLOW or TRAJECTORY directly. It should only keep meta uncertainty and stabilization from snapping to fully healed.

Prolonged grinding deterioration is handled reasonably by TRAJECTORY and SLOW, but the shared story can still alternate among MIXED, TRAJECTORY_DEGRADATION, and FAST_SHOCK when intermittent volatility occurs. The architecture would benefit from a contextual persistence buffer that records recent instability burden without declaring a regime.

Panic stabilization currently has asymmetric healing, but the implementation is too threshold-like: stabilization increases by a fixed amount when overall stress is below `0.30` and declines in proportion to stress otherwise. That produces understandable but sometimes psychologically abrupt readings.

Double-dip structures are detected as renewed stress, and uncertainty remains non-trivial. The refinement need is not stronger detection. It is maintaining an explicit recovery-skepticism residue between the first rebound and the second dip so the double-dip feels like a continuation of unresolved healing rather than a fresh unrelated episode.

## 6. Uncertainty-Inertia Findings

Uncertainty is computed from inter-engine disagreement, confidence deficit, and evolution instability, then smoothed with a fixed `0.8/0.2` EMA. This is simple and bounded, which is good.

The weakness is symmetric smoothing. Ambiguity after a shock should generally decay more slowly than it rises, especially when TRAJECTORY remains skeptical or SLOW persistence has recently been elevated. Conversely, clean stabilization should relax uncertainty gradually but decisively.

Disagreement can disappear too quickly if current risks converge after a turbulent path, because uncertainty only sees current disagreement plus evolution score. There is no explicit ambiguity half-life tied to recent instability burden. Transitional states can therefore feel under-remembered.

Coherence is instantaneous: `1.0 - std(risks)`. It is useful as a current alignment measure, but not as a continuity measure. A system can look coherent today because all engines are low or moderately aligned, while the recent path still warrants ambiguity.

## 7. Abruptness/Discontinuity Findings

Primary abruptness points:

- SLOW persistence uses consecutive breach streaks, so one clean break can sharply reduce stress memory.
- Stabilization growth is based on a hard `overall_stress < 0.30` threshold.
- Stabilization can saturate at `1.00` even while uncertainty and trajectory skepticism remain meaningful.
- Stabilization can stay pinned near `0.00` in recovery paths without expressing partial healing.
- Recovery lifecycle code exists in `layer3/convergence/recovery.py`, but it is not wired into `run_convergence`, and its enum/config dependencies are not reflected in the current schema/config.
- `smooth_risk` and `smooth_confidence` exist but the convergence manager does not use them for published outputs.
- `docs/convergence_design.md` describes `overall_risk` and `overall_confidence`, but the current `MetaDynamicsOutput` exposes only stabilization, uncertainty, coherence, and dominant field.

These are not architecture failures. They are places where implementation and temporal narrative are slightly misaligned.

## 8. Recommended Temporal Refinements

Introduce convergence-local contextual memory primitives, not engine coupling:

- `recent_instability_memory`: bounded scalar updated from current adjusted FAST/SLOW/TRAJECTORY stress burden, with faster rise than decay.
- `recovery_skepticism_residue`: bounded scalar that rises after high stress and failed/ambiguous rebound evidence, then decays only when stabilization quality is clean.
- `uncertainty_inertia`: bounded buffer that keeps ambiguity elevated after recent disagreement or rapid evolution, with asymmetric decay.
- `healing_lag`: bounded scalar that delays full stabilization after severe stress but decays under sustained low stress and improving coherence.

These should live inside `ConvergenceState`, use only current and previous convergence-visible outputs, and never feed back into FAST, SLOW, or TRAJECTORY. They should influence only meta-dynamics and explanatory continuity, not raw engine probabilities.

Keep all primitives scalar, capped to `[0, 1]`, and non-recursive beyond their own previous value:

```text
memory_t = clip(decay * memory_{t-1} + input_gain * current_context, 0, cap)
```

No primitive should consume another primitive as its primary input. That avoids hidden feedback chains.

## 9. Persistence-Governance Findings

EMA weight smoothing is appropriately conservative and should not be aggressively increased. It solves allocation flicker, not scar memory.

Uncertainty smoothing should become asymmetric: rise faster during disagreement/evolution, decay slower after severe stress, and decay faster only during sustained clean stabilization. This is a governance refinement, not a recalibration of uncertainty meaning.

Stabilization should evolve from fixed-threshold growth toward quality-aware healing. It should consider low current stress, improving coherence, declining uncertainty, and low recovery skepticism. It should not instantly reward a single low-stress observation after a severe shock.

SLOW streak persistence should remain simple, but a convergence-level `recent_instability_memory` can compensate for streak discontinuity without modifying SLOW identity.

TRAJECTORY half-life logic is appropriately domain-local. Do not add new horizons or indicators. The refinement should be to let TRAJECTORY skepticism be narratively visible in META memory, not to strengthen TRAJECTORY itself.

## 10. Lightweight Memory Recommendations

Recommended minimal implementation set:

1. Add four optional scalar fields to `ConvergenceState`: `recent_instability_memory`, `recovery_skepticism_residue`, `uncertainty_inertia`, and `healing_lag`.
2. Update them once per convergence call after adjusted risks and uncertainty are computed.
3. Use them to modulate only `stabilization_strength`, `uncertainty_pressure`, and possibly an internal explanatory score if added later.
4. Do not alter engine outputs, dynamic weights, LSTM advisory, or partner influence caps.
5. Add targeted validation scenarios for V-shaped recovery, fake recovery, panic stabilization, double-dip, and grinding deterioration.

Suggested bounds:

- Caps should remain modest, preferably `0.25-0.40` effect on meta outputs rather than direct risk outputs.
- Decay should be half-life-like and short enough to avoid permanent paranoia.
- Rise should be event-sensitive but capped, especially after one-day shock artifacts.
- Recovery should require sustained evidence, but not a hard phase machine.

## 11. Remaining Temporal Risks

If memory primitives are too weak, the system will continue to feel parallel: FAST cools, SLOW/TRAJECTORY remain elevated, and META does not explain the relationship.

If memory primitives are too strong, the system will become sticky and institutionally paranoid after shocks. This would violate the user requirement and the existing CRIS philosophy.

If primitives are allowed to feed each other recursively, the architecture could quietly reintroduce the coupling and feedback problems that earlier phases intentionally removed.

If stabilization is over-conditioned on uncertainty, clean recoveries may be punished unnecessarily. Recovery skepticism must remain bounded and able to heal.

If the report recommendations are implemented without tests, the architecture could preserve continuity in one scenario while degrading recovery normalization in long calm windows.

## 12. Final Temporal-Coherence Assessment

Layer 3 already has the correct architectural foundation for organic temporal continuity: independent engines, bounded convergence, uncertainty preservation, and smoothing governors. The current weakness is that shared temporal context is implicit and scattered rather than explicit and lightly governed.

The recommended refinement is not stronger coupling. It is convergence-local contextual memory: bounded scar memory, recovery skepticism residue, uncertainty inertia, and healing lag. These primitives would let the system remember evolving probabilistic context while preserving decoupling, interpretability, boundedness, and non-binary uncertainty.

Final assessment: CRIS Layer 3 is architecturally sound and ready for subtle temporal-cohesion refinement. The next pass should make META remember the story of the last few weeks without letting that memory become a hidden regime classifier.
