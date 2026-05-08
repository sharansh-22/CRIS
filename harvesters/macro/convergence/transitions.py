"""
transitions.py — StressField evolution tracking for convergence layer.

Tracks how the stress_field landscape is evolving over time:
  - Which engine is gaining influence
  - How rapidly the stress_field is shifting
  - Whether we're in a stable or transitional state

This is purely observational — it does NOT drive decisions.
It provides interpretability context for the human operator.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field

from harvesters.macro.schema import DominantField
from configs.macro_config import FIELD_EVOLUTION_WINDOW


@dataclass
class EvolutionTracker:
    """Tracks stress_field evolution history for interpretability.

    Maintains a rolling buffer of recent dominant stress_fields and risk levels.
    """
    risk_history: List[float] = field(default_factory=list)
    dominant_history: List[str] = field(default_factory=list)
    weight_history: List[Tuple[float, float, float]] = field(default_factory=list)
    max_history: int = 60  # Keep ~3 months of daily history


def determine_dominant_field(
    fast_weight: float,
    slow_weight: float,
    decay_weight: float,
    fast_risk: float,
    slow_risk: float,
    decay_risk: float,
) -> DominantField:
    """Determine which stress_field engine is currently dominant.

    Dominance is based on weight × risk product — the engine with
    the highest influence-adjusted risk is dominant.

    If multiple engines have similar influence, the stress_field is MIXED.
    """
    # Influence = weight × risk × confidence-implied-strength
    fast_influence = fast_weight * fast_risk
    slow_influence = slow_weight * slow_risk
    decay_influence = decay_weight * decay_risk

    influences = {
        DominantField.FAST_SHOCK: fast_influence,
        DominantField.SLOW_STRUCTURAL: slow_influence,
        DominantField.TRAJECTORY_DEGRADATION: decay_influence,
    }

    max_influence = max(influences.values())

    # If all influences are very low, no dominant stress_field
    if max_influence < 0.05:
        return DominantField.NONE

    # Check if multiple engines are close (within 30% of max)
    close_count = sum(1 for v in influences.values() if v >= max_influence * 0.70)
    if close_count >= 2:
        return DominantField.MIXED

    # Single dominant engine
    return max(influences, key=influences.get)


def compute_evolution_score(tracker: EvolutionTracker) -> float:
    """Measure how rapidly the stress_field landscape is changing.

    Returns:
        Score in [0, 1]. 0 = completely stable, 1 = rapid stress_field shift.
    """
    n = min(FIELD_EVOLUTION_WINDOW, len(tracker.risk_history))
    if n < 3:
        return 0.0

    recent_risk = tracker.risk_history[-n:]

    # Risk volatility (how much risk is bouncing around)
    risk_vol = float(np.std(recent_risk))

    # Risk trend (is risk rising or falling?)
    risk_change = abs(recent_risk[-1] - recent_risk[0])

    # Dominant stress_field changes
    if len(tracker.dominant_history) >= n:
        recent_dominant = tracker.dominant_history[-n:]
        changes = sum(1 for i in range(1, len(recent_dominant))
                      if recent_dominant[i] != recent_dominant[i-1])
        stress_field_instability = changes / (n - 1)
    else:
        stress_field_instability = 0.0

    # Composite evolution score
    evolution = float(np.clip(
        0.35 * risk_vol * 5.0 + 0.35 * risk_change * 3.0 + 0.30 * stress_field_instability,
        0.0, 1.0
    ))

    return round(evolution, 4)


def update_tracker(
    tracker: EvolutionTracker,
    overall_risk: float,
    dominant: DominantField,
    weights: Tuple[float, float, float],
) -> EvolutionTracker:
    """Append current state to evolution tracker.

    Automatically trims history to max_history length.
    """
    tracker.risk_history.append(overall_risk)
    tracker.dominant_history.append(dominant.value)
    tracker.weight_history.append(weights)

    # Trim if too long
    if len(tracker.risk_history) > tracker.max_history:
        tracker.risk_history = tracker.risk_history[-tracker.max_history:]
        tracker.dominant_history = tracker.dominant_history[-tracker.max_history:]
        tracker.weight_history = tracker.weight_history[-tracker.max_history:]

    return tracker
