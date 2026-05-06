"""
smoothing.py — Temporal smoothing for convergence transitions.

Ensures that weights, risk scores, and stress_field assessments evolve
smoothly over time rather than jumping between states.

Uses exponential moving average (EMA) smoothing with configurable
decay rates. This is the primary mechanism preventing oscillation
and ensuring stable stress_field evolution.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass, field

from ..config import WEIGHT_SMOOTHING_ALPHA


@dataclass
class SmootherState:
    """Persistent state for the EMA smoother across timesteps.

    This must be carried forward between calls to maintain
    temporal continuity.
    """
    prev_weights: Optional[Tuple[float, float, float]] = None
    prev_risk: Optional[float] = None
    prev_confidence: Optional[float] = None
    prev_fast_risk: Optional[float] = None
    prev_slow_risk: Optional[float] = None
    prev_decay_risk: Optional[float] = None
    steps: int = 0


def smooth_weights(
    raw_weights: Tuple[float, float, float],
    state: SmootherState,
    alpha: float = WEIGHT_SMOOTHING_ALPHA,
) -> Tuple[float, float, float]:
    """Apply EMA smoothing to dynamic weights.

    EMA: smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_{t-1}

    Lower alpha = smoother (more inertia).
    Higher alpha = more responsive to changes.

    Args:
        raw_weights: Current unsmoothed weights (fast, slow, decay)
        state: Smoother state from previous timestep
        alpha: EMA decay factor (0.05 to 0.30 recommended)

    Returns:
        Smoothed weights (fast, slow, decay), normalized to sum to 1.0
    """
    if state.prev_weights is None or state.steps == 0:
        # First timestep: use raw weights directly
        return raw_weights

    smoothed = tuple(
        alpha * raw + (1 - alpha) * prev
        for raw, prev in zip(raw_weights, state.prev_weights)
    )

    # Re-normalize after smoothing
    total = sum(smoothed)
    if total > 0:
        smoothed = tuple(w / total for w in smoothed)

    return (round(smoothed[0], 4), round(smoothed[1], 4), round(smoothed[2], 4))


def smooth_risk(
    raw_risk: float,
    state: SmootherState,
    alpha: float = WEIGHT_SMOOTHING_ALPHA,
) -> float:
    """Apply EMA smoothing to the overall risk score.

    Prevents risk from spiking or collapsing instantaneously.
    """
    if state.prev_risk is None or state.steps == 0:
        return raw_risk

    smoothed = alpha * raw_risk + (1 - alpha) * state.prev_risk
    return round(float(np.clip(smoothed, 0.0, 1.0)), 4)


def smooth_confidence(
    raw_confidence: float,
    state: SmootherState,
    alpha: float = WEIGHT_SMOOTHING_ALPHA,
) -> float:
    """Apply EMA smoothing to the overall confidence score."""
    if state.prev_confidence is None or state.steps == 0:
        return raw_confidence

    smoothed = alpha * raw_confidence + (1 - alpha) * state.prev_confidence
    return round(float(np.clip(smoothed, 0.0, 1.0)), 4)


def update_state(
    state: SmootherState,
    weights: Tuple[float, float, float],
    overall_risk: float,
    overall_confidence: float,
    fast_risk: float = 0.0,
    slow_risk: float = 0.0,
    decay_risk: float = 0.0,
) -> SmootherState:
    """Update smoother state after processing a timestep.

    This state must be persisted between calls for proper smoothing.
    """
    state.prev_weights = weights
    state.prev_risk = overall_risk
    state.prev_confidence = overall_confidence
    state.prev_fast_risk = fast_risk
    state.prev_slow_risk = slow_risk
    state.prev_decay_risk = decay_risk
    state.steps += 1
    return state
