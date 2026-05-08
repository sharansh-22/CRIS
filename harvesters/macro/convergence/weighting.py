"""
weighting.py — Dynamic weight computation for convergence layer.

Determines how much influence each stress_field engine gets in the final
assessment. Weights are driven by:
  - Each engine's own confidence
  - Persistence of signals
  - Time horizon relevance

Weights evolve GRADUALLY. No abrupt switching.

Design constraints:
  - Weights always sum to 1.0
  - No weight can drop to zero (minimum floor maintained)
  - Persistence-based transitions are bounded and gradual
"""

import numpy as np
from typing import Tuple

from configs.macro_config import (
    WEIGHT_PRIOR_FAST,
    WEIGHT_PRIOR_SLOW,
    WEIGHT_PRIOR_DECAY,
    PERSISTENCE_SLOW_BOOST_RATE,
    PERSISTENCE_DECAY_BOOST_RATE,
)


# Minimum weight floor — no engine is ever fully silenced
_WEIGHT_FLOOR = 0.08


def compute_dynamic_weights(
    fast_confidence: float,
    slow_confidence: float,
    decay_confidence: float,
    fast_persistence: int,
    slow_persistence: int,
    decay_persistence: int,
) -> Tuple[float, float, float]:
    """Compute dynamic weights for stress_field engines.

    The weight for each engine is determined by:
    1. A prior (base allocation reflecting time-horizon importance)
    2. Confidence scaling (engines with higher confidence get more weight)
    3. Persistence adjustment (sustained signals shift weight toward
       longer-horizon engines GRADUALLY)

    Returns:
        Tuple of (fast_weight, slow_weight, decay_weight) summing to 1.0
    """
    # ── 1. Start from priors ──
    w_fast = WEIGHT_PRIOR_FAST
    w_slow = WEIGHT_PRIOR_SLOW
    w_decay = WEIGHT_PRIOR_DECAY

    # ── 2. Confidence scaling ──
    # Each engine's weight is scaled by its confidence
    w_fast *= (0.3 + 0.7 * fast_confidence)
    w_slow *= (0.3 + 0.7 * slow_confidence)
    w_decay *= (0.3 + 0.7 * decay_confidence)

    # ── 3. Persistence-based gradual shift ──
    # Repeated fast instability slowly increases slow weight
    # Prolonged slow stress gradually increases decay weight
    # These transitions are BOUNDED and GRADUAL
    slow_boost = min(0.15, fast_persistence * PERSISTENCE_SLOW_BOOST_RATE)
    decay_boost = min(0.12, slow_persistence * PERSISTENCE_DECAY_BOOST_RATE)

    w_slow += slow_boost
    w_decay += decay_boost

    # ── 4. Decay persistence bonus ──
    # Long decay persistence further reinforces decay weight
    if decay_persistence > 30:
        decay_self_boost = min(0.10, (decay_persistence - 30) * 0.002)
        w_decay += decay_self_boost

    # ── 5. Normalize with floor ──
    w_fast, w_slow, w_decay = _normalize_with_floor(w_fast, w_slow, w_decay)

    return (round(w_fast, 4), round(w_slow, 4), round(w_decay, 4))


def _normalize_with_floor(w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
    """Normalize weights to sum to 1.0, enforcing a minimum floor.

    No engine is ever fully silenced — this prevents hard switching.
    """
    # Enforce floor
    w1 = max(_WEIGHT_FLOOR, w1)
    w2 = max(_WEIGHT_FLOOR, w2)
    w3 = max(_WEIGHT_FLOOR, w3)

    # Normalize
    total = w1 + w2 + w3
    if total > 0:
        w1 /= total
        w2 /= total
        w3 /= total

    return (w1, w2, w3)
