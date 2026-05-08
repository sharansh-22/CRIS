"""
arbitration.py — Bounded inter-layer influence for convergence.

Implements the "soft influence" mechanism where engines can gently
bias each other's probabilities WITHOUT redefining reality.

Design philosophy:
  - 90-95% own reasoning, 5-10% partner influence
  - Influence is BOUNDED and CAPPED
  - No runaway feedback loops
  - Influence decays when underlying conditions normalize

This module is the ONLY place where inter-layer communication occurs.
The individual detectors (fast, slow, decay) remain fully independent.
"""

import numpy as np
from typing import Tuple

from configs.macro_config import PARTNER_INFLUENCE_CAP


def apply_partner_influence(
    fast_risk: float,
    fast_confidence: float,
    slow_risk: float,
    slow_confidence: float,
    decay_risk: float,
    decay_confidence: float,
) -> Tuple[float, float, float]:
    """Apply bounded partner influence to each engine's risk score.

    Rules:
    1. Fast instability gently nudges slow risk upward (early warning)
    2. Slow persistence gently nudges decay risk upward (escalation)
    3. Decay evidence does NOT feed back to fast (prevents circular loops)

    The influence is:
    - Proportional to the partner's risk × confidence
    - Capped at PARTNER_INFLUENCE_CAP (default 10%)
    - One-directional: fast→slow→decay (no circular paths)

    Returns:
        Tuple of adjusted (fast_risk, slow_risk, decay_risk)
    """
    cap = PARTNER_INFLUENCE_CAP

    # ── Fast is purely independent — no partner influence ──
    adjusted_fast = fast_risk

    # ── Slow receives gentle nudge from fast ──
    # If fast is alarming, slow gets a small upward bias
    fast_to_slow = fast_risk * fast_confidence * cap
    fast_to_slow = min(fast_to_slow, cap)  # Hard cap
    adjusted_slow = slow_risk * (1 - cap) + (slow_risk + fast_to_slow) * cap
    adjusted_slow = float(np.clip(adjusted_slow, 0.0, 1.0))

    # ── Decay receives gentle nudge from slow ──
    # If slow is persistently stressed, decay gets a small upward bias
    slow_to_decay = slow_risk * slow_confidence * cap
    slow_to_decay = min(slow_to_decay, cap)  # Hard cap
    adjusted_decay = decay_risk * (1 - cap) + (decay_risk + slow_to_decay) * cap
    adjusted_decay = float(np.clip(adjusted_decay, 0.0, 1.0))

    # NO reverse influence: decay does NOT affect fast or slow
    # NO skip influence: fast does NOT directly affect decay
    # This prevents circular feedback loops

    return (
        round(adjusted_fast, 4),
        round(adjusted_slow, 4),
        round(adjusted_decay, 4),
    )


def validate_no_circular_feedback(
    original_fast: float,
    original_slow: float,
    original_decay: float,
    adjusted_fast: float,
    adjusted_slow: float,
    adjusted_decay: float,
) -> bool:
    """Validate that partner influence hasn't created runaway amplification.

    Returns True if the adjustments are within safe bounds.
    """
    cap = PARTNER_INFLUENCE_CAP

    # Fast should never be modified by partners
    if abs(adjusted_fast - original_fast) > 1e-6:
        return False

    # Slow adjustment should be bounded
    if abs(adjusted_slow - original_slow) > cap * 1.5:
        return False

    # Decay adjustment should be bounded
    if abs(adjusted_decay - original_decay) > cap * 1.5:
        return False

    return True
