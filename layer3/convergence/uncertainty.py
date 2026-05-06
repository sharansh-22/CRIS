"""
uncertainty.py — Uncertainty quantification for convergence layer.

Markets are often ambiguous. The system must be capable of admitting
uncertainty rather than forcing a confident classification.

Detects:
  - MIXED stress_fields (multiple engines disagree)
  - TRANSITIONAL states (stress_field is actively shifting)
  - UNCLEAR situations (low confidence across all engines)
"""

import numpy as np

from ..schema import DominantField
from ..config import (
    UNCERTAINTY_DISAGREEMENT_THRESHOLD,
    UNCERTAINTY_LOW_CONFIDENCE_THRESHOLD,
    FIELD_EVOLUTION_WINDOW,
)


def compute_uncertainty_score(
    fast_risk: float,
    slow_risk: float,
    decay_risk: float,
    fast_confidence: float,
    slow_confidence: float,
    decay_confidence: float,
    evolution_score: float = 0.0,
) -> float:
    """Compute overall uncertainty about the stress_field assessment.

    High uncertainty when:
      - Engines disagree about risk (high variance)
      - All engines have low confidence
      - The stress_field is rapidly evolving

    Returns:
        Uncertainty score [0, 1]. 0 = very clear, 1 = very uncertain.
    """
    risks = [fast_risk, slow_risk, decay_risk]
    confs = [fast_confidence, slow_confidence, decay_confidence]

    # 1. Inter-engine disagreement (variance in risk scores)
    risk_spread = float(np.std(risks))
    disagreement = min(1.0, risk_spread / 0.3)

    # 2. Overall confidence deficit
    avg_confidence = float(np.mean(confs))
    confidence_deficit = max(0.0, 1.0 - avg_confidence * 1.5)

    # 3. StressField evolution instability
    evolution_uncertainty = float(np.clip(evolution_score, 0.0, 1.0))

    # Composite
    uncertainty = float(np.clip(
        0.40 * disagreement + 0.35 * confidence_deficit + 0.25 * evolution_uncertainty,
        0.0, 1.0
    ))

    return round(uncertainty, 2)


def classify_uncertainty_state(
    dominant: DominantField,
    uncertainty: float,
    fast_risk: float,
    slow_risk: float,
    decay_risk: float,
    fast_confidence: float,
    slow_confidence: float,
    decay_confidence: float,
    evolution_score: float = 0.0,
) -> DominantField:
    """Potentially reclassify the dominant stress_field to an uncertainty state.

    Overrides the dominant stress_field determination when:
      - High inter-engine disagreement → MIXED
      - Rapid evolution → TRANSITIONAL
      - Low confidence everywhere → UNCLEAR

    Returns:
        DominantField (potentially overridden to MIXED/TRANSITIONAL/UNCLEAR)
    """
    risks = [fast_risk, slow_risk, decay_risk]
    confs = [fast_confidence, slow_confidence, decay_confidence]

    # Check for UNCLEAR: all engines have low confidence
    if all(c < UNCERTAINTY_LOW_CONFIDENCE_THRESHOLD for c in confs):
        max_risk = max(risks)
        if max_risk < 0.3:
            return DominantField.UNCLEAR

    # Check for MIXED: multiple engines show significant risk but disagree
    elevated_count = sum(1 for r in risks if r > 0.3)
    if elevated_count >= 2:
        risk_spread = max(risks) - min(risks)
        if risk_spread > UNCERTAINTY_DISAGREEMENT_THRESHOLD:
            return DominantField.MIXED

    # Check for TRANSITIONAL: stress_field is actively shifting
    if evolution_score > 0.5 and uncertainty > 0.4:
        return DominantField.TRANSITIONAL

    # No override — keep the original determination
    return dominant
