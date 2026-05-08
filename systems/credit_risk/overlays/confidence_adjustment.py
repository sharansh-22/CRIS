"""
Environmental confidence adjustment for credit governance.

The output is not a prediction target.  It is an audit signal describing how
comfortable governance should be with borrower-only decisions under the current
Layer 3 environment.
"""

from __future__ import annotations

import numpy as np


def environmental_confidence(
    uncertainty_pressure: np.ndarray,
    stress_score: np.ndarray,
    uncertainty_weight: float = 0.65,
) -> np.ndarray:
    """Map Layer 3 uncertainty and stress to an interpretable confidence score."""

    uncertainty_pressure = np.asarray(uncertainty_pressure, dtype=float)
    stress_score = np.asarray(stress_score, dtype=float)
    stress_weight = 1.0 - uncertainty_weight
    penalty = uncertainty_weight * uncertainty_pressure + stress_weight * stress_score
    return np.clip(1.0 - penalty, 0.0, 1.0)
