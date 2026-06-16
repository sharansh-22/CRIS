"""
contagion.py — Systemic Contagion Analysis.

This module provides convenience wrappers for contagion-specific queries.
The core contagion computation lives in correlation.py
(correlation density, coupling, acceleration).

This module adds higher-level contagion interpretation.
"""

import pandas as pd
from harvesters.macro.systemic.correlation import (
    compute_contagion_acceleration,
    compute_rolling_correlation_density,
)


def is_contagion_building(returns: pd.DataFrame, threshold: float = 0.3) -> bool:
    """Quick check: is correlation acceleration above the contagion threshold?"""
    accel = compute_contagion_acceleration(returns)
    return accel > threshold


def contagion_severity(returns: pd.DataFrame) -> str:
    """Classify current contagion severity level.

    Returns one of: NONE, ELEVATED, HIGH, CRITICAL.
    """
    density = compute_rolling_correlation_density(returns, window=63)
    accel = compute_contagion_acceleration(returns)

    combined = 0.6 * density + 0.4 * accel

    if combined < 0.2:
        return "NONE"
    elif combined < 0.4:
        return "ELEVATED"
    elif combined < 0.7:
        return "HIGH"
    else:
        return "CRITICAL"
