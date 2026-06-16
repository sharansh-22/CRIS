"""
fragmentation.py — Market Fragmentation Analysis.

Provides convenience wrappers for fragmentation-specific queries.
Core fragmentation computations live in dispersion.py
(cross-sectional dispersion, leadership instability).
"""

import pandas as pd
from harvesters.macro.systemic.dispersion import (
    compute_cross_sectional_dispersion,
    compute_leadership_instability,
)


def is_market_fragmenting(returns: pd.DataFrame, threshold: float = 0.4) -> bool:
    """Quick check: is cross-sectional dispersion above the fragmentation threshold?"""
    disp = compute_cross_sectional_dispersion(returns)
    return disp > threshold


def fragmentation_severity(returns: pd.DataFrame) -> str:
    """Classify current market fragmentation level.

    Returns one of: COHERENT, DIVERGING, FRAGMENTING, DISLOCATED.
    """
    disp = compute_cross_sectional_dispersion(returns)
    instab = compute_leadership_instability(returns)

    combined = 0.5 * disp + 0.5 * instab

    if combined < 0.15:
        return "COHERENT"
    elif combined < 0.35:
        return "DIVERGING"
    elif combined < 0.6:
        return "FRAGMENTING"
    else:
        return "DISLOCATED"
