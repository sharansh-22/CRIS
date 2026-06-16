"""
synchronization.py — Cross-Sector Synchronization Analysis.

Measures how tightly sectors are moving in lockstep.
High synchronization = systemic risk is concentrated.
Low synchronization = healthy independent dynamics.

Wraps correlation-based coupling detection with
sector-specific interpretation.
"""

import pandas as pd
from harvesters.macro.systemic.correlation import (
    compute_cross_sector_coupling,
    compute_diversification_failure,
)


def synchronization_level(returns: pd.DataFrame) -> str:
    """Classify the degree of cross-sector synchronization.

    Returns one of: INDEPENDENT, MILD_COUPLING, SYNCHRONIZED, LOCKSTEP.
    """
    coupling = compute_cross_sector_coupling(returns)
    div_fail = compute_diversification_failure(returns)

    combined = 0.5 * coupling + 0.5 * div_fail

    if combined < 0.2:
        return "INDEPENDENT"
    elif combined < 0.4:
        return "MILD_COUPLING"
    elif combined < 0.7:
        return "SYNCHRONIZED"
    else:
        return "LOCKSTEP"


def is_diversification_intact(returns: pd.DataFrame, threshold: float = 0.5) -> bool:
    """Quick check: is diversification still functional?"""
    return compute_diversification_failure(returns) < threshold
