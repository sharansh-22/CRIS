"""
signals.py — Short-term alarm signals for Fast Shock detection.

Uses FAST entropy (short-horizon) — NOT slow_structural entropy.
"""

import pandas as pd
import numpy as np
from .entropy import compute_fast_entropy_series
from ..config import ROLLING_WINDOW_SMALL, PERM_ALARM_DROP_THRESHOLD


def compute_permutation_alarm(returns: pd.Series, baseline_perm_entropy: float) -> dict:
    """Detect directional trends via short-horizon permutation entropy drops."""
    perm_series = compute_fast_entropy_series(returns, window=ROLLING_WINDOW_SMALL)

    alarm_threshold = baseline_perm_entropy - PERM_ALARM_DROP_THRESHOLD
    if len(perm_series.dropna()) > 0:
        current_perm = float(perm_series.dropna().iloc[-1])
        alarm_active = current_perm < alarm_threshold
    else:
        current_perm = float(baseline_perm_entropy)
        alarm_active = False

    return {
        "alarm_active": alarm_active,
        "current_perm_entropy": current_perm,
        "alarm_threshold": alarm_threshold
    }
