"""
persistence.py — Gating and streak logic for Slow StressField detection.
"""

import pandas as pd
import numpy as np

def compute_streak(breach_series: pd.Series) -> int:
    """Compute the current consecutive streak of True values at the end of the series."""
    values = breach_series.dropna().values
    if len(values) == 0:
        return 0
    streak = 0
    for v in reversed(values):
        if v:
            streak += 1
        else:
            break
    return streak

def check_persistence(streak: int, threshold: int) -> bool:
    """Return True if the streak meets or exceeds the threshold."""
    return streak >= threshold
