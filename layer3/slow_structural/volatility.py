"""
volatility.py — Volatility ratio logic for Slow StressField detection.
"""

import pandas as pd
from ..shared.rolling import compute_rolling_vol
from ..config import ROLLING_WINDOW_SMALL

def compute_vol_ratio(returns: pd.Series, baseline_vol: float, window: int = ROLLING_WINDOW_SMALL) -> pd.Series:
    """Compute ratio of current rolling vol to historical baseline."""
    rolling_vol = compute_rolling_vol(returns, window)
    return rolling_vol / baseline_vol
