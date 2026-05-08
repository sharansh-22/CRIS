"""
volatility.py — Short-term volatility bursts for Fast Shock detection.
"""

import pandas as pd
from harvesters.shared.rolling import compute_rolling_vol

def compute_short_term_vol(returns: pd.Series, window: int = 5) -> pd.Series:
    """Compute 5-day rolling volatility to catch sudden bursts."""
    return compute_rolling_vol(returns, window)
