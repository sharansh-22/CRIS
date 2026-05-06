"""
rolling.py — Generic rolling window utilities for financial time-series.
"""

import pandas as pd
import numpy as np
from typing import Callable

def apply_rolling(series: pd.Series, window: int, func: Callable, **kwargs) -> pd.Series:
    """Apply a function over a rolling window of a pandas Series."""
    result = pd.Series(np.nan, index=series.index, dtype=float)
    min_periods = window // 2
    
    for i in range(len(series)):
        start = max(0, i - window + 1)
        chunk = series.iloc[start : i + 1]
        if len(chunk) < min_periods:
            continue
        result.iloc[i] = func(chunk.values, **kwargs)
    return result

def compute_rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    """Compute rolling mean absolute return (a proxy for volatility)."""
    return returns.abs().rolling(window).mean()
