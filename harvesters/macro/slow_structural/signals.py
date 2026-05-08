"""
signals.py — Signal aggregation for Slow StressField detection.
"""

import pandas as pd
from .entropy import compute_sample_entropy
from .volatility import compute_vol_ratio
from harvesters.shared.rolling import apply_rolling
from configs.macro_config import ROLLING_WINDOW_LARGE

def get_slow_signals(returns: pd.Series, baseline_vol: float, baseline_entropy: float) -> dict:
    """Aggregate slow-moving signals (Entropy Delta and Vol Ratio)."""
    
    entropy_series = apply_rolling(returns, ROLLING_WINDOW_LARGE, compute_sample_entropy)
    current_entropy = float(entropy_series.dropna().iloc[-1]) if len(entropy_series.dropna()) > 0 else baseline_entropy
    
    vol_ratio_series = compute_vol_ratio(returns, baseline_vol)
    current_vol_ratio = float(vol_ratio_series.dropna().iloc[-1]) if len(vol_ratio_series.dropna()) > 0 else 1.0
    
    return {
        "entropy_delta": current_entropy - baseline_entropy,
        "vol_ratio": current_vol_ratio,
        "vol_ratio_series": vol_ratio_series
    }
