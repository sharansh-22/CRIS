"""
deterioration.py — Structural deterioration metrics for Decay StressField.

Measures drawdown depth, recovery failure, and momentum exhaustion —
the hallmarks of a slow grind-down that doesn't trigger vol-based
detectors.
"""

import numpy as np
import pandas as pd


def compute_drawdown_depth(prices: pd.Series) -> float:
    """Compute current drawdown from the running peak.

    Returns:
        Drawdown as a positive fraction [0, 1].
        0.0 = at peak, 0.35 = 35% below peak.
    """
    if len(prices) < 2:
        return 0.0

    running_max = prices.cummax()
    current_dd = (running_max.iloc[-1] - prices.iloc[-1]) / running_max.iloc[-1]
    return float(max(0.0, current_dd))


def compute_drawdown_duration(prices: pd.Series) -> int:
    """Compute how many days since the last all-time high.

    Returns:
        Number of trading days since the peak.
    """
    if len(prices) < 2:
        return 0

    running_max = prices.cummax()
    at_peak = prices >= running_max
    if not at_peak.any():
        return len(prices)

    last_peak_idx = at_peak[::-1].idxmax()
    last_peak_pos = prices.index.get_loc(last_peak_idx)
    return len(prices) - 1 - last_peak_pos


def compute_momentum_exhaustion(returns: pd.Series, window: int = 30) -> float:
    """Measure momentum exhaustion: how negative is recent cumulative return.

    Returns:
        Score in [0, 1]. 0 = strong positive momentum, 1 = fully exhausted.
    """
    if len(returns) < window // 2:
        return 0.0

    recent = returns.iloc[-window:]
    cum_return = float((1 + recent).prod() - 1)

    # Map: -20% → 1.0 exhaustion, +10% → 0.0
    # Piecewise linear
    if cum_return >= 0.0:
        return 0.0
    elif cum_return <= -0.20:
        return 1.0
    else:
        return float(abs(cum_return) / 0.20)


def compute_recovery_failure(prices: pd.Series, lookback: int = 60) -> float:
    """Measure failed recovery attempts — rallies that don't reclaim previous highs.

    Returns:
        Score in [0, 1]. 0 = successful recoveries, 1 = persistent failure.
    """
    if len(prices) < lookback:
        return 0.0

    recent = prices.iloc[-lookback:]
    running_max = recent.cummax()

    # Fraction of days below the running max
    below_peak = (recent < running_max * 0.98).mean()  # 2% buffer

    return float(np.clip(below_peak, 0.0, 1.0))


def compute_lower_highs(prices: pd.Series, segment_size: int = 20) -> float:
    """Detect lower-highs pattern (characteristic of slow deterioration).

    Splits recent price history into segments and checks if peaks
    are progressively lower.

    Returns:
        Score in [0, 1]. 0 = no lower highs, 1 = consistent lower highs.
    """
    if len(prices) < segment_size * 3:
        return 0.0

    n_segments = min(5, len(prices) // segment_size)
    if n_segments < 3:
        return 0.0

    segment_highs = []
    for i in range(n_segments):
        start = -(n_segments - i) * segment_size
        end = start + segment_size if i < n_segments - 1 else None
        segment = prices.iloc[start:end] if end else prices.iloc[start:]
        segment_highs.append(float(segment.max()))

    # Count how many consecutive peaks are lower
    lower_count = 0
    for i in range(1, len(segment_highs)):
        if segment_highs[i] < segment_highs[i - 1]:
            lower_count += 1

    return float(lower_count / (len(segment_highs) - 1))
