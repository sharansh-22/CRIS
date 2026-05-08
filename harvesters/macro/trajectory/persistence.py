"""
persistence.py — Persistence logic for Decay StressField detection.

Tracks how long deterioration conditions have been active.
Unlike fast/slow persistence which counts vol breaches,
decay persistence counts structural trend degradation duration.
"""

import pandas as pd
import numpy as np


def compute_decay_persistence(prices: pd.Series, threshold_dd: float = 0.05) -> int:
    """Count consecutive days the price has been in drawdown above threshold.

    Args:
        prices: Price series
        threshold_dd: Minimum drawdown fraction to count as deterioration

    Returns:
        Number of consecutive days in meaningful drawdown at end of series.
    """
    if len(prices) < 2:
        return 0

    running_max = prices.cummax()
    drawdowns = (running_max - prices) / running_max

    streak = 0
    for dd in reversed(drawdowns.values):
        if dd >= threshold_dd:
            streak += 1
        else:
            break

    return streak


def compute_trend_persistence(returns: pd.Series, window: int = 20) -> int:
    """Count consecutive windows where rolling mean return is negative.

    Returns:
        Number of consecutive negative-mean periods at end of series.
    """
    if len(returns) < window:
        return 0

    rolling_mean = returns.rolling(window).mean().dropna()
    if len(rolling_mean) == 0:
        return 0

    streak = 0
    for val in reversed(rolling_mean.values):
        if val < 0:
            streak += 1
        else:
            break

    return streak


def compute_combined_decay_persistence(
    prices: pd.Series,
    returns: pd.Series,
    dd_threshold: float = 0.05,
    trend_window: int = 20,
) -> int:
    """Combine drawdown and trend persistence into a single metric.

    Returns the maximum of the two persistence measures.
    """
    dd_persist = compute_decay_persistence(prices, dd_threshold)
    trend_persist = compute_trend_persistence(returns, trend_window)
    return max(dd_persist, trend_persist)
