"""
breadth.py — Market Breadth Intelligence for CRIS.

Measures participation quality inside markets.
Healthy markets have broad participation.
Fragile markets become narrow and concentrated.

Input: DataFrame of daily returns for multiple sector ETFs.
Output: BreadthOutput (Pydantic typed contract).
"""

import numpy as np
import pandas as pd
from configs.macro_config import CONFIDENCE_FLOOR


def compute_advance_decline_ratio(returns: pd.DataFrame) -> float:
    """Compute normalized advance/decline ratio from the most recent day.

    Returns a value in [0, 1] where:
      - 1.0 = all assets advancing
      - 0.5 = balanced
      - 0.0 = all assets declining
    """
    if returns.empty or len(returns) < 1:
        return 0.5
    latest = returns.iloc[-1]
    n_total = len(latest.dropna())
    if n_total == 0:
        return 0.5
    n_advancing = int((latest > 0).sum())
    return float(n_advancing / n_total)


def compute_pct_above_sma(prices: pd.DataFrame, sma_window: int = 50) -> float:
    """Compute the fraction of assets trading above their N-day SMA.

    This is a classic breadth indicator — a healthy market has most
    constituents above their moving average.
    """
    if prices.empty or len(prices) < sma_window:
        return 0.5
    sma = prices.rolling(window=sma_window).mean()
    latest_prices = prices.iloc[-1]
    latest_sma = sma.iloc[-1]
    valid = latest_prices.dropna().index.intersection(latest_sma.dropna().index)
    if len(valid) == 0:
        return 0.5
    above = (latest_prices[valid] > latest_sma[valid]).sum()
    return float(above / len(valid))


def compute_participation_decay(returns: pd.DataFrame, window: int = 21) -> float:
    """Measure how rapidly market participation is narrowing.

    Compares the advance/decline ratio over a trailing window
    to detect deterioration trends.
    Returns a value in [0, 1] where higher = more decay.
    """
    if returns.empty or len(returns) < window:
        return 0.0
    ad_series = []
    for i in range(max(0, len(returns) - window), len(returns)):
        row = returns.iloc[i]
        valid = row.dropna()
        if len(valid) == 0:
            ad_series.append(0.5)
            continue
        ad_series.append(float((valid > 0).sum()) / len(valid))

    ad = np.array(ad_series)
    if len(ad) < 5:
        return 0.0

    # Fit a simple linear trend — negative slope = participation decay
    x = np.arange(len(ad))
    slope = float(np.polyfit(x, ad, 1)[0])

    # Normalize: a slope of -0.02/day is severe decay
    decay = float(np.clip(-slope / 0.02, 0.0, 1.0))
    return decay


def compute_breadth_collapse_velocity(returns: pd.DataFrame, window: int = 10) -> float:
    """Detect sudden breadth collapse — a rapid shift from broad to narrow.

    Measures the acceleration of participation narrowing over a short window.
    """
    if returns.empty or len(returns) < window + 5:
        return 0.0

    ad_recent = []
    for i in range(len(returns) - window, len(returns)):
        row = returns.iloc[i]
        valid = row.dropna()
        if len(valid) == 0:
            ad_recent.append(0.5)
            continue
        ad_recent.append(float((valid > 0).sum()) / len(valid))

    ad = np.array(ad_recent)
    if len(ad) < 3:
        return 0.0

    # Velocity = max single-day drop in breadth
    diffs = np.diff(ad)
    worst_drop = float(-np.min(diffs)) if len(diffs) > 0 else 0.0

    # Normalize: a 0.4 drop in one day is extreme
    return float(np.clip(worst_drop / 0.4, 0.0, 1.0))


def run_breadth(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    sma_window: int = 50,
) -> "BreadthOutput":
    """Execute the full breadth intelligence module.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for sector ETFs / constituents. Columns = tickers.
    prices : pd.DataFrame
        Daily close prices for the same tickers. Columns = tickers.
    sma_window : int
        Window for SMA-based breadth computation.

    Returns
    -------
    BreadthOutput
        Typed Pydantic output.
    """
    from harvesters.macro.systemic.schema import BreadthOutput

    ad_ratio = compute_advance_decline_ratio(returns)
    pct_sma = compute_pct_above_sma(prices, sma_window)
    decay = compute_participation_decay(returns)
    collapse_vel = compute_breadth_collapse_velocity(returns)

    # Confidence: based on data sufficiency
    n_assets = returns.shape[1] if not returns.empty else 0
    n_days = len(returns)
    data_confidence = min(1.0, n_assets / 5.0) * min(1.0, n_days / 60.0)
    confidence = float(np.clip(data_confidence, CONFIDENCE_FLOOR, 1.0))

    return BreadthOutput(
        advance_decline_ratio=round(ad_ratio, 4),
        pct_above_sma=round(pct_sma, 4),
        participation_decay=round(decay, 4),
        breadth_collapse_velocity=round(collapse_vel, 4),
        confidence=round(confidence, 4),
    )
