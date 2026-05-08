"""
recovery.py — Recovery analysis for Decay StressField.

Detects failed recoveries, fake rebounds, and recovery strength.
This is what allows DECAY to differentiate between:
  - genuine trend reversals (exit decay)
  - dead cat bounces (stay in decay)
  - sustained recovery (confirm normalization)
"""

import numpy as np
import pandas as pd


def compute_recovery_strength(prices: pd.Series, lookback: int = 30) -> float:
    """Measure how strong the current recovery attempt is.

    Compares recent price action to the trough:
      - 0.0 = no recovery / still declining
      - 0.5 = partial recovery
      - 1.0 = full recovery to prior peak

    Returns:
        Recovery strength score [0, 1].
    """
    if len(prices) < lookback:
        return 0.0

    recent = prices.iloc[-lookback:]
    trough = float(recent.min())
    peak = float(prices.iloc[:-lookback].max()) if len(prices) > lookback else float(recent.max())
    current = float(prices.iloc[-1])

    if peak <= trough:
        return 0.0

    recovery_pct = (current - trough) / (peak - trough)
    return float(np.clip(recovery_pct, 0.0, 1.0))


def detect_failed_recovery(prices: pd.Series, returns: pd.Series, lookback: int = 40) -> float:
    """Detect whether a recovery attempt has stalled or reversed.

    Looks for the pattern: decline → bounce → renewed decline.

    Returns:
        Failed recovery score [0, 1]. 0 = healthy, 1 = definitively failed.
    """
    if len(prices) < lookback:
        return 0.0

    recent = prices.iloc[-lookback:]

    # Split into thirds: early, mid, late
    third = lookback // 3
    early = recent.iloc[:third]
    mid = recent.iloc[third:2*third]
    late = recent.iloc[2*third:]

    early_mean = float(early.mean())
    mid_mean = float(mid.mean())
    late_mean = float(late.mean())

    # Pattern: early_low → mid_higher → late_low_again = failed recovery
    if mid_mean > early_mean and late_mean < mid_mean:
        # How deep is the re-decline?
        decline_from_bounce = (mid_mean - late_mean) / mid_mean if mid_mean > 0 else 0.0
        return float(np.clip(decline_from_bounce * 5.0, 0.0, 1.0))

    # Alternative: continuous decline (no bounce attempt at all)
    if late_mean < early_mean:
        decline = (early_mean - late_mean) / early_mean if early_mean > 0 else 0.0
        return float(np.clip(decline * 3.0, 0.0, 0.7))  # Cap at 0.7 — no bounce = not a "failed" recovery

    return 0.0


def detect_fake_rebound(returns: pd.Series, window: int = 10) -> float:
    """Detect short-lived bounces that don't sustain (dead cat bounces).

    Looks for: sharp positive returns followed by resumed decline.

    Returns:
        Fake rebound score [0, 1]. 0 = no fake rebound, 1 = definitive fake.
    """
    if len(returns) < window * 2:
        return 0.0

    first_half = returns.iloc[-2*window:-window]
    second_half = returns.iloc[-window:]

    first_cum = float((1 + first_half).prod() - 1)
    second_cum = float((1 + second_half).prod() - 1)

    # Pattern: positive first half, negative second half = fake rebound
    if first_cum > 0.02 and second_cum < -0.01:
        magnitude = abs(second_cum) / first_cum if first_cum > 0 else 0.0
        return float(np.clip(magnitude, 0.0, 1.0))

    return 0.0
