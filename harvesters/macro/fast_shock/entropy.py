"""
entropy.py — Short-horizon entropy analysis for Fast Shock detection.

FAST entropy is fundamentally different from SLOW entropy:
  - FAST uses short windows (5-10 days) to detect sudden discontinuities
  - SLOW uses long windows (20-30 days) to detect structural shifts

This module is INDEPENDENT from harvesters.macro.slow_structural/entropy.py.
It shares mathematical primitives but applies them on different timescales.
"""

import numpy as np
import pandas as pd
import antropy
from harvesters.shared.rolling import apply_rolling
from configs.macro_config import ROLLING_WINDOW_SMALL


def compute_fast_permutation_entropy(returns: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Compute permutation entropy on a SHORT window.

    Detects sudden changes in the ordinal pattern structure of returns.
    A sharp drop indicates emerging directional trend (potential shock).
    """
    arr = returns[~np.isnan(returns)]
    if len(arr) < order:
        return 1.0  # Maximum entropy = maximum randomness = no shock
    try:
        perm_en = antropy.perm_entropy(arr, order=order, delay=delay, normalize=True)
    except Exception:
        return 1.0
    return float(np.clip(perm_en, 0.0, 1.0))


def compute_fast_entropy_series(returns: pd.Series, window: int = ROLLING_WINDOW_SMALL) -> pd.Series:
    """Compute rolling permutation entropy over short windows."""
    return apply_rolling(returns, window, compute_fast_permutation_entropy)


def compute_entropy_spike(
    returns: pd.Series,
    baseline_entropy: float,
    window: int = ROLLING_WINDOW_SMALL,
) -> dict:
    """Detect sudden entropy drops (directional stress_field emergence).

    Returns:
        Dictionary with spike detection results:
          - current_entropy: latest rolling permutation entropy
          - entropy_drop: how far below baseline (positive = dropped)
          - spike_detected: whether the drop exceeds alarm threshold
    """
    entropy_series = compute_fast_entropy_series(returns, window=window)
    valid = entropy_series.dropna()

    if len(valid) == 0:
        return {
            "current_entropy": baseline_entropy,
            "entropy_drop": 0.0,
            "spike_detected": False,
        }

    current = float(valid.iloc[-1])
    drop = max(0.0, baseline_entropy - current)

    return {
        "current_entropy": current,
        "entropy_drop": drop,
        "spike_detected": drop > 0.05,
    }
