"""
entropy.py — Shannon, Permutation, and Sample Entropy logic for Slow StressField detection.
"""

import numpy as np
import pandas as pd
import antropy
from ..config import DEFAULT_LOOKBACK_WINDOW

# Fixed normalization reference for Sample Entropy.
_SAMPLE_ENTROPY_NORM_CONSTANT = np.log(DEFAULT_LOOKBACK_WINDOW)

def compute_shannon_entropy(returns: np.ndarray, n_bins: int = 50) -> float:
    arr = returns[~np.isnan(returns)]
    if len(arr) == 0: return 0.0
    n_unique = len(np.unique(arr))
    if n_unique <= 1: return 0.0
    effective_bins = min(n_bins, n_unique)
    counts, _ = np.histogram(arr, bins=effective_bins)
    total = counts.sum()
    if total == 0: return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(effective_bins)
    return float(np.clip(entropy / max_entropy, 0.0, 1.0)) if max_entropy > 0 else 0.0

def compute_permutation_entropy(returns: np.ndarray, order: int = 3, delay: int = 1) -> float:
    arr = returns[~np.isnan(returns)]
    if len(arr) < order: return 0.0
    try:
        perm_en = antropy.perm_entropy(arr, order=order, delay=delay, normalize=True)
    except Exception:
        return 0.0
    return float(np.clip(perm_en, 0.0, 1.0))

def compute_sample_entropy(returns: np.ndarray, order: int = 2, tolerance: float = 0.005) -> float:
    arr = returns[~np.isnan(returns)]
    if len(arr) <= order: return 0.0
    try:
        samp_en = antropy.sample_entropy(arr, order=order, metric="chebyshev", tolerance=tolerance)
    except Exception:
        samp_en = np.nan
    if np.isnan(samp_en) or np.isinf(samp_en): return 1.0
    return float(np.clip(samp_en / _SAMPLE_ENTROPY_NORM_CONSTANT, 0.0, 1.0))

def compute_tsallis_entropy(returns: np.ndarray, q: float = 0.5, n_bins: int = 50) -> float:
    """Compute normalised Tsallis Entropy (q < 1 amplifies rare events)."""
    if np.isclose(q, 1.0): return compute_shannon_entropy(returns, n_bins=n_bins)
    arr = returns[~np.isnan(returns)]
    if len(arr) == 0: return 0.0
    n_unique = len(np.unique(arr))
    if n_unique <= 1: return 0.0
    effective_bins = min(n_bins, n_unique)
    counts, _ = np.histogram(arr, bins=effective_bins)
    total = counts.sum()
    if total == 0: return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    s_q = (1.0 - np.sum(np.power(probs, q))) / (q - 1.0)
    s_q_max = (1.0 - np.power(effective_bins, 1.0 - q)) / (q - 1.0)
    return float(np.clip(s_q / s_q_max, 0.0, 1.0)) if s_q_max > 0 else 0.0

def compute_entropy_acceleration(entropy_series: pd.Series) -> pd.Series:
    """Compute rolling 5-day mean of absolute day-over-day entropy changes."""
    return entropy_series.diff().abs().rolling(window=5, min_periods=1).mean()
