"""
normalization.py — Normalization and scaling utilities for time-series features.
"""

import numpy as np

def z_score(series: np.ndarray) -> np.ndarray:
    """Compute standard Z-score of a distribution."""
    mu = np.mean(series)
    sigma = np.std(series)
    if sigma == 0: return np.zeros_like(series)
    return (series - mu) / sigma

def min_max_scale(series: np.ndarray) -> np.ndarray:
    """Scale a series to the [0, 1] range."""
    s_min = np.min(series)
    s_max = np.max(series)
    if s_max == s_min: return np.zeros_like(series)
    return (series - s_min) / (s_max - s_min)
