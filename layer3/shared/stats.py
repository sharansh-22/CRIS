"""
stats.py — Statistical utility functions for return distributions.
"""

from scipy import stats
import numpy as np

def compute_return_stats(returns: np.ndarray) -> dict:
    """Compute standard higher-order moments (skew, kurtosis)."""
    rets = returns[~np.isnan(returns)]
    if len(rets) < 2:
        return {"skew": 0.0, "kurtosis": 0.0}
    return {
        "skew": float(stats.skew(rets)),
        "kurtosis": float(stats.kurtosis(rets))
    }
