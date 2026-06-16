"""
correlation.py — Correlation Compression Intelligence for CRIS.

Detects systemic contagion and diversification collapse.
During stress, everything starts moving together.
Rising correlations signal that diversification is failing
and systemic risk is building.

Input: DataFrame of daily returns for multiple sector ETFs.
Output: CorrelationCompressionOutput (Pydantic typed contract).
"""

import numpy as np
import pandas as pd
from configs.macro_config import CONFIDENCE_FLOOR


def compute_rolling_correlation_density(returns: pd.DataFrame, window: int = 63) -> float:
    """Compute the average pairwise correlation across all assets.

    High density = everything is moving together (contagion).
    Low density = healthy diversification.
    """
    if returns.empty or returns.shape[1] < 2 or len(returns) < window:
        return 0.0

    recent = returns.iloc[-window:]
    corr_matrix = recent.corr()

    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    pairwise = corr_matrix.values[mask]
    pairwise = pairwise[~np.isnan(pairwise)]

    if len(pairwise) == 0:
        return 0.0

    avg_corr = float(np.mean(pairwise))
    # Normalize: correlation ranges [-1, 1], map to [0, 1]
    # avg_corr of 0.8+ is extreme compression
    return float(np.clip(avg_corr, 0.0, 1.0))


def compute_cross_sector_coupling(returns: pd.DataFrame, window: int = 63) -> float:
    """Measure how tightly sectors are coupled using the first eigenvalue.

    The fraction of variance explained by the first principal component
    of the correlation matrix indicates systemic coupling.
    High value = one common factor dominates (contagion).
    """
    if returns.empty or returns.shape[1] < 3 or len(returns) < window:
        return 0.0

    recent = returns.iloc[-window:].dropna(axis=1)
    if recent.shape[1] < 3:
        return 0.0

    corr_matrix = recent.corr().values
    # Handle NaN in correlation matrix
    if np.any(np.isnan(corr_matrix)):
        return 0.0

    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

    total = float(np.sum(eigenvalues))
    if total <= 0:
        return 0.0

    # Fraction explained by the first eigenvalue
    first_explained = float(eigenvalues[0] / total)

    # Normalize: in a perfectly uncorrelated market of N assets,
    # first eigenvalue explains 1/N. In contagion, it approaches 1.0.
    n = len(eigenvalues)
    baseline = 1.0 / n
    coupling = float(np.clip((first_explained - baseline) / (1.0 - baseline), 0.0, 1.0))
    return coupling


def compute_contagion_acceleration(
    returns: pd.DataFrame,
    short_window: int = 21,
    long_window: int = 63,
) -> float:
    """Detect rapid increases in correlation density.

    Compares short-term average correlation to long-term average.
    Rising short-term correlation relative to long-term signals
    accelerating contagion.
    """
    if returns.empty or returns.shape[1] < 2 or len(returns) < long_window:
        return 0.0

    short_corr = compute_rolling_correlation_density(returns, short_window)
    long_corr = compute_rolling_correlation_density(returns, long_window)

    # Acceleration = how much short-term exceeds long-term
    acceleration = short_corr - long_corr

    # Normalize: a 0.2 increase in avg correlation is significant
    return float(np.clip(acceleration / 0.20, 0.0, 1.0))


def compute_diversification_failure(returns: pd.DataFrame, window: int = 63) -> float:
    """Measure the degree to which portfolio diversification is collapsing.

    Uses the effective number of independent bets (1 / HHI of eigenvalues)
    as a proxy. When this drops, diversification is failing.
    """
    if returns.empty or returns.shape[1] < 3 or len(returns) < window:
        return 0.0

    recent = returns.iloc[-window:].dropna(axis=1)
    if recent.shape[1] < 3:
        return 0.0

    corr_matrix = recent.corr().values
    if np.any(np.isnan(corr_matrix)):
        return 0.0

    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)  # Clip negative eigenvalues
    total = float(np.sum(eigenvalues))
    if total <= 0:
        return 0.0

    # Normalized eigenvalue weights
    weights = eigenvalues / total
    # HHI of eigenvalue distribution
    hhi = float(np.sum(weights ** 2))

    # Effective number of independent sources = 1/HHI
    n = len(eigenvalues)
    # In a perfectly diversified market, effective_n = N
    # In complete contagion, effective_n = 1
    effective_n = 1.0 / hhi if hhi > 0 else n

    # Map to failure: 1 - (effective_n / N)
    failure = float(np.clip(1.0 - (effective_n / n), 0.0, 1.0))
    return failure


def run_correlation(
    returns: pd.DataFrame,
    short_window: int = 21,
    long_window: int = 63,
) -> "CorrelationCompressionOutput":
    """Execute the full correlation compression intelligence module.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for sector ETFs. Columns = tickers.
    short_window : int
        Short lookback for acceleration detection.
    long_window : int
        Long lookback for baseline correlation.

    Returns
    -------
    CorrelationCompressionOutput
        Typed Pydantic output.
    """
    from harvesters.macro.systemic.schema import CorrelationCompressionOutput

    corr_density = compute_rolling_correlation_density(returns, long_window)
    coupling = compute_cross_sector_coupling(returns, long_window)
    acceleration = compute_contagion_acceleration(returns, short_window, long_window)
    div_failure = compute_diversification_failure(returns, long_window)

    n_assets = returns.shape[1] if not returns.empty else 0
    n_days = len(returns)
    data_confidence = min(1.0, n_assets / 5.0) * min(1.0, n_days / 63.0)
    confidence = float(np.clip(data_confidence, CONFIDENCE_FLOOR, 1.0))

    return CorrelationCompressionOutput(
        correlation_density=round(corr_density, 4),
        cross_sector_coupling=round(coupling, 4),
        contagion_acceleration=round(acceleration, 4),
        diversification_failure=round(div_failure, 4),
        confidence=round(confidence, 4),
    )
