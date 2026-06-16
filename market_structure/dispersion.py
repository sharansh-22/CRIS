"""
dispersion.py — Cross-Sectional Dispersion Intelligence for CRIS.

Measures disagreement and fragmentation inside the market.
High dispersion often appears during regime transitions,
uncertainty escalation, and fragmentation events.

Input: DataFrame of daily returns for multiple sector ETFs.
Output: DispersionOutput (Pydantic typed contract).
"""

import numpy as np
import pandas as pd
from configs.macro_config import CONFIDENCE_FLOOR


# Sector classification for rotation analysis
CYCLICAL_SECTORS = {"XLY", "XLI", "XLB", "XLF", "XLE"}
DEFENSIVE_SECTORS = {"XLU", "XLP", "XLV", "XLRE"}


def compute_cross_sectional_dispersion(returns: pd.DataFrame, window: int = 21) -> float:
    """Compute the rolling cross-sectional standard deviation of returns.

    High dispersion = assets are disagreeing about direction.
    Normalized to [0, 1] using empirical thresholds.
    """
    if returns.empty or len(returns) < window:
        return 0.0

    # Rolling mean of daily cross-sectional std
    daily_xstd = returns.iloc[-window:].std(axis=1)
    avg_dispersion = float(daily_xstd.mean())

    # Normalize: 0.03 daily cross-sectional std is extreme
    return float(np.clip(avg_dispersion / 0.03, 0.0, 1.0))


def compute_sector_dispersion(returns: pd.DataFrame, window: int = 21) -> float:
    """Compute dispersion among sector-level cumulative returns.

    Measures how much sectors are diverging from each other
    over the trailing window.
    """
    if returns.empty or len(returns) < window:
        return 0.0

    # Cumulative returns over the window for each sector
    cum_returns = (1 + returns.iloc[-window:]).prod() - 1
    valid = cum_returns.dropna()
    if len(valid) < 3:
        return 0.0

    sector_std = float(valid.std())
    # Normalize: 0.15 spread among sectors over 21 days is extreme
    return float(np.clip(sector_std / 0.15, 0.0, 1.0))


def compute_leadership_instability(returns: pd.DataFrame, window: int = 21) -> float:
    """Measure how unstable sector leadership rankings are.

    Computes rank correlation between first-half and second-half
    performance within the window. Low correlation = unstable leadership.
    """
    if returns.empty or len(returns) < window:
        return 0.0

    half = window // 2
    recent = returns.iloc[-window:]
    first_half = (1 + recent.iloc[:half]).prod() - 1
    second_half = (1 + recent.iloc[half:]).prod() - 1

    valid_cols = first_half.dropna().index.intersection(second_half.dropna().index)
    if len(valid_cols) < 3:
        return 0.0

    # Spearman rank correlation
    rank1 = first_half[valid_cols].rank()
    rank2 = second_half[valid_cols].rank()
    corr = float(rank1.corr(rank2))

    # Transform: low correlation = high instability
    instability = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    return instability


def compute_defensive_rotation_pressure(
    returns: pd.DataFrame,
    window: int = 21,
    cyclical_cols: set = None,
    defensive_cols: set = None,
) -> float:
    """Detect rotation from cyclical to defensive sectors.

    When defensive sectors outperform cyclical sectors consistently,
    it signals risk-off behavior and environmental stress.
    """
    cyclical = cyclical_cols or CYCLICAL_SECTORS
    defensive = defensive_cols or DEFENSIVE_SECTORS

    if returns.empty or len(returns) < window:
        return 0.0

    available_cyclical = [c for c in returns.columns if c in cyclical]
    available_defensive = [c for c in returns.columns if c in defensive]

    if not available_cyclical or not available_defensive:
        return 0.0

    recent = returns.iloc[-window:]
    cyc_perf = float((1 + recent[available_cyclical].mean(axis=1)).prod() - 1)
    def_perf = float((1 + recent[available_defensive].mean(axis=1)).prod() - 1)

    # Positive rotation = defensive outperforming cyclical
    rotation_spread = def_perf - cyc_perf

    # Normalize: a 0.10 spread over 21 days is strong rotation
    return float(np.clip(rotation_spread / 0.10, 0.0, 1.0))


def run_dispersion(
    returns: pd.DataFrame,
    window: int = 21,
) -> "DispersionOutput":
    """Execute the full dispersion intelligence module.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for sector ETFs. Columns = tickers.
    window : int
        Lookback window for dispersion computation.

    Returns
    -------
    DispersionOutput
        Typed Pydantic output.
    """
    from harvesters.macro.systemic.schema import DispersionOutput

    xs_disp = compute_cross_sectional_dispersion(returns, window)
    sec_disp = compute_sector_dispersion(returns, window)
    lead_instab = compute_leadership_instability(returns, window)
    def_rot = compute_defensive_rotation_pressure(returns, window)

    n_assets = returns.shape[1] if not returns.empty else 0
    n_days = len(returns)
    data_confidence = min(1.0, n_assets / 5.0) * min(1.0, n_days / 60.0)
    confidence = float(np.clip(data_confidence, CONFIDENCE_FLOOR, 1.0))

    return DispersionOutput(
        cross_sectional_dispersion=round(xs_disp, 4),
        sector_dispersion=round(sec_disp, 4),
        leadership_instability=round(lead_instab, 4),
        defensive_rotation_pressure=round(def_rot, 4),
        confidence=round(confidence, 4),
    )
