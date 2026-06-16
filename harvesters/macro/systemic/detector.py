"""
detector.py — Unified Market Structure Intelligence Detector.

Orchestrates breadth, dispersion, and correlation compression signals
into a single SystemicHealthOutput contract.

This detector is PARALLEL to and INDEPENDENT of the Layer 3 macro engines.
It does NOT modify Layer3Output, convergence logic, or calibration systems.

Usage:
    from harvesters.macro.systemic.detector import run_market_structure
    output = run_market_structure(returns_df, prices_df)
    print(output.summary())
"""

import numpy as np
import pandas as pd
from configs.macro_config import CONFIDENCE_FLOOR

from harvesters.macro.systemic.schema import SystemicHealthOutput
from harvesters.macro.systemic.breadth import run_breadth
from harvesters.macro.systemic.dispersion import run_dispersion
from harvesters.macro.systemic.correlation import run_correlation


def run_market_structure(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    sma_window: int = 50,
    dispersion_window: int = 21,
    correlation_short_window: int = 21,
    correlation_long_window: int = 63,
) -> SystemicHealthOutput:
    """Execute the complete market structure intelligence pipeline.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns for sector ETFs or constituents. Columns = tickers.
    prices : pd.DataFrame
        Daily close prices for the same tickers. Columns = tickers.
    sma_window : int
        Window for SMA-based breadth computation.
    dispersion_window : int
        Lookback window for dispersion signals.
    correlation_short_window : int
        Short lookback for correlation acceleration.
    correlation_long_window : int
        Long lookback for baseline correlation density.

    Returns
    -------
    SystemicHealthOutput
        Typed Pydantic output with breadth, dispersion, correlation,
        and composite summary fields.
    """
    # ── 1. Run independent signal modules ──
    breadth_out = run_breadth(returns, prices, sma_window)
    dispersion_out = run_dispersion(returns, dispersion_window)
    correlation_out = run_correlation(returns, correlation_short_window, correlation_long_window)

    # ── 2. Composite market health score ──
    # Health is the INVERSE of fragility signals.
    # Healthy = broad participation + low dispersion + low correlation compression
    breadth_health = breadth_out.advance_decline_ratio * (1.0 - breadth_out.participation_decay)
    dispersion_penalty = dispersion_out.cross_sectional_dispersion * 0.3 + dispersion_out.leadership_instability * 0.2
    correlation_penalty = correlation_out.correlation_density * 0.3 + correlation_out.diversification_failure * 0.2

    market_health_score = float(np.clip(
        breadth_health - dispersion_penalty - correlation_penalty,
        0.0, 1.0,
    ))

    # ── 3. Structural fragility composite ──
    # Fragility rises when breadth collapses, dispersion spikes, and correlations compress
    fragility = float(np.clip(
        0.30 * breadth_out.participation_decay
        + 0.15 * breadth_out.breadth_collapse_velocity
        + 0.15 * dispersion_out.sector_dispersion
        + 0.10 * dispersion_out.defensive_rotation_pressure
        + 0.15 * correlation_out.correlation_density
        + 0.15 * correlation_out.contagion_acceleration,
        0.0, 1.0,
    ))

    # ── 4. Composite confidence ──
    # Weighted average of sub-module confidences
    confidence = float(np.clip(
        0.35 * breadth_out.confidence
        + 0.30 * dispersion_out.confidence
        + 0.35 * correlation_out.confidence,
        CONFIDENCE_FLOOR, 1.0,
    ))

    return SystemicHealthOutput(
        breadth=breadth_out,
        dispersion=dispersion_out,
        correlation=correlation_out,
        market_health_score=round(market_health_score, 4),
        structural_fragility=round(fragility, 4),
        confidence=round(confidence, 4),
    )
