"""
detector.py — Fast Shock instability field.

Pure short-horizon instability estimation.
Detects discontinuities, volatility explosions, entropy spikes.
Confidence naturally decays as persistence increases (prolonged stress is not a shock).
"""

import pandas as pd
import numpy as np
from .signals import compute_permutation_alarm
from .volatility import compute_short_term_vol
from harvesters.macro.schema import FastShockOutput
from configs.macro_config import (
    FAST_VOL_SPIKE_THRESHOLD,
    FAST_VOL_EXTREME_THRESHOLD,
    CONFIDENCE_FLOOR,
)

def run_fast_shock(
    returns: pd.Series,
    baseline_vol: float,
    baseline_perm_entropy: float,
) -> FastShockOutput:
    if len(returns) < 5:
        return FastShockOutput(
            shock_intensity=0.0,
            liquidity_disruption=0.0,
            instability_velocity=0.0,
            confidence=CONFIDENCE_FLOOR,
        )

    # ── 1. Volatility / Shock Intensity ──
    short_vol = compute_short_term_vol(returns, window=5)
    current_vol = float(short_vol.dropna().iloc[-1]) if len(short_vol.dropna()) > 0 else baseline_vol
    vol_ratio = current_vol / baseline_vol if baseline_vol > 0 else 1.0

    shock_intensity = _sigmoid_risk(vol_ratio, center=FAST_VOL_SPIKE_THRESHOLD,
                             steepness=1.5, saturation=FAST_VOL_EXTREME_THRESHOLD)

    # ── 2. Entropy / Instability Velocity ──
    alarm_info = compute_permutation_alarm(returns, baseline_perm_entropy)
    perm_current = alarm_info["current_perm_entropy"]
    entropy_drop = max(0.0, baseline_perm_entropy - perm_current)
    instability_velocity = min(1.0, entropy_drop / 0.15)

    # ── 3. Return Magnitude / Liquidity Disruption ──
    recent_returns = returns.iloc[-5:]
    max_abs_return = float(recent_returns.abs().max())
    sigma_daily = float(returns.std()) if len(returns) > 1 else baseline_vol
    return_sigma_ratio = max_abs_return / sigma_daily if sigma_daily > 0 else 0.0

    # Large moves relative to historical vol map to liquidity disruption
    liquidity_disruption = min(1.0, max(0.0, (return_sigma_ratio - 2.0) / 4.0))

    # ── 4. Confidence and Persistence Decay ──
    signals = [shock_intensity, instability_velocity, liquidity_disruption]
    signal_strength = float(np.mean(signals))
    signal_agreement = 1.0 - float(np.std(signals))

    # Calculate persistence to penalize confidence (FAST is temporary)
    persistence = _compute_fast_persistence(short_vol, baseline_vol, threshold=FAST_VOL_SPIKE_THRESHOLD)
    
    # Decay confidence if persistence is high (> 10 days)
    persistence_penalty = 0.0
    if persistence > 10:
        persistence_penalty = min(0.8, (persistence - 10) * 0.05)

    base_confidence = float(np.clip(
        0.6 * signal_strength + 0.4 * signal_agreement,
        CONFIDENCE_FLOOR, 1.0
    ))
    
    confidence = float(np.clip(base_confidence - persistence_penalty, CONFIDENCE_FLOOR, 1.0))

    return FastShockOutput(
        shock_intensity=round(shock_intensity, 2),
        liquidity_disruption=round(liquidity_disruption, 2),
        instability_velocity=round(instability_velocity, 2),
        confidence=round(confidence, 2)
    )

def _sigmoid_risk(value: float, center: float, steepness: float, saturation: float) -> float:
    if value <= 1.0:
        return 0.0
    x = (value - 1.0) / (saturation - 1.0)
    x = max(0.0, min(x, 2.0))
    result = 1.0 / (1.0 + np.exp(-steepness * (x * 4.0 - 2.0)))
    return float(np.clip(result, 0.0, 1.0))

def _compute_fast_persistence(vol_series: pd.Series, baseline_vol: float, threshold: float) -> int:
    if len(vol_series.dropna()) == 0:
        return 0
    ratios = vol_series.dropna() / baseline_vol if baseline_vol > 0 else vol_series.dropna()
    streak = 0
    for v in reversed(ratios.values):
        if v > threshold:
            streak += 1
        else:
            break
    return streak
