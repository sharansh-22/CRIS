"""
detector.py — Slow StressField structural stress_field.

Persistent structural stress estimation.
Tracks prolonged instability, stress_field fragility, and stress accumulation.
Outputs continuous probabilistic intensities. No stress_field labels.
"""

import pandas as pd
import numpy as np
from .signals import get_slow_signals
from .persistence import compute_streak
from harvesters.macro.schema import SlowStructuralOutput
from configs.macro_config import (
    VOL_STRESS_MULTIPLIER,
    VOL_CRITICAL_MULTIPLIER,
    STRESS_PERSISTENCE_DAYS,
    CRITICAL_PERSISTENCE_DAYS,
    ENTROPY_STRESS_THRESHOLD,
    ENTROPY_CRITICAL_THRESHOLD,
    CONFIDENCE_FLOOR,
)

def run_slow_structural(
    returns: pd.Series,
    baseline_vol: float,
    baseline_entropy: float,
) -> SlowStructuralOutput:
    if len(returns) < 20:
        return SlowStructuralOutput(
            structural_instability=0.0,
            stress_persistence=0.0,
            fragility_pressure=0.0,
            confidence=CONFIDENCE_FLOOR,
        )

    signals = get_slow_signals(returns, baseline_vol, baseline_entropy)
    entropy_delta = signals["entropy_delta"]
    vol_ratio = signals["vol_ratio"]
    vol_ratio_series = signals["vol_ratio_series"]

    # ── 1. Structural Instability ──
    vol_risk = _gradual_risk(
        vol_ratio,
        low=1.0,
        mid=VOL_STRESS_MULTIPLIER,
        high=VOL_CRITICAL_MULTIPLIER,
    )

    entropy_risk = _gradual_risk(
        max(0.0, entropy_delta),
        low=0.0,
        mid=ENTROPY_STRESS_THRESHOLD,
        high=ENTROPY_CRITICAL_THRESHOLD,
    )
    
    structural_instability = float(np.clip(0.6 * vol_risk + 0.4 * entropy_risk, 0.0, 1.0))

    # ── 2. Stress Persistence ──
    stress_breach = vol_ratio_series > VOL_STRESS_MULTIPLIER
    critical_breach = vol_ratio_series > VOL_CRITICAL_MULTIPLIER
    stress_streak = compute_streak(stress_breach)
    critical_streak = compute_streak(critical_breach)

    effective_streak = critical_streak if critical_streak >= CRITICAL_PERSISTENCE_DAYS else stress_streak
    
    # Map streak directly to a continuous persistence field
    stress_persistence = float(np.clip(effective_streak / (1.5 * STRESS_PERSISTENCE_DAYS), 0.0, 1.0))

    # ── 3. Fragility Pressure ──
    # Combines current instability level with its persistence (how close is the system to breaking)
    fragility_pressure = float(np.clip(structural_instability * stress_persistence * 1.2, 0.0, 1.0))

    # ── 4. Confidence ──
    # High confidence if signals agree and stress is persistent
    signal_agreement = 1.0 - abs(vol_risk - entropy_risk)
    confidence = float(np.clip(
        0.4 * signal_agreement + 0.4 * stress_persistence + 0.2 * structural_instability,
        CONFIDENCE_FLOOR, 1.0
    ))

    return SlowStructuralOutput(
        structural_instability=round(structural_instability, 2),
        stress_persistence=round(stress_persistence, 2),
        fragility_pressure=round(fragility_pressure, 2),
        confidence=round(confidence, 2)
    )

def _gradual_risk(value: float, low: float, mid: float, high: float) -> float:
    if value <= low:
        return 0.0
    elif value <= mid:
        return 0.5 * (value - low) / (mid - low) if (mid - low) > 0 else 0.0
    elif value <= high:
        return 0.5 + 0.5 * (value - mid) / (high - mid) if (high - mid) > 0 else 0.5
    else:
        return 1.0
