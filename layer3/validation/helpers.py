"""
helpers.py — Shared test utilities for the validation suite.

Synthetic data generators and walk-forward runner used across all test files.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from layer3.orchestrator import run_layer3, Layer3State


# ──────────────────────────────────────────────────────────
#  Synthetic Data Generators
# ──────────────────────────────────────────────────────────

def generate_calm(n: int = 300, seed: int = 42) -> pd.Series:
    """Calm market: low vol, slight uptrend."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.008, n), index=pd.bdate_range("2020-01-01", periods=n))


def generate_sudden_shock(n: int = 300, shock_day: int = 250, seed: int = 42) -> pd.Series:
    """Calm market → sudden shock (5 days of extreme negative returns)."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.008, n)
    for i in range(shock_day, min(shock_day + 5, n)):
        ret[i] = rng.normal(-0.05, 0.02)
    return pd.Series(ret, index=pd.bdate_range("2020-01-01", periods=n))


def generate_persistent_crisis(n: int = 350, crisis_start: int = 260, seed: int = 42) -> pd.Series:
    """Calm market → persistent high-vol stress."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(n):
        ret[i] = rng.normal(0.0003, 0.008) if i < crisis_start else rng.normal(-0.005, 0.035)
    return pd.Series(ret, index=pd.bdate_range("2020-01-01", periods=n))


def generate_slow_grind(n: int = 400, grind_start: int = 150, seed: int = 42) -> pd.Series:
    """Slow grind-down: low vol, persistent negative drift."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(n):
        ret[i] = rng.normal(0.0003, 0.008) if i < grind_start else rng.normal(-0.002, 0.010)
    return pd.Series(ret, index=pd.bdate_range("2019-01-01", periods=n))


def generate_recovery(n: int = 300, crash_at: int = 150, recovery_at: int = 180, seed: int = 42) -> pd.Series:
    """Crash followed by sustained recovery."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(n):
        if i < crash_at:
            ret[i] = rng.normal(0.0003, 0.008)
        elif i < recovery_at:
            ret[i] = rng.normal(-0.03, 0.04)
        else:
            ret[i] = rng.normal(0.005, 0.012)
    return pd.Series(ret, index=pd.bdate_range("2020-01-01", periods=n))


def generate_double_dip(n: int = 350, first_crash: int = 150, bounce: int = 180,
                        second_crash: int = 220, recovery: int = 260, seed: int = 42) -> pd.Series:
    """Crash → bounce → second crash → real recovery."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(n):
        if i < first_crash:
            ret[i] = rng.normal(0.0003, 0.008)
        elif i < bounce:
            ret[i] = rng.normal(-0.025, 0.035)
        elif i < second_crash:
            ret[i] = rng.normal(0.004, 0.015)   # Bounce
        elif i < recovery:
            ret[i] = rng.normal(-0.02, 0.03)    # Second dip
        else:
            ret[i] = rng.normal(0.005, 0.012)   # Real recovery
    return pd.Series(ret, index=pd.bdate_range("2020-01-01", periods=n))


def generate_fake_spike(n: int = 300, spike_day: int = 250, seed: int = 42) -> pd.Series:
    """Calm market with 1-day false volatility spike."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.008, n)
    ret[spike_day] = -0.06
    ret[spike_day + 1] = 0.04
    return pd.Series(ret, index=pd.bdate_range("2020-01-01", periods=n))


def generate_oscillating(n: int = 400, seed: int = 42) -> pd.Series:
    """Oscillating market: alternating calm/stress every 30 days."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(n):
        cycle = (i // 30) % 2
        if cycle == 0:
            ret[i] = rng.normal(0.0003, 0.008)
        else:
            ret[i] = rng.normal(-0.003, 0.025)
    return pd.Series(ret, index=pd.bdate_range("2019-01-01", periods=n))


def generate_mixed_stress_field(n: int = 300, seed: int = 42) -> pd.Series:
    """Ambiguous market with conflicting signals."""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n)
    for i in range(n):
        # Mix of small trends and occasional vol bursts
        trend = 0.001 * np.sin(i / 20.0)
        noise = rng.normal(0, 0.015)
        ret[i] = trend + noise
    return pd.Series(ret, index=pd.bdate_range("2020-01-01", periods=n))


# ──────────────────────────────────────────────────────────
#  Walk-Forward Runner
# ──────────────────────────────────────────────────────────

def returns_to_prices(returns: pd.Series, S0: float = 100.0) -> pd.Series:
    return S0 * (1 + returns).cumprod()


def run_walk_forward(returns: pd.Series, warmup: int = 200) -> pd.DataFrame:
    """Run the orchestrator day-by-day, collecting outputs.

    Maps Layer3Output fields to a flat DataFrame using the canonical schema.
    All stress fields use uniform 'high = more stress' polarity.
    """
    prices = returns_to_prices(returns)
    state = Layer3State()
    baseline_vol = float(returns.iloc[:warmup].abs().mean()) if len(returns) > warmup else 0.008

    results = []
    start = max(warmup, 50)

    for i in range(start, len(returns)):
        r = returns.iloc[max(0, i - 252): i + 1]
        p = prices.iloc[max(0, i - 252): i + 1]

        output, state = run_layer3(r, p, ticker="TEST", baseline_vol=baseline_vol, state=state)
        results.append({
            "date": returns.index[i],
            # ── FAST (Short-Horizon Instability) ──
            "fast_risk": output.fast.shock_intensity,
            "fast_conf": output.fast.confidence,
            "fast_liq": output.fast.liquidity_disruption,
            "fast_vel": output.fast.instability_velocity,
            # ── SLOW (Persistent Structural Stress) ──
            "slow_risk": output.slow.structural_instability,
            "slow_conf": output.slow.confidence,
            "slow_persist": output.slow.stress_persistence,
            "slow_fragility": output.slow.fragility_pressure,
            # ── DECAY (Trajectory Degradation) ──
            "decay_risk": output.decay.erosion_strength,
            "decay_conf": output.decay.confidence,
            "decay_rebound_fail": output.decay.rebound_failure,
            "decay_resilience_def": output.decay.resilience_deficit,
            "decay_frag": output.decay.trajectory_fragility,
            "decay_hold_fail": output.decay.holding_failure,
            # ── META (Convergence Dynamics) ──
            "overall_risk": (
                output.fast.shock_intensity * 0.4
                + output.slow.structural_instability * 0.35
                + output.decay.erosion_strength * 0.25
            ),
            "overall_conf": (
                output.fast.confidence * 0.4
                + output.slow.confidence * 0.35
                + output.decay.confidence * 0.25
            ),
            "dominant": output.meta.dominant_field.value,
            "uncertainty": output.meta.uncertainty_pressure,
            "stab_strength": output.meta.stabilization_strength,
            "coherence": output.meta.signal_coherence,
        })

    return pd.DataFrame(results)

