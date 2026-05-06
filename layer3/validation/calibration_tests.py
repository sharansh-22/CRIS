"""
calibration_tests.py - Validate slow environmental calibration governance.

These tests verify that adaptive anchors behave like slow climate adaptation,
not fast threshold chasing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from layer3.calibration import (
    CalibrationState,
    compute_calibration_candidate,
    update_calibration_state,
)
from layer3.orchestrator import Layer3State, run_layer3
from layer3.validation.helpers import returns_to_prices


def _returns(n: int, sigma: float, seed: int = 42, drift: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(drift, sigma, n),
        index=pd.bdate_range("2015-01-01", periods=n),
    )


def test_crisis_freeze_resists_normalization():
    print("=" * 60)
    print("TEST: Calibration freeze resists crisis normalization")
    print("=" * 60)

    state = CalibrationState().ensure_initialized(baseline_vol=0.01)
    old_anchor = state.volatility_anchor
    crisis_returns = _returns(1300, 0.035)

    state = update_calibration_state(
        state,
        crisis_returns,
        stress_context={
            "fast_shock": 0.95,
            "slow_structural": 0.20,
            "decay_erosion": 0.10,
            "uncertainty": 0.40,
        },
        force=True,
    )

    assert state.freeze_active, "Calibration should freeze during extreme FAST stress"
    assert state.freeze_reason == "FAST_STRESS_FREEZE"
    assert state.volatility_anchor == old_anchor, "Frozen calibration must not absorb crisis volatility"
    print("  PASSED\n")


def test_bounded_adaptation_velocity():
    print("=" * 60)
    print("TEST: Anchor movement is velocity capped")
    print("=" * 60)

    state = CalibrationState().ensure_initialized(baseline_vol=0.01)
    high_vol_stable = _returns(1300, 0.035)

    state = update_calibration_state(
        state,
        high_vol_stable,
        stress_context={
            "fast_shock": 0.10,
            "slow_structural": 0.10,
            "decay_erosion": 0.10,
            "uncertainty": 0.10,
        },
        force=True,
    )

    assert state.last_update_approved, "Low-stress long history should permit a bounded update"
    assert state.volatility_anchor <= 0.0105 + 1e-12, \
        f"Anchor moved too fast: {state.volatility_anchor}"
    assert state.anchor_version == 1, "Approved bounded update should increment version"
    print("  PASSED\n")


def test_multi_horizon_memory_blends_slowly():
    print("=" * 60)
    print("TEST: Multi-horizon candidate is not dominated by recent year")
    print("=" * 60)

    low = _returns(1008, 0.006, seed=1)
    high = _returns(252, 0.030, seed=2)
    high.index = pd.bdate_range(low.index[-1] + pd.offsets.BDay(1), periods=252)
    transition = pd.concat([low, high])

    candidate = compute_calibration_candidate(transition)
    short_anchor = candidate.short_anchor["volatility_anchor"]
    blended_anchor = candidate.volatility_anchor

    assert blended_anchor < short_anchor, \
        "Deep/medium anchors should prevent the recent year from dominating"
    assert candidate.deep_anchor, "Deep anchor should be present with five years of data"
    print("  PASSED\n")


def test_output_metadata_uses_previous_anchor():
    print("=" * 60)
    print("TEST: Inference uses approved prior anchor before updating")
    print("=" * 60)

    returns = _returns(1300, 0.012)
    prices = returns_to_prices(returns)
    state = Layer3State()
    state.calibration.ensure_initialized(baseline_vol=0.01)

    output, state = run_layer3(
        returns=returns,
        prices=prices,
        ticker="CAL",
        baseline_vol=0.01,
        state=state,
    )

    used_anchor = output.calibration["volatility_anchor"]
    updated_anchor = state.calibration.volatility_anchor

    assert used_anchor == 0.01, "Output metadata should reflect anchor used for that inference"
    assert 0.0095 - 1e-12 <= updated_anchor <= 0.0105 + 1e-12, \
        "Next anchor should move only by governed velocity cap"
    assert state.calibration.anchor_version == 1, "State should version the post-inference update"
    print("  PASSED\n")


def test_insufficient_history_is_inert():
    print("=" * 60)
    print("TEST: Insufficient history does not update calibration")
    print("=" * 60)

    state = CalibrationState().ensure_initialized(baseline_vol=0.01)
    short_history = _returns(300, 0.020)

    state = update_calibration_state(
        state,
        short_history,
        stress_context={"fast_shock": 0.10, "uncertainty": 0.10},
        force=True,
    )

    assert state.anchor_version == 0, "Short history must not approve adaptive calibration"
    assert state.last_update_reason == "INSUFFICIENT_HISTORY"
    assert state.volatility_anchor == 0.01
    print("  PASSED\n")


def run_all():
    tests = [
        test_crisis_freeze_resists_normalization,
        test_bounded_adaptation_velocity,
        test_multi_horizon_memory_blends_slowly,
        test_output_metadata_uses_previous_anchor,
        test_insufficient_history_is_inert,
    ]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except (AssertionError, Exception) as exc:
            print(f"  FAILED: {exc}\n")
            failed += 1
    return passed, failed


if __name__ == "__main__":
    p, f = run_all()
    print(f"Calibration: {p} passed, {f} failed")
