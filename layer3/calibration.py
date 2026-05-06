"""
calibration.py - Slow environmental baseline governance for Layer 3.

This module adapts only environmental anchors, not stress semantics.
It does not optimize outcomes, tune thresholds, or learn from Layer 3
performance. Approved anchors move slowly, are versioned, and freeze during
stress so crisis behavior is not absorbed as normal background structure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    ENTROPY_SAMPLE_BASELINE,
    SPY_BASELINE_PERM_ENTROPY,
    SPY_BASELINE_VOL,
)
from .fast_shock.entropy import compute_fast_entropy_series
from .shared.rolling import apply_rolling
from .slow_structural.entropy import compute_sample_entropy


MIN_CALIBRATION_WINDOW = 756       # about 3 trading years
SHORT_ANCHOR_WINDOW = 252          # about 1 trading year
MEDIUM_ANCHOR_WINDOW = 756         # about 3 trading years
DEEP_ANCHOR_WINDOW = 1260          # about 5 trading years
UPDATE_INTERVAL_STEPS = 63         # quarterly in daily data

MAX_ANCHOR_CHANGE_PER_UPDATE = 0.05
VOL_ANCHOR_FLOOR = 0.0005
VOL_ANCHOR_CAP = 0.08
ENTROPY_ANCHOR_FLOOR = 0.05
ENTROPY_ANCHOR_CAP = 1.0

FAST_FREEZE_THRESHOLD = 0.85
SLOW_FREEZE_THRESHOLD = 0.80
DECAY_FREEZE_THRESHOLD = 0.80
UNCERTAINTY_FREEZE_THRESHOLD = 0.70


@dataclass
class CalibrationCandidate:
    """Candidate environmental anchors computed from long raw history."""

    volatility_anchor: float
    fast_perm_entropy_anchor: float
    slow_sample_entropy_anchor: float
    calibration_window_days: int
    short_anchor: Dict[str, float] = field(default_factory=dict)
    medium_anchor: Dict[str, float] = field(default_factory=dict)
    deep_anchor: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "volatility_anchor": round(self.volatility_anchor, 8),
            "fast_perm_entropy_anchor": round(self.fast_perm_entropy_anchor, 6),
            "slow_sample_entropy_anchor": round(self.slow_sample_entropy_anchor, 6),
            "calibration_window_days": self.calibration_window_days,
            "short_anchor": self.short_anchor,
            "medium_anchor": self.medium_anchor,
            "deep_anchor": self.deep_anchor,
        }


@dataclass
class CalibrationState:
    """Versioned approved environmental anchors for Layer 3 normalization.

    The state answers five governance questions:
      - what adapts: volatility and entropy anchors only
      - why: long-run environmental normalization
      - how fast: quarterly with capped anchor movement
      - what constrains it: hard bounds, robust statistics, stress freeze
      - what prevents runaway drift: no outcome feedback or recursion
    """

    volatility_anchor: Optional[float] = None
    fast_perm_entropy_anchor: Optional[float] = None
    slow_sample_entropy_anchor: Optional[float] = None
    anchor_version: int = 0
    effective_date: Optional[str] = None
    calibration_window_days: int = 0
    observations_since_update: int = UPDATE_INTERVAL_STEPS
    freeze_active: bool = False
    freeze_reason: str = "UNINITIALIZED"
    last_candidate: Optional[Dict[str, Any]] = None
    last_update_approved: bool = False
    last_update_reason: str = "UNINITIALIZED"
    max_allowed_change: float = MAX_ANCHOR_CHANGE_PER_UPDATE

    def ensure_initialized(
        self,
        baseline_vol: float = SPY_BASELINE_VOL,
        baseline_perm: float = SPY_BASELINE_PERM_ENTROPY,
        baseline_sample: float = ENTROPY_SAMPLE_BASELINE,
    ) -> "CalibrationState":
        if self.volatility_anchor is None:
            self.volatility_anchor = float(np.clip(baseline_vol, VOL_ANCHOR_FLOOR, VOL_ANCHOR_CAP))
        if self.fast_perm_entropy_anchor is None:
            self.fast_perm_entropy_anchor = float(np.clip(baseline_perm, ENTROPY_ANCHOR_FLOOR, ENTROPY_ANCHOR_CAP))
        if self.slow_sample_entropy_anchor is None:
            self.slow_sample_entropy_anchor = float(np.clip(baseline_sample, ENTROPY_ANCHOR_FLOOR, ENTROPY_ANCHOR_CAP))
        if self.effective_date is None:
            self.effective_date = datetime.utcnow().date().isoformat()
        if self.freeze_reason == "UNINITIALIZED":
            self.freeze_reason = "NONE"
            self.last_update_reason = "SEEDED_FROM_CONFIG"
        return self

    def current_baselines(self) -> Tuple[float, float, float]:
        self.ensure_initialized()
        return (
            float(self.volatility_anchor),
            float(self.fast_perm_entropy_anchor),
            float(self.slow_sample_entropy_anchor),
        )

    def to_metadata(self) -> Dict[str, Any]:
        self.ensure_initialized()
        return {
            "anchor_version": self.anchor_version,
            "effective_date": self.effective_date,
            "volatility_anchor": round(float(self.volatility_anchor), 8),
            "fast_perm_entropy_anchor": round(float(self.fast_perm_entropy_anchor), 6),
            "slow_sample_entropy_anchor": round(float(self.slow_sample_entropy_anchor), 6),
            "calibration_window_days": self.calibration_window_days,
            "observations_since_update": self.observations_since_update,
            "freeze_active": self.freeze_active,
            "freeze_reason": self.freeze_reason,
            "last_update_approved": self.last_update_approved,
            "last_update_reason": self.last_update_reason,
            "max_allowed_change": self.max_allowed_change,
            "last_candidate": self.last_candidate,
        }


def update_calibration_state(
    state: CalibrationState,
    returns: pd.Series,
    stress_context: Optional[Dict[str, float]] = None,
    effective_date: Optional[str] = None,
    force: bool = False,
) -> CalibrationState:
    """Evaluate a governed anchor update from long-horizon raw returns.

    The update is intentionally post-inference. Current Layer 3 probabilities
    should be computed from previously approved anchors; this function prepares
    anchors for future calls.
    """
    state.ensure_initialized()
    clean_returns = _clean_returns(returns)
    state.observations_since_update += 1

    if len(clean_returns) < MIN_CALIBRATION_WINDOW:
        state.freeze_active = False
        state.freeze_reason = "NONE"
        state.last_update_approved = False
        state.last_update_reason = "INSUFFICIENT_HISTORY"
        state.calibration_window_days = len(clean_returns)
        return state

    freeze_reason = _freeze_reason(stress_context)
    if freeze_reason:
        state.freeze_active = True
        state.freeze_reason = freeze_reason
        state.last_update_approved = False
        state.last_update_reason = "FROZEN_DURING_STRESS"
        state.calibration_window_days = len(clean_returns)
        state.last_candidate = compute_calibration_candidate(clean_returns).to_dict()
        return state

    state.freeze_active = False
    state.freeze_reason = "NONE"

    if not force and state.observations_since_update < UPDATE_INTERVAL_STEPS:
        state.last_update_approved = False
        state.last_update_reason = "AWAITING_GOVERNED_UPDATE_INTERVAL"
        state.calibration_window_days = len(clean_returns)
        return state

    candidate = compute_calibration_candidate(clean_returns)
    state.last_candidate = candidate.to_dict()

    old = (
        float(state.volatility_anchor),
        float(state.fast_perm_entropy_anchor),
        float(state.slow_sample_entropy_anchor),
    )
    new = (
        _bounded_anchor_step(old[0], candidate.volatility_anchor, state.max_allowed_change, VOL_ANCHOR_FLOOR, VOL_ANCHOR_CAP),
        _bounded_anchor_step(old[1], candidate.fast_perm_entropy_anchor, state.max_allowed_change, ENTROPY_ANCHOR_FLOOR, ENTROPY_ANCHOR_CAP),
        _bounded_anchor_step(old[2], candidate.slow_sample_entropy_anchor, state.max_allowed_change, ENTROPY_ANCHOR_FLOOR, ENTROPY_ANCHOR_CAP),
    )

    changed = any(abs(a - b) > 1e-10 for a, b in zip(old, new))
    state.volatility_anchor, state.fast_perm_entropy_anchor, state.slow_sample_entropy_anchor = new
    state.calibration_window_days = candidate.calibration_window_days
    state.last_update_approved = changed
    state.last_update_reason = "APPROVED_BOUNDED_UPDATE" if changed else "NO_ANCHOR_CHANGE"
    state.observations_since_update = 0
    if changed:
        state.anchor_version += 1
        state.effective_date = effective_date or _last_index_date(clean_returns) or datetime.utcnow().date().isoformat()
    return state


def compute_calibration_candidate(returns: pd.Series) -> CalibrationCandidate:
    """Compute robust multi-horizon environmental anchors from raw returns."""
    clean_returns = _clean_returns(returns)
    if len(clean_returns) < MIN_CALIBRATION_WINDOW:
        raise ValueError("Calibration candidate requires at least 756 observations.")

    fast_entropy = compute_fast_entropy_series(clean_returns).dropna()
    slow_entropy = apply_rolling(clean_returns, 30, compute_sample_entropy).dropna()

    short = _horizon_anchors(clean_returns, fast_entropy, slow_entropy, SHORT_ANCHOR_WINDOW)
    medium = _horizon_anchors(clean_returns, fast_entropy, slow_entropy, MEDIUM_ANCHOR_WINDOW)
    deep = _horizon_anchors(clean_returns, fast_entropy, slow_entropy, DEEP_ANCHOR_WINDOW)

    available = []
    if short:
        available.append((0.10, short))
    if medium:
        available.append((0.30, medium))
    if deep:
        available.append((0.60, deep))
    total_weight = sum(w for w, _ in available)

    blended = {}
    for key in ("volatility_anchor", "fast_perm_entropy_anchor", "slow_sample_entropy_anchor"):
        blended[key] = sum((w / total_weight) * anchors[key] for w, anchors in available)

    return CalibrationCandidate(
        volatility_anchor=float(np.clip(blended["volatility_anchor"], VOL_ANCHOR_FLOOR, VOL_ANCHOR_CAP)),
        fast_perm_entropy_anchor=float(np.clip(blended["fast_perm_entropy_anchor"], ENTROPY_ANCHOR_FLOOR, ENTROPY_ANCHOR_CAP)),
        slow_sample_entropy_anchor=float(np.clip(blended["slow_sample_entropy_anchor"], ENTROPY_ANCHOR_FLOOR, ENTROPY_ANCHOR_CAP)),
        calibration_window_days=len(clean_returns),
        short_anchor=short,
        medium_anchor=medium,
        deep_anchor=deep,
    )


def _horizon_anchors(
    returns: pd.Series,
    fast_entropy: pd.Series,
    slow_entropy: pd.Series,
    window: int,
) -> Dict[str, float]:
    if len(returns) < window:
        return {}

    r = returns.iloc[-window:]
    calm_returns = _stress_filtered_returns(r)
    if len(calm_returns) < max(60, window // 3):
        calm_returns = _winsorize_abs_returns(r)

    fast = fast_entropy.loc[fast_entropy.index.intersection(r.index)]
    slow = slow_entropy.loc[slow_entropy.index.intersection(r.index)]

    return {
        "volatility_anchor": round(float(np.median(calm_returns.abs())), 8),
        "fast_perm_entropy_anchor": round(_robust_entropy_anchor(fast, SPY_BASELINE_PERM_ENTROPY), 6),
        "slow_sample_entropy_anchor": round(_robust_entropy_anchor(slow, ENTROPY_SAMPLE_BASELINE), 6),
    }


def _stress_filtered_returns(returns: pd.Series) -> pd.Series:
    """Exclude extreme absolute-return windows from environmental anchors."""
    abs_ret = returns.abs().dropna()
    if len(abs_ret) == 0:
        return returns.dropna()
    median = float(abs_ret.median())
    mad = float((abs_ret - median).abs().median())
    robust_cap = median + 5.0 * mad if mad > 0 else float(abs_ret.quantile(0.95))
    percentile_cap = float(abs_ret.quantile(0.95))
    cap = max(median, min(robust_cap, percentile_cap))
    return returns[returns.abs() <= cap].dropna()


def _winsorize_abs_returns(returns: pd.Series) -> pd.Series:
    cap = float(returns.abs().quantile(0.95))
    return returns.clip(lower=-cap, upper=cap).dropna()


def _robust_entropy_anchor(series: pd.Series, fallback: float) -> float:
    valid = series.dropna()
    if len(valid) == 0:
        return float(fallback)
    q_low = valid.quantile(0.05)
    q_high = valid.quantile(0.95)
    trimmed = valid[(valid >= q_low) & (valid <= q_high)]
    if len(trimmed) == 0:
        trimmed = valid
    return float(np.clip(trimmed.median(), ENTROPY_ANCHOR_FLOOR, ENTROPY_ANCHOR_CAP))


def _bounded_anchor_step(current: float, target: float, max_change: float, floor: float, cap: float) -> float:
    current = float(np.clip(current, floor, cap))
    target = float(np.clip(target, floor, cap))
    if current == 0:
        return target
    lower = current * (1.0 - max_change)
    upper = current * (1.0 + max_change)
    return float(np.clip(target, max(floor, lower), min(cap, upper)))


def _freeze_reason(stress_context: Optional[Dict[str, float]]) -> Optional[str]:
    if not stress_context:
        return None
    checks = [
        ("FAST_STRESS_FREEZE", stress_context.get("fast_shock", 0.0), FAST_FREEZE_THRESHOLD),
        ("SLOW_STRESS_FREEZE", stress_context.get("slow_structural", 0.0), SLOW_FREEZE_THRESHOLD),
        ("TRAJECTORY_STRESS_FREEZE", stress_context.get("decay_erosion", 0.0), DECAY_FREEZE_THRESHOLD),
        ("UNCERTAINTY_FREEZE", stress_context.get("uncertainty", 0.0), UNCERTAINTY_FREEZE_THRESHOLD),
        ("SYSTEMIC_STRESS_FREEZE", stress_context.get("systemic_stress", 0.0), UNCERTAINTY_FREEZE_THRESHOLD),
    ]
    triggered = [name for name, value, threshold in checks if value >= threshold]
    return ",".join(triggered) if triggered else None


def _clean_returns(returns: pd.Series) -> pd.Series:
    if returns is None:
        return pd.Series(dtype=float)
    return pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def _last_index_date(series: pd.Series) -> Optional[str]:
    if len(series) == 0:
        return None
    idx = series.index[-1]
    if hasattr(idx, "date"):
        return idx.date().isoformat()
    return str(idx)
