"""
recovery.py — Recovery dynamics for the convergence layer.

Models the lifecycle of recovery from stress events:
  NONE → (stress detected) → EARLY_RECOVERY → SUSTAINED → CONFIRMED
                                    ↓
                              FAILED_RECOVERY → back to stress

Key behaviors:
  - Confidence RELAXES after stress subsides (exponential decay)
  - Recovery must be SUSTAINED (not instant normalization)
  - Failed rebounds re-escalate (system doesn't stay permanently fearful)
  - System should NOT remain permanently elevated after a crisis
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..schema import RecoveryPhase
from ..config import (
    RECOVERY_DECAY_RATE,
    RECOVERY_CONFIRM_DAYS,
    RECOVERY_EARLY_DAYS,
    RECOVERY_FAILED_REBOUND_THRESHOLD,
)


@dataclass
class RecoveryState:
    """Tracks the recovery lifecycle."""
    phase: RecoveryPhase = RecoveryPhase.NONE
    calm_streak: int = 0        # Consecutive days of low risk
    peak_risk_memory: float = 0.0  # Highest risk seen during last stress event
    days_since_peak: int = 0    # Days since peak risk


def update_recovery(
    current_risk: float,
    state: RecoveryState,
    risk_threshold: float = 0.30,
) -> RecoveryState:
    """Update recovery state based on current overall risk.

    The recovery lifecycle:
    1. NONE: No recent stress. System is calm.
    2. When risk rises above threshold → track peak, reset calm streak
    3. When risk falls below threshold → enter EARLY recovery
    4. After EARLY_DAYS of sustained calm → SUSTAINED
    5. After CONFIRM_DAYS of sustained calm → CONFIRMED (fully normal)
    6. If risk spikes during EARLY/SUSTAINED → FAILED_RECOVERY

    Returns:
        Updated RecoveryState
    """
    # Track peak risk
    if current_risk > state.peak_risk_memory:
        state.peak_risk_memory = current_risk
        state.days_since_peak = 0
    else:
        state.days_since_peak += 1

    # Is the market currently calm?
    is_calm = current_risk < risk_threshold

    if is_calm:
        state.calm_streak += 1
    else:
        # Risk elevated — check if this is a failed recovery
        if state.phase in (RecoveryPhase.EARLY_RECOVERY, RecoveryPhase.SUSTAINED_RECOVERY):
            if current_risk > RECOVERY_FAILED_REBOUND_THRESHOLD:
                state.phase = RecoveryPhase.FAILED_RECOVERY
                state.calm_streak = 0
                return state

        # Reset calm streak if risk is elevated
        if current_risk > risk_threshold * 1.5:
            state.calm_streak = 0

    # Phase transitions
    if state.phase == RecoveryPhase.NONE:
        if not is_calm and state.peak_risk_memory > risk_threshold:
            # Stress detected, stay in NONE (waiting for calm)
            pass
        elif is_calm and state.peak_risk_memory > risk_threshold:
            # Just became calm after stress
            state.phase = RecoveryPhase.EARLY_RECOVERY

    elif state.phase == RecoveryPhase.EARLY_RECOVERY:
        if state.calm_streak >= RECOVERY_EARLY_DAYS:
            state.phase = RecoveryPhase.SUSTAINED_RECOVERY

    elif state.phase == RecoveryPhase.SUSTAINED_RECOVERY:
        if state.calm_streak >= RECOVERY_CONFIRM_DAYS:
            state.phase = RecoveryPhase.CONFIRMED_NORMAL
            state.peak_risk_memory = 0.0  # Reset memory

    elif state.phase == RecoveryPhase.CONFIRMED_NORMAL:
        state.phase = RecoveryPhase.NONE  # Return to baseline
        state.peak_risk_memory = 0.0
        state.calm_streak = 0

    elif state.phase == RecoveryPhase.FAILED_RECOVERY:
        # After a failed recovery, we need sustained calm to try again
        if state.calm_streak >= RECOVERY_EARLY_DAYS:
            state.phase = RecoveryPhase.EARLY_RECOVERY

    return state


def apply_recovery_relaxation(
    raw_risk: float,
    state: RecoveryState,
    decay_rate: float = RECOVERY_DECAY_RATE,
) -> float:
    """Apply confidence relaxation during recovery phases.

    During recovery, risk should gradually decay toward baseline.
    This prevents the system from remaining permanently fearful.

    The relaxation is exponential: risk_t = raw_risk * exp(-decay_rate * calm_days)
    """
    if state.phase == RecoveryPhase.NONE:
        return raw_risk  # No relaxation needed

    if state.phase == RecoveryPhase.FAILED_RECOVERY:
        return raw_risk  # Don't relax during failed recovery

    if state.phase == RecoveryPhase.CONFIRMED_NORMAL:
        return raw_risk * 0.5  # Strong dampening for confirmed recovery

    # Exponential relaxation during EARLY and SUSTAINED recovery
    relaxation_factor = np.exp(-decay_rate * state.calm_streak)
    relaxed = raw_risk * relaxation_factor

    # Floor: don't relax below 10% of original during EARLY
    if state.phase == RecoveryPhase.EARLY_RECOVERY:
        relaxed = max(relaxed, raw_risk * 0.3)

    return float(np.clip(relaxed, 0.0, 1.0))
