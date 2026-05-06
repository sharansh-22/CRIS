"""
orchestrator.py — Main orchestrator for Layer 3 StressField Intelligence.

Wires three independent stress_field engines to the convergence dispatcher.
Each engine processes raw market data independently; the convergence
layer coordinates their outputs probabilistically.

Architecture:
  raw data ──→ [FAST]  ──→
  raw data ──→ [SLOW]  ──→ [CONVERGENCE] ──→ Layer3Output
  raw data ──→ [DECAY] ──→

The convergence layer NEVER sees raw data. It only sees engine outputs.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .schema import Layer3Output
from .fast_shock.detector import run_fast_shock
from .slow_structural.detector import run_slow_structural
from .trajectory_engine.detector import run_trajectory_engine
from .trajectory_engine.lstm_model import LSTMState, train_on_history
from .convergence.manager import run_convergence, ConvergenceState
from .calibration import CalibrationState, update_calibration_state
from .config import (
    SPY_BASELINE_VOL,
    ENTROPY_SAMPLE_BASELINE,
    SPY_BASELINE_PERM_ENTROPY,
)


@dataclass
class Layer3State:
    """Complete persistent state for the Layer 3 pipeline.

    Carries:
      - Convergence state (smoothing, evolution, recovery)
      - LSTM state (trained model on GPU)
      - Calibration state (slow environmental anchors)
    """
    convergence: ConvergenceState = None
    lstm: LSTMState = None
    calibration: CalibrationState = None

    def __post_init__(self):
        if self.convergence is None:
            self.convergence = ConvergenceState()
        if self.lstm is None:
            self.lstm = LSTMState()
        if self.calibration is None:
            self.calibration = CalibrationState()


def run_layer3(
    returns: pd.Series,
    prices: pd.Series,
    ticker: str = "SPY",
    baseline_vol: float = SPY_BASELINE_VOL,
    baseline_perm: float = SPY_BASELINE_PERM_ENTROPY,
    baseline_sample: float = ENTROPY_SAMPLE_BASELINE,
    state: Optional[Layer3State] = None,
    enable_adaptive_calibration: bool = True,
) -> tuple:
    """Run the complete Layer 3 stress_field intelligence pipeline.

    Pipeline:
    1. Fast Shock:   Analyzes raw returns for sudden instability
    2. Slow StressField:  Analyzes raw returns for persistent stress
    3. Decay StressField: Analyzes raw prices/returns for trend deterioration
                     (with optional LSTM advisory)
    4. Convergence:  Coordinates engine outputs probabilistically
                     (with recovery dynamics and uncertainty)

    Each engine is FULLY INDEPENDENT. They share no state, no signals,
    and no intermediate computations. The convergence layer applies
    bounded inter-layer influence AFTER all engines have reported.

    Args:
        returns: Daily return series
        prices: Close price series
        ticker: Asset identifier
        baseline_vol: Historical baseline daily volatility
        baseline_perm: Historical baseline permutation entropy
        baseline_sample: Historical baseline sample entropy
        state: Persistent state (convergence + LSTM)
        enable_adaptive_calibration: Use slow, versioned environmental anchors
            when enough long-horizon history is available.

    Returns:
        Tuple of (Layer3Output, updated Layer3State)
    """
    if state is None:
        state = Layer3State()

    state.calibration.ensure_initialized(
        baseline_vol=baseline_vol,
        baseline_perm=baseline_perm,
        baseline_sample=baseline_sample,
    )
    if enable_adaptive_calibration:
        active_baseline_vol, active_baseline_perm, active_baseline_sample = state.calibration.current_baselines()
    else:
        active_baseline_vol, active_baseline_perm, active_baseline_sample = (
            baseline_vol,
            baseline_perm,
            baseline_sample,
        )
    calibration_metadata = state.calibration.to_metadata() if enable_adaptive_calibration else None

    # ── 1. Independent Engine Execution ──
    fast_out = run_fast_shock(
        returns=returns,
        baseline_vol=active_baseline_vol,
        baseline_perm_entropy=active_baseline_perm,
    )

    slow_out = run_slow_structural(
        returns=returns,
        baseline_vol=active_baseline_vol,
        baseline_entropy=active_baseline_sample,
    )

    decay_out = run_trajectory_engine(
        prices=prices,
        returns=returns,
        lstm_state=state.lstm if state.lstm.is_trained else None,
    )

    # ── 2. Convergence Coordination ──
    meta_out, state.convergence = run_convergence(
        fast=fast_out,
        slow=slow_out,
        decay=decay_out,
        state=state.convergence,
    )

    # ── 3. Assemble Output ──
    output = Layer3Output(
        ticker=ticker,
        fast=fast_out,
        slow=slow_out,
        decay=decay_out,
        meta=meta_out,
        calibration=calibration_metadata,
    )

    if enable_adaptive_calibration:
        state.calibration = update_calibration_state(
            state.calibration,
            returns=returns,
            stress_context={
                "fast_shock": fast_out.shock_intensity,
                "slow_structural": slow_out.structural_instability,
                "decay_erosion": decay_out.erosion_strength,
                "uncertainty": meta_out.uncertainty_pressure,
            },
        )

    return output, state


def train_lstm(
    prices: pd.Series,
    returns: pd.Series,
    state: Optional[Layer3State] = None,
    epochs: int = 30,
) -> Layer3State:
    """Train the LSTM on historical deterioration patterns.

    Call this ONCE with historical data before running live inference.
    The trained model is stored in the Layer3State and runs on GPU.

    Args:
        prices: Full historical price series
        returns: Full historical return series
        state: Layer3State to update
        epochs: Training epochs

    Returns:
        Updated Layer3State with trained LSTM
    """
    if state is None:
        state = Layer3State()

    state.lstm = train_on_history(prices, returns, state.lstm, epochs=epochs)
    return state
