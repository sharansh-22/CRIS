"""
run_macro_harvesters.py — Main orchestrator for Layer 3 StressField Intelligence.
Hardened for publication with absolute imports and dynamic root discovery.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Dynamic project root discovery
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Absolute Imports from the new architecture
from harvesters.macro.schema import Layer3Output
from harvesters.macro.fast_shock.detector import run_fast_shock
from harvesters.macro.slow_structural.detector import run_slow_structural
from harvesters.macro.trajectory.detector import run_trajectory_engine
from harvesters.macro.trajectory.lstm_model import LSTMState, train_on_history
from harvesters.macro.convergence.manager import run_convergence, ConvergenceState
from harvesters.macro.calibration.calibration import CalibrationState, update_calibration_state
from configs.macro_config import STRESS_THRESHOLD, CAUTION_THRESHOLD

# Default Constants (if not in config)
SPY_BASELINE_VOL = 0.012
SPY_BASELINE_PERM_ENTROPY = 0.92
ENTROPY_SAMPLE_BASELINE = 1.4

@dataclass
class Layer3State:
    """Complete persistent state for the Layer 3 pipeline."""
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
    """Run the complete Layer 3 stress_field intelligence pipeline."""
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
    """Train the LSTM on historical deterioration patterns."""
    if state is None:
        state = Layer3State()

    state.lstm = train_on_history(prices, returns, state.lstm, epochs=epochs)
    return state

if __name__ == "__main__":
    logger.info("Macro Harvester Orchestrator Initialized.")
