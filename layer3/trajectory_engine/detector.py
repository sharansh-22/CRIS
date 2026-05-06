"""
detector.py — Decay StressField structural deterioration field.

Analyzes trajectory evolution over multi-horizon windows.
Estimates resilience degradation and recovery capability rather than trend direction.
Outputs continuous probabilistic archetypes.
"""

import pandas as pd
import numpy as np
from typing import Optional
from .signals import get_decay_signals
from .lstm_model import LSTMState
from ..schema import DecayTrajectoryOutput
from ..config import CONFIDENCE_FLOOR, LSTM_INFLUENCE_CAP

def run_trajectory_engine(
    prices: pd.Series,
    returns: pd.Series,
    lstm_state: Optional[LSTMState] = None,
) -> DecayTrajectoryOutput:
    if len(prices) < 21:
        return DecayTrajectoryOutput(
            erosion_strength=0.0,
            rebound_failure=0.0,
            resilience_deficit=0.0,
            trajectory_fragility=0.0,
            holding_failure=0.0,
            confidence=CONFIDENCE_FLOOR,
        )

    signals = get_decay_signals(prices, returns, lstm_state=lstm_state)

    # ── Horizon Weighting ──
    # We emphasize 3m and 6m windows for structural decay. 
    # 1m is too noisy, 12m is too slow to react.
    weights = {"1m": 0.15, "3m": 0.35, "6m": 0.35, "12m": 0.15}
    
    def aggregate_feature(feature_suffix: str) -> float:
        val = 0.0
        for horizon, w in weights.items():
            key = f"{horizon}_{feature_suffix}"
            val += w * signals.get(key, 0.0)
        return float(np.clip(val, 0.0, 1.0))

    # ── 1. Rebound Failure ──
    # Aggregated failed bounces across multiple timeframes
    rebound_failure = aggregate_feature("failed_bounce_risk")

    # ── 2. Holding Failure ──
    # How poorly the asset holds levels after drops (inverse of stabilization consistency)
    stab_risk = aggregate_feature("stab_risk")
    holding_failure = stab_risk

    # ── 3. Resilience Deficit ──
    # High deficit = slow recovery half-life and weak upside participation
    hl_risk = aggregate_feature("half_life_risk")
    part_risk = aggregate_feature("participation_risk")
    
    resilience_deficit = float(np.clip(0.6 * hl_risk + 0.4 * part_risk, 0.0, 1.0))

    # ── 4. Trajectory Fragility ──
    # Archetype similarity to deterioration. Combines poor resilience and failing recoveries.
    trajectory_base = float(np.clip(0.5 * rebound_failure + 0.5 * resilience_deficit, 0.0, 1.0))
    
    # LSTM advisory influence (max 10%) learns the historical sequence similarity
    lstm_prob = signals.get("lstm_deterioration_prob", 0.0)
    trajectory_fragility = float(np.clip(
        trajectory_base * (1.0 - LSTM_INFLUENCE_CAP) + (lstm_prob * LSTM_INFLUENCE_CAP),
        0.0, 1.0
    ))

    # ── 5. Erosion Strength ──
    # The final, overall structural weakening metric.
    # It requires both fragility and a failure to hold levels.
    erosion_strength = float(np.clip(
        trajectory_fragility * (1.0 + holding_failure * 0.5),
        0.0, 1.0
    ))

    # ── 6. Confidence ──
    # High confidence if all timeframes agree on the trajectory structure.
    # We measure variance across horizons for the primary feature (resilience_risk)
    hl_vars = [signals.get(f"{h}_half_life_risk", 0.0) for h in weights.keys()]
    horizon_agreement = 1.0 - float(np.std(hl_vars))
    
    duration_confidence = min(1.0, len(prices) / 126.0) # Need ~6 months for high confidence
    
    confidence = float(np.clip(
        0.5 * horizon_agreement + 0.5 * duration_confidence,
        CONFIDENCE_FLOOR, 1.0
    ))

    return DecayTrajectoryOutput(
        erosion_strength=round(erosion_strength, 2),
        rebound_failure=round(rebound_failure, 2),
        resilience_deficit=round(resilience_deficit, 2),
        trajectory_fragility=round(trajectory_fragility, 2),
        holding_failure=round(holding_failure, 2),
        confidence=round(confidence, 2)
    )
