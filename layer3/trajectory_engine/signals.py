"""
signals.py — Trajectory and resilience signals for the DECAY engine.

Computes multi-horizon (1m, 3m, 6m, 12m) structural trajectory features:
- Recovery half-life (how long it takes to heal)
- Failed rebound accumulation
- Resilience degradation
- Structural participation quality

Ignores simple directional slope to prevent trend-following bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .lstm_model import LSTMState, predict_deterioration

def compute_multi_horizon_trajectory(prices: pd.Series, returns: pd.Series) -> Dict[str, float]:
    """Analyze trajectory evolution over multiple overlapping horizons."""
    
    horizons = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    signals = {}
    
    n = len(prices)
    if n < 21:
        return {k: 0.0 for k in _get_all_keys(horizons)}
        
    for name, w in horizons.items():
        if n < w:
            w_eff = n
        else:
            w_eff = w
            
        p_window = prices.iloc[-w_eff:]
        r_window = returns.iloc[-w_eff:]
        
        # ── 1. Recovery Half-Life ──
        # Measure median time to recover from drawdowns > 2%
        running_max = p_window.cummax()
        drawdown = (running_max - p_window) / running_max
        is_dd = drawdown > 0.02
        
        recovery_days = []
        current_streak = 0
        for dd in is_dd:
            if dd:
                current_streak += 1
            elif current_streak > 0:
                recovery_days.append(current_streak)
                current_streak = 0
                
        half_life = float(np.median(recovery_days)) if recovery_days else 0.0
        # Normalize to 0-1 risk score (longer half-life = higher risk)
        hl_risk = min(1.0, half_life / 30.0) 
        
        # ── 2. Failed Rebound Accumulation ──
        # Count of sharp up-days (>1.5%) followed by lower-lows within 5 days
        failed_bounces = 0
        for i in range(len(r_window) - 5):
            if r_window.iloc[i] > 0.015:
                # Did price drop below pre-bounce level within 5 days?
                pre_bounce_p = p_window.iloc[max(0, i-1)]
                min_future_p = p_window.iloc[i+1:i+6].min()
                if min_future_p < pre_bounce_p:
                    failed_bounces += 1
                    
        # Normalize: > 3 failed bounces in a 3m window is severe
        fb_risk = min(1.0, failed_bounces / max(1, (w_eff / 21)))

        # ── 3. Stabilization Consistency ──
        # How well does the asset hold levels after drops? (Positive skewness of returns)
        skew = r_window.skew()
        skew_val = float(skew) if not pd.isna(skew) else 0.0
        stab_risk = float(np.clip((-skew_val + 0.5) / 1.5, 0.0, 1.0))
        
        # ── 4. Participation Quality Proxy ──
        # Since we only have pure price/returns, we proxy breadth/participation via UP vs DOWN volume/volatility.
        # High Downside Volatility vs Low Upside Volatility = Weak Participation
        down_vol = float(r_window[r_window < 0].std()) if len(r_window[r_window < 0]) > 1 else 0.0
        up_vol = float(r_window[r_window > 0].std()) if len(r_window[r_window > 0]) > 1 else 0.0
        part_risk = 0.0
        if up_vol > 0:
            vol_ratio = down_vol / up_vol
            part_risk = float(np.clip((vol_ratio - 1.0) / 1.0, 0.0, 1.0))
        elif down_vol > 0:
            part_risk = 1.0 # high risk if no upside vol but there is downside vol
            
        signals[f"{name}_half_life_risk"] = hl_risk
        signals[f"{name}_failed_bounce_risk"] = fb_risk
        signals[f"{name}_stab_risk"] = stab_risk
        signals[f"{name}_participation_risk"] = part_risk

    return signals

def _get_all_keys(horizons: Dict[str, int]) -> list:
    keys = []
    for name in horizons.keys():
        keys.extend([
            f"{name}_half_life_risk",
            f"{name}_failed_bounce_risk",
            f"{name}_stab_risk",
            f"{name}_participation_risk"
        ])
    return keys

def get_decay_signals(
    prices: pd.Series,
    returns: pd.Series,
    lstm_state: Optional[LSTMState] = None,
) -> Dict[str, Any]:
    """Aggregate trajectory evolution signals for DECAY."""
    
    signals = compute_multi_horizon_trajectory(prices, returns)
    
    # LSTM trajectory pattern match (max 10% influence handled in detector)
    lstm_prob = 0.0
    if lstm_state is not None and lstm_state.is_trained:
        lstm_prob = predict_deterioration(prices, returns, lstm_state)
        
    signals["lstm_deterioration_prob"] = lstm_prob
    
    return signals
