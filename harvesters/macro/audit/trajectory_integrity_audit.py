"""
trajectory_integrity_audit.py — Institutional Audit Suite

Evaluates the integrity of the Trajectory Erosion Engine, specifically focusing on 
decoupling structural degradation from pure price-momentum and validating LSTM 
advisory coherence under out-of-distribution stress_fields.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Dynamic project root discovery
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.run_macro_harvesters import run_layer3, Layer3State, train_lstm
from validation.macro_validation.helpers import (
    generate_calm, generate_sudden_shock, generate_persistent_crisis,
    generate_slow_grind, generate_recovery, generate_double_dip,
    returns_to_prices
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _setup_style():
    plt.style.use('dark_background')

def run_scenario(returns: pd.Series, warmup: int = 100, baseline_vol: float = 0.01) -> pd.DataFrame:
    prices = returns_to_prices(returns)
    state = Layer3State()
    
    # Train LSTM on full sequence so it's active
    state = train_lstm(prices, returns, state, epochs=15)
        
    results = []
    start = max(warmup, 50)

    for i in range(start, len(returns)):
        r = returns.iloc[max(0, i - 252): i + 1]
        p = prices.iloc[max(0, i - 252): i + 1]

        output, state = run_layer3(r, p, ticker="AUDIT", baseline_vol=baseline_vol, state=state)
        results.append({
            "date": returns.index[i],
            "fast_shock": output.fast.shock_intensity,
            "slow_struc": output.slow.structural_instability,
            "decay_ero": output.decay.erosion_strength,
            "meta_stab": output.meta.stabilization_strength,
            "meta_unc": output.meta.uncertainty_pressure,
            "lstm_prob": output.decay.lstm_deterioration_prob if hasattr(output.decay, 'lstm_deterioration_prob') else 0.0
        })

    return pd.DataFrame(results)

def audit_decay_trend_bias():
    rng = np.random.default_rng(42)
    # Secular bull pullback (10 days drop, then resume)
    ret_pullback = np.zeros(250)
    for i in range(250):
        if 150 <= i <= 160: ret_pullback[i] = rng.normal(-0.015, 0.01)
        else: ret_pullback[i] = rng.normal(0.001, 0.005)
    
    # Prolonged healthy downtrend (slow drift but no massive crashes or failed recoveries)
    ret_downtrend = np.zeros(250)
    for i in range(250):
        if i > 100: ret_downtrend[i] = rng.normal(-0.002, 0.005)
        else: ret_downtrend[i] = rng.normal(0.001, 0.005)
        
    df_p = run_scenario(pd.Series(ret_pullback, index=pd.bdate_range("2020-01-01", periods=250)))
    df_d = run_scenario(pd.Series(ret_downtrend, index=pd.bdate_range("2020-01-01", periods=250)))
    
    _setup_style()
    plt.figure(figsize=(10, 5))
    plt.plot(df_p['decay_ero'], label='Bull Pullback', color='orange')
    plt.plot(df_d['decay_ero'], label='Healthy Slow Downtrend', color='blue')
    plt.title('Audit 1: DECAY Trend Bias')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'audit1_decay.png')
    plt.close()
    
    return {
        "max_erosion_pullback": float(df_p['decay_ero'].max()),
        "max_erosion_downtrend": float(df_d['decay_ero'].max())
    }

def audit_lstm_mismatch():
    rng = np.random.default_rng(42)
    ret_unseen = np.zeros(250)
    for i in range(250):
        # Crazy unseen vol structure: huge jumps but mean reverting daily
        ret_unseen[i] = rng.choice([-0.05, 0.05]) + rng.normal(0, 0.01)
        
    df = run_scenario(pd.Series(ret_unseen, index=pd.bdate_range("2020-01-01", periods=250)))
    
    return {
        "avg_lstm_unseen": float(df['lstm_prob'].mean()),
        "max_lstm_unseen": float(df['lstm_prob'].max()),
        "std_lstm_unseen": float(df['lstm_prob'].std())
    }

def audit_downstream_coherence():
    returns = generate_persistent_crisis(n=300, crisis_start=150)
    df = run_scenario(returns)
    
    # Simulate an allocator that triggers when Any Field > 0.6
    reduce_signal = (df['fast_shock'] > 0.6) | (df['slow_struc'] > 0.6) | (df['decay_ero'] > 0.6)
    flips = (reduce_signal != reduce_signal.shift(1)).sum()
    
    return {
        "signal_flips": int(flips),
        "avg_uncertainty": float(df['meta_unc'].mean()),
        "max_uncertainty": float(df['meta_unc'].max())
    }

def audit_recovery_oversmoothing():
    rng = np.random.default_rng(42)
    ret_v = np.zeros(250)
    for i in range(250):
        if 150 <= i <= 155: ret_v[i] = rng.normal(-0.04, 0.01) # fast crash
        elif 156 <= i <= 165: ret_v[i] = rng.normal(0.03, 0.01) # fast V rebound
        else: ret_v[i] = rng.normal(0.001, 0.005)
        
    df = run_scenario(pd.Series(ret_v, index=pd.bdate_range("2020-01-01", periods=250)))
    
    _setup_style()
    plt.figure(figsize=(10, 5))
    plt.plot(df['fast_shock'], label='Shock Intensity', color='red')
    plt.plot(df['meta_stab'], label='Stabilization', color='purple')
    plt.title('Audit 4: V-Shape Recovery Oversmoothing')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'audit4_recovery.png')
    plt.close()
    
    crash_idx = 150 - 100 # offset by warmup
    rebound_end_idx = 165 - 100
    
    stab_during_crash = df.iloc[crash_idx:crash_idx+5]['meta_stab'].min()
    stab_post_rebound = df.iloc[rebound_end_idx:rebound_end_idx+20]['meta_stab'].max()
    
    return {
        "min_stab_during_crash": float(stab_during_crash),
        "max_stab_20d_post_rebound": float(stab_post_rebound)
    }

def audit_asset_normalization():
    returns_spy = generate_sudden_shock(n=250)
    # Proxy TSLA: multiply returns by 4
    returns_tsla = returns_spy * 4
    
    df_spy = run_scenario(returns_spy, baseline_vol=0.01)
    df_tsla = run_scenario(returns_tsla, baseline_vol=0.04) # Normalized by giving it correct baseline
    
    _setup_style()
    plt.figure(figsize=(10, 5))
    plt.plot(df_spy['fast_shock'], label='SPY Fast Shock', color='blue')
    plt.plot(df_tsla['fast_shock'], label='TSLA Proxy Fast Shock', color='orange', linestyle='--')
    plt.title('Audit 5: Asset Normalization (SPY vs High-Vol TSLA Proxy)')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'audit5_normalization.png')
    plt.close()
    
    diff = (df_spy['fast_shock'] - df_tsla['fast_shock']).abs().mean()
    
    return {
        "mean_absolute_difference": float(diff),
        "spy_max_shock": float(df_spy['fast_shock'].max()),
        "tsla_max_shock": float(df_tsla['fast_shock'].max())
    }

if __name__ == "__main__":
    import json
    res = {
        "decay": audit_decay_trend_bias(),
        "lstm": audit_lstm_mismatch(),
        "downstream": audit_downstream_coherence(),
        "recovery": audit_recovery_oversmoothing(),
        "normalization": audit_asset_normalization()
    }
    with open(OUTPUT_DIR / "audit_results_v3.json", "w") as f:
        json.dump(res, f, indent=4)
    print("V3 Audit Complete.")
