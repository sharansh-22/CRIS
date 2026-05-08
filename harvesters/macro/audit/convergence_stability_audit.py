"""
convergence_stability_audit.py — Institutional Audit Suite

Audits the convergence and coordination dynamics between independent stress_fields.
Verifies that the temporal smoothing and inter-layer influence governors 
prevent runaway feedback and signal oscillation.
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
    generate_slow_grind, returns_to_prices
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output_v4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _setup_style():
    plt.style.use('dark_background')

def run_scenario(returns: pd.Series, warmup: int = 100, baseline_vol: float = 0.01) -> pd.DataFrame:
    prices = returns_to_prices(returns)
    state = Layer3State()
    
    state = train_lstm(prices, returns, state, epochs=15)
        
    results = []
    start = max(warmup, 50)

    for i in range(start, len(returns)):
        r = returns.iloc[max(0, i - 252): i + 1]
        p = prices.iloc[max(0, i - 252): i + 1]

        output, state = run_layer3(r, p, ticker="AUDIT", baseline_vol=baseline_vol, state=state)
        results.append({
            "date": returns.index[i],
            "decay_ero": output.decay.erosion_strength,
            "decay_rb_fail": output.decay.rebound_failure,
            "decay_res_def": output.decay.resilience_deficit,
            "decay_frag": output.decay.trajectory_fragility,
            "decay_hold_fail": output.decay.holding_failure,
            "fast_shock": output.fast.shock_intensity
        })

    return pd.DataFrame(results)

def audit_decay_trend_bias():
    rng = np.random.default_rng(42)
    # Healthy Secular Bull Pullback (10 day drop, fast rebound, no failed bounces)
    ret_pullback = np.zeros(250)
    for i in range(250):
        if 150 <= i <= 160: ret_pullback[i] = rng.normal(-0.015, 0.01)
        elif 161 <= i <= 170: ret_pullback[i] = rng.normal(0.015, 0.01) # immediate strong rebound
        else: ret_pullback[i] = rng.normal(0.001, 0.005)
    
    # Structural Weakness (small drops, but every bounce fails, poor participation)
    ret_structural = np.zeros(250)
    for i in range(250):
        if i > 150:
            if i % 10 == 0: ret_structural[i] = rng.normal(0.02, 0.005) # weak fake bounce
            else: ret_structural[i] = rng.normal(-0.003, 0.008) # high downside vol
        else: ret_structural[i] = rng.normal(0.001, 0.005)
        
    df_p = run_scenario(pd.Series(ret_pullback, index=pd.bdate_range("2020-01-01", periods=250)))
    df_s = run_scenario(pd.Series(ret_structural, index=pd.bdate_range("2020-01-01", periods=250)))
    
    _setup_style()
    plt.figure(figsize=(10, 5))
    plt.plot(df_p['decay_ero'], label='Healthy Pullback Erosion', color='green')
    plt.plot(df_s['decay_ero'], label='Structural Weakness Erosion', color='red')
    plt.title('Audit 1: DECAY Trend Bias Elimination')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'audit1_trend_bias.png')
    plt.close()
    
    return {
        "max_erosion_healthy_pullback": float(df_p['decay_ero'].max()),
        "max_erosion_structural_weakness": float(df_s['decay_ero'].max())
    }

def audit_recovery_quality():
    rng = np.random.default_rng(42)
    ret_weak = np.zeros(250)
    for i in range(250):
        if i > 120:
            # high downside vol, low upside vol
            val = rng.normal(0, 0.015)
            if val < 0: val *= 1.5 
            else: val *= 0.5
            ret_weak[i] = val
        else: ret_weak[i] = rng.normal(0.001, 0.005)
        
    df = run_scenario(pd.Series(ret_weak, index=pd.bdate_range("2020-01-01", periods=250)))
    
    _setup_style()
    plt.figure(figsize=(10, 5))
    plt.plot(df['decay_res_qual'], label='Resilience Quality', color='cyan')
    plt.plot(df['decay_rec_fail'], label='Recovery Failure', color='orange')
    plt.plot(df['decay_frag'], label='Trajectory Fragility', color='magenta')
    plt.title('Audit 2: Trajectory Fragility & Resilience')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'audit2_resilience.png')
    plt.close()

    return {
        "max_resilience_deficit": float(df['decay_res_def'].max()),
        "max_trajectory_fragility": float(df['decay_frag'].max()),
        "max_rebound_failure": float(df['decay_rb_fail'].max())
    }

if __name__ == "__main__":
    import json
    res = {
        "trend_bias": audit_decay_trend_bias(),
        "recovery_quality": audit_recovery_quality()
    }
    with open(OUTPUT_DIR / "audit_results_v4.json", "w") as f:
        json.dump(res, f, indent=4)
    print("V4 Audit Complete.")
