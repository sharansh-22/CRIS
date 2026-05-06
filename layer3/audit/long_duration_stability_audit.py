"""
long_duration_stability_audit.py — Institutional Audit Suite

Executes a full-scale, 10-year behavioral stress audit. Evaluates long-term 
probabilistic drift, signal normalization across historical eras, and 
downstream operational stability.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from layer3.orchestrator import run_layer3, Layer3State, train_lstm
from layer3.validation.helpers import returns_to_prices

OUTPUT_DIR = Path(__file__).resolve().parent / "output_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _setup_style():
    plt.style.use('dark_background')

def run_scenario(returns: pd.Series, warmup: int = 100, baseline_vol: float = 0.01) -> pd.DataFrame:
    prices = returns_to_prices(returns)
    state = Layer3State()
    state = train_lstm(prices, returns, state, epochs=10)
    
    results = []
    start = max(warmup, 50)
    
    # Store daily fields
    for i in range(start, len(returns)):
        r = returns.iloc[max(0, i - 252): i + 1]
        p = prices.iloc[max(0, i - 252): i + 1]
        
        output, state = run_layer3(r, p, ticker="AUDIT", baseline_vol=baseline_vol, state=state)
        
        results.append({
            "date": returns.index[i],
            "fast_shock": output.fast.shock_intensity,
            "slow_struc": output.slow.structural_instability,
            "decay_ero": output.decay.erosion_strength,
            "decay_fail": output.decay.rebound_failure,
            "decay_deficit": output.decay.resilience_deficit,
            "meta_stab": output.meta.stabilization_strength,
            "meta_unc": output.meta.uncertainty_pressure,
            "meta_coh": output.meta.signal_coherence,
            "lstm_prob": output.decay.trajectory_fragility # proxy using fragility
        })
    return pd.DataFrame(results)

def generate_multi_year_scenario(n=2500) -> pd.Series:
    """Simulates 10 years of market data with various stress_fields."""
    rng = np.random.default_rng(42)
    ret = np.zeros(n)
    
    # Base bull
    ret[:] = rng.normal(0.0005, 0.007, n)
    
    # Crises
    # Y1: Flash crash
    ret[200:205] = rng.normal(-0.04, 0.01, 5)
    ret[205:220] = rng.normal(0.015, 0.01, 15) # V rebound
    
    # Y3: Slow structural erosion (bear market)
    for i in range(500, 750):
        if i % 10 == 0: ret[i] = rng.normal(0.01, 0.01) # fake bounce
        else: ret[i] = rng.normal(-0.002, 0.012)
        
    # Y5: Prolonged sideways chop
    ret[1000:1250] = rng.normal(0, 0.008, 250)
    
    # Y7: Massive macro shock (Covid style)
    ret[1600:1620] = rng.normal(-0.03, 0.02, 20)
    ret[1620:1680] = rng.normal(0.01, 0.02, 60) # High vol rebound
    
    return pd.Series(ret, index=pd.bdate_range("2010-01-01", periods=n))

def run_all_validations():
    print("Running multi-year long-duration simulation...")
    returns_long = generate_multi_year_scenario(2500)
    df_long = run_scenario(returns_long, warmup=100)
    
    _setup_style()
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axs[0].plot(df_long['date'], df_long['fast_shock'], color='red', alpha=0.7, label='FAST Shock')
    axs[0].legend(loc='upper right')
    
    axs[1].plot(df_long['date'], df_long['slow_struc'], color='blue', alpha=0.7, label='SLOW Structural')
    axs[1].legend(loc='upper right')
    
    axs[2].plot(df_long['date'], df_long['decay_ero'], color='green', alpha=0.7, label='DECAY Erosion')
    axs[2].legend(loc='upper right')
    
    axs[3].plot(df_long['date'], df_long['meta_stab'], color='purple', alpha=0.7, label='META Stabilization')
    axs[3].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'long_duration_stability.png')
    plt.close()
    
    # 1. Long-Duration Drift Check
    late_period = df_long.iloc[-250:] # Last year (should be calm bull)
    drift_metrics = {
        "final_year_avg_fast": float(late_period['fast_shock'].mean()),
        "final_year_avg_slow": float(late_period['slow_struc'].mean()),
        "final_year_avg_decay": float(late_period['decay_ero'].mean()),
        "final_year_avg_stab": float(late_period['meta_stab'].mean())
    }
    
    # 2. Downstream Operationalization (Signal Flips)
    # Allocator rule: reduce risk if any stress > 0.6
    reduce_signal = (df_long['fast_shock'] > 0.6) | (df_long['slow_struc'] > 0.6) | (df_long['decay_ero'] > 0.6)
    signal_flips = (reduce_signal != reduce_signal.shift(1)).sum()
    downstream_metrics = {
        "total_flips_in_10_years": int(signal_flips),
        "avg_uncertainty_pressure": float(df_long['meta_unc'].mean()),
        "max_uncertainty_pressure": float(df_long['meta_unc'].max())
    }
    
    # 3. FAST Layer Dynamics
    # Isolate flash crash (idx 100-120 relative to dataframe)
    fc_idx = 200 - 100
    fc_window = df_long.iloc[fc_idx:fc_idx+30]
    fast_metrics = {
        "peak_shock_reaction": float(fc_window['fast_shock'].max()),
        "days_to_decay_below_0.3": int((fc_window['fast_shock'] < 0.3).idxmax() - fc_window.index[0]) if (fc_window['fast_shock'] < 0.3).any() else 30
    }
    
    # 4. SLOW Layer Dynamics
    # Isolate structural bear market (idx 400-650 relative)
    bear_idx = 500 - 100
    bear_window = df_long.iloc[bear_idx:bear_idx+250]
    slow_metrics = {
        "peak_structural_stress": float(bear_window['slow_struc'].max()),
        "avg_structural_stress_during_bear": float(bear_window['slow_struc'].mean())
    }
    
    # 5. DECAY Layer Dynamics
    decay_metrics = {
        "peak_erosion_during_bear": float(bear_window['decay_ero'].max()),
        "peak_erosion_during_flash_crash": float(fc_window['decay_ero'].max())
    }
    
    # 6. Recovery & Convergence Smoothing
    covid_idx = 1600 - 100
    covid_crash = df_long.iloc[covid_idx:covid_idx+20]
    covid_rebound = df_long.iloc[covid_idx+20:covid_idx+80]
    recovery_metrics = {
        "min_stab_during_crash": float(covid_crash['meta_stab'].min()),
        "max_stab_during_rebound": float(covid_rebound['meta_stab'].max())
    }
    
    # 7. LSTM Generalization
    # Test on completely unseen alien stress_field
    rng = np.random.default_rng(99)
    ret_alien = np.zeros(300)
    for i in range(300):
        ret_alien[i] = np.sin(i / 10.0) * 0.05 + rng.normal(0, 0.01) # Sinewave returns
    df_alien = run_scenario(pd.Series(ret_alien, index=pd.bdate_range("2020-01-01", periods=300)))
    lstm_metrics = {
        "max_fragility_in_alien_stress_field": float(df_alien['lstm_prob'].max()),
        "avg_fragility_in_alien_stress_field": float(df_alien['lstm_prob'].mean())
    }

    results = {
        "long_duration_drift": drift_metrics,
        "downstream_usability": downstream_metrics,
        "fast_layer": fast_metrics,
        "slow_layer": slow_metrics,
        "decay_layer": decay_metrics,
        "recovery_dynamics": recovery_metrics,
        "lstm_generalization": lstm_metrics
    }
    
    import json
    with open(OUTPUT_DIR / "audit_results_v5.json", "w") as f:
        json.dump(results, f, indent=4)
    print("V5 Final Audit Complete.")

if __name__ == "__main__":
    run_all_validations()
