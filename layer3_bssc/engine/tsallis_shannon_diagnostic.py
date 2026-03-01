"""
tsallis_shannon_diagnostic.py
Diagnostic script to verify the 1.00 correlation between Tsallis and Shannon entropy.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Redirect stdout to also write to a file
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "simulation_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tee = Tee(OUTPUT_DIR / "tsallis_shannon_diagnostic.txt", "w")

from layer3_bssc.engine.entropy import (
    compute_shannon_entropy,
    compute_tsallis_entropy,
    compute_rolling_multi_entropy,
    _load_ohlcv
)

def run_diagnostic():
    # Step 1: Load SPY returns
    csv_path = DATA_DIR / "Indices" / "SPY.csv"
    df = _load_ohlcv(str(csv_path))
    close = df["Close"].dropna().astype(float)
    log_returns = np.log(close / close.shift(1)).dropna()
    log_returns = log_returns.loc[~log_returns.index.duplicated(keep="first")]
    
    event_returns = log_returns.loc["2020-02-15":"2020-03-31"]
    returns_array = event_returns.values
    
    # Step 2: Run both methods
    shannon_val = compute_shannon_entropy(returns_array, n_bins=50)
    tsallis_val = compute_tsallis_entropy(returns_array, q=0.5, n_bins=50)
    
    # Step 3: Print intermediate probability distributions
    bins = np.histogram(returns_array, bins=50)
    counts = bins[0]
    
    # Shannon
    probabilities_shannon = counts / counts.sum()
    nonzero_shannon = probabilities_shannon[probabilities_shannon > 0]
    shannon_manual = -np.sum(nonzero_shannon * np.log2(nonzero_shannon))
    # Note: entropy.py normalizes shannon by log2(n_unique) or similar max entropy
    print("Shannon probability distribution (nonzero bins):")
    print(nonzero_shannon)
    print(f"Shannon manual H: {shannon_manual:.6f}")
    print(f"Shannon from function: {shannon_val:.6f}")
    
    # Tsallis
    probabilities_tsallis = counts / counts.sum()
    nonzero_tsallis = probabilities_tsallis[probabilities_tsallis > 0]
    q = 0.5
    S_q = (1 - np.sum(nonzero_tsallis ** q)) / (q - 1)
    S_q_max = (1 - len(nonzero_tsallis) ** (1 - q)) / (q - 1)
    tsallis_manual = S_q / S_q_max if S_q_max != 0 else 0.0
    print("\nTsallis probability distribution (nonzero bins):")
    print(nonzero_tsallis)
    print(f"Tsallis manual S_q: {tsallis_manual:.6f}")
    print(f"Tsallis from function: {tsallis_val:.6f}")
    
    # Step 4: Compare distributions
    are_distributions_identical = np.allclose(nonzero_shannon, nonzero_tsallis, atol=1e-10)
    are_final_values_identical = abs(shannon_val - tsallis_val) < 1e-10
    
    print("\n--- DIAGNOSTIC VERDICT ---")
    print(f"Probability distributions identical: {are_distributions_identical}")
    print(f"Final values identical: {are_final_values_identical}")
    print(f"Shannon value:  {shannon_val:.6f}")
    print(f"Tsallis value:  {tsallis_val:.6f}")
    print(f"Absolute difference: {abs(shannon_val - tsallis_val):.8f}")
    
    # Step 5: Test with different q values
    q_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5, 2.0]
    print("\n--- TSALLIS q SENSITIVITY ---")
    print(f"{'q value':<12} {'Tsallis':>10} {'Shannon':>10} {'Difference':>12}")
    print("-" * 46)
    for q_val in q_values:
        if abs(q_val - 1.0) < 1e-9:
            t_val = shannon_val
        else:
            t_val = compute_tsallis_entropy(returns_array, q=q_val, n_bins=50)
        diff = abs(t_val - shannon_val)
        print(f"{q_val:<12} {t_val:>10.6f} {shannon_val:>10.6f} {diff:>12.8f}")
        
    diff_divergence_q = None
    for q_val in q_values:
        t_val = compute_tsallis_entropy(returns_array, q=q_val, n_bins=50)
        if abs(t_val - shannon_val) > 0.05:
            diff_divergence_q = q_val
            break
            
    # Step 6: Rolling correlation test 
    # Use full log_returns
    events = {
        "COVID": {"start": "2020-02-01", "end": "2020-03-31"},
        "Q4 2018": {"start": "2018-10-01", "end": "2018-12-31"},
        "2022": {"start": "2022-01-01", "end": "2022-06-30"}
    }
    
    min_periods = 15
    import warnings
    warnings.filterwarnings('ignore')
    
    # Calculate for required q values
    roll_shan = pd.Series(index=log_returns.index, dtype=float)
    roll_ts_01 = pd.Series(index=log_returns.index, dtype=float)
    roll_ts_05 = pd.Series(index=log_returns.index, dtype=float)
    roll_ts_20 = pd.Series(index=log_returns.index, dtype=float)
    
    window = 30
    for i in range(len(log_returns)):
        start_idx = max(0, i - window + 1)
        chunk = log_returns.iloc[start_idx : i + 1]
        if len(chunk) < min_periods:
            continue
        vals = chunk.values
        roll_shan.iloc[i] = compute_shannon_entropy(vals, n_bins=20)   # Matching the exact parameters from compute_rolling_multi_entropy
        roll_ts_01.iloc[i] = compute_tsallis_entropy(vals, q=0.1, n_bins=20)
        roll_ts_05.iloc[i] = compute_tsallis_entropy(vals, q=0.5, n_bins=20)
        roll_ts_20.iloc[i] = compute_tsallis_entropy(vals, q=2.0, n_bins=20)
        
    print("\n--- ROLLING CORRELATION WITH SHANNON BY q VALUE ---")
    print(f"{'q value':<12} {'COVID':>10} {'Q4 2018':>10} {'2022':>10} {'Mean':>10}")
    print("-" * 46)
    
    q_test_vals = [0.1, 0.5, 2.0]
    roll_ts_dict = {0.1: roll_ts_01, 0.5: roll_ts_05, 2.0: roll_ts_20}
    
    best_q = None
    max_divergence_mean = 1.0 # Looking for lowest correlation
    
    for q_val in q_test_vals:
        corrs = []
        for ev_name, ev_dates in events.items():
            ev_start = ev_dates["start"]
            ev_end = ev_dates["end"]
            s_ev = roll_shan.loc[ev_start:ev_end]
            t_ev = roll_ts_dict[q_val].loc[ev_start:ev_end]
            
            # Dropna for correlation
            valid = ~s_ev.isna() & ~t_ev.isna()
            c = s_ev[valid].corr(t_ev[valid]) if valid.sum() > 1 else 0.0
            corrs.append(c)
            
        mean_c = np.mean(corrs)
        if mean_c < max_divergence_mean:
            max_divergence_mean = mean_c
            best_q = q_val
            
        print(f"{q_val:<12.1f} {corrs[0]:>10.4f} {corrs[1]:>10.4f} {corrs[2]:>10.4f} {mean_c:>10.4f}")
        
    print("\n")
    if are_distributions_identical:
        print("FINDING: BUG CONFIRMED — Both methods use identical")
        print("probability distributions. The 1.00 correlation is")  
        print("a computational artifact. Tsallis function requires")
        print("investigation of its binning implementation.")
    elif not are_distributions_identical and abs(shannon_val - tsallis_val) < 0.1:
        print("FINDING: REAL MATHEMATICAL RESULT — Distributions differ")
        print("but produce similar values for q=0.5 on SPY returns.")
        print(f"Divergence begins at q = {diff_divergence_q}")
        print(f"Recommendation: Use q={best_q} for future Tsallis experiments")
    else:
        print("FINDING: METHODS DIVERGE — The 1.00 correlation was")
        print("a statistical artifact of these specific event windows.")
        print("Methods are genuinely independent.")

if __name__ == "__main__":
    run_diagnostic()
