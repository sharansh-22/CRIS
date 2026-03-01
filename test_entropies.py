import os
import numpy as np
import pandas as pd
from layer3_bssc.engine.entropy import (
    compute_permutation_entropy,
    compute_sample_entropy,
    compute_tsallis_entropy,
    plot_multi_entropy_comparison,
    run_entropy_method_selection
)

def test_1():
    print("--- Test 1 — Basic ordering check ---")
    np.random.seed(42)
    normal = np.random.normal(0, 0.01, 252)
    crash = np.concatenate([
        np.random.normal(0, 0.01, 200),
        np.random.normal(-0.03, 0.005, 52)
    ])
    
    pe_normal = compute_permutation_entropy(normal)
    pe_crash = compute_permutation_entropy(crash)
    
    se_normal = compute_sample_entropy(normal)
    se_crash = compute_sample_entropy(crash)
    
    te_normal = compute_tsallis_entropy(normal)
    te_crash = compute_tsallis_entropy(crash)
    
    print(f"Permutation — Normal: {pe_normal:.4f} | Crash: {pe_crash:.4f}")
    print(f"Sample      — Normal: {se_normal:.4f} | Crash: {se_crash:.4f}")
    print(f"Tsallis     — Normal: {te_normal:.4f} | Crash: {te_crash:.4f}")
    
    assert se_crash > se_normal, "SampEn must rise during crash"
    print("Test 1 PASSED")

def test_2():
    print("\n--- Test 2 — Historical SPY validation ---")
    res = plot_multi_entropy_comparison(
        ticker="SPY",
        csv_path="data/Indices/SPY.csv",
        event_start="2020-02-01",
        event_end="2020-03-31"
    )
    print("Correlation matrix:")
    print(res["correlation_matrix"])
    print(f"Plot saved at: {res['plot_path']}")
    assert os.path.exists(res['plot_path']), "Plot file not found!"
    print("Test 2 PASSED")

def test_3():
    print("\n--- Test 3 — Tsallis q sensitivity ---")
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 500)
    returns[400:410] = -0.08
    
    t_q05 = compute_tsallis_entropy(returns, q=0.5)
    t_q10 = compute_tsallis_entropy(returns, q=1.0)
    t_q20 = compute_tsallis_entropy(returns, q=2.0)
    
    print(f"q=0.5 (fat tail sensitive): {t_q05:.4f}")
    print(f"q=1.0 (same as Shannon):    {t_q10:.4f}")  
    print(f"q=2.0 (common event focus): {t_q20:.4f}")
    
    assert t_q05 > t_q20, "q=0.5 must be more sensitive to fat tails"
    print("Test 3 PASSED")

def test_4():
    print("\n--- Test 4 — Method selection runs cleanly ---")
    result = run_entropy_method_selection(
        ticker="SPY",
        csv_path="data/Indices/SPY.csv"
    )
    assert "primary_method" in result
    assert "confirmation_method" in result
    assert result["primary_method"] != result["confirmation_method"]
    assert os.path.exists("data/simulation_output/entropy_method_selection.json")
    print("Test 4 PASSED")
    print(f"Primary method selected: {result['primary_method']}")
    print(f"Confirmation method: {result['confirmation_method']}")

if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
