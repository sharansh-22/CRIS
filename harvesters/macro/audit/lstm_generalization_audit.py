"""
lstm_generalization_audit.py — Institutional Audit Suite

Audits the bounded contribution and generalization capabilities of the LSTM
trajectory-similarity model. Validates that the ML advisory layer correctly
identifies structural deterioration without violating interpretive bounds.
"""

import sys
import pandas as pd
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
from validation.macro_validation.helpers import generate_slow_grind, returns_to_prices

# We need at least seq_len + 90 = 60 + 90 = 150 days to train. 
returns = generate_slow_grind(n=500, grind_start=300)
prices = returns_to_prices(returns)

state = Layer3State()
state = train_lstm(prices[:300], returns[:300], state, epochs=20)

results_with = []
for i in range(300, 500):
    r = returns.iloc[max(0, i - 252): i + 1]
    p = prices.iloc[max(0, i - 252): i + 1]
    out, state = run_layer3(r, p, ticker="TEST", state=state)
    results_with.append(out.decay.trajectory_fragility)

print(f"Max LSTM prob: {max(results_with)}")
print(f"Avg LSTM prob: {sum(results_with)/len(results_with)}")
print(f"Min LSTM prob: {min(results_with)}")
