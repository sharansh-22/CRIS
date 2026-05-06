"""
utils.py — Shared helper functions for Layer 3.
"""

import pandas as pd
import numpy as np

def load_returns(csv_path: str) -> pd.Series:
    """Load OHLCV data and convert to log-returns."""
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    close_col = [c for c in df.columns if c[0] == "Close"]
    close = df[close_col[0]].dropna().astype(float)
    return np.log(close / close.shift(1)).dropna()
