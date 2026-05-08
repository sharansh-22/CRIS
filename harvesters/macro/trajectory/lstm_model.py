"""
lstm_model.py — Bounded interpretable LSTM for Decay StressField deterioration detection.

Runs on GPU (RTX 4060) when available, falls back to CPU.

This LSTM does NOT:
  - predict prices
  - predict returns
  - become a black-box forecasting engine

It ONLY:
  - learns deterioration SEQUENCE STRUCTURE
  - recognizes long-term weakening PATTERNS
  - estimates deterioration PROBABILITY

It operates on ENGINEERED DETERIORATION FEATURES, not raw prices:
  - rolling drift
  - trend slope
  - recovery weakness
  - drawdown persistence
  - momentum exhaustion
  - lower-highs score

The LSTM remains:
  - advisory (capped influence)
  - bounded (output clipped to [0, 1])
  - interpretable (single probability output)
  - secondary to structural logic
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass, field

from configs.macro_config import (
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_SEQUENCE_LENGTH,
    LSTM_FEATURE_DIM,
    LSTM_INFLUENCE_CAP,
)


# ──────────────────────────────────────────────────────────
#  Device selection — prefer GPU
# ──────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    """Select best available device (GPU > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = _get_device()


# ──────────────────────────────────────────────────────────
#  LSTM Architecture
# ──────────────────────────────────────────────────────────

class DeteriorationLSTM(nn.Module):
    """Lightweight LSTM for deterioration sequence recognition.

    Architecture:
      Input (6 features) → LSTM (32 hidden) → Linear → Sigmoid

    Output: single probability [0, 1] representing the likelihood
    that the market is in a deterioration sequence.
    """

    def __init__(
        self,
        input_dim: int = LSTM_FEATURE_DIM,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, features)

        Returns:
            Tensor of shape (batch, 1) — deterioration probability
        """
        lstm_out, _ = self.lstm(x)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


# ──────────────────────────────────────────────────────────
#  Feature Engineering (from raw market data → deterioration features)
# ──────────────────────────────────────────────────────────

def engineer_deterioration_features(
    prices: pd.Series,
    returns: pd.Series,
    seq_len: int = LSTM_SEQUENCE_LENGTH,
) -> Optional[np.ndarray]:
    """Engineer 6 structural resilience features for the LSTM.

    Features (computed per-day using trailing windows):
      1. Recovery half-life proxy (30d) — high = unable to recover
      2. Failed bounce frequency (20d)  — high = failed recoveries
      3. Stabilization risk (20d skew)  — high = frequent sharp drops
      4. Participation risk (20d vol ratio) — high = downside > upside
      5. Momentum exhaustion (20d)      — negative cumulative return
      6. Lower-highs indicator (60d)    — declining local maxima

    Returns:
        Array of shape (seq_len, 6) or None if insufficient data.
    """
    min_required = seq_len + 60  # Need lookback for feature computation
    if len(prices) < min_required or len(returns) < min_required:
        return None

    n = len(returns)
    features = np.zeros((n, 6))

    for i in range(60, n):
        w20_r = returns.iloc[max(0, i-20):i+1]
        w30_p = prices.iloc[max(0, i-30):i+1]
        
        # 1. Recovery half-life proxy (rolling drawdown depth vs time)
        peak = w30_p.max()
        current = prices.iloc[i]
        features[i, 0] = float((peak - current) / peak) if peak > 0 else 0.0
        
        # 2. Failed bounce frequency (simplified proxy)
        fb = 0
        for j in range(len(w20_r)-3):
            if w20_r.iloc[j] > 0.01 and w20_r.iloc[j+1:j+4].min() < -0.01:
                fb += 1
        features[i, 1] = float(fb)
        
        # 3. Stabilization risk (negative skewness)
        skew = w20_r.skew()
        features[i, 2] = -float(skew) if not pd.isna(skew) else 0.0
        
        # 4. Participation risk (downside vol / upside vol)
        down = w20_r[w20_r < 0].std()
        up = w20_r[w20_r > 0].std()
        if not pd.isna(down) and not pd.isna(up) and up > 0:
            features[i, 3] = float(down / up)
        else:
            features[i, 3] = 1.0
            
        # 5. Momentum exhaustion
        cum_ret = float((1 + w20_r).prod() - 1)
        features[i, 4] = cum_ret * 10
        
        # 6. Lower-highs
        recent_high = float(prices.iloc[max(0, i-20):i+1].max())
        prior_high = float(prices.iloc[max(0, i-60):max(0, i-20)].max())
        if prior_high > 0:
            features[i, 5] = (recent_high - prior_high) / prior_high
        else:
            features[i, 5] = 0.0

    # Extract the last seq_len rows
    sequence = features[-seq_len:]
    sequence = np.clip(sequence, -5.0, 5.0)

    return sequence.astype(np.float32)


# ──────────────────────────────────────────────────────────
#  LSTM State Management
# ──────────────────────────────────────────────────────────

@dataclass
class LSTMState:
    """Persistent state for the LSTM model."""
    model: Optional[DeteriorationLSTM] = None
    is_trained: bool = False
    device: torch.device = field(default_factory=_get_device)


def _ensure_model(state: LSTMState) -> LSTMState:
    """Lazily initialize the LSTM model on the correct device."""
    if state.model is None:
        state.model = DeteriorationLSTM().to(state.device)
        state.model.eval()
    return state


# ──────────────────────────────────────────────────────────
#  Self-Supervised Training
# ──────────────────────────────────────────────────────────

def train_on_history(
    prices: pd.Series,
    returns: pd.Series,
    state: LSTMState,
    epochs: int = 30,
    lr: float = 0.001,
) -> LSTMState:
    """Self-supervised training: teach the LSTM to recognize resilience breakdown.

    Labels are derived from CURRENT structural weakness:
      - High recovery failure (f2) + weak participation (f4) + lower highs (f6)
    """
    state = _ensure_model(state)
    seq_len = LSTM_SEQUENCE_LENGTH
    min_required = seq_len + 60 

    if len(prices) < min_required:
        return state

    all_features = engineer_deterioration_features(prices, returns, seq_len=len(returns) - 60)
    if all_features is None:
        return state

    sequences = []
    labels = []

    for i in range(len(all_features) - seq_len):
        seq = all_features[i:i + seq_len]
        
        f_rec_fail = seq[-1, 1]     # failed bounces
        f_part_risk = seq[-1, 3]    # participation risk
        f_lower_highs = seq[-1, 5]  # lower highs
        
        # High failed bounces, poor participation (downside vol > upside), making lower highs
        is_deteriorating = (f_rec_fail >= 2) and (f_part_risk > 1.2) and (f_lower_highs < -0.05)
        label = 1.0 if is_deteriorating else 0.0

        sequences.append(seq)
        labels.append(label)

    if len(sequences) < 20 or sum(labels) == 0:
        return state

    X = torch.tensor(np.array(sequences), dtype=torch.float32).to(state.device)
    y = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1).to(state.device)

    state.model.train()
    optimizer = torch.optim.Adam(state.model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = state.model(X)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
        optimizer.step()

    state.model.eval()
    state.is_trained = True
    return state


def predict_deterioration(
    prices: pd.Series,
    returns: pd.Series,
    state: LSTMState,
) -> float:
    """Predict deterioration probability using the LSTM."""
    if not state.is_trained or state.model is None:
        return 0.0

    features = engineer_deterioration_features(prices, returns)
    if features is None:
        return 0.0

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(state.device)
        prob = state.model(x)
        result = float(prob.item())

    # Return pure probability [0, 1]. Influence cap is applied by the caller.
    return result
