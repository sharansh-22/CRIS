"""
slippage.py — Execution cost modeling for Fast Shocks.
"""

import numpy as np
from configs.macro_config import VOL_STRESS_MULTIPLIER, VOL_CRITICAL_MULTIPLIER

_STRESS_MULTIPLIERS = {
    "NORMAL": 1.0,
    "STRESS": 1.5,
    "CRITICAL": 2.5,
}

def compute_implementation_shortfall(decision_price: float, fill_prices: np.ndarray, side: str = "sell") -> float:
    """Calculate gap between decision price and avg fill in bps."""
    if decision_price <= 0 or len(fill_prices) == 0: return 0.0
    avg_fill = float(np.mean(fill_prices))
    is_bps = ((decision_price - avg_fill) / decision_price) * 10_000 if side == "sell" else ((avg_fill - decision_price) / decision_price) * 10_000
    return max(0.0, float(is_bps))

def compute_market_impact(sigma_daily: float, order_frac: float = 0.01, eta: float = 0.3) -> float:
    """Almgren-Chriss square-root impact model."""
    return float(eta * sigma_daily * np.sqrt(order_frac) * 10_000)

def compute_spread_cost(base_spread_bps: float = 2.0, stress_field_multiplier: float = 1.0) -> float:
    """Execution cost from spread widening."""
    return float(base_spread_bps * stress_field_multiplier)

def get_stress_field_multiplier(market_state: str) -> float:
    return _STRESS_MULTIPLIERS.get(market_state, 1.0)
