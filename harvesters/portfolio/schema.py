"""
schema.py - Data contracts for Portfolio Intelligence Harvesters.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class AssetSignal(BaseModel):
    """Signal structure for individual assets."""
    ticker: str
    timestamp: datetime
    volatility_z_score: float
    liquidity_depth_score: float
    macro_sensitivity_beta: float
    drawdown_intensity: float
    confidence_score: float

class PortfolioDiagnostics(BaseModel):
    """Aggregate diagnostics for the user portfolio."""
    timestamp: datetime
    total_value: float
    total_drawdown: float
    concentration_hhi: float
    correlation_matrix_density: float
    systemic_risk_contribution: float
    active_risk_flags: List[str]

class ProbabilisticForecast(BaseModel):
    """Probabilistic outcome distribution for a given horizon."""
    ticker: str
    horizon_days: int
    expected_return: float
    volatility_estimate: float
    percentiles: Dict[str, float]  # e.g., {"p5": -0.05, "p95": 0.08}
    reassessment_error_history: List[float]
    confidence: float
