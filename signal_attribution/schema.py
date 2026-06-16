"""
schema.py — Data contracts for Signal Attribution Engine (SAE).

Defines typed contracts for attribution results, stability metrics,
and entropy analysis. Strictly diagnostic, not advisory.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional
from enum import Enum


class SignalSource(str, Enum):
    """Origin classification of a signal."""
    LAYER3_FAST = "Layer3.Fast"
    LAYER3_SLOW = "Layer3.Slow"
    LAYER3_DECAY = "Layer3.Decay"
    LAYER3_META = "Layer3.Meta"
    MARKET_STRUCTURE = "MarketStructure"
    COMPOSITE = "Composite"


# ── Signal registry: every signal mapped to its source ──
SIGNAL_REGISTRY: Dict[str, SignalSource] = {
    # Layer 3 — Fast Shock
    "shock_intensity": SignalSource.LAYER3_FAST,
    "liquidity_disruption": SignalSource.LAYER3_FAST,
    "instability_velocity": SignalSource.LAYER3_FAST,
    # Layer 3 — Slow Structural
    "structural_instability": SignalSource.LAYER3_SLOW,
    "stress_persistence": SignalSource.LAYER3_SLOW,
    "structural_fragility": SignalSource.LAYER3_SLOW,
    # Layer 3 — Decay Trajectory
    "erosion_strength": SignalSource.LAYER3_DECAY,
    "rebound_failure": SignalSource.LAYER3_DECAY,
    "resilience_deficit": SignalSource.LAYER3_DECAY,
    "trajectory_fragility": SignalSource.LAYER3_DECAY,
    # Layer 3 — Meta Dynamics
    "stabilization_strength": SignalSource.LAYER3_META,
    "uncertainty_pressure": SignalSource.LAYER3_META,
    "signal_coherence": SignalSource.LAYER3_META,
    # Market Structure Intelligence
    "breadth_health": SignalSource.MARKET_STRUCTURE,
    "breadth_deterioration": SignalSource.MARKET_STRUCTURE,
    "market_structure_fragility": SignalSource.MARKET_STRUCTURE,
    "dispersion_pressure": SignalSource.MARKET_STRUCTURE,
    "correlation_density": SignalSource.MARKET_STRUCTURE,
}


class SignalAttribution(BaseModel):
    """Attribution result for a single signal."""
    model_config = ConfigDict(extra="ignore")

    signal_name: str = Field(description="Name of the environmental signal")
    source: str = Field(description="Origin subsystem of the signal")
    correlation_strength: float = Field(description="Rank correlation with default rate")
    predictive_lift_auc: float = Field(description="Marginal AUC improvement when signal added")
    predictive_lift_brier: float = Field(description="Marginal Brier improvement when signal added")
    temporal_stability: float = Field(ge=0.0, le=1.0, description="Consistency of attribution across time windows")
    regime_stability: float = Field(ge=0.0, le=1.0, description="Consistency across stress vs. calm regimes")
    raw_score: float = Field(ge=0.0, description="Combined raw attribution score")
    attribution_weight: float = Field(ge=0.0, le=1.0, description="Normalized probability weight (sums to 1)")


class TemporalWindow(BaseModel):
    """Attribution snapshot for a single time window."""
    model_config = ConfigDict(extra="ignore")

    window_label: str = Field(description="Human-readable period label")
    start_year: int
    end_year: int
    n_loans: int = Field(ge=0, description="Number of loans in window")
    n_defaults: int = Field(ge=0, description="Number of defaults in window")
    default_rate: float = Field(ge=0.0, le=1.0)
    signal_weights: Dict[str, float] = Field(description="Attribution weights for this window")


class EntropyAnalysis(BaseModel):
    """Information concentration analysis."""
    model_config = ConfigDict(extra="ignore")

    attribution_entropy: float = Field(description="Shannon entropy of attribution distribution")
    max_possible_entropy: float = Field(description="Maximum entropy (uniform distribution)")
    normalized_entropy: float = Field(ge=0.0, le=1.0, description="Entropy / max_entropy; 1.0 = perfectly uniform")
    concentration_ratio_top3: float = Field(ge=0.0, le=1.0, description="Sum of top 3 signal weights")
    concentration_ratio_top5: float = Field(ge=0.0, le=1.0, description="Sum of top 5 signal weights")
    interpretation: str = Field(description="Human-readable interpretation")


class AttributionReport(BaseModel):
    """Complete SAE output contract."""
    model_config = ConfigDict(extra="ignore")

    signals: List[SignalAttribution] = Field(description="Per-signal attribution results")
    temporal_windows: List[TemporalWindow] = Field(description="Per-window attribution snapshots")
    entropy: EntropyAnalysis = Field(description="Information concentration analysis")
    n_total_loans: int
    n_total_defaults: int
    overall_default_rate: float
    validation_status: str = Field(description="Walk-forward safety validation result")
