"""
schema.py — Data contracts for the Layer 3 continuous probabilistic framework.

All engines output continuous stress/instability intensities.
There are NO hard stress_field labels, NO boolean recovery states, and NO false precision.
The system models evolving market forces, not categorical buckets.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum


# ──────────────────────────────────────────────────────────
#  CONTINUOUS PROBABILISTIC ENGINE OUTPUTS
# ──────────────────────────────────────────────────────────

class DominantField(str, Enum):
    """Enumeration of the dominant stress_fields and uncertainty states."""
    NONE = "NONE"
    FAST_SHOCK = "FAST_SHOCK"
    SLOW_STRUCTURAL = "SLOW_STRUCTURAL"
    TRAJECTORY_DEGRADATION = "TRAJECTORY_DEGRADATION"
    MIXED = "MIXED"
    TRANSITIONAL = "TRANSITIONAL"
    UNCLEAR = "UNCLEAR"


class FastShockOutput(BaseModel):
    """Output from the short-horizon instability field (FAST)."""
    model_config = ConfigDict(extra='ignore')

    shock_intensity: float = Field(ge=0.0, le=1.0, description="Overall short-horizon shock severity")
    liquidity_disruption: float = Field(ge=0.0, le=1.0, description="Slippage and jump probability")
    instability_velocity: float = Field(ge=0.0, le=1.0, description="Rate of entropy breakdown")
    confidence: float = Field(ge=0.0, le=1.0, description="Signal strength and coherence")


class SlowStructuralOutput(BaseModel):
    """Output from the persistent structural stress_field (SLOW)."""
    model_config = ConfigDict(extra='ignore')

    structural_instability: float = Field(ge=0.0, le=1.0, description="Overall structural stress")
    stress_persistence: float = Field(ge=0.0, le=1.0, description="Duration-weighted stress memory")
    fragility_pressure: float = Field(ge=0.0, le=1.0, description="How close the market is to breaking")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence derived from persistence")


class DecayTrajectoryOutput(BaseModel):
    """Output from the long-horizon deterioration field (DECAY)."""
    model_config = ConfigDict(extra='ignore')

    erosion_strength: float = Field(ge=0.0, le=1.0, description="Overall structural weakening (composite of fragility and holding failure)")
    rebound_failure: float = Field(ge=0.0, le=1.0, description="Rate at which sharp bounces are immediately reversed")
    resilience_deficit: float = Field(ge=0.0, le=1.0, description="Slowness of recovery and weakness of upside participation (high = poor resilience)")
    trajectory_fragility: float = Field(ge=0.0, le=1.0, description="Archetype similarity to long-term deterioration sequences")
    holding_failure: float = Field(ge=0.0, le=1.0, description="Inability to maintain price levels after drawdowns (high = levels break)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in structural decay estimation (multi-horizon agreement)")


class MetaDynamicsOutput(BaseModel):
    """Temporal coordination and emergent system dynamics."""
    model_config = ConfigDict(extra='ignore')

    stabilization_strength: float = Field(ge=0.0, le=1.0, description="Continuous asymmetric recovery metric")
    uncertainty_pressure: float = Field(ge=0.0, le=1.0, description="Inter-engine conflict / ambiguity")
    signal_coherence: float = Field(ge=0.0, le=1.0, description="Alignment of market stress forces")
    dominant_field: DominantField = Field(default=DominantField.NONE, description="Currently dominant stress_field or uncertainty state")


# ──────────────────────────────────────────────────────────
#  UNIFIED LAYER 3 OUTPUT
# ──────────────────────────────────────────────────────────

class Layer3Output(BaseModel):
    """Complete continuous probabilistic output of Layer 3."""
    model_config = ConfigDict(extra='ignore')

    ticker: str
    fast: FastShockOutput
    slow: SlowStructuralOutput
    decay: DecayTrajectoryOutput
    meta: MetaDynamicsOutput
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def summary(self) -> str:
        return (
            "╔══════════════════════════════════════════════════╗\n"
            "║  CRIS LAYER 3 — CONTINUOUS PROBABILISTIC FIELD  ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            f"║  Ticker:       {self.ticker:<33s}║\n"
            "╠══════════════════════════════════════════════════╣\n"
            f"║  [FAST]  Shock: {self.fast.shock_intensity:.2f} | Liq: {self.fast.liquidity_disruption:.2f} | Vel: {self.fast.instability_velocity:.2f}    ║\n"
            f"║  [SLOW]  Struc: {self.slow.structural_instability:.2f} | Per: {self.slow.stress_persistence:.2f} | Fra: {self.slow.fragility_pressure:.2f}    ║\n"
            f"║  [DECAY] Erosn: {self.decay.erosion_strength:.2f} | RbF: {self.decay.rebound_failure:.2f} | Fra: {self.decay.trajectory_fragility:.2f}    ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            f"║  [META]  Stab:  {self.meta.stabilization_strength:.2f} | Unc: {self.meta.uncertainty_pressure:.2f} | Coh: {self.meta.signal_coherence:.2f}    ║\n"
            f"║  [FIELD] Dom:   {self.meta.dominant_field.value:<33s}║\n"
            "╚══════════════════════════════════════════════════╝"
        )
