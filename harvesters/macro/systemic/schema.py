"""
schema.py — Data contracts for Market Structure Intelligence.

Parallel output contract for systemic health signals.
This schema is INDEPENDENT of Layer3Output and does NOT modify it.
Downstream consumers (portfolio intelligence, governance advisory)
consume this contract separately.
"""

from pydantic import BaseModel, Field, ConfigDict


class BreadthOutput(BaseModel):
    """Market participation health signals."""
    model_config = ConfigDict(extra='ignore')

    advance_decline_ratio: float = Field(ge=0.0, le=1.0, description="Normalized advance/decline breadth (0.5 = balanced, <0.3 = deteriorating)")
    pct_above_sma: float = Field(ge=0.0, le=1.0, description="Fraction of sector constituents above their 50-day SMA")
    participation_decay: float = Field(ge=0.0, le=1.0, description="Rate of narrowing market participation")
    breadth_collapse_velocity: float = Field(ge=0.0, le=1.0, description="Speed at which breadth is deteriorating")
    confidence: float = Field(ge=0.0, le=1.0, description="Signal reliability based on data sufficiency")


class DispersionOutput(BaseModel):
    """Cross-sectional disagreement and fragmentation signals."""
    model_config = ConfigDict(extra='ignore')

    cross_sectional_dispersion: float = Field(ge=0.0, le=1.0, description="Normalized standard deviation of cross-sectional returns")
    sector_dispersion: float = Field(ge=0.0, le=1.0, description="Dispersion among sector-level returns")
    leadership_instability: float = Field(ge=0.0, le=1.0, description="Instability in sector leadership rankings")
    defensive_rotation_pressure: float = Field(ge=0.0, le=1.0, description="Strength of rotation from cyclical to defensive sectors")
    confidence: float = Field(ge=0.0, le=1.0, description="Signal reliability")


class CorrelationCompressionOutput(BaseModel):
    """Systemic contagion and diversification collapse signals."""
    model_config = ConfigDict(extra='ignore')

    correlation_density: float = Field(ge=0.0, le=1.0, description="Average pairwise correlation across sectors (high = contagion)")
    cross_sector_coupling: float = Field(ge=0.0, le=1.0, description="Degree to which sectors are moving together")
    contagion_acceleration: float = Field(ge=0.0, le=1.0, description="Rate of increase in correlation density")
    diversification_failure: float = Field(ge=0.0, le=1.0, description="Degree to which portfolio diversification is collapsing")
    confidence: float = Field(ge=0.0, le=1.0, description="Signal reliability")


class SystemicHealthOutput(BaseModel):
    """Unified market structure intelligence output.

    This is a PARALLEL contract to Layer3Output.
    It does NOT replace or modify Layer3Output.
    """
    model_config = ConfigDict(extra='ignore')

    breadth: BreadthOutput
    dispersion: DispersionOutput
    correlation: CorrelationCompressionOutput

    # Composite summary fields
    market_health_score: float = Field(ge=0.0, le=1.0, description="Aggregate internal market health (1.0 = healthy, 0.0 = fragile)")
    structural_fragility: float = Field(ge=0.0, le=1.0, description="Composite fragility from breadth + dispersion + correlation")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence in market structure assessment")

    def summary(self) -> str:
        return (
            "╔══════════════════════════════════════════════════╗\n"
            "║  CRIS — MARKET STRUCTURE INTELLIGENCE            ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            f"║  [BREADTH]  A/D: {self.breadth.advance_decline_ratio:.2f} | %SMA: {self.breadth.pct_above_sma:.2f} | Decay: {self.breadth.participation_decay:.2f}  ║\n"
            f"║  [DISPERS]  XS:  {self.dispersion.cross_sectional_dispersion:.2f} | Sec:  {self.dispersion.sector_dispersion:.2f} | Rot:  {self.dispersion.defensive_rotation_pressure:.2f}  ║\n"
            f"║  [CORREL]   Den: {self.correlation.correlation_density:.2f} | Coup: {self.correlation.cross_sector_coupling:.2f} | Div:  {self.correlation.diversification_failure:.2f}  ║\n"
            "╠══════════════════════════════════════════════════╣\n"
            f"║  Health: {self.market_health_score:.2f} | Fragility: {self.structural_fragility:.2f} | Conf: {self.confidence:.2f}      ║\n"
            "╚══════════════════════════════════════════════════╝"
        )
