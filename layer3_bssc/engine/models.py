"""
models.py — Layer 3 Data Contracts

Defines the typed Pydantic models that carry outputs
between Layer 3 pipeline stages and into the convergence
engine. Contains zero computation.

Design decision: Pydantic BaseModel over dataclasses.
Rationale: runtime enforcement — invalid field values
raise ValidationError at object creation, not silently
downstream. In a financial risk system, silent errors
are more dangerous than loud ones.

TS-001 decisions are structurally encoded as Literals:
  primary_method:      Literal["sample"]
  confirmation_method: Literal["permutation"]
  market_state:        Literal["NORMAL","STRESS","BLACK_SWAN"]
Pydantic rejects any other value at runtime.

Pipeline position:
  simulation.py → SimulationResult
  entropy.py    → EntropyResult
  slippage.py   → SlippageResult
  auditor       → Layer3Report (assembled from all three)
  convergence   → reads Layer3Report
"""

from pydantic import (
    BaseModel,
    model_validator,
    Field,
    ConfigDict,
)
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime


# ──────────────────────────────────────────────────────────
#  MODEL 1 — SimulationResult
# ──────────────────────────────────────────────────────────

class SimulationResult(BaseModel):
    """Carries the output of simulation.py."""
    model_config = ConfigDict(extra='ignore')

    ticker: str
    """The asset ticker e.g. 'SPY'."""

    S0: float
    """Starting price from calibration."""

    mu: float
    """Annualised drift from calibration."""

    sigma: float
    """Annualised volatility from calibration."""

    n_paths: int
    """Number of paths simulated per mode."""

    kurtosis_gbm: float
    """Kurtosis of GBM return distribution. Validated ~0.158 for SPY."""

    kurtosis_jd: float
    """Kurtosis of Jump-Diffusion return distribution. Validated ~80.58 for SPY."""

    skewness_jd: float
    """Skewness of Jump-Diffusion returns. Validated ~-6.83 for SPY.
    Negative = fat left tail = correct for equity crashes."""

    min_return_gbm: float
    """Minimum single-period return in GBM paths. Validated ~-2.58% for SPY."""

    min_return_jd: float
    """Minimum single-period return in JD paths. Validated ~-21.29% for SPY."""

    simulation_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export,
        WandB logging, and convergence engine consumption."""
        return self.model_dump()


# ──────────────────────────────────────────────────────────
#  MODEL 2 — EntropyResult
# ──────────────────────────────────────────────────────────

class EntropyResult(BaseModel):
    """Carries the output of entropy.py."""
    model_config = ConfigDict(extra='ignore')

    ticker: str

    primary_method: Literal["sample", "volatility_ratio"]
    """Always 'sample' — decided empirically by TS-001.
    Literal enforces this — Pydantic will reject any other value."""

    confirmation_method: Literal[
        "permutation", "permutation_alarm"
    ]
    """Always 'permutation' — decided empirically by TS-001.
    Literal enforces this."""

    baseline_entropy: float
    """Calm period entropy used as reference point.
    Computed from first 252 trading days if calm period not in data."""

    event_entropy: float
    """Peak entropy measured during the event window."""

    entropy_delta: float
    """event_entropy minus baseline_entropy.
    Positive = entropy rose = disorder increased."""

    peak_entropy_date: str
    """Date of maximum entropy during event window. ISO format string."""

    market_state: Literal["NORMAL", "STRESS", "BLACK_SWAN"]
    """Output of classify_market_state().
    Pydantic enforces only these three values.
    Primary signal passed to slippage.py."""

    breach_duration_days: int
    """Days entropy remained above stress threshold.
    Crashes = 18+ days sustained. Positive shocks = 3-6 days.
    This asymmetry is the natural false positive filter."""

    baseline_period_used: str
    """Description of what period was used as baseline.
    e.g. '2018-01-02 to 2018-12-31 (252 day fallback)'."""

    entropy_acceleration_peak: Optional[float] = None
    """Peak rate of change of entropy during event.
    Detects rapid structural breakdown."""

    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export,
        WandB logging, and convergence engine consumption."""
        return self.model_dump()


# ──────────────────────────────────────────────────────────
#  MODEL 3 — SlippageResult
# ──────────────────────────────────────────────────────────

class SlippageResult(BaseModel):
    """Carries the output of slippage.py."""
    model_config = ConfigDict(extra='ignore')

    ticker: str

    mode: Literal["gbm", "jump_diffusion"]
    """Which simulation mode produced these results."""

    n_paths: int
    """Number of Monte Carlo paths used. Validated at 2000 for stable P99."""

    mean_bps: float
    """Mean Implementation Shortfall in basis points.
    Validated GBM: ~214.59 bps. Validated JD: ~490.58 bps."""

    median_bps: float
    """Median IS in basis points.
    Validated GBM: ~210.55 bps. Validated JD: ~241.91 bps.
    Note: median << mean for JD confirms fat right tail working correctly."""

    p95_bps: float
    """95th percentile IS in basis points.
    Validated GBM: ~447.25 bps. Validated JD: ~1948.61 bps."""

    p99_bps: float
    """99th percentile IS in basis points.
    Validated GBM: ~548.48 bps. Validated JD: ~2729.41 bps."""

    max_bps: float
    """Maximum observed IS across all paths.
    Validated GBM: ~696.72 bps. Validated JD: ~4492.89 bps."""

    std_bps: float
    """Standard deviation of IS distribution."""

    jd_gbm_ratio: float
    """JD mean / GBM mean. Validation gate: must exceed 1.5x.
    Validated value: 2.29x."""

    validation_passed: bool = False
    """Computed automatically by model_validator. Never set manually."""

    regime_breakdown: Dict[str, int]
    """Count of paths classified per regime.
    Keys: 'NORMAL', 'STRESS', 'BLACK_SWAN'.
    Validated: GBM paths are always NORMAL,
    JD paths contain STRESS and BLACK_SWAN."""

    entropy_state_at_execution: str
    """The market_state from EntropyResult at time
    of execution anchoring in slippage.py."""

    wandb_run_url: Optional[str] = None
    """URL of WandB run that produced these results."""

    validation_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @model_validator(mode="after")
    def compute_validation_gate(self):
        """Auto-compute validation_passed from jd_gbm_ratio."""
        self.validation_passed = self.jd_gbm_ratio > 1.5
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export,
        WandB logging, and convergence engine consumption."""
        return self.model_dump()


# ──────────────────────────────────────────────────────────
#  MODEL 4 — Layer3Report
# ──────────────────────────────────────────────────────────

class Layer3Report(BaseModel):
    """Assembled output of the entire Layer 3 pipeline.
    This is what the auditor builds and the convergence engine reads.
    Contains the three sub-models plus derived fields."""
    model_config = ConfigDict(extra='ignore')

    ticker: str

    simulation: SimulationResult
    """Full simulation sub-result."""

    entropy: EntropyResult
    """Full entropy sub-result."""

    slippage: SlippageResult
    """Full slippage sub-result."""

    overall_risk_level: Literal["NORMAL", "STRESS", "BLACK_SWAN"] = "NORMAL"
    """Derived from entropy.market_state.
    Primary Layer 3 signal to convergence engine.
    Computed automatically by validator."""

    recommended_action: Literal["HOLD", "REDUCE", "LIQUIDATE"] = "HOLD"
    """Derived from overall_risk_level and slippage.p99_bps.
    Computed automatically by validator."""

    pipeline_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )

    notes: Optional[str] = None

    @model_validator(mode="after")
    def derive_risk_and_action(self):
        """Derive overall_risk_level and recommended_action
        from sub-model values."""
        self.overall_risk_level = self.entropy.market_state

        if self.overall_risk_level == "NORMAL":
            self.recommended_action = "HOLD"
        elif self.overall_risk_level == "STRESS":
            self.recommended_action = "REDUCE"
        else:
            if self.slippage.p99_bps >= 1000:
                self.recommended_action = "LIQUIDATE"
            else:
                self.recommended_action = "REDUCE"

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to nested dictionary for JSON export,
        WandB logging, and convergence engine consumption."""
        return {
            "ticker": self.ticker,
            "overall_risk_level": self.overall_risk_level,
            "recommended_action": self.recommended_action,
            "pipeline_timestamp": self.pipeline_timestamp,
            "simulation": self.simulation.model_dump(),
            "entropy": self.entropy.model_dump(),
            "slippage": self.slippage.model_dump(),
            "consistency_warnings": self.validate_pipeline_consistency(),
        }

    def validate_pipeline_consistency(self) -> List[str]:
        """Checks that the three sub-results tell a coherent story.
        Returns list of warning strings.
        Empty list = fully consistent pipeline.
        Non-empty = warnings for the auditor to log and investigate."""
        warnings: List[str] = []

        # Check 1 — Entropy-Slippage regime alignment
        if (
            self.entropy.market_state == "BLACK_SWAN"
            and self.slippage.entropy_state_at_execution == "NORMAL"
        ):
            warnings.append(
                "WARNING: Entropy regime mismatch — "
                "entropy.py classified BLACK_SWAN but "
                "slippage.py executed under NORMAL regime. "
                "Results may reflect different time windows."
            )

        # Check 2 — Validation gate vs risk level
        if (
            self.slippage.validation_passed is False
            and self.overall_risk_level == "BLACK_SWAN"
        ):
            warnings.append(
                "WARNING: Validation gate failed "
                "(JD/GBM ratio below 1.5x) but "
                "overall_risk_level is BLACK_SWAN. "
                "Slippage estimates may be unreliable. "
                "Do not use p99 for position sizing."
            )

        # Check 3 — Breach duration vs market state
        if (
            self.entropy.market_state == "BLACK_SWAN"
            and self.entropy.breach_duration_days < 5
        ):
            warnings.append(
                "WARNING: BLACK_SWAN classification with "
                "breach duration below 5 days. Expected "
                "18+ days for genuine crisis. Possible "
                "false positive — verify manually."
            )

        # Check 4 — Ticker consistency
        tickers = {
            self.simulation.ticker,
            self.entropy.ticker,
            self.slippage.ticker,
        }
        if len(tickers) > 1:
            warnings.append(
                f"WARNING: Ticker mismatch across pipeline stages. "
                f"simulation={self.simulation.ticker} "
                f"entropy={self.entropy.ticker} "
                f"slippage={self.slippage.ticker}. "
                f"Results may be from mixed runs and should not be combined."
            )

        # Check 5 — JD kurtosis sanity
        if self.simulation.kurtosis_jd < 10:
            warnings.append(
                f"WARNING: JD kurtosis below 10 "
                f"(actual={self.simulation.kurtosis_jd}). "
                f"Jump parameters may be miscalibrated. "
                f"Fat tail behavior not confirmed. "
                f"Expected ~80.58 for SPY."
            )

        # Check 6 — P99 vs mean sanity
        if self.slippage.p99_bps < self.slippage.mean_bps:
            warnings.append(
                "WARNING: P99 slippage below mean slippage. "
                "This is statistically impossible. "
                "Indicates a computation error in "
                "slippage.py percentile calculation."
            )

        return warnings

    def summary(self) -> str:
        """Clean human-readable summary of the full Layer 3 pipeline result."""
        gate_str = "PASSED" if self.slippage.validation_passed else "FAILED"

        box = (
            "╔══════════════════════════════════════════════╗\n"
            "║  CRIS LAYER 3 — PIPELINE SUMMARY            ║\n"
            "╠══════════════════════════════════════════════╣\n"
            f"║  Ticker:          {self.ticker:<25s}║\n"
            f"║  Timestamp:       {self.pipeline_timestamp:<25s}║\n"
            "╠══════════════════════════════════════════════╣\n"
            "║  SIMULATION                                 ║\n"
            f"║  Kurtosis GBM:    {self.simulation.kurtosis_gbm:<25.3f}║\n"
            f"║  Kurtosis JD:     {self.simulation.kurtosis_jd:<25.2f}║\n"
            f"║  Min Return JD:   {self.simulation.min_return_jd:<24.2f}%║\n"
            "╠══════════════════════════════════════════════╣\n"
            "║  ENTROPY                                    ║\n"
            f"║  Market State:    {self.entropy.market_state:<25s}║\n"
            f"║  Breach Duration: {self.entropy.breach_duration_days:<24d}d║\n"
            f"║  Entropy Delta:   {self.entropy.entropy_delta:<25.4f}║\n"
            "╠══════════════════════════════════════════════╣\n"
            "║  SLIPPAGE                                   ║\n"
            f"║  Mean IS:         {self.slippage.mean_bps:<21.2f} bps ║\n"
            f"║  P99 IS:          {self.slippage.p99_bps:<21.2f} bps ║\n"
            f"║  JD/GBM Ratio:    {self.slippage.jd_gbm_ratio:<24.2f}x║\n"
            f"║  Gate:            {gate_str:<25s}║\n"
            "╠══════════════════════════════════════════════╣\n"
            "║  LAYER 3 VERDICT                            ║\n"
            f"║  Risk Level:      {self.overall_risk_level:<25s}║\n"
            f"║  Action:          {self.recommended_action:<25s}║\n"
            "╚══════════════════════════════════════════════╝"
        )

        consistency = self.validate_pipeline_consistency()
        if not consistency:
            box += "\n✅ Pipeline consistency: OK"
        else:
            for w in consistency:
                box += f"\n⚠️  {w}"

        return box
