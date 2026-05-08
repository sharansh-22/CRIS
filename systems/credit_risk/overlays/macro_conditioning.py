"""
Layer 3 macro-state conditioning helpers for credit overlays.

Only compact Layer 3 environmental descriptors are exposed here.  Raw market
series are intentionally not joined onto loan records, which keeps the Phase 2
experiment from becoming a generic macro feature model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ALLOWED_STATE_COLUMNS = [
    "uncertainty_pressure",
    "structural_fragility",
    "liquidity_disruption",
    "stabilization_strength",
    "trajectory_fragility",
    "dominant_field",
]


@dataclass(frozen=True)
class MacroStressWeights:
    uncertainty_pressure: float = 0.25
    structural_fragility: float = 0.25
    liquidity_disruption: float = 0.15
    trajectory_fragility: float = 0.20
    stabilization_weakness: float = 0.15


def compute_macro_stress_score(
    states: pd.DataFrame,
    weights: MacroStressWeights = MacroStressWeights(),
) -> pd.Series:
    """Create a bounded audit score from permitted Layer 3 descriptors."""

    score = (
        weights.uncertainty_pressure * states["uncertainty_pressure"].astype(float)
        + weights.structural_fragility * states["structural_fragility"].astype(float)
        + weights.liquidity_disruption * states["liquidity_disruption"].astype(float)
        + weights.trajectory_fragility * states["trajectory_fragility"].astype(float)
        + weights.stabilization_weakness
        * (1.0 - states["stabilization_strength"].astype(float))
    )
    return pd.Series(np.clip(score, 0.0, 1.0), index=states.index, name="macro_stress_score")


def align_market_states_to_loans(
    issue_dates: pd.Series,
    market_states: pd.DataFrame,
) -> pd.DataFrame:
    """Join each issue month to the most recent available Layer 3 state.

    This uses pandas.merge_asof with backward direction, enforcing the rule that
    no market information after loan issuance can enter the loan record.
    """

    loans = pd.DataFrame(
        {
            "row_id": np.arange(len(issue_dates)),
            "issue_d": pd.to_datetime(issue_dates).values,
        }
    ).sort_values("issue_d")

    states = market_states.copy()
    states["state_date"] = pd.to_datetime(states["state_date"])
    states = states.sort_values("state_date")

    aligned = pd.merge_asof(
        loans,
        states,
        left_on="issue_d",
        right_on="state_date",
        direction="backward",
    ).sort_values("row_id")

    return aligned.drop(columns=["row_id"]).reset_index(drop=True)
