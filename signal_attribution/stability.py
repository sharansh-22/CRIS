"""
stability.py — Temporal and regime stability analysis for signals.

Measures how consistent a signal's informational value is across
different time windows and market regimes.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from scipy.stats import spearmanr

logger = logging.getLogger("CRIS.SAE.stability")

# Temporal windows for attribution stability analysis
TEMPORAL_WINDOWS = [
    ("2007–2010", 2007, 2010),
    ("2010–2013", 2010, 2013),
    ("2013–2016", 2013, 2016),
    ("2016–2018", 2016, 2018),
]


def compute_window_correlations(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    target_col: str = "target",
    windows: List[Tuple[str, int, int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute correlation between each signal and default rate in each window.

    Returns
    -------
    dict of signal_name -> dict of window_label -> correlation
    """
    windows = windows or TEMPORAL_WINDOWS
    merged_df = merged_df.copy()
    merged_df["year"] = pd.to_datetime(merged_df["issue_month"]).dt.year

    result = {}
    for signal in signal_names:
        if signal not in merged_df.columns:
            continue
        window_corrs = {}
        for label, y_start, y_end in windows:
            mask = (merged_df["year"] >= y_start) & (merged_df["year"] <= y_end)
            subset = merged_df.loc[mask]
            if len(subset) < 20:
                window_corrs[label] = 0.0
                continue

            monthly = subset.groupby(
                pd.to_datetime(subset["issue_month"]).dt.to_period("M")
            ).agg({signal: "mean", target_col: "mean"})

            if len(monthly) < 3:
                window_corrs[label] = 0.0
                continue

            corr, _ = spearmanr(monthly[signal], monthly[target_col])
            window_corrs[label] = float(abs(corr)) if not np.isnan(corr) else 0.0

        result[signal] = window_corrs
    return result


def compute_temporal_stability(
    window_correlations: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute temporal stability as inverse of coefficient of variation.

    A signal that has similar correlation strength across all windows
    receives high stability (close to 1.0). A signal that appears
    strong in one window but disappears in others receives low stability.
    """
    stability = {}
    for signal, corrs in window_correlations.items():
        values = list(corrs.values())
        if not values or max(values) < 1e-6:
            stability[signal] = 0.1
            continue

        mean_corr = np.mean(values)
        std_corr = np.std(values)

        if mean_corr < 1e-6:
            stability[signal] = 0.1
        else:
            cv = std_corr / mean_corr
            # Low CV = high stability; CV of 0 = perfect stability
            stability[signal] = float(np.clip(1.0 - cv, 0.05, 1.0))

    return stability


def compute_regime_stability(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    target_col: str = "target",
    stress_col: str = "macro_stress_score",
    stress_threshold: float = None,
) -> Dict[str, float]:
    """Compare signal usefulness in high-stress vs. low-stress regimes.

    Stability is high when the signal's correlation with defaults
    remains similar across both calm and stressed environments.

    Uses median-based splitting by default to ensure adequate sample
    sizes in both regimes regardless of the stress distribution.
    """
    merged_df = merged_df.copy()
    if stress_threshold is None:
        stress_threshold = float(merged_df[stress_col].median())
    high_stress = merged_df[stress_col] >= stress_threshold
    low_stress = ~high_stress

    stability = {}
    for signal in signal_names:
        if signal not in merged_df.columns:
            stability[signal] = 0.5
            continue

        corrs = []
        for mask in [high_stress, low_stress]:
            subset = merged_df.loc[mask]
            if len(subset) < 20:
                corrs.append(0.0)
                continue

            monthly = subset.groupby(
                pd.to_datetime(subset["issue_month"]).dt.to_period("M")
            ).agg({signal: "mean", target_col: "mean"})

            if len(monthly) < 3:
                corrs.append(0.0)
                continue

            c, _ = spearmanr(monthly[signal], monthly[target_col])
            corrs.append(float(abs(c)) if not np.isnan(c) else 0.0)

        if max(corrs) < 1e-6:
            stability[signal] = 0.1
        else:
            # Stability = 1 - |high_stress_corr - low_stress_corr| / max_corr
            diff = abs(corrs[0] - corrs[1])
            stability[signal] = float(np.clip(1.0 - diff / max(max(corrs), 1e-6), 0.05, 1.0))

    return stability


def build_temporal_window_snapshots(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    target_col: str = "target",
    borrower_pd_col: str = "borrower_pd",
    windows: List[Tuple[str, int, int]] = None,
) -> List[Dict]:
    """Build per-window attribution snapshots.

    Runs a lightweight attribution (correlation-only) within each window
    to show how the attribution distribution shifts over time.
    """
    from signal_attribution.attribution import (
        compute_correlation_strength,
        normalize_to_distribution,
    )

    windows = windows or TEMPORAL_WINDOWS
    merged_df = merged_df.copy()
    merged_df["year"] = pd.to_datetime(merged_df["issue_month"]).dt.year

    snapshots = []
    for label, y_start, y_end in windows:
        mask = (merged_df["year"] >= y_start) & (merged_df["year"] <= y_end)
        subset = merged_df.loc[mask]

        n_loans = len(subset)
        n_defaults = int(subset[target_col].sum()) if n_loans > 0 else 0
        default_rate = float(subset[target_col].mean()) if n_loans > 0 else 0.0

        # Compute per-signal correlation in this window
        monthly = subset.groupby(
            pd.to_datetime(subset["issue_month"]).dt.to_period("M")
        )

        raw_scores = {}
        for signal in signal_names:
            if signal not in subset.columns:
                raw_scores[signal] = 1e-6
                continue

            monthly_agg = monthly.agg({signal: "mean", target_col: "mean"})
            if len(monthly_agg) < 3:
                raw_scores[signal] = 1e-6
                continue

            corr = compute_correlation_strength(monthly_agg[signal], monthly_agg[target_col])
            raw_scores[signal] = max(corr, 1e-6)

        weights = normalize_to_distribution(raw_scores)

        snapshots.append({
            "window_label": label,
            "start_year": y_start,
            "end_year": y_end,
            "n_loans": n_loans,
            "n_defaults": n_defaults,
            "default_rate": round(default_rate, 4),
            "signal_weights": {k: round(v, 6) for k, v in weights.items()},
        })

    return snapshots
