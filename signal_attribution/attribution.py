"""
attribution.py — Core signal attribution computation engine.

Measures informational contribution of each environmental signal
to credit deterioration through multiple lenses:
  1. Correlation strength (rank correlation with default rate)
  2. Predictive contribution (marginal model improvement)
  3. Combined attribution score

All computations are walk-forward safe.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

from signal_attribution.schema import SIGNAL_REGISTRY

logger = logging.getLogger("CRIS.SAE.attribution")


def compute_correlation_strength(
    signal_series: pd.Series,
    default_rate_series: pd.Series,
) -> float:
    """Compute Spearman rank correlation between a signal and monthly default rate.

    Uses rank correlation to avoid sensitivity to non-linear relationships.
    Returns absolute value since both positive and negative relationships
    carry informational value.
    """
    common_idx = signal_series.dropna().index.intersection(default_rate_series.dropna().index)
    if len(common_idx) < 5:
        return 0.0
    corr = signal_series.loc[common_idx].corr(default_rate_series.loc[common_idx], method="spearman")
    return float(abs(corr)) if not pd.isna(corr) else 0.0


def compute_predictive_contribution(
    X_base: pd.DataFrame,
    y: pd.Series,
    signal_col: str,
    signal_values: pd.Series,
    seed: int = 42,
) -> Dict[str, float]:
    """Measure marginal AUC and Brier improvement from adding a single signal.

    Compares a baseline logistic model (borrower PD only) against
    baseline + signal. Walk-forward safe: uses only the data provided.

    Returns dict with 'auc_lift' and 'brier_lift'.
    """
    mask = signal_values.notna() & y.notna() & X_base.iloc[:, 0].notna()
    X_b = X_base.loc[mask].values.reshape(-1, 1) if X_base.ndim == 1 else X_base.loc[mask].values
    y_clean = y.loc[mask].values
    sig = signal_values.loc[mask].values.reshape(-1, 1)

    if len(y_clean) < 50 or y_clean.sum() < 5 or (y_clean == 0).sum() < 5:
        return {"auc_lift": 0.0, "brier_lift": 0.0}

    try:
        # Baseline model: borrower PD only
        lr_base = LogisticRegression(random_state=seed, max_iter=500, solver="lbfgs")
        lr_base.fit(X_b, y_clean)
        pred_base = lr_base.predict_proba(X_b)[:, 1]
        auc_base = roc_auc_score(y_clean, pred_base)
        brier_base = brier_score_loss(y_clean, pred_base)

        # Augmented model: borrower PD + signal
        X_aug = np.hstack([X_b, sig])
        lr_aug = LogisticRegression(random_state=seed, max_iter=500, solver="lbfgs")
        lr_aug.fit(X_aug, y_clean)
        pred_aug = lr_aug.predict_proba(X_aug)[:, 1]
        auc_aug = roc_auc_score(y_clean, pred_aug)
        brier_aug = brier_score_loss(y_clean, pred_aug)

        return {
            "auc_lift": float(auc_aug - auc_base),
            "brier_lift": float(brier_base - brier_aug),  # Positive = improvement
        }
    except Exception as e:
        logger.warning(f"Predictive contribution failed for {signal_col}: {e}")
        return {"auc_lift": 0.0, "brier_lift": 0.0}


def compute_raw_attribution_score(
    correlation: float,
    auc_lift: float,
    brier_lift: float,
    temporal_stability: float,
    regime_stability: float,
) -> float:
    """Combine evidence streams into a single raw attribution score.

    Weights:
      - Correlation strength:    25%
      - Predictive AUC lift:     30%
      - Predictive Brier lift:   15%
      - Temporal stability:      15%
      - Regime stability:        15%
    """
    # Normalize AUC lift: typical lifts are 0.001–0.02
    norm_auc = float(np.clip(auc_lift / 0.015, 0.0, 1.0))
    # Normalize Brier lift: typical improvements are 0.0001–0.005
    norm_brier = float(np.clip(brier_lift / 0.003, 0.0, 1.0))

    score = (
        0.25 * correlation
        + 0.30 * norm_auc
        + 0.15 * norm_brier
        + 0.15 * temporal_stability
        + 0.15 * regime_stability
    )
    return max(float(score), 1e-6)  # Floor to prevent hard elimination


def normalize_to_distribution(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Convert raw scores to a probability distribution (sums to 1).

    No signal is hard-eliminated; weak signals receive small weights.
    """
    total = sum(raw_scores.values())
    if total <= 0:
        n = len(raw_scores)
        return {k: 1.0 / n for k in raw_scores}
    return {k: v / total for k, v in raw_scores.items()}


def run_full_attribution(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    borrower_pd_col: str = "borrower_pd",
    target_col: str = "target",
    temporal_stability_scores: Dict[str, float] = None,
    regime_stability_scores: Dict[str, float] = None,
) -> List[Dict]:
    """Execute full attribution computation for all signals.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Loan-level data with signal columns, borrower PD, and default target.
    signal_names : list of str
        Signal columns to evaluate.
    borrower_pd_col : str
        Column name for the baseline borrower PD.
    target_col : str
        Binary default label column.

    Returns
    -------
    list of dict
        Per-signal attribution results ready for SignalAttribution schema.
    """
    temporal_stability_scores = temporal_stability_scores or {}
    regime_stability_scores = regime_stability_scores or {}

    # Compute monthly default rates for correlation analysis
    merged_df["issue_month_str"] = pd.to_datetime(merged_df["issue_month"]).dt.strftime("%Y-%m")
    monthly_defaults = merged_df.groupby("issue_month_str")[target_col].mean()

    results = []
    for signal in signal_names:
        if signal not in merged_df.columns:
            logger.warning(f"Signal {signal} not found in merged data — skipping")
            continue

        source = SIGNAL_REGISTRY.get(signal, "Unknown")

        # 1. Correlation strength
        monthly_signal = merged_df.groupby("issue_month_str")[signal].mean()
        corr = compute_correlation_strength(monthly_signal, monthly_defaults)

        # 2. Predictive contribution
        pred_result = compute_predictive_contribution(
            X_base=merged_df[[borrower_pd_col]],
            y=merged_df[target_col],
            signal_col=signal,
            signal_values=merged_df[signal],
        )

        # 3. Stability scores
        temp_stab = temporal_stability_scores.get(signal, 0.5)
        regime_stab = regime_stability_scores.get(signal, 0.5)

        # 4. Combined score
        raw_score = compute_raw_attribution_score(
            corr, pred_result["auc_lift"], pred_result["brier_lift"],
            temp_stab, regime_stab,
        )

        results.append({
            "signal_name": signal,
            "source": source.value if hasattr(source, "value") else str(source),
            "correlation_strength": round(corr, 6),
            "predictive_lift_auc": round(pred_result["auc_lift"], 6),
            "predictive_lift_brier": round(pred_result["brier_lift"], 6),
            "temporal_stability": round(temp_stab, 4),
            "regime_stability": round(regime_stab, 4),
            "raw_score": round(raw_score, 6),
            "attribution_weight": 0.0,  # Filled after normalization
        })

    # Normalize
    raw_scores = {r["signal_name"]: r["raw_score"] for r in results}
    weights = normalize_to_distribution(raw_scores)
    for r in results:
        r["attribution_weight"] = round(weights[r["signal_name"]], 6)

    # Sort by weight descending
    results.sort(key=lambda x: x["attribution_weight"], reverse=True)
    return results
