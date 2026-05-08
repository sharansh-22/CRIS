"""
Interpretable Bayesian-style macro conditioning for borrower PDs.

This module deliberately does not train a replacement default model.  It
applies a small, validation-calibrated log-odds pressure adjustment to an
existing borrower probability when Layer 3 indicates environmental stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


EPS = 1e-6


@dataclass(frozen=True)
class BayesianOverlayConfig:
    """Configuration for a bounded macro pressure overlay.

    stress_anchor is the validation-era normal level.  Only excess pressure
    above this anchor can raise default probabilities; the overlay never grants
    credit for benign markets by reducing borrower risk below the standalone
    estimate.
    """

    stress_anchor: float
    beta: float
    max_logit_shift: float = 0.35


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return np.log(probability / (1.0 - probability))


def inverse_logit(log_odds: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(log_odds, dtype=float)))


def excess_macro_pressure(stress_score: np.ndarray, stress_anchor: float) -> np.ndarray:
    """Return non-negative pressure above the validation-era anchor."""

    stress_score = np.asarray(stress_score, dtype=float)
    return np.clip(stress_score - stress_anchor, 0.0, 1.0)


def apply_bayesian_pressure_overlay(
    borrower_pd: np.ndarray,
    stress_score: np.ndarray,
    config: BayesianOverlayConfig,
) -> np.ndarray:
    """Apply a bounded log-odds adjustment to borrower PDs."""

    pressure = excess_macro_pressure(stress_score, config.stress_anchor)
    shift = np.clip(config.beta * pressure, 0.0, config.max_logit_shift)
    return inverse_logit(logit(borrower_pd) + shift)


def fit_bayesian_pressure_overlay(
    borrower_pd: np.ndarray,
    y_true: np.ndarray,
    stress_score: np.ndarray,
    stress_anchor: float | None = None,
    max_logit_shift: float = 0.35,
) -> BayesianOverlayConfig:
    """Fit one non-negative overlay coefficient on validation data.

    The grid is intentionally small and monotone to keep the overlay auditable.
    Selection uses Brier score first and log loss as a tie-breaker.
    """

    borrower_pd = np.asarray(borrower_pd, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    stress_score = np.asarray(stress_score, dtype=float)
    if stress_anchor is None:
        stress_anchor = float(np.nanmedian(stress_score))

    candidates = np.linspace(0.0, 1.5, 61)
    best_beta = 0.0
    best_score = (np.inf, np.inf)

    for beta in candidates:
        config = BayesianOverlayConfig(
            stress_anchor=float(stress_anchor),
            beta=float(beta),
            max_logit_shift=float(max_logit_shift),
        )
        adjusted = apply_bayesian_pressure_overlay(borrower_pd, stress_score, config)
        score = (
            brier_score_loss(y_true, adjusted),
            log_loss(y_true, adjusted, labels=[0, 1]),
        )
        if score < best_score:
            best_score = score
            best_beta = float(beta)

    return BayesianOverlayConfig(
        stress_anchor=float(stress_anchor),
        beta=best_beta,
        max_logit_shift=float(max_logit_shift),
    )
