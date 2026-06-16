"""
ablation.py — Ablation study implementation for environmental signals.

Runs baseline and ablation models using both Logistic Regression and LightGBM,
using identical train/test splits and hyperparameters.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
import lightgbm as lgb
from typing import Dict, List, Tuple

from configs.credit_config import SEED

logger = logging.getLogger("CRIS.SAE.ablation")

# Define signal families
SIGNAL_FAMILIES = {
    "Layer3.Fast": [
        "shock_intensity",
        "liquidity_disruption",
        "instability_velocity",
    ],
    "Layer3.Slow": [
        "structural_instability",
        "stress_persistence",
        "structural_fragility",
    ],
    "Layer3.Decay": [
        "erosion_strength",
        "rebound_failure",
        "resilience_deficit",
        "trajectory_fragility",
    ],
    "Layer3.Meta": [
        "stabilization_strength",
        "uncertainty_pressure",
        "signal_coherence",
    ],
    "MarketStructure": [
        "breadth_health",
        "breadth_deterioration",
        "market_structure_fragility",
        "dispersion_pressure",
        "correlation_density",
    ],
}

TOP_SIGNALS = [
    "rebound_failure",
    "uncertainty_pressure",
    "trajectory_fragility",
    "erosion_strength",
    "signal_coherence",
]

ALL_SIGNALS = [sig for family in SIGNAL_FAMILIES.values() for sig in family]


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    total_n = len(y_prob)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.20) -> Dict[str, float]:
    """Calculate ROC-AUC, PR-AUC, Brier Score, ECE, Accuracy, and Default Capture."""
    # Binary decisions at threshold (PD > threshold is reject, so approved is PD <= threshold)
    y_pred_approve = (y_prob <= threshold).astype(int)
    y_pred_reject = (y_prob > threshold).astype(int)
    
    # Default capture rate = percentage of actual defaults that have predicted PD > threshold (i.e. rejected)
    defaults = (y_true == 1)
    default_capture = float(y_pred_reject[defaults].mean()) if defaults.sum() > 0 else 0.0
    
    # Accuracy of the final decision status
    acc = float(accuracy_score(y_true, y_pred_reject))  # 1 means default, y_pred_reject=1 means reject/flagged
    
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": calculate_ece(y_true, y_prob),
        "accuracy": acc,
        "default_capture": default_capture,
    }


def run_single_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = "lgbm",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Train a model on train_df and evaluate on both train_df (in-sample) and test_df (out-of-sample)."""
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].values
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df["target"].values
    
    if model_type == "lr":
        # Scale for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
        clf.fit(X_train_scaled, y_train)
        probs_train = clf.predict_proba(X_train_scaled)[:, 1]
        probs_test = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        # LightGBM
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_train, y_train)
        probs_train = clf.predict_proba(X_train)[:, 1]
        probs_test = clf.predict_proba(X_test)[:, 1]
        
    metrics_train = calculate_metrics(y_train, probs_train)
    metrics_test = calculate_metrics(y_test, probs_test)
    return metrics_train, metrics_test


def run_all_ablation_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Execute all ablation experiments for both Logistic Regression and LightGBM on train and test splits."""
    results = {
        "lr": {"train": {}, "test": {}},
        "lgbm": {"train": {}, "test": {}},
    }
    
    # Define features for each experiment
    experiments = {
        "Baseline A (Credit Only)": [],
        "Baseline B (Full CRIS)": ALL_SIGNALS,
        "Remove Layer3.Fast": [s for s in ALL_SIGNALS if s not in SIGNAL_FAMILIES["Layer3.Fast"]],
        "Remove Layer3.Slow": [s for s in ALL_SIGNALS if s not in SIGNAL_FAMILIES["Layer3.Slow"]],
        "Remove Layer3.Decay": [s for s in ALL_SIGNALS if s not in SIGNAL_FAMILIES["Layer3.Decay"]],
        "Remove Layer3.Meta": [s for s in ALL_SIGNALS if s not in SIGNAL_FAMILIES["Layer3.Meta"]],
        "Remove Market Structure": [s for s in ALL_SIGNALS if s not in SIGNAL_FAMILIES["MarketStructure"]],
        "Top-Signal Only": TOP_SIGNALS,
    }
    
    for model_type in ["lr", "lgbm"]:
        logger.info(f"Running ablation experiments for {model_type.upper()}...")
        for exp_name, signals in experiments.items():
            # Features = borrower_pd + signals
            features = ["borrower_pd"] + signals
            logger.info(f"  Running: {exp_name} (features: {len(features)})")
            metrics_tr, metrics_te = run_single_experiment(train_df, test_df, features, model_type=model_type)
            results[model_type]["train"][exp_name] = metrics_tr
            results[model_type]["test"][exp_name] = metrics_te
            
    return results
