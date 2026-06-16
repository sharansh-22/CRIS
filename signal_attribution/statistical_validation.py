"""
statistical_validation.py — Statistical Validation Framework for CRIS Phase 2.

Implements bootstrapping, rank stability, permutation testing, ablation significance,
top signal robustness, and temporal rolling window analysis.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Tuple, Any
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score
import lightgbm as lgb
from joblib import Parallel, delayed

from configs.credit_config import SEED
from signal_attribution.schema import SIGNAL_REGISTRY
from signal_attribution.attribution import (
    compute_correlation_strength,
    compute_predictive_contribution,
    compute_raw_attribution_score,
    normalize_to_distribution,
)
from signal_attribution.stability import (
    compute_window_correlations,
    compute_temporal_stability,
    compute_regime_stability,
)
from signal_attribution.ablation import calculate_ece, SIGNAL_FAMILIES, TOP_SIGNALS

logger = logging.getLogger("CRIS.SAE.statistical_validation")


def _run_single_bootstrap_attribution(
    sample_df: pd.DataFrame,
    signal_names: List[str],
    borrower_pd_col: str,
    target_col: str,
    seed: int,
) -> Dict[str, float]:
    """Helper function to run a single attribution calculation on a resampled dataset."""
    sample_df = sample_df.copy()
    sample_df["issue_month_str"] = pd.to_datetime(sample_df["issue_month"]).dt.strftime("%Y-%m")
    monthly_defaults = sample_df.groupby("issue_month_str")[target_col].mean()
    
    # 1. Stability scores (simplified temporal stability for speed during bootstrap)
    window_corrs = compute_window_correlations(sample_df, signal_names, target_col=target_col)
    temp_stability = compute_temporal_stability(window_corrs)
    regime_stability = compute_regime_stability(sample_df, signal_names, target_col=target_col)
    
    raw_scores = {}
    for signal in signal_names:
        monthly_signal = sample_df.groupby("issue_month_str")[signal].mean()
        corr = compute_correlation_strength(monthly_signal, monthly_defaults)
        
        pred_result = compute_predictive_contribution(
            X_base=sample_df[[borrower_pd_col]],
            y=sample_df[target_col],
            signal_col=signal,
            signal_values=sample_df[signal],
            seed=seed,
        )
        
        raw_score = compute_raw_attribution_score(
            corr,
            pred_result["auc_lift"],
            pred_result["brier_lift"],
            temp_stability.get(signal, 0.5),
            regime_stability.get(signal, 0.5),
        )
        raw_scores[signal] = raw_score
        
    return normalize_to_distribution(raw_scores)


def run_bootstrap_attribution(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    borrower_pd_col: str = "borrower_pd",
    target_col: str = "target",
    n_iterations: int = 100,
    sample_size: int = 100000,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap observations to determine confidence intervals and rank stability for each signal and family."""
    logger.info(f"Running {n_iterations} bootstrap iterations on training data (sample_size={sample_size})...")
    
    def one_iter(i):
        # Sample with replacement
        sample_df = merged_df.sample(n=sample_size, replace=True, random_state=SEED + i)
        weights = _run_single_bootstrap_attribution(
            sample_df, signal_names, borrower_pd_col, target_col, SEED + i
        )
        return weights

    results = Parallel(n_jobs=n_jobs)(delayed(one_iter)(i) for i in range(n_iterations))
    
    # Process weights and ranks
    weight_records = []
    rank_records = []
    
    for i, weights in enumerate(results):
        # Sort to find ranks
        sorted_signals = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        ranks = {sig: rank for rank, (sig, _) in enumerate(sorted_signals, 1)}
        
        for sig, w in weights.items():
            weight_records.append({"iteration": i, "signal": sig, "weight": w})
            rank_records.append({"iteration": i, "signal": sig, "rank": ranks[sig]})
            
    df_weights = pd.DataFrame(weight_records)
    df_ranks = pd.DataFrame(rank_records)
    return df_weights, df_ranks


def run_permutation_test(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    observed_weights: Dict[str, float],
    borrower_pd_col: str = "borrower_pd",
    target_col: str = "target",
    n_permutations: int = 100,
    sample_size: int = 100000,
    n_jobs: int = -1,
) -> Dict[str, float]:
    """Permutation test by shuffling defaults (target) to determine p-values for signal attribution."""
    logger.info(f"Running {n_permutations} permutation iterations (sample_size={sample_size})...")
    
    # Sub-sample to keep runtimes extremely fast
    sub_df = merged_df.sample(n=sample_size, replace=False, random_state=SEED).copy()
    
    def one_iter(i):
        # Permute target column
        perm_df = sub_df.copy()
        perm_df[target_col] = np.random.RandomState(SEED + i).permutation(perm_df[target_col].values)
        weights = _run_single_bootstrap_attribution(
            perm_df, signal_names, borrower_pd_col, target_col, SEED + i
        )
        return weights

    results = Parallel(n_jobs=n_jobs)(delayed(one_iter)(i) for i in range(n_permutations))
    
    # Compute p-values: proportion of permuted weights >= observed weight
    p_values = {}
    for sig in signal_names:
        obs_w = observed_weights.get(sig, 0.0)
        perm_ws = [r[sig] for r in results]
        greater_eq = sum(1 for w in perm_ws if w >= obs_w)
        p_values[sig] = float(greater_eq) / n_permutations
        
    return p_values


def run_ablation_bootstrap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_iterations: int = 200,
    n_jobs: int = -1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Bootstrap the evaluation set to determine statistical significance of ablation losses and top signal performance."""
    logger.info(f"Fitting pre-ablation models for bootstrap evaluation...")
    
    experiments = {
        "Baseline A (Credit Only)": [],
        "Baseline B (Full CRIS)": [s for f in SIGNAL_FAMILIES.values() for s in f],
        "Remove Layer3.Fast": [s for f in SIGNAL_FAMILIES.values() for s in f if s not in SIGNAL_FAMILIES["Layer3.Fast"]],
        "Remove Layer3.Slow": [s for f in SIGNAL_FAMILIES.values() for s in f if s not in SIGNAL_FAMILIES["Layer3.Slow"]],
        "Remove Layer3.Decay": [s for f in SIGNAL_FAMILIES.values() for s in f if s not in SIGNAL_FAMILIES["Layer3.Decay"]],
        "Remove Layer3.Meta": [s for f in SIGNAL_FAMILIES.values() for s in f if s not in SIGNAL_FAMILIES["Layer3.Meta"]],
        "Remove Market Structure": [s for f in SIGNAL_FAMILIES.values() for s in f if s not in SIGNAL_FAMILIES["MarketStructure"]],
        "Top-Signal Only": TOP_SIGNALS,
    }
    
    # We train all 7 models once on train_df
    models = {}
    for exp_name, signals in experiments.items():
        features = ["borrower_pd"] + signals
        
        # Fit LR
        scaler = StandardScaler()
        X_tr_lr = scaler.fit_transform(train_df[features])
        lr_model = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
        lr_model.fit(X_tr_lr, train_df["target"].values)
        
        # Fit LGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1
        )
        lgb_model.fit(train_df[features], train_df["target"].values)
        
        models[exp_name] = {
            "lr": lr_model,
            "lgbm": lgb_model,
            "scaler": scaler,
            "features": features,
        }
        
    logger.info(f"Running {n_iterations} bootstrap evaluation iterations on the test split...")
    
    # Pre-generate predictions on the entire test set to make bootstrap incredibly fast
    y_test = test_df["target"].values
    probs_dict = {"lr": {}, "lgbm": {}}
    for exp_name in experiments:
        lr_m = models[exp_name]["lr"]
        lgb_m = models[exp_name]["lgbm"]
        features = models[exp_name]["features"]
        scaler = models[exp_name]["scaler"]
        
        # Predictions
        probs_dict["lr"][exp_name] = lr_m.predict_proba(scaler.transform(test_df[features]))[:, 1]
        probs_dict["lgbm"][exp_name] = lgb_m.predict_proba(test_df[features])[:, 1]

    # Bootstrap the indices of the test set
    n_samples = len(test_df)
    
    def one_iter(i):
        rng = np.random.RandomState(SEED + i)
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        
        y_boot = y_test[boot_idx]
        
        iter_res = {"lr": {}, "lgbm": {}}
        for m_type in ["lr", "lgbm"]:
            for exp_name in experiments:
                p_boot = probs_dict[m_type][exp_name][boot_idx]
                iter_res[m_type][exp_name] = {
                    "auc": roc_auc_score(y_boot, p_boot),
                    "pr_auc": average_precision_score(y_boot, p_boot),
                    "brier": brier_score_loss(y_boot, p_boot),
                    "ece": calculate_ece(y_boot, p_boot),
                }
        return iter_res

    results = Parallel(n_jobs=n_jobs)(delayed(one_iter)(i) for i in range(n_iterations))
    
    # Reorganize results into arrays
    boot_data = {"lr": {}, "lgbm": {}}
    for m_type in ["lr", "lgbm"]:
        for exp_name in experiments:
            boot_data[m_type][exp_name] = {
                "auc": np.array([r[m_type][exp_name]["auc"] for r in results]),
                "pr_auc": np.array([r[m_type][exp_name]["pr_auc"] for r in results]),
                "brier": np.array([r[m_type][exp_name]["brier"] for r in results]),
                "ece": np.array([r[m_type][exp_name]["ece"] for r in results]),
            }
            
    return boot_data


def run_temporal_rolling_windows(
    merged_df: pd.DataFrame,
    signal_names: List[str],
    borrower_pd_col: str = "borrower_pd",
    target_col: str = "target",
) -> List[Dict[str, Any]]:
    """Compute rolling temporal windows to measure signal attribution stability through time."""
    logger.info("Computing rolling temporal window snapshots...")
    merged_df = merged_df.copy()
    merged_df["year"] = pd.to_datetime(merged_df["issue_month"]).dt.year
    
    # Define rolling 3-year windows
    start_year = merged_df["year"].min()
    end_year = merged_df["year"].max()
    
    rolling_windows = []
    for yr in range(start_year, end_year - 1):
        rolling_windows.append((f"{yr}–{yr+2}", yr, yr+2))
        
    window_results = []
    for label, y_start, y_end in rolling_windows:
        mask = (merged_df["year"] >= y_start) & (merged_df["year"] <= y_end)
        subset = merged_df.loc[mask]
        
        if len(subset) < 1000:
            continue
            
        # Run simplified attribution on this window to measure drift
        # Precompute correlations
        window_corrs = compute_window_correlations(subset, signal_names, target_col=target_col)
        temp_stability = compute_temporal_stability(window_corrs)
        regime_stability = compute_regime_stability(subset, signal_names, target_col=target_col)
        
        subset_defaults = subset.groupby("issue_month")[target_col].mean()
        
        raw_scores = {}
        for sig in signal_names:
            subset_signal = subset.groupby("issue_month")[sig].mean()
            corr = compute_correlation_strength(subset_signal, subset_defaults)
            
            # Sub-sample predictive contribution to avoid slow runs on window subsets
            sub_predict = subset.sample(min(50000, len(subset)), random_state=SEED)
            pred_result = compute_predictive_contribution(
                X_base=sub_predict[[borrower_pd_col]],
                y=sub_predict[target_col],
                signal_col=sig,
                signal_values=sub_predict[sig],
                seed=SEED,
            )
            
            raw_score = compute_raw_attribution_score(
                corr,
                pred_result["auc_lift"],
                pred_result["brier_lift"],
                temp_stability.get(sig, 0.5),
                regime_stability.get(sig, 0.5),
            )
            raw_scores[sig] = raw_score
            
        weights = normalize_to_distribution(raw_scores)
        
        # Calculate family weights
        family_weights = {}
        for family, signals in SIGNAL_FAMILIES.items():
            family_weights[family] = sum(weights.get(s, 0.0) for s in signals)
            
        # Calculate normalized entropy
        p = np.array(list(weights.values()))
        entropy = -np.sum(p * np.log2(p + 1e-12)) / np.log2(len(p))
        
        window_results.append({
            "window": label,
            "n_loans": len(subset),
            "signal_weights": weights,
            "family_weights": family_weights,
            "entropy": float(entropy),
        })
        
    return window_results
