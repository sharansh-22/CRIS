"""
run_ablation_study.py — Master orchestrator for CRIS Phase 1.5.

Executes all ablation experiments, benchmarks Logistic Regression vs LightGBM,
computes metrics, validates findings against SAE weights, and generates charts.
"""

import sys
import logging
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict


# Dynamic project root discovery
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.credit_config import OUTPUT_DIR, MODEL_DIR
from signal_attribution.ablation import run_all_ablation_experiments, SIGNAL_FAMILIES
from signal_attribution.ablation_reporting import (
    plot_performance_deltas,
    plot_attribution_vs_loss,
    generate_ablation_markdown_report,
)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger("CRIS.SAE.ablation_orchestrator")

SAE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "signal_attribution"
SAE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sae_weights() -> Dict[str, float]:
    """Load the computed weights from Phase 1 results JSON."""
    json_path = SAE_OUTPUT_DIR / "attribution_results.json"
    if not json_path.exists():
        logger.warning(f"Attribution results not found at {json_path}. Using fallback weights.")
        return {
            "Layer3.Decay": 0.3511,
            "Layer3.Meta": 0.2337,
            "MarketStructure": 0.2181,
            "Layer3.Slow": 0.1127,
            "Layer3.Fast": 0.0844,
        }
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    weights = {}
    for s in data["signals"]:
        source = s["source"]
        weights[source] = weights.get(source, 0.0) + s["attribution_weight"]
    return weights


def main():
    t0 = time.time()
    
    print()
    print("=" * 60)
    print("  CRIS SIGNAL ATTRIBUTION VALIDATION (ABLATION STUDY)")
    print("=" * 60)
    print()
    
    # ── 1. Load data ──
    logger.info("Loading dataset and merging environmental signals...")
    eng = pd.read_parquet(OUTPUT_DIR / "engineered_data.parquet")
    eng["issue_d"] = pd.to_datetime(eng["issue_d"])
    eng["issue_month"] = eng["issue_d"].dt.strftime("%Y-%m-01")

    macro = pd.read_csv(OUTPUT_DIR / "phase2_layer3_macro_states.csv")
    macro["issue_month"] = pd.to_datetime(macro["issue_month"]).dt.strftime("%Y-%m-01")

    # Load borrower model
    logger.info("Loading borrower PD model...")
    model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    model_features = model.feature_name_
    original_cols = {c.replace(" ", "_"): c for c in eng.columns}
    needed_cols = [original_cols.get(f, f) for f in model_features]
    X = eng[needed_cols].copy()
    X.columns = model_features
    eng["borrower_pd"] = model.predict_proba(X)[:, 1]

    # Merge
    merged = eng.merge(macro, on="issue_month", how="left")
    merged = merged.dropna(subset=["macro_stress_score"])
    merged["year"] = merged["issue_d"].dt.year

    # Train/Test Split (identical across all experiments)
    train_df = merged[merged["year"] <= 2015].copy()
    test_df = merged[merged["year"] >= 2018].copy()
    
    logger.info(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")
    
    # ── 2. Run experiments ──
    results = run_all_ablation_experiments(train_df, test_df)
    
    # ── 3. Load SAE weights ──
    family_weights = load_sae_weights()
    
    # ── 4. Generate visual & text reports ──
    logger.info("Generating plots and markdown report...")
    plot_performance_deltas(results, SAE_OUTPUT_DIR)
    plot_attribution_vs_loss(results, family_weights, SAE_OUTPUT_DIR)
    generate_ablation_markdown_report(results, family_weights, SAE_OUTPUT_DIR)
    
    # Save raw results JSON
    with open(SAE_OUTPUT_DIR / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save comparison table CSV
    rows = []
    for model_type in ["lr", "lgbm"]:
        for split in ["train", "test"]:
            for exp_name, m in results[model_type][split].items():
                rows.append({
                    "model_type": model_type,
                    "split": split,
                    "experiment": exp_name,
                    "auc": m["auc"],
                    "pr_auc": m["pr_auc"],
                    "brier": m["brier"],
                    "ece": m["ece"],
                    "accuracy": m["accuracy"],
                    "default_capture": m["default_capture"],
                })
    pd.DataFrame(rows).to_csv(SAE_OUTPUT_DIR / "ablation_comparison_table.csv", index=False)
    
    # ── 5. Print summary ──
    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("  EXPERIMENT SUMMARY: LIGHTGBM (TEST SPLIT)")
    print("=" * 60)
    print(f"  {'Experiment':<30s} {'AUC':>8s} {'PR-AUC':>8s} {'Brier':>8s} {'ECE':>8s} {'DefCap':>8s}")
    print("  " + "─" * 78)
    for exp_name, m in results["lgbm"]["test"].items():
        print(f"  {exp_name:<30s} {m['auc']:>8.4f} {m['pr_auc']:>8.4f} {m['brier']:>8.5f} {m['ece']:>8.4f} {m['default_capture']:>8.1%}")
    print("=" * 60)
    print(f"  Ablation study complete in {elapsed:.1f}s.")
    print(f"  Outputs written to: {SAE_OUTPUT_DIR}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
