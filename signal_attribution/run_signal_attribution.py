"""
run_signal_attribution.py — Master orchestrator for SAE Research Mode V1.

Executes the complete signal attribution pipeline:
  1. Load and merge credit + macro signal data
  2. Compute temporal and regime stability
  3. Run full attribution analysis
  4. Compute entropy metrics
  5. Validate walk-forward safety
  6. Generate visual reports

Usage:
    conda activate CRIS
    python -m signal_attribution.run_signal_attribution
"""

import sys
import logging
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# Dynamic project root
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.credit_config import OUTPUT_DIR, MODEL_DIR

from signal_attribution.schema import (
    SIGNAL_REGISTRY,
    SignalAttribution,
    TemporalWindow,
    AttributionReport,
)
from signal_attribution.attribution import run_full_attribution
from signal_attribution.stability import (
    compute_window_correlations,
    compute_temporal_stability,
    compute_regime_stability,
    build_temporal_window_snapshots,
)
from signal_attribution.entropy import compute_attribution_entropy
from signal_attribution.validation import run_full_validation
from signal_attribution.reporting import (
    plot_attribution_ranking,
    plot_attribution_through_time,
    plot_stability_analysis,
    plot_correlation_heatmap,
    generate_text_report,
)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy libraries
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("fontTools").setLevel(logging.WARNING)
logger = logging.getLogger("CRIS.SAE")

# ── Output directory ──
SAE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "signal_attribution"
SAE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Signal universe ──
SIGNAL_NAMES = list(SIGNAL_REGISTRY.keys())

DIVIDER = "=" * 60


def load_and_merge_data() -> pd.DataFrame:
    """Load credit data and merge with environmental signals.

    Walk-forward safe: each loan is joined to the macro state
    from its issue month (which was computed using only data
    available up to that month).
    """
    logger.info("Loading engineered credit data...")
    eng = pd.read_parquet(OUTPUT_DIR / "engineered_data.parquet")
    eng["issue_d"] = pd.to_datetime(eng["issue_d"])
    eng["issue_month"] = eng["issue_d"].dt.strftime("%Y-%m-01")

    logger.info("Loading macro + market structure signals...")
    macro = pd.read_csv(OUTPUT_DIR / "phase2_layer3_macro_states.csv")
    macro["issue_month"] = pd.to_datetime(macro["issue_month"]).dt.strftime("%Y-%m-01")

    logger.info("Loading borrower PD model...")
    model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    model_features = model.feature_name_
    original_cols = {c.replace(" ", "_"): c for c in eng.columns}
    needed_cols = [original_cols.get(f, f) for f in model_features]
    X = eng[needed_cols].copy()
    eng["borrower_pd"] = model.predict_proba(X)[:, 1]

    logger.info("Merging loan-level data with environmental signals...")
    merged = eng.merge(macro, on="issue_month", how="left")
    merged = merged.dropna(subset=["macro_stress_score"])

    # Verify available signals
    available = [s for s in SIGNAL_NAMES if s in merged.columns]
    missing = [s for s in SIGNAL_NAMES if s not in merged.columns]
    logger.info(f"Available signals: {len(available)}/{len(SIGNAL_NAMES)}")
    if missing:
        logger.warning(f"Missing signals: {missing}")

    logger.info(f"Merged dataset: {len(merged):,} loans, {int(merged['target'].sum()):,} defaults")
    return merged


def run_sae():
    """Execute the complete Signal Attribution Engine pipeline."""
    t0 = time.time()

    print()
    print(DIVIDER)
    print("  CRIS SIGNAL ATTRIBUTION ENGINE — RESEARCH MODE V1")
    print(DIVIDER)
    print()

    # ── Stage 1: Data Loading ──
    logger.info("[1/6] Loading and merging data...")
    merged = load_and_merge_data()

    available_signals = [s for s in SIGNAL_NAMES if s in merged.columns]

    # ── Stage 2: Stability Analysis ──
    logger.info("[2/6] Computing stability analysis...")
    window_corrs = compute_window_correlations(merged, available_signals)
    temporal_stability = compute_temporal_stability(window_corrs)
    regime_stability = compute_regime_stability(merged, available_signals)

    # ── Stage 3: Attribution ──
    logger.info("[3/6] Computing signal attribution...")
    attribution_results = run_full_attribution(
        merged,
        available_signals,
        borrower_pd_col="borrower_pd",
        target_col="target",
        temporal_stability_scores=temporal_stability,
        regime_stability_scores=regime_stability,
    )

    # ── Stage 4: Temporal Snapshots ──
    logger.info("[4/6] Building temporal window snapshots...")
    temporal_snapshots = build_temporal_window_snapshots(
        merged, available_signals,
        target_col="target",
        borrower_pd_col="borrower_pd",
    )

    # ── Stage 5: Entropy ──
    logger.info("[5/6] Computing entropy analysis...")
    weights = {r["signal_name"]: r["attribution_weight"] for r in attribution_results}
    entropy_result = compute_attribution_entropy(weights)

    # ── Stage 6: Validation ──
    logger.info("[6/6] Running walk-forward validation...")
    validation_results = run_full_validation(merged, available_signals)
    validation_status = validation_results.get("OVERALL", "UNKNOWN")

    # ── Build Report ──
    report = AttributionReport(
        signals=[SignalAttribution(**r) for r in attribution_results],
        temporal_windows=[TemporalWindow(**s) for s in temporal_snapshots],
        entropy=entropy_result,
        n_total_loans=len(merged),
        n_total_defaults=int(merged["target"].sum()),
        overall_default_rate=float(merged["target"].mean()),
        validation_status=validation_status,
    )

    # ── Save Results ──
    logger.info("Saving results...")

    # JSON export
    json_path = SAE_OUTPUT_DIR / "attribution_results.json"
    json_path.write_text(json.dumps(report.model_dump(), indent=2, default=str))

    # CSV of attribution table
    attr_df = pd.DataFrame(attribution_results)
    attr_df.to_csv(SAE_OUTPUT_DIR / "attribution_table.csv", index=False)

    # Visual reports
    logger.info("Generating visual reports...")
    plot_attribution_ranking(report, SAE_OUTPUT_DIR)
    plot_attribution_through_time(report, SAE_OUTPUT_DIR)
    plot_stability_analysis(report, SAE_OUTPUT_DIR)
    plot_correlation_heatmap(report, SAE_OUTPUT_DIR)
    generate_text_report(report, validation_results, SAE_OUTPUT_DIR)

    # ── Print Summary ──
    elapsed = time.time() - t0

    print()
    print(DIVIDER)
    print("  SIGNAL ATTRIBUTION DISTRIBUTION")
    print(DIVIDER)
    print()
    print(f"  {'Signal':<30s} {'Weight':>8s}  {'Source':<20s}")
    print(f"  {'─'*30} {'─'*8}  {'─'*20}")
    cumulative = 0.0
    for s in report.signals:
        cumulative += s.attribution_weight
        print(f"  {s.signal_name:<30s} {s.attribution_weight:>7.4f}  {s.source:<20s}")
    print(f"  {'─'*30} {'─'*8}")
    print(f"  {'Σ':<30s} {cumulative:>7.4f}")

    print()
    print(f"  Entropy:          {entropy_result.normalized_entropy:.4f} (normalized)")
    print(f"  Top-3 Conc.:      {entropy_result.concentration_ratio_top3:.1%}")
    print(f"  Top-5 Conc.:      {entropy_result.concentration_ratio_top5:.1%}")
    print(f"  Interpretation:   {entropy_result.interpretation[:80]}...")

    print()
    print(f"  Validation:       {validation_status}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print(f"  Outputs:          {SAE_OUTPUT_DIR}")
    print()
    print(DIVIDER)
    print("  SAE RESEARCH PIPELINE COMPLETE")
    print(DIVIDER)
    print()

    return report


if __name__ == "__main__":
    run_sae()
