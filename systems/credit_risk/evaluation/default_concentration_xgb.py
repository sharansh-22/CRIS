"""
default_concentration_xgb.py — Phase 2A-XGB: XGBoost Default Concentration Analysis.
"""

import sys
import logging
import time
import shutil
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Discover project root
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.credit_config import SEED, OUTPUT_DIR

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CreditRisk.DefaultConcentrationXGB")

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_IMAGES_DIR = REPORTS_DIR / "images"
REPORTS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = Path("/home/sharansh/.gemini/antigravity/brain/768bd776-6aca-4d09-83b2-425908f2859a/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "models" / "saved_models"

def main():
    t0 = time.time()
    logger.info("Starting Phase 2A-XGB Default Concentration Analysis...")
    
    # Load data
    engineered_path = OUTPUT_DIR / "engineered_data.parquet"
    df_all = pd.read_parquet(engineered_path)
    df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])
    df_all['year'] = df_all['issue_d'].dt.year
    
    test_all = df_all[df_all["year"] >= 2018]
    test_df = test_all.sample(50000, random_state=SEED).copy()
    
    # Load saved XGBoost model and scaler
    logger.info("Loading saved XGBoost model...")
    xgb_model = joblib.load(MODEL_DIR / "xgboost.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    
    features_spaces = list(scaler.feature_names_in_)
    features_underscores = [c.replace(' ', '_') for c in features_spaces]
    
    X_test_spaces = test_df[features_spaces].fillna(0)
    X_test_underscores = X_test_spaces.copy()
    X_test_underscores.columns = features_underscores
    
    # Predict Probability of Default (PD) using XGBoost
    logger.info("Generating predictions...")
    # XGBoost needs values array if feature name mismatch occurs, or matching features
    probs = xgb_model.predict_proba(X_test_underscores.values)[:, 1]
    test_df["pred_pd"] = probs
    
    # Sort test set from Lowest Predicted Risk to Highest Predicted Risk
    test_df_sorted = test_df.sort_values(by="pred_pd", ascending=True).reset_index(drop=True)
    
    # Create 10 equal-sized risk buckets (deciles)
    n_total = len(test_df_sorted)
    decile_size = n_total // 10
    
    decile_records = []
    total_defaults_all = int(test_df_sorted["target"].sum())
    logger.info(f"Total borrowers: {n_total} | Total defaults: {total_defaults_all} | Baseline Default Rate: {test_df_sorted['target'].mean():.2%}")
    
    for i in range(10):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < 9 else n_total
        
        decile_df = test_df_sorted.iloc[start_idx:end_idx]
        b_count = len(decile_df)
        defaults = int(decile_df["target"].sum())
        non_defaults = b_count - defaults
        default_rate = defaults / b_count
        avg_pred_pd = float(decile_df["pred_pd"].mean())
        share_of_defaults = defaults / total_defaults_all
        
        decile_records.append({
            "Decile": f"D{i+1}",
            "Borrowers": b_count,
            "Defaults": defaults,
            "Non-Defaults": non_defaults,
            "Default Rate": default_rate,
            "Avg Predicted PD": avg_pred_pd,
            "Share of Defaults": share_of_defaults
        })
        
    decile_summary_df = pd.DataFrame(decile_records)
    print("\nXGBoost Decile Summary Table:")
    print(decile_summary_df.to_string(index=False))
    
    # Calculate Risk Segmentation Metrics
    lowest_decile_dr = decile_summary_df.loc[0, "Default Rate"] # D1
    highest_decile_dr = decile_summary_df.loc[9, "Default Rate"] # D10
    segmentation_ratio = highest_decile_dr / lowest_decile_dr if lowest_decile_dr > 0 else np.nan
    
    print(f"\nLowest Decile Default Rate: {lowest_decile_dr:.2%}")
    print(f"Highest Decile Default Rate: {highest_decile_dr:.2%}")
    print(f"Risk Segmentation Ratio (D10 / D1): {segmentation_ratio:.4f}")
    
    # ── Graph 1: xgb_default_share_by_decile.png ──
    logger.info("Generating Graph 1: xgb_default_share_by_decile.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Decile",
        y=decile_summary_df["Share of Defaults"] * 100,
        data=decile_summary_df,
        palette="magma",
        ax=ax
    )
    ax.set_title("XGBoost: Share of Total Defaults (%) by Risk Decile (D1 = Safest, D10 = Riskiest)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Risk Decile")
    ax.set_ylabel("Share of Total Defaults (%)")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    chart1_path = REPORTS_IMAGES_DIR / "xgb_default_share_by_decile.png"
    fig.savefig(chart1_path, dpi=150)
    plt.close(fig)
    
    # ── Graph 2: xgb_default_rate_by_decile.png ──
    logger.info("Generating Graph 2: xgb_default_rate_by_decile.png...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x="Decile",
        y=decile_summary_df["Default Rate"] * 100,
        data=decile_summary_df,
        marker="o",
        linewidth=2.5,
        color="crimson",
        label="Actual Default Rate (%)",
        ax=ax
    )
    sns.lineplot(
        x="Decile",
        y=decile_summary_df["Avg Predicted PD"] * 100,
        data=decile_summary_df,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        color="navy",
        label="Average Predicted PD (%)",
        ax=ax
    )
    ax.set_title("XGBoost: Actual Default Rate vs. Average Predicted PD by Risk Decile", fontsize=12, fontweight="bold")
    ax.set_xlabel("Risk Decile")
    ax.set_ylabel("Rate / Probability (%)")
    ax.legend(loc="upper left")
    for i, row in decile_summary_df.iterrows():
        ax.annotate(f"{row['Default Rate']*100:.2f}%", (row['Decile'], row['Default Rate']*100 + 0.5),
                    ha="center", va="bottom", fontsize=9, color="crimson")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    chart2_path = REPORTS_IMAGES_DIR / "xgb_default_rate_by_decile.png"
    fig.savefig(chart2_path, dpi=150)
    plt.close(fig)
    
    # ── Graph 3: xgb_cumulative_default_capture_curve.png ──
    logger.info("Generating Graph 3: xgb_cumulative_default_capture_curve.png...")
    x_vals = [0] + [i * 10 for i in range(1, 11)]
    reversed_shares = list(decile_summary_df["Share of Defaults"].values)[::-1]
    cum_shares = [0] + list(np.cumsum(reversed_shares) * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=x_vals,
        y=cum_shares,
        marker="o",
        linewidth=2.5,
        color="darkorange",
        label="XGBoost Capture (CAP)",
        ax=ax
    )
    sns.lineplot(
        x=[0, 100],
        y=[0, 100],
        color="gray",
        linestyle="--",
        label="Random Selection Baseline",
        ax=ax
    )
    ax.set_title("XGBoost: Cumulative Default Capture Curve (CAP / Power Curve)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Riskiest Borrowers Included (% of Total Population)")
    ax.set_ylabel("Cumulative Defaults Captured (% of Total Defaults)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    
    # Annotate points
    for x, y in zip(x_vals[1:6], cum_shares[1:6]):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight="bold")
        
    plt.tight_layout()
    chart3_path = REPORTS_IMAGES_DIR / "xgb_cumulative_default_capture_curve.png"
    fig.savefig(chart3_path, dpi=150)
    plt.close(fig)
    
    # Copy charts to artifacts
    shutil.copy(chart1_path, ARTIFACTS_DIR / "xgb_default_share_by_decile.png")
    shutil.copy(chart2_path, ARTIFACTS_DIR / "xgb_default_rate_by_decile.png")
    shutil.copy(chart3_path, ARTIFACTS_DIR / "xgb_cumulative_default_capture_curve.png")
    
    logger.info("XGBoost graph generation and copying complete.")
    logger.info(f"Phase 2A-XGB Default Concentration Analysis completed successfully in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
