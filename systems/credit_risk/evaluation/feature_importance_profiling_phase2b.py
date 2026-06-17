"""
feature_importance_profiling_phase2b.py — Phase 2B: Feature Importance & Borrower Profiling.
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
from sklearn.inspection import permutation_importance
import shap

# Discover project root
PROJECT_ROOT = Path(__file__).resolve().parent
while not (PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.credit_config import SEED, OUTPUT_DIR

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CreditRisk.FeatureImportancePhase2B")

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_IMAGES_DIR = REPORTS_DIR / "images"
REPORTS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = Path("/home/sharansh/.gemini/antigravity/brain/768bd776-6aca-4d09-83b2-425908f2859a/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "models" / "saved_models"

def main():
    t0 = time.time()
    logger.info("Starting Phase 2B Feature Importance & Borrower Profiling...")
    
    # Load data
    engineered_path = OUTPUT_DIR / "engineered_data.parquet"
    df_all = pd.read_parquet(engineered_path)
    df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])
    df_all['year'] = df_all['issue_d'].dt.year
    
    test_all = df_all[df_all["year"] >= 2018]
    test_df = test_all.sample(50000, random_state=SEED).copy()
    
    # Load saved LightGBM model and scaler
    logger.info("Loading saved LightGBM model...")
    lgbm_model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    
    features_spaces = list(scaler.feature_names_in_)
    features_underscores = [c.replace(' ', '_') for c in features_spaces]
    
    X_test_spaces = test_df[features_spaces].fillna(0)
    X_test_underscores = X_test_spaces.copy()
    X_test_underscores.columns = features_underscores
    y_test = test_df["target"]
    
    # Score test set
    logger.info("Scoring test dataset...")
    probs = lgbm_model.predict_proba(X_test_underscores)[:, 1]
    test_df["pred_pd"] = probs
    
    # ── PART 1: FEATURE IMPORTANCE ANALYSIS ──
    logger.info("Running Feature Importance Analysis...")
    
    # 1. Native LightGBM Importance (Gain & Split)
    logger.info("Extracting native LightGBM feature importance...")
    booster = lgbm_model.booster_
    gain_imp = booster.feature_importance(importance_type="gain")
    split_imp = booster.feature_importance(importance_type="split")
    
    gain_df = pd.DataFrame({"Feature": features_underscores, "Gain": gain_imp}).sort_values(by="Gain", ascending=False).reset_index(drop=True)
    split_df = pd.DataFrame({"Feature": features_underscores, "Split": split_imp}).sort_values(by="Split", ascending=False).reset_index(drop=True)
    
    gain_df.to_csv(REPORTS_DIR / "feature_importance_gain.csv", index=False)
    split_df.to_csv(REPORTS_DIR / "feature_importance_split.csv", index=False)
    
    # 2. Permutation Importance
    logger.info("Calculating Permutation Importance on a 5,000-sample test subset...")
    # Use 5,000 samples for computational speed
    perm_sub_df = X_test_underscores.sample(5000, random_state=SEED)
    perm_sub_y = y_test.loc[perm_sub_df.index]
    
    perm_result = permutation_importance(
        lgbm_model, perm_sub_df, perm_sub_y,
        scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=-1
    )
    
    perm_df = pd.DataFrame({
        "Feature": features_underscores,
        "Permutation_Importance": perm_result.importances_mean,
        "Permutation_Std": perm_result.importances_std
    }).sort_values(by="Permutation_Importance", ascending=False).reset_index(drop=True)
    
    perm_df.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    
    # 3. SHAP Analysis
    logger.info("Calculating SHAP values on the same 5,000-sample subset...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(perm_sub_df)
    
    # In shap 0.45+, shap_values for binary classification can be a list or an array of shape (N, D, 2) or (N, D).
    # Let's inspect shap_values structure
    if isinstance(shap_values, list):
        # usually class 1 is index 1
        shap_vals_class1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values
        
    mean_abs_shap = np.mean(np.abs(shap_vals_class1), axis=0)
    shap_df = pd.DataFrame({"Feature": features_underscores, "SHAP_Importance": mean_abs_shap}).sort_values(by="SHAP_Importance", ascending=False).reset_index(drop=True)
    
    shap_df.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    
    # 4. Consensus Rank aggregation
    logger.info("Computing Consensus Rank...")
    # Add ranks (1 = highest importance)
    gain_df["Gain_Rank"] = gain_df.index + 1
    perm_df["Perm_Rank"] = perm_df.index + 1
    shap_df["SHAP_Rank"] = shap_df.index + 1
    
    # Merge ranks
    consensus_df = gain_df[["Feature", "Gain_Rank"]].merge(
        perm_df[["Feature", "Perm_Rank"]], on="Feature"
    ).merge(
        shap_df[["Feature", "SHAP_Rank"]], on="Feature"
    )
    
    consensus_df["Consensus_Score"] = (consensus_df["Gain_Rank"] + consensus_df["Perm_Rank"] + consensus_df["SHAP_Rank"]) / 3.0
    consensus_df = consensus_df.sort_values(by="Consensus_Score", ascending=True).reset_index(drop=True)
    
    consensus_df.to_csv(REPORTS_DIR / "consensus_feature_importance.csv", index=False)
    
    print("\nTop 10 Consensus Features:")
    print(consensus_df.head(10).to_string(index=False))
    
    # ── PART 2: BORROWER PROFILING ──
    logger.info("Running Borrower Profiling...")
    # Sort test set by predicted PD
    test_df_sorted = test_df.sort_values(by="pred_pd", ascending=True).reset_index(drop=True)
    n_total = len(test_df_sorted)
    decile_size = n_total // 10
    
    # D1 = Safest 10%
    d1_df = test_df_sorted.iloc[0:decile_size]
    # D10 = Riskiest 10%
    d10_df = test_df_sorted.iloc[9*decile_size:n_total]
    # Mid-Risk = D5 + D6 (representing 40% to 60% deciles)
    d5_d6_df = test_df_sorted.iloc[4*decile_size:6*decile_size]
    
    profile_cols = {
        "fico_range_low": "Average FICO",
        "annual_inc": "Average Annual Income",
        "dti": "Average DTI",
        "revol_util": "Average Revolving Utilization",
        "cr_hist_years": "Average Credit History Length",
        "loan_amnt": "Average Loan Amount",
        "delinq_2yrs": "Average Delinquencies",
        "pub_rec": "Average Public Records",
        "open_acc": "Average Open Credit Lines"
    }
    
    profiles = {}
    for name, df_sub in [("Low Risk", d1_df), ("Mid Risk", d5_d6_df), ("High Risk", d10_df)]:
        profile_vals = {}
        for col, label in profile_cols.items():
            profile_vals[label] = float(df_sub[col].mean())
        profiles[name] = profile_vals
        
    profile_df = pd.DataFrame(profiles)
    profile_df.index.name = "Metric"
    profile_df = profile_df.reset_index()
    
    # Write Profiles
    d1_profile = pd.DataFrame(list(profiles["Low Risk"].items()), columns=["Metric", "Value"])
    d1_profile.to_csv(REPORTS_DIR / "low_risk_borrower_profile.csv", index=False)
    
    mid_profile = pd.DataFrame(list(profiles["Mid Risk"].items()), columns=["Metric", "Value"])
    mid_profile.to_csv(REPORTS_DIR / "mid_risk_borrower_profile.csv", index=False)
    
    d10_profile = pd.DataFrame(list(profiles["High Risk"].items()), columns=["Metric", "Value"])
    d10_profile.to_csv(REPORTS_DIR / "high_risk_borrower_profile.csv", index=False)
    
    print("\nBorrower Profiling Table:")
    print(profile_df.to_string(index=False))
    
    # ── PART 3: RISK DRIVER ANALYSIS ──
    logger.info("Running Risk Driver Analysis...")
    driver_records = []
    
    for col, label in profile_cols.items():
        low_val = float(d1_df[col].mean())
        high_val = float(d10_df[col].mean())
        
        # Relative difference calculation
        # FICO: higher is safer, DTI: lower is safer. Let's calculate relative diff as:
        # (High Risk - Low Risk) / Low Risk * 100
        # If low_val is 0, handle it
        rel_diff = ((high_val - low_val) / low_val * 100) if low_val != 0 else np.nan
        
        driver_records.append({
            "Metric": label,
            "Low Risk (D1)": low_val,
            "High Risk (D10)": high_val,
            "Absolute Difference": high_val - low_val,
            "Relative Difference (%)": rel_diff
        })
        
    driver_df = pd.DataFrame(driver_records)
    # Sort by absolute of relative difference to identify strongest risk drivers
    driver_df["Abs_Rel_Diff"] = driver_df["Relative Difference (%)"].abs()
    driver_df_sorted = driver_df.sort_values(by="Abs_Rel_Diff", ascending=False).drop(columns="Abs_Rel_Diff").reset_index(drop=True)
    
    print("\nRisk Driver Differences (Sorted by Relative Difference magnitude):")
    print(driver_df_sorted.to_string(index=False))
    
    # ── GRAPH GENERATION ──
    logger.info("Generating plots...")
    
    # Plot 1: feature_importance_gain.png
    fig, ax = plt.subplots(figsize=(10, 6))
    top_gain = gain_df.head(20)
    sns.barplot(x="Gain", y="Feature", data=top_gain, palette="viridis", ax=ax)
    ax.set_title("LightGBM Native Gain Feature Importance (Top 20)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Cumulative Gain")
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "feature_importance_gain.png", dpi=150)
    plt.close(fig)
    
    # Plot 2: permutation_importance.png
    fig, ax = plt.subplots(figsize=(10, 6))
    top_perm = perm_df.head(20)
    sns.barplot(x="Permutation_Importance", y="Feature", data=top_perm, palette="plasma", ax=ax)
    ax.set_title("Permutation Importance on Test Set (Top 20)", fontsize=12, fontweight="bold")
    ax.set_xlabel("ROC-AUC Degradation")
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "permutation_importance.png", dpi=150)
    plt.close(fig)
    
    # Plot 3: shap_importance.png
    fig, ax = plt.subplots(figsize=(10, 6))
    top_shap = shap_df.head(20)
    sns.barplot(x="SHAP_Importance", y="Feature", data=top_shap, palette="inferno", ax=ax)
    ax.set_title("Mean Absolute SHAP Feature Importance (Top 20)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value| (log-odds scale)")
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)
    
    # Plot 4: consensus_feature_importance.png
    fig, ax = plt.subplots(figsize=(10, 6))
    top_consensus = consensus_df.head(15).copy()
    # Invert Consensus Score so that a lower rank has a longer bar for visual clarity
    top_consensus["Visual_Score"] = 100.0 / (top_consensus["Consensus_Score"] + 1)
    sns.barplot(x="Visual_Score", y="Feature", data=top_consensus, palette="mako", ax=ax)
    ax.set_title("Consensus Feature Importance Rank (Top 15, Lower Score is Better)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Consensus Strength Score (Inverted Avg Rank)")
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "consensus_feature_importance.png", dpi=150)
    plt.close(fig)
    
    # Plot 5: borrower_profile_comparison.png
    # Let's standardize some metrics for comparison (FICO, DTI, Income/100, Util*100, Credit History*10)
    fig, ax = plt.subplots(figsize=(10, 6))
    profile_data = {
        "Group": ["Low Risk", "Mid Risk", "High Risk"],
        "FICO": [profiles["Low Risk"]["Average FICO"], profiles["Mid Risk"]["Average FICO"], profiles["High Risk"]["Average FICO"]],
        "DTI (%)": [profiles["Low Risk"]["Average DTI"], profiles["Mid Risk"]["Average DTI"], profiles["High Risk"]["Average DTI"]],
        "Utilization (%)": [profiles["Low Risk"]["Average Revolving Utilization"], profiles["Mid Risk"]["Average Revolving Utilization"], profiles["High Risk"]["Average Revolving Utilization"]],
        "Credit Hist (Years)": [profiles["Low Risk"]["Average Credit History Length"], profiles["Mid Risk"]["Average Credit History Length"], profiles["High Risk"]["Average Credit History Length"]]
    }
    df_plot_prof = pd.DataFrame(profile_data)
    df_plot_prof_melted = df_plot_prof.melt(id_vars="Group", var_name="Metric", value_name="Value")
    
    sns.barplot(x="Metric", y="Value", hue="Group", data=df_plot_prof_melted, palette="Set2", ax=ax)
    ax.set_title("Borrower Profile Key Metric Comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "borrower_profile_comparison.png", dpi=150)
    plt.close(fig)
    
    # Plot 6: risk_driver_differences.png
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x="Relative Difference (%)",
        y="Metric",
        data=driver_df_sorted,
        palette="coolwarm",
        ax=ax
    )
    ax.set_title("Relative Difference (%) in Borrower Profile: D10 vs. D1", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Difference (%)")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    for p in ax.patches:
        val = p.get_width()
        ax.annotate(f"{val:+.1f}%", (val + (2 if val >= 0 else -10), p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "risk_driver_differences.png", dpi=150)
    plt.close(fig)
    
    # Copy all CSVs and PNGs to artifacts
    for f_path in REPORTS_DIR.glob("*.csv"):
        shutil.copy(f_path, ARTIFACTS_DIR / f_path.name)
    for f_path in REPORTS_IMAGES_DIR.glob("*.png"):
        shutil.copy(f_path, ARTIFACTS_DIR / f_path.name)
        
    logger.info("All deliverables generated and copied to artifacts directory.")
    logger.info(f"Phase 2B completed successfully in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
