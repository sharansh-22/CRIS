"""
borrower_only_audit_phase2c.py — Phase 2C: Borrower-Only Credit Risk Audit.
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from lightgbm import LGBMClassifier

# Discover project root
PROJECT_ROOT = Path(__file__).resolve().parent
while not (ENVIRONMENT_FILE := PROJECT_ROOT / "environment.yml").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.credit_config import SEED, OUTPUT_DIR

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CreditRisk.BorrowerOnlyAudit")

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_IMAGES_DIR = REPORTS_DIR / "images"
REPORTS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = Path("/home/sharansh/.gemini/antigravity/brain/768bd776-6aca-4d09-83b2-425908f2859a/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "models" / "saved_models"

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return float(ece)

def calculate_metrics_with_optimized_threshold(y_true: np.ndarray, y_prob: np.ndarray):
    """Calculate ROC-AUC, PR-AUC, Brier score, and F1/Precision/Recall/Accuracy at optimized threshold."""
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    
    prec, rec, thrs = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    opt_idx = np.argmax(f1_scores)
    opt_thr = thrs[opt_idx] if opt_idx < len(thrs) else 0.5
    y_pred = (y_prob >= opt_thr).astype(int)
    
    accuracy = float((y_pred == y_true).mean())
    f1 = float(f1_scores[opt_idx])
    precision = float(prec[opt_idx])
    recall = float(rec[opt_idx])
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    t0 = time.time()
    logger.info("Starting Phase 2C Borrower-Only Credit Risk Audit...")
    
    # Load LendingClub data
    logger.info("Loading LendingClub data...")
    engineered_path = OUTPUT_DIR / "engineered_data.parquet"
    df_all = pd.read_parquet(engineered_path)
    df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])
    df_all['year'] = df_all['issue_d'].dt.year
    
    # Replicate split used in Phase 1
    # Train <= 2015, Sample size 100,000
    # Test >= 2018, Sample size 50,000
    train_all = df_all[df_all["year"] <= 2015]
    test_all = df_all[df_all["year"] >= 2018]
    
    train_df = train_all.sample(100000, random_state=SEED).copy()
    test_df = test_all.sample(50000, random_state=SEED).copy()
    
    y_train = train_df["target"]
    y_test = test_df["target"]
    
    # Load baseline model and scaler features
    logger.info("Loading full baseline model and scaler...")
    full_model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    
    features_spaces = list(scaler.feature_names_in_)
    features_underscores = [c.replace(' ', '_') for c in features_spaces]
    
    # Map features of train and test datasets
    X_train_spaces = train_df[features_spaces].fillna(0)
    X_train_full = X_train_spaces.copy()
    X_train_full.columns = features_underscores
    
    X_test_spaces = test_df[features_spaces].fillna(0)
    X_test_full = X_test_spaces.copy()
    X_test_full.columns = features_underscores
    
    # Define feature groups
    # Group B features: Underwriting/Contract-related
    # We will exclude any features related to: int_rate, term_months, installment, grade, sub_grade
    group_b_patterns = ["int_rate", "term_months", "installment", "grade"]
    group_b_features = []
    group_a_features = []
    
    for f in features_underscores:
        # Check if f matches Group B pattern
        is_group_b = False
        for p in group_b_patterns:
            if p in f:
                is_group_b = True
                break
        if is_group_b:
            group_b_features.append(f)
        else:
            group_a_features.append(f)
            
    logger.info(f"Group A Features (Borrower-Intrinsic) Count: {len(group_a_features)}")
    logger.info(f"Group B Features (Lender-Underwriting) Count: {len(group_b_features)}")
    
    # ── STEP 2: BUILD BORROWER-ONLY MODEL ──
    logger.info("Training Borrower-Only LightGBM Model...")
    X_train_borrower = X_train_full[group_a_features]
    X_test_borrower = X_test_full[group_a_features]
    
    borrower_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1
    )
    
    t_start = time.time()
    borrower_model.fit(X_train_borrower, y_train)
    t_train = time.time() - t_start
    logger.info(f"Borrower-Only Model training completed in {t_train:.2f} seconds.")
    
    # ── STEP 3: PERFORMANCE COMPARISON ──
    logger.info("Evaluating predictive performance...")
    # Score test data with full model
    probs_full = full_model.predict_proba(X_test_full)[:, 1]
    # Score test data with borrower-only model
    probs_borrower = borrower_model.predict_proba(X_test_borrower)[:, 1]
    
    metrics_full = calculate_metrics_with_optimized_threshold(y_test.values, probs_full)
    metrics_borrower = calculate_metrics_with_optimized_threshold(y_test.values, probs_borrower)
    
    # Save the borrower model to disk (optional, let's keep it in reports directory for reference)
    joblib.dump(borrower_model, REPORTS_DIR / "borrower_only_lightgbm.joblib")
    
    # Print metrics
    print("\nModel Performance Comparison:")
    for metric, full_val in metrics_full.items():
        borrower_val = metrics_borrower[metric]
        delta = borrower_val - full_val
        print(f"{metric.upper():<12} | Full Model: {full_val:.5f} | Borrower-Only: {borrower_val:.5f} | Delta: {delta:+.5f}")
        
    # ── STEP 4: DEFAULT CONCENTRATION REPLICATION ──
    logger.info("Running Default Concentration for Borrower-Only Model...")
    test_df_sorted_borrower = test_df.copy()
    test_df_sorted_borrower["pred_pd"] = probs_borrower
    test_df_sorted_borrower = test_df_sorted_borrower.sort_values(by="pred_pd", ascending=True).reset_index(drop=True)
    
    decile_size = len(test_df_sorted_borrower) // 10
    total_defaults_all = int(test_df_sorted_borrower["target"].sum())
    
    decile_records_b = []
    for i in range(10):
        start_idx = i * decile_size
        end_idx = (i + 1) * decile_size if i < 9 else len(test_df_sorted_borrower)
        
        decile_df = test_df_sorted_borrower.iloc[start_idx:end_idx]
        b_count = len(decile_df)
        defaults = int(decile_df["target"].sum())
        non_defaults = b_count - defaults
        default_rate = defaults / b_count
        avg_pred_pd = float(decile_df["pred_pd"].mean())
        share_of_defaults = defaults / total_defaults_all
        
        decile_records_b.append({
            "Decile": f"D{i+1}",
            "Borrowers": b_count,
            "Defaults": defaults,
            "Non-Defaults": non_defaults,
            "Default Rate": default_rate,
            "Avg Predicted PD": avg_pred_pd,
            "Share of Defaults": share_of_defaults
        })
        
    decile_summary_b = pd.DataFrame(decile_records_b)
    print("\nBorrower-Only Decile Summary:")
    print(decile_summary_b.to_string(index=False))
    
    lowest_decile_dr = decile_summary_b.loc[0, "Default Rate"] # D1
    highest_decile_dr = decile_summary_b.loc[9, "Default Rate"] # D10
    segmentation_ratio_b = highest_decile_dr / lowest_decile_dr
    
    # ── STEP 5: FEATURE IMPORTANCE REPLICATION ──
    logger.info("Generating feature importance for Borrower-Only Model...")
    
    # 1. Gain
    booster_b = borrower_model.booster_
    gain_imp_b = booster_b.feature_importance(importance_type="gain")
    gain_df_b = pd.DataFrame({"Feature": group_a_features, "Gain": gain_imp_b}).sort_values(by="Gain", ascending=False).reset_index(drop=True)
    gain_df_b["Gain_Rank"] = gain_df_b.index + 1
    
    # 2. Permutation
    logger.info("Permutation importance for Borrower-Only Model...")
    perm_sub_df = X_test_borrower.sample(5000, random_state=SEED)
    perm_sub_y = y_test.loc[perm_sub_df.index]
    
    perm_result_b = permutation_importance(
        borrower_model, perm_sub_df, perm_sub_y,
        scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=-1
    )
    perm_df_b = pd.DataFrame({
        "Feature": group_a_features,
        "Permutation_Importance": perm_result_b.importances_mean
    }).sort_values(by="Permutation_Importance", ascending=False).reset_index(drop=True)
    perm_df_b["Perm_Rank"] = perm_df_b.index + 1
    
    # 3. SHAP
    logger.info("SHAP importance for Borrower-Only Model...")
    explainer_b = shap.TreeExplainer(borrower_model)
    shap_values_b = explainer_b.shap_values(perm_sub_df)
    
    if isinstance(shap_values_b, list):
        shap_vals_class1_b = shap_values_b[1]
    elif isinstance(shap_values_b, np.ndarray) and len(shap_values_b.shape) == 3:
        shap_vals_class1_b = shap_values_b[:, :, 1]
    else:
        shap_vals_class1_b = shap_values_b
        
    mean_abs_shap_b = np.mean(np.abs(shap_vals_class1_b), axis=0)
    shap_df_b = pd.DataFrame({"Feature": group_a_features, "SHAP_Importance": mean_abs_shap_b}).sort_values(by="SHAP_Importance", ascending=False).reset_index(drop=True)
    shap_df_b["SHAP_Rank"] = shap_df_b.index + 1
    
    # Consensus Feature Importance
    consensus_df_b = gain_df_b[["Feature", "Gain_Rank"]].merge(
        perm_df_b[["Feature", "Perm_Rank"]], on="Feature"
    ).merge(
        shap_df_b[["Feature", "SHAP_Rank"]], on="Feature"
    )
    consensus_df_b["Consensus_Score"] = (consensus_df_b["Gain_Rank"] + consensus_df_b["Perm_Rank"] + consensus_df_b["SHAP_Rank"]) / 3.0
    consensus_df_b = consensus_df_b.sort_values(by="Consensus_Score", ascending=True).reset_index(drop=True)
    consensus_df_b.to_csv(REPORTS_DIR / "borrower_only_feature_importance.csv", index=False)
    
    print("\nTop 10 Borrower-Only Risk Drivers (Consensus):")
    print(consensus_df_b.head(10).to_string(index=False))
    
    # ── STEP 6: BORROWER PROFILE VALIDATION ──
    logger.info("Validating borrower risk profiles...")
    d1_df_b = test_df_sorted_borrower.iloc[0:decile_size]
    d10_df_b = test_df_sorted_borrower.iloc[9*decile_size:len(test_df_sorted_borrower)]
    d5_d6_df_b = test_df_sorted_borrower.iloc[4*decile_size:6*decile_size]
    
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
    
    profiles_b = {}
    for name, df_sub in [("Low Risk", d1_df_b), ("Mid Risk", d5_d6_df_b), ("High Risk", d10_df_b)]:
        profile_vals = {}
        for col, label in profile_cols.items():
            profile_vals[label] = float(df_sub[col].mean())
        profiles_b[name] = profile_vals
        
    profile_df_b = pd.DataFrame(profiles_b)
    profile_df_b.index.name = "Metric"
    profile_df_b = profile_df_b.reset_index()
    print("\nBorrower-Only Profiles:")
    print(profile_df_b.to_string(index=False))
    
    # ── STEP 7: VISUALIZATIONS ──
    logger.info("Generating borrower-only visualizations...")
    
    # Plot 1: borrower_only_auc_comparison.png (ROC curve comparison)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC Curves
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr_f, tpr_f, _ = roc_curve(y_test, probs_full)
    fpr_b, tpr_b, _ = roc_curve(y_test, probs_borrower)
    
    axes[0].plot(fpr_f, tpr_f, color="navy", lw=2, label=f"Full Model (AUC = {metrics_full['roc_auc']:.4f})")
    axes[0].plot(fpr_b, tpr_b, color="crimson", lw=2, linestyle="--", label=f"Borrower-Only Model (AUC = {metrics_borrower['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    axes[0].set_title("Out-of-Sample ROC Curves", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.15)
    
    # PR Curves
    prec_f, rec_f, _ = precision_recall_curve(y_test, probs_full)
    prec_b, rec_b, _ = precision_recall_curve(y_test, probs_borrower)
    axes[1].plot(rec_f, prec_f, color="navy", lw=2, label=f"Full Model (PR-AUC = {metrics_full['pr_auc']:.4f})")
    axes[1].plot(rec_b, prec_b, color="crimson", lw=2, linestyle="--", label=f"Borrower-Only Model (PR-AUC = {metrics_borrower['pr_auc']:.4f})")
    axes[1].set_title("Out-of-Sample PR Curves", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.15)
    
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "borrower_only_auc_comparison.png", dpi=150)
    plt.close(fig)
    
    # Plot 2: borrower_only_default_concentration.png
    # Load LGBM default rate by decile from prior results (D1-D10: 3.02%, 5.96%, 7.74%, 9.70%, 12.84%, 14.82%, 18.58%, 21.80%, 27.10%, 35.74%)
    lgbm_decile_dr = [3.02, 5.96, 7.74, 9.70, 12.84, 14.82, 18.58, 21.80, 27.10, 35.74]
    borrower_decile_dr = list(decile_summary_b["Default Rate"].values * 100)
    decile_labels = [f"D{i}" for i in range(1, 11)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_indices = np.arange(10)
    bar_width = 0.35
    
    ax.bar(x_indices - bar_width/2, lgbm_decile_dr, bar_width, label="Full Model", color="navy")
    ax.bar(x_indices + bar_width/2, borrower_decile_dr, bar_width, label="Borrower-Only Model", color="crimson")
    ax.set_title("Default Rate (%) by Decile: Full vs. Borrower-Only Model", fontsize=12, fontweight="bold")
    ax.set_xlabel("Risk Decile")
    ax.set_ylabel("Actual Default Rate (%)")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(decile_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "borrower_only_default_concentration.png", dpi=150)
    plt.close(fig)
    
    # Plot 3: borrower_only_feature_importance.png
    fig, ax = plt.subplots(figsize=(10, 6))
    top_consensus_b = consensus_df_b.head(15).copy()
    top_consensus_b["Visual_Score"] = 100.0 / (top_consensus_b["Consensus_Score"] + 1)
    sns.barplot(x="Visual_Score", y="Feature", data=top_consensus_b, palette="rocket", ax=ax)
    ax.set_title("Borrower-Only Consensus Feature Importance (Top 15)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Consensus Strength Score (Inverted Avg Rank)")
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "borrower_only_feature_importance.png", dpi=150)
    plt.close(fig)
    
    # Plot 4: borrower_only_risk_profiles.png
    fig, ax = plt.subplots(figsize=(10, 6))
    profile_data_b = {
        "Group": ["Low Risk", "Mid Risk", "High Risk"],
        "FICO": [profiles_b["Low Risk"]["Average FICO"], profiles_b["Mid Risk"]["Average FICO"], profiles_b["High Risk"]["Average FICO"]],
        "DTI (%)": [profiles_b["Low Risk"]["Average DTI"], profiles_b["Mid Risk"]["Average DTI"], profiles_b["High Risk"]["Average DTI"]],
        "Utilization (%)": [profiles_b["Low Risk"]["Average Revolving Utilization"], profiles_b["Mid Risk"]["Average Revolving Utilization"], profiles_b["High Risk"]["Average Revolving Utilization"]],
        "Credit Hist (Years)": [profiles_b["Low Risk"]["Average Credit History Length"], profiles_b["Mid Risk"]["Average Credit History Length"], profiles_b["High Risk"]["Average Credit History Length"]]
    }
    df_plot_prof_b = pd.DataFrame(profile_data_b)
    df_plot_prof_melted_b = df_plot_prof_b.melt(id_vars="Group", var_name="Metric", value_name="Value")
    
    sns.barplot(x="Metric", y="Value", hue="Group", data=df_plot_prof_melted_b, palette="Set1", ax=ax)
    ax.set_title("Borrower-Only Key Metric Comparison Across Risk Groups", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    fig.savefig(REPORTS_IMAGES_DIR / "borrower_only_risk_profiles.png", dpi=150)
    plt.close(fig)
    
    # Copy all CSVs and PNGs to artifacts
    for f_path in REPORTS_DIR.glob("borrower_only_*.csv"):
        shutil.copy(f_path, ARTIFACTS_DIR / f_path.name)
    shutil.copy(REPORTS_DIR / "consensus_feature_importance.csv", ARTIFACTS_DIR / "consensus_feature_importance.csv")
    for f_path in REPORTS_IMAGES_DIR.glob("borrower_only_*.png"):
        shutil.copy(f_path, ARTIFACTS_DIR / f_path.name)
        
    logger.info("All borrower-only deliverables generated and copied to artifacts.")
    logger.info(f"Phase 2C borrower-only audit completed successfully in {time.time() - t0:.2f} seconds.")

if __name__ == "__main__":
    main()
