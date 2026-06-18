"""
stealth_defaulter_study.py — Population Audit, Decile Location, Borrower Archetypes, and SHAP Analysis.
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix
import shap

# Configure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.credit_config import SEED, OUTPUT_DIR, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CRA.StealthDefaulterStudy")

# Setup output folders
AN_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "cr_analysis"
TABLES_DIR = AN_DIR / "outputs" / "tables"
FIGURES_DIR = AN_DIR / "outputs" / "figures"
DATA_DIR = AN_DIR / "outputs" / "data"

for d in [TABLES_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_data_and_predictions():
    logger.info("Loading engineered data and champion LightGBM model...")
    engineered_path = OUTPUT_DIR / "engineered_data.parquet"
    if not engineered_path.exists():
        raise FileNotFoundError(f"Missing LendingClub engineered data: {engineered_path}")
    df_all = pd.read_parquet(engineered_path)
    df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])
    df_all['year'] = df_all['issue_d'].dt.year

    test_all = df_all[df_all["year"] >= 2018]
    test_df = test_all.sample(50000, random_state=SEED).copy()

    y_test = test_df["target"].values

    full_model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    features_spaces = list(scaler.feature_names_in_)
    features_underscores = [c.replace(' ', '_') for c in features_spaces]

    X_test_spaces = test_df[features_spaces].fillna(0)
    X_test_full = X_test_spaces.copy()
    X_test_full.columns = features_underscores

    probs = full_model.predict_proba(X_test_full)[:, 1]
    
    # Calculate optimized F1 threshold on the test set
    prec, rec, thrs = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    opt_idx = np.argmax(f1_scores)
    opt_thr = thrs[opt_idx]
    
    test_df["pred_pd"] = probs
    test_df["pred_target"] = (probs >= opt_thr).astype(int)
    
    return test_df, X_test_full, y_test, probs, opt_thr, full_model

def run_population_audit(test_df, y_test, probs, opt_thr):
    logger.info("Analysis 1: Running Population Audit...")
    y_pred = test_df["pred_target"].values
    
    total_borrowers = len(test_df)
    total_defaults = int(y_test.sum())
    total_non_defaults = total_borrowers - total_defaults
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Stealth defaulters are False Negatives (FN)
    false_negatives = int(fn)
    true_positives = int(tp)
    false_positives = int(fp)
    true_negatives = int(tn)
    
    fn_rate = false_negatives / total_defaults
    tp_rate = true_positives / total_defaults
    
    pop_summary = pd.DataFrame({
        "Metric": [
            "Total Evaluation Cohort",
            "Total Defaults",
            "Total Non-Defaults",
            "Captured Defaulters (True Positives)",
            "Stealth Defaulters (False Negatives)",
            "False Positives (Good Borrowers Flagged Risk)",
            "True Negatives (Good Borrowers Approved)",
            "False Negative Rate (Stealth / Total Defaults)",
            "Model Decision Threshold"
        ],
        "Value": [
            float(total_borrowers),
            float(total_defaults),
            float(total_non_defaults),
            float(true_positives),
            float(false_negatives),
            float(false_positives),
            float(true_negatives),
            float(fn_rate),
            float(opt_thr)
        ],
        "Formatted": [
            f"{total_borrowers:,}",
            f"{total_defaults:,}",
            f"{total_non_defaults:,}",
            f"{true_positives:,}",
            f"{false_negatives:,}",
            f"{false_positives:,}",
            f"{true_negatives:,}",
            f"{fn_rate:.2%}",
            f"{opt_thr:.5f}"
        ]
    })
    
    pop_summary.to_csv(TABLES_DIR / "population_summary.csv", index=False)
    logger.info(f"Population Audit summary saved to {TABLES_DIR / 'population_summary.csv'}")
    return false_negatives, true_positives

def run_decile_analysis(test_df):
    logger.info("Analysis 2: Running Decile Location Analysis...")
    # Rank by predicted PD
    test_df_sorted = test_df.sort_values(by="pred_pd", ascending=True).reset_index(drop=True)
    decile_size = len(test_df_sorted) // 10
    
    test_df_sorted["decile"] = np.minimum(test_df_sorted.index // decile_size + 1, 10)
    
    total_stealth_in_pop = int(((test_df_sorted["target"] == 1) & (test_df_sorted["pred_target"] == 0)).sum())
    
    decile_stats = []
    for d in range(1, 11):
        decile_df = test_df_sorted[test_df_sorted["decile"] == d]
        total_borrowers = len(decile_df)
        total_defaults = int(decile_df["target"].sum())
        # Stealth defaulters in this decile
        stealth_in_decile = int(((decile_df["target"] == 1) & (decile_df["pred_target"] == 0)).sum())
        captured_in_decile = total_defaults - stealth_in_decile
        
        decile_stats.append({
            "Decile": f"D{d}",
            "Total Borrowers": total_borrowers,
            "Total Defaults": total_defaults,
            "Stealth Defaulters": stealth_in_decile,
            "Captured Defaulters": captured_in_decile,
            "Stealth Share of Decile Defaults": stealth_in_decile / (total_defaults + 1e-8),
            "Stealth Share of Total Stealth": stealth_in_decile / (total_stealth_in_pop + 1e-8)
        })
        
    df_decile = pd.DataFrame(decile_stats)
    df_decile.to_csv(TABLES_DIR / "stealth_decile_location.csv", index=False)
    logger.info(f"Decile analysis saved to {TABLES_DIR / 'stealth_decile_location.csv'}")
    
    # Plot 1: Count of Stealth Defaulters by Decile
    plt.figure(figsize=(8, 5))
    plt.bar(df_decile["Decile"], df_decile["Stealth Defaulters"], color="#da3637", edgecolor="#30363d")
    plt.title("Stealth Defaulter Count by Predicted Risk Decile", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Risk Decile")
    plt.ylabel("Stealth Defaulters (Count)")
    plt.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stealth_count_by_decile.png", dpi=150)
    plt.close()
    
    # Plot 2: Percentage of Stealth Defaulters by Decile
    plt.figure(figsize=(8, 5))
    plt.bar(df_decile["Decile"], df_decile["Stealth Share of Total Stealth"] * 100, color="#f0883e", edgecolor="#30363d")
    plt.title("Percentage of Total Stealth Defaulters by Risk Decile", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Risk Decile")
    plt.ylabel("Share of Total Stealth Defaulters (%)")
    plt.grid(axis="y", alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stealth_pct_by_decile.png", dpi=150)
    plt.close()
    
    logger.info("Decile plots generated.")

def run_borrower_archetype_analysis(test_df):
    logger.info("Analysis 3: Running Borrower Archetype Analysis...")
    
    # Group A: Good Borrowers (Non-defaults)
    # Group B: Captured Defaulters (True Positives)
    # Group C: Stealth Defaulters (False Negatives)
    test_df["archetype"] = "Unknown"
    test_df.loc[(test_df["target"] == 0), "archetype"] = "Group A: Good Borrowers"
    test_df.loc[(test_df["target"] == 1) & (test_df["pred_target"] == 1), "archetype"] = "Group B: Captured Defaulters"
    test_df.loc[(test_df["target"] == 1) & (test_df["pred_target"] == 0), "archetype"] = "Group C: Stealth Defaulters"
    
    # Key borrower-only metrics
    metrics = {
        "fico_range_low": "FICO",
        "dti": "DTI (%)",
        "annual_inc": "Annual Income ($)",
        "revol_util": "Revolving Utilization (%)",
        "loan_amnt": "Loan Amount ($)",
        "cr_hist_years": "Credit History Length (Years)",
        "delinq_2yrs": "Delinquency Count (2 Years)",
        "pub_rec": "Public Records",
        "open_acc": "Open Credit Lines",
        "emp_length_num": "Employment Length (Years)",
        "tot_hi_cred_lim": "Total High Credit Limit ($)"
    }
    
    summary_records = []
    
    for col, name in metrics.items():
        if col not in test_df.columns:
            continue
        row = {"Feature": name}
        for grp in ["Group A: Good Borrowers", "Group B: Captured Defaulters", "Group C: Stealth Defaulters"]:
            sub_df = test_df[test_df["archetype"] == grp][col]
            mean_val = sub_df.mean()
            median_val = sub_df.median()
            p10 = sub_df.quantile(0.10)
            p90 = sub_df.quantile(0.90)
            row[f"{grp} Mean"] = mean_val
            row[f"{grp} Median"] = median_val
            row[f"{grp} P10"] = p10
            row[f"{grp} P90"] = p90
        summary_records.append(row)
        
    df_archetype = pd.DataFrame(summary_records)
    df_archetype.to_csv(TABLES_DIR / "borrower_archetype_comparison.csv", index=False)
    logger.info(f"Borrower archetype comparison table saved to {TABLES_DIR / 'borrower_archetype_comparison.csv'}")
    
    # Save the labeled dataset for downstream analyses
    test_df.to_parquet(DATA_DIR / "labeled_stealth_test_df.parquet", index=False)
    
    # Generate distribution plots for FICO, DTI, Income, and Utilization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    groups = ["Group A: Good Borrowers", "Group B: Captured Defaulters", "Group C: Stealth Defaulters"]
    colors = ["#58a6ff", "#da3637", "#f0883e"]
    
    # Plot 1: FICO Distribution
    for g, c in zip(groups, colors):
        sns.kdeplot(data=test_df[test_df["archetype"] == g], x="fico_range_low", label=g, color=c, ax=axes[0, 0], fill=True, alpha=0.1)
    axes[0, 0].set_title("FICO Score Distribution Comparison", fontweight="bold")
    axes[0, 0].set_xlabel("FICO Score")
    axes[0, 0].legend()
    
    # Plot 2: DTI Distribution
    for g, c in zip(groups, colors):
        # Clip DTI for visual clarity
        sns.kdeplot(data=test_df[test_df["archetype"] == g], x="dti", label=g, color=c, ax=axes[0, 1], fill=True, alpha=0.1)
    axes[0, 1].set_title("Debt-to-Income (DTI) Ratio Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("DTI (%)")
    axes[0, 1].set_xlim(0, 50)
    axes[0, 1].legend()
    
    # Plot 3: Annual Income Distribution
    for g, c in zip(groups, colors):
        # Clip Income for visual clarity
        sns.kdeplot(data=test_df[test_df["archetype"] == g], x="annual_inc", label=g, color=c, ax=axes[1, 0], fill=True, alpha=0.1)
    axes[1, 0].set_title("Annual Income Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Annual Income ($)")
    axes[1, 0].set_xlim(0, 150000)
    axes[1, 0].legend()
    
    # Plot 4: Revolving Utilization Distribution
    for g, c in zip(groups, colors):
        sns.kdeplot(data=test_df[test_df["archetype"] == g], x="revol_util", label=g, color=c, ax=axes[1, 1], fill=True, alpha=0.1)
    axes[1, 1].set_title("Revolving Utilization (%) Distribution", fontweight="bold")
    axes[1, 1].set_xlabel("Revolving Utilization (%)")
    axes[1, 1].set_xlim(0, 120)
    axes[1, 1].legend()
    
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "archetype_distributions.png", dpi=150)
    plt.close()
    
    logger.info("Archetype plots generated.")

def run_shap_analysis(X_test_full, test_df, full_model):
    logger.info("Analysis 4: Running SHAP Analysis comparison...")
    # Filter Group B (Captured Defaulters) and Group C (Stealth Defaulters)
    idx_captured = test_df[test_df["archetype"] == "Group B: Captured Defaulters"].index
    idx_stealth = test_df[test_df["archetype"] == "Group C: Stealth Defaulters"].index
    
    # To keep it fast, sample 500 of each
    np.random.seed(SEED)
    sample_size = min(500, len(idx_captured), len(idx_stealth))
    
    captured_sample_idx = np.random.choice(idx_captured, size=sample_size, replace=False)
    stealth_sample_idx = np.random.choice(idx_stealth, size=sample_size, replace=False)
    
    X_captured = X_test_full.loc[captured_sample_idx]
    X_stealth = X_test_full.loc[stealth_sample_idx]
    
    explainer = shap.TreeExplainer(full_model)
    
    shap_captured = explainer.shap_values(X_captured)
    shap_stealth = explainer.shap_values(X_stealth)
    
    # Extract class 1 (default) SHAP values
    if isinstance(shap_captured, list):
        shap_captured_c1 = shap_captured[1]
        shap_stealth_c1 = shap_stealth[1]
    elif isinstance(shap_captured, np.ndarray) and len(shap_captured.shape) == 3:
        shap_captured_c1 = shap_captured[:, :, 1]
        shap_stealth_c1 = shap_stealth[:, :, 1]
    else:
        shap_captured_c1 = shap_captured
        shap_stealth_c1 = shap_stealth
        
    mean_shap_captured = np.mean(shap_captured_c1, axis=0)
    mean_shap_stealth = np.mean(shap_stealth_c1, axis=0)
    
    shap_df = pd.DataFrame({
        "Feature": X_test_full.columns,
        "Captured_Mean_SHAP": mean_shap_captured,
        "Stealth_Mean_SHAP": mean_shap_stealth,
        "SHAP_Difference": mean_shap_stealth - mean_shap_captured
    }).sort_values(by="SHAP_Difference", ascending=True).reset_index(drop=True)
    
    shap_df.to_csv(TABLES_DIR / "shap_stealth_comparison.csv", index=False)
    logger.info(f"SHAP comparison saved to {TABLES_DIR / 'shap_stealth_comparison.csv'}")
    
    # Plot top 15 features with largest difference in SHAP value (contributing to lower predicted risk for stealth)
    top_diff_features = pd.concat([shap_df.head(10), shap_df.tail(10)]).drop_duplicates().sort_values(by="SHAP_Difference")
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_diff_features, x="SHAP_Difference", y="Feature", palette="coolwarm_r")
    plt.title("SHAP Value Differences (Stealth Defaulters vs Captured Defaulters)\nNegative values indicate feature drives predicted PD lower in Stealth Defaulters", fontsize=11, fontweight="bold")
    plt.xlabel("Mean SHAP Value Difference (Stealth - Captured)")
    plt.grid(axis="x", alpha=0.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_stealth_comparison.png", dpi=150)
    plt.close()
    
    # Identify representative case studies of stealth defaulters
    # Sort stealth defaulters by predicted PD ascending (the most "stealthy" ones)
    stealth_sorted = test_df[test_df["archetype"] == "Group C: Stealth Defaulters"].sort_values(by="pred_pd").head(3)
    
    case_studies = []
    for idx, row in stealth_sorted.iterrows():
        case_studies.append({
            "FICO": row["fico_range_low"],
            "DTI": row["dti"],
            "Annual_Income": row["annual_inc"],
            "Revolving_Utilization": row["revol_util"],
            "Loan_Amount": row["loan_amnt"],
            "Predicted_PD": row["pred_pd"],
            "Target": row["target"]
        })
    pd.DataFrame(case_studies).to_csv(TABLES_DIR / "stealth_case_studies.csv", index=False)
    logger.info("SHAP analysis completed.")

def main():
    test_df, X_test_full, y_test, probs, opt_thr, full_model = load_data_and_predictions()
    run_population_audit(test_df, y_test, probs, opt_thr)
    run_decile_analysis(test_df)
    run_borrower_archetype_analysis(test_df)
    run_shap_analysis(X_test_full, test_df, full_model)
    logger.info("stealth_defaulter_study.py complete.")

if __name__ == "__main__":
    main()
