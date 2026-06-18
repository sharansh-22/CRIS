"""
noise_vs_structure_test.py — Predictive Modeling to test if Stealth Defaulters are predictable structure or random noise.
"""

import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from lightgbm import LGBMClassifier

# Configure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.credit_config import SEED, OUTPUT_DIR, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CRA.NoiseVsStructureTest")

# Setup output folders
AN_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "cr_analysis"
TABLES_DIR = AN_DIR / "outputs" / "tables"
DATA_DIR = AN_DIR / "outputs" / "data"

def run_noise_test():
    logger.info("Loading engineered data and champion LightGBM model for Noise vs Structure test...")
    engineered_path = OUTPUT_DIR / "engineered_data.parquet"
    if not engineered_path.exists():
        raise FileNotFoundError(f"Missing LendingClub engineered data: {engineered_path}")
    df_all = pd.read_parquet(engineered_path)
    df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])
    df_all['year'] = df_all['issue_d'].dt.year

    train_all = df_all[df_all["year"] <= 2015]
    test_all = df_all[df_all["year"] >= 2018]
    
    train_df = train_all.sample(100000, random_state=SEED).copy()
    test_df = test_all.sample(50000, random_state=SEED).copy()

    # Load champion model and scaler
    full_model = joblib.load(MODEL_DIR / "lightgbm.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")

    features_spaces = list(scaler.feature_names_in_)
    features_underscores = [c.replace(' ', '_') for c in features_spaces]

    # Map features
    X_train_spaces = train_df[features_spaces].fillna(0)
    X_train_full = X_train_spaces.copy()
    X_train_full.columns = features_underscores
    
    X_test_spaces = test_df[features_spaces].fillna(0)
    X_test_full = X_test_spaces.copy()
    X_test_full.columns = features_underscores

    # Predict PDs
    probs_train = full_model.predict_proba(X_train_full)[:, 1]
    probs_test = full_model.predict_proba(X_test_full)[:, 1]
    
    # Calculate optimized F1 threshold on test set
    y_test = test_df["target"].values
    prec, rec, thrs = precision_recall_curve(y_test, probs_test)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    opt_idx = np.argmax(f1_scores)
    opt_thr = thrs[opt_idx]
    
    # Define stealth target (1 if defaulted but predicted PD < threshold, 0 otherwise)
    y_train_stealth = ((train_df["target"] == 1) & (probs_train < opt_thr)).astype(int).values
    y_test_stealth = ((test_df["target"] == 1) & (probs_test < opt_thr)).astype(int).values
    
    logger.info(f"Train stealth defaulter rate: {y_train_stealth.mean():.2%} ({y_train_stealth.sum()} samples)")
    logger.info(f"Test stealth defaulter rate: {y_test_stealth.mean():.2%} ({y_test_stealth.sum()} samples)")
    
    # Filter features: Exclude lender variables
    group_b_patterns = ["int_rate", "term_months", "installment", "grade"]
    features_borrower = [f for f in features_underscores if not any(pat in f for pat in group_b_patterns)]
    
    X_train_borrower = X_train_full[features_borrower]
    X_test_borrower = X_test_full[features_borrower]
    
    # Train Stealth Defaulter Classifier
    logger.info("Training Stealth Defaulter Classifier...")
    stealth_clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1
    )
    stealth_clf.fit(X_train_borrower, y_train_stealth)
    
    # Evaluate
    probs_stealth = stealth_clf.predict_proba(X_test_borrower)[:, 1]
    
    auc_stealth = roc_auc_score(y_test_stealth, probs_stealth)
    pr_auc_stealth = average_precision_score(y_test_stealth, probs_stealth)
    
    # Threshold optimization for F1 on stealth prediction
    p_s, r_s, t_s = precision_recall_curve(y_test_stealth, probs_stealth)
    f1_s = 2 * (p_s * r_s) / (p_s + r_s + 1e-8)
    opt_idx_s = np.argmax(f1_s)
    opt_thr_s = t_s[opt_idx_s] if opt_idx_s < len(t_s) else 0.5
    
    y_pred_stealth = (probs_stealth >= opt_thr_s).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_stealth, y_pred_stealth).ravel()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / len(y_test_stealth)
    
    # ── STEP 2: BOOTSTRAP SIGNIFICANCE TESTING ──
    logger.info("Running bootstrap validation (50 trials) for confidence intervals...")
    n_boot = 50
    boot_aucs = []
    boot_praucs = []
    
    rng = np.random.RandomState(SEED)
    for _ in range(n_boot):
        idx = rng.choice(len(y_test_stealth), size=len(y_test_stealth), replace=True)
        y_b = y_test_stealth[idx]
        probs_b = probs_stealth[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, probs_b))
        boot_praucs.append(average_precision_score(y_b, probs_b))
        
    auc_ci = np.percentile(boot_aucs, [2.5, 97.5])
    prauc_ci = np.percentile(boot_praucs, [2.5, 97.5])
    
    logger.info(f"Stealth Defaulter Classifier ROC-AUC: {auc_stealth:.5f} (95% CI: [{auc_ci[0]:.5f}, {auc_ci[1]:.5f}])")
    logger.info(f"Stealth Defaulter Classifier PR-AUC: {pr_auc_stealth:.5f} (95% CI: [{prauc_ci[0]:.5f}, {prauc_ci[1]:.5f}])")
    
    # Save results
    results_df = pd.DataFrame({
        "Metric": ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1 Score", "Optimized Threshold"],
        "Value": [auc_stealth, pr_auc_stealth, accuracy, precision, recall, f1_s[opt_idx_s], opt_thr_s],
        "CI_Lower": [auc_ci[0], prauc_ci[0], np.nan, np.nan, np.nan, np.nan, np.nan],
        "CI_Upper": [auc_ci[1], prauc_ci[1], np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    
    results_df.to_csv(TABLES_DIR / "noise_vs_structure_results.csv", index=False)
    
    # Save model and predictions for report generation
    test_df["stealth_pred_prob"] = probs_stealth
    test_df["stealth_target"] = y_test_stealth
    test_df.to_parquet(DATA_DIR / "stealth_predictions_df.parquet", index=False)
    
    # Save model
    joblib.dump(stealth_clf, DATA_DIR / "stealth_classifier.joblib")
    
    # Write verdict based on ROC-AUC
    # Let's set standard credit scoring threshold: ROC-AUC < 0.55 = random noise; >= 0.55 = weak structure; >= 0.65 = moderate structure
    if auc_stealth >= 0.65:
        verdict = "Structured and partially predictable borrower segment."
    elif auc_stealth >= 0.55:
        verdict = "Weakly structured; dominated by high variance and irreducible noise."
    else:
        verdict = "Fundamentally random noise; behaves like irreducible credit risk variance."
        
    verdict_df = pd.DataFrame({
        "Attribute": ["Stealth Predictability Verdict", "ROC-AUC", "PR-AUC", "Baseline Default Rate"],
        "Value": [verdict, f"{auc_stealth:.5f}", f"{pr_auc_stealth:.5f}", f"{y_test_stealth.mean():.4f}"]
    })
    verdict_df.to_csv(TABLES_DIR / "noise_vs_structure_verdict.csv", index=False)
    logger.info("Noise vs Structure test completed successfully.")

if __name__ == "__main__":
    run_noise_test()
