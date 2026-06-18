"""
stealth_detection_experiments.py — Experiments to improve stealth default detection using interaction features and segment-specific models.
"""

import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from lightgbm import LGBMClassifier

# Configure project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from configs.credit_config import SEED, OUTPUT_DIR, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("CRA.StealthDetectionExperiments")

# Setup output folders
AN_DIR = PROJECT_ROOT / "systems" / "credit_risk" / "cr_analysis"
TABLES_DIR = AN_DIR / "outputs" / "tables"
DATA_DIR = AN_DIR / "outputs" / "data"

def load_data():
    engineered_path = OUTPUT_DIR / "engineered_data.parquet"
    df_all = pd.read_parquet(engineered_path)
    df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])
    df_all['year'] = df_all['issue_d'].dt.year

    train_df = df_all[df_all["year"] <= 2015].sample(100000, random_state=SEED).copy()
    test_df = df_all[df_all["year"] >= 2018].sample(50000, random_state=SEED).copy()
    
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    features_spaces = list(scaler.feature_names_in_)
    features_underscores = [c.replace(' ', '_') for c in features_spaces]

    X_train_full = train_df[features_spaces].fillna(0).copy()
    X_train_full.columns = features_underscores
    
    X_test_full = test_df[features_spaces].fillna(0).copy()
    X_test_full.columns = features_underscores
    
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    
    # Filter features: Exclude lender variables
    group_b_patterns = ["int_rate", "term_months", "installment", "grade"]
    features_borrower = [f for f in features_underscores if not any(pat in f for pat in group_b_patterns)]
    
    X_train_borrower = X_train_full[features_borrower].copy()
    X_test_borrower = X_test_full[features_borrower].copy()
    
    return X_train_borrower, X_test_borrower, y_train, y_test, train_df, test_df

def run_interaction_experiment(X_train, X_test, y_train, y_test):
    logger.info("Running Interaction Features Experiment...")
    
    # 1. Baseline Borrower-Only Model
    clf_base = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1)
    clf_base.fit(X_train, y_train)
    probs_base = clf_base.predict_proba(X_test)[:, 1]
    
    auc_base = roc_auc_score(y_test, probs_base)
    prauc_base = average_precision_score(y_test, probs_base)
    
    prec_b, rec_b, thrs_b = precision_recall_curve(y_test, probs_base)
    f1_b = 2 * (prec_b * rec_b) / (prec_b + rec_b + 1e-8)
    opt_thr_base = thrs_b[np.argmax(f1_b)]
    y_pred_base = (probs_base >= opt_thr_base).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_base).ravel()
    
    # 2. Add Interaction Features
    # cr_hist_years * revol_util
    # cr_hist_years * dti
    # annual_inc * revol_util
    # fico_range_low * revol_util
    # fico_range_low * dti
    X_train_int = X_train.copy()
    X_test_int = X_test.copy()
    
    interactions = {
        "Age_x_Util": ("cr_hist_years", "revol_util"),
        "Age_x_DTI": ("cr_hist_years", "dti"),
        "Income_x_Util": ("annual_inc", "revol_util"),
        "FICO_x_Util": ("fico_range_low", "revol_util"),
        "FICO_x_DTI": ("fico_range_low", "dti")
    }
    
    for name, (col1, col2) in interactions.items():
        if col1 in X_train.columns and col2 in X_train.columns:
            X_train_int[name] = X_train[col1] * X_train[col2]
            X_test_int[name] = X_test[col1] * X_test[col2]
            
    clf_int = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1)
    clf_int.fit(X_train_int, y_train)
    probs_int = clf_int.predict_proba(X_test_int)[:, 1]
    
    auc_int = roc_auc_score(y_test, probs_int)
    prauc_int = average_precision_score(y_test, probs_int)
    
    prec_i, rec_i, thrs_i = precision_recall_curve(y_test, probs_int)
    f1_i = 2 * (prec_i * rec_i) / (prec_i + rec_i + 1e-8)
    opt_thr_int = thrs_i[np.argmax(f1_i)]
    y_pred_int = (probs_int >= opt_thr_int).astype(int)
    tn_i, fp_i, fn_i, tp_i = confusion_matrix(y_test, y_pred_int).ravel()
    
    logger.info(f"Baseline AUC: {auc_base:.5f} | Interaction AUC: {auc_int:.5f} (Delta: {auc_int - auc_base:+.5f})")
    logger.info(f"Baseline PR-AUC: {prauc_base:.5f} | Interaction PR-AUC: {prauc_int:.5f} (Delta: {prauc_int - prauc_base:+.5f})")
    logger.info(f"Baseline FNs: {fn_b} | Interaction FNs: {fn_i} (FN Change: {fn_i - fn_b})")
    
    results = [
        {"Model": "Borrower Baseline", "ROC-AUC": auc_base, "PR-AUC": prauc_base, "False Negatives": int(fn_b), "Threshold": opt_thr_base},
        {"Model": "Borrower + Interactions", "ROC-AUC": auc_int, "PR-AUC": prauc_int, "False Negatives": int(fn_i), "Threshold": opt_thr_int}
    ]
    return pd.DataFrame(results), clf_base, clf_int

def run_segment_experiments(X_train, X_test, y_train, y_test):
    logger.info("Running Segment-Specific Modeling Experiment...")
    
    # We define three segment conditions:
    # 1. Older borrowers (cr_hist_years >= median)
    # 2. Low utilization borrowers (revol_util < median)
    # 3. High income borrowers (annual_inc >= median)
    
    medians = {
        "cr_hist_years": X_train["cr_hist_years"].median(),
        "revol_util": X_train["revol_util"].median(),
        "annual_inc": X_train["annual_inc"].median()
    }
    
    segments = {
        "Older Borrowers": (X_train["cr_hist_years"] >= medians["cr_hist_years"], X_test["cr_hist_years"] >= medians["cr_hist_years"]),
        "Low Util Borrowers": (X_train["revol_util"] < medians["revol_util"], X_test["revol_util"] < medians["revol_util"]),
        "High Income Borrowers": (X_train["annual_inc"] >= medians["annual_inc"], X_test["annual_inc"] >= medians["annual_inc"])
    }
    
    segment_results = []
    
    # Train general model first
    general_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1)
    general_clf.fit(X_train, y_train)
    
    for seg_name, (train_mask, test_mask) in segments.items():
        logger.info(f"Evaluating segment: {seg_name}...")
        
        # Segment subsets
        X_tr_seg = X_train[train_mask]
        y_tr_seg = y_train[train_mask]
        
        X_te_seg = X_test[test_mask]
        y_te_seg = y_test[test_mask]
        
        # 1. General model scored on this segment
        probs_general = general_clf.predict_proba(X_te_seg)[:, 1]
        auc_general = roc_auc_score(y_te_seg, probs_general)
        prauc_general = average_precision_score(y_te_seg, probs_general)
        
        prec_g, rec_g, thrs_g = precision_recall_curve(y_te_seg, probs_general)
        f1_g = 2 * (prec_g * rec_g) / (prec_g + rec_g + 1e-8)
        opt_thr_g = thrs_g[np.argmax(f1_g)] if len(thrs_g) > 0 else 0.5
        y_pred_g = (probs_general >= opt_thr_g).astype(int)
        fn_g = int(confusion_matrix(y_te_seg, y_pred_g).ravel()[2]) if len(np.unique(y_te_seg)) > 1 else 0
        
        # 2. Segment-specific model
        seg_clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=SEED, n_jobs=-1, verbosity=-1)
        seg_clf.fit(X_tr_seg, y_tr_seg)
        probs_seg = seg_clf.predict_proba(X_te_seg)[:, 1]
        
        auc_seg = roc_auc_score(y_te_seg, probs_seg)
        prauc_seg = average_precision_score(y_te_seg, probs_seg)
        
        prec_s, rec_s, thrs_s = precision_recall_curve(y_te_seg, probs_seg)
        f1_s = 2 * (prec_s * rec_s) / (prec_s + rec_s + 1e-8)
        opt_thr_s = thrs_s[np.argmax(f1_s)] if len(thrs_s) > 0 else 0.5
        y_pred_s = (probs_seg >= opt_thr_s).astype(int)
        fn_s = int(confusion_matrix(y_te_seg, y_pred_s).ravel()[2]) if len(np.unique(y_te_seg)) > 1 else 0
        
        segment_results.append({
            "Segment": seg_name,
            "General Model AUC": auc_general,
            "Segment Model AUC": auc_seg,
            "AUC Delta": auc_seg - auc_general,
            "General Model PR-AUC": prauc_general,
            "Segment Model PR-AUC": prauc_seg,
            "PR-AUC Delta": prauc_seg - prauc_general,
            "General Model FNs": fn_g,
            "Segment Model FNs": fn_s,
            "FN Change": fn_s - fn_g
        })
        
    return pd.DataFrame(segment_results)

def main():
    X_train, X_test, y_train, y_test, train_df, test_df = load_data()
    
    # Run Experiments
    df_int, clf_base, clf_int = run_interaction_experiment(X_train, X_test, y_train, y_test)
    df_seg = run_segment_experiments(X_train, X_test, y_train, y_test)
    
    # Save output tables
    df_int.to_csv(TABLES_DIR / "detection_improvement_interactions.csv", index=False)
    df_seg.to_csv(TABLES_DIR / "detection_improvement_segments.csv", index=False)
    
    logger.info("Detection Improvement Study complete.")

if __name__ == "__main__":
    main()
