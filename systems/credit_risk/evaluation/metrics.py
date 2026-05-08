"""
Evaluation and Metrics for Credit Risk Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             precision_recall_curve, confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.credit_config import OUTPUT_DIR, MODEL_DIR, SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_models(models, X_test, y_test, X_test_scaled):
    """Compute performance and calibration metrics for all models."""
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Use scaled data for LR
        X = X_test_scaled if name == 'Logistic Regression' else X_test
        
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        
        auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_prob)
        
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'auc': auc,
            'pr_auc': pr_auc,
            'f1': f1,
            'brier': brier,
            'cm': cm.tolist(),
            'y_prob': y_prob
        }
        
    return results

def plot_calibration(results, y_test):
    """Generate calibration plot."""
    plt.figure(figsize=(10, 7))
    for name, res in results.items():
        prob_true, prob_pred = calibration_curve(y_test, res['y_prob'], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title('Probability Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "calibration_plot.png")
    plt.close()

def generate_evaluation_report(results):
    """Document evaluation findings."""
    report_path = OUTPUT_DIR / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("# Institutional Evaluation Report - Phase 1\n\n")
        f.write("## Model Performance Comparison (Test Set 2018)\n\n")
        
        metrics_df = pd.DataFrame({
            name: {
                'ROC-AUC': f"{res['auc']:.4f}",
                'PR-AUC': f"{res['pr_auc']:.4f}",
                'F1-Score': f"{res['f1']:.4f}",
                'Brier Score': f"{res['brier']:.4f}"
            } for name, res in results.items()
        }).T
        
        f.write(metrics_df.to_markdown() + "\n\n")
        
        f.write("## Confusion Matrices\n")
        for name, res in results.items():
            cm = np.array(res['cm'])
            f.write(f"### {name}\n")
            f.write(f"```\n{cm}\n```\n")
            f.write(f"- TN: {cm[0,0]}, FP: {cm[0,1]}\n")
            f.write(f"- FN: {cm[1,0]}, TP: {cm[1,1]}\n\n")
            
        f.write("## Calibration Analysis\n")
        f.write("The Brier score and calibration curve (saved as `calibration_plot.png`) indicate how well the predicted probabilities match actual default rates.\n")
        
    logger.info(f"Evaluation report generated at {report_path}")

if __name__ == "__main__":
    # Load test data
    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
    y_test = pd.read_parquet(OUTPUT_DIR / "y_test.parquet").iloc[:, 0]
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    X_test_scaled = scaler.transform(X_test)
    
    # Load models
    models = {}
    for p in MODEL_DIR.glob("*.joblib"):
        if p.stem != 'scaler':
            name = p.stem.replace('_', ' ').title()
            models[name] = joblib.load(p)
            
    results = evaluate_models(models, X_test, y_test, X_test_scaled)
    plot_calibration(results, y_test)
    generate_evaluation_report(results)
    logger.info("Evaluation complete.")
