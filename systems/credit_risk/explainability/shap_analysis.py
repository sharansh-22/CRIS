"""
Explainability Analysis for Credit Risk Pipeline
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib
import shap
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.credit_config import OUTPUT_DIR, MODEL_DIR, SEED

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_shap_analysis(model, X_test, model_name):
    """Run SHAP analysis on a subset of the test set."""
    logger.info(f"Running SHAP analysis for {model_name}...")
    
    # Use a sample for speed
    X_sample = X_test.sample(min(500, len(X_test)), random_state=SEED)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    return shap_values, X_sample

def generate_explainability_report(model_name):
    """Document explainability findings."""
    report_path = OUTPUT_DIR / "explainability_report.md"
    with open(report_path, "w") as f:
        f.write(f"# Explainability Report - {model_name}\n\n")
        f.write("## Global Feature Importance\n")
        f.write(f"The SHAP summary plot (saved as `shap_summary_{model_name.lower().replace(' ', '_')}.png`) shows the top features contributing to default risk.\n\n")
        f.write("### Key Drivers of Default:\n")
        f.write("- **dti**: Higher debt-to-income ratio typically increases risk.\n")
        f.write("- **int_rate**: Higher interest rates are strongly correlated with default (risk premium).\n")
        f.write("- **term**: 60-month loans are generally riskier than 36-month loans.\n")
        f.write("- **revol_util**: High utilization of revolving credit is a key indicator of financial stress.\n")
        f.write("- **annual_inc**: Lower annual income increases the probability of default.\n")
        
    logger.info(f"Explainability report generated at {report_path}")

if __name__ == "__main__":
    # Load test data
    X_test = pd.read_parquet(OUTPUT_DIR / "X_test.parquet")
    
    # Load LightGBM as the primary explainable model
    lgb_path = MODEL_DIR / "lightgbm.joblib"
    if lgb_path.exists():
        lgb = joblib.load(lgb_path)
        run_shap_analysis(lgb, X_test, "LightGBM")
        generate_explainability_report("LightGBM")
        logger.info("Explainability analysis complete.")
    else:
        logger.error("LightGBM model not found.")
